# -*- coding: utf-8 -*-


import os
import sys
import datetime

from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from IPython.display import display, Image, clear_output
import ipywidgets as widgets

try:
    import cv2
except ImportError:
    warnings.warn("opencv not available!")
import skimage

from . import __version__
from . import *

# def imshow(img):
#     import cv2
#     import IPython
#     _,ret = cv2.imencode('.jpg', img) 
#     i = IPython.display.Image(data=ret)
#     IPython.display.display(i)


class _stereoViewMatch(object):
    def __init__(self, case, config, version=__version__, markParticles=True, increaseContrast=True):
        
        self.case = case
        self.config = config
        self.version = version
        self.markParticles = markParticles
        self.increaseContrast = increaseContrast
        self.cameras = [config.leader, config.follower]

        self.open()
        self.rr = 0

    def open(self):

        fL = files.FindFiles(self.case, self.config.leader, self.config, self.version)
        assert(len(fL.listFiles("level0")) == 1), f"Please select case so that this {fL.fnamesPattern.level0} results only in one file"
        fL1 = files.Filenames(fL.listFiles("level0")[0], self.config)

        #  open all the files
        fnamesLv0 = {}
        self.meta = {}
        self.lv1detect = {}
        self.videos = {}
        self.idDiffs = {}
        self.imagesL1detect = {}

        fnamesLv0[self.config.leader] = fL.listFiles("level0")[0]
        fnames0F = fL1.filenamesOtherCamera(graceInterval=-1, level="level0")
        if len(fnames0F)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnames0F[1:])
            fnames0F = fnames0F[:1]
        fnamesLv0[self.config.follower] = fnames0F[0]

        self.meta[self.config.leader] = tools.open_mfmetaFrames(fL.listFiles("metaFrames"), self.config)
        self.lv1detect[self.config.leader] = tools.open_mflevel1detect(fL.listFiles("level1detect"), self.config)
        self.imagesL1detect[self.config.leader] = fL.listFiles("imagesL1detect")[0]
        fnamesMF = fL1.filenamesOtherCamera(graceInterval=-1, level="metaFrames")
        if len(fnamesMF)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnamesMF[1:])
            fnamesMF = fnamesMF[:1]
        fnames1F = fL1.filenamesOtherCamera(graceInterval=-1, level="level1detect")
        if len(fnames1F)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnames1F[1:])
            fnames1F = fnames1F[:1]

        self.meta[self.config.follower] = tools.open_mfmetaFrames(fnamesMF, self.config)
        self.lv1detect[self.config.follower] = tools.open_mflevel1detect(fnames1F, self.config)
        
        self.lv1match = tools.open_mflevel1match(fL.listFiles("level1match"), self.config)
        self.imagesL1detect[self.config.follower] = fL1.filenamesOtherCamera(graceInterval=-1, level="imagesL1detect")[0]

        print(self.imagesL1detect)
        # #get capture ID diffs
        # self.idDiff, nMatched = tools.estimateCaptureIdDiffCore(*self.meta.values(), "capture_time", nPoints=500, timeDim="record_time")
        # idDiff2, nMatched = tools.estimateCaptureIdDiffCore(*self.lv1detect.values(), "fpid", nPoints=500, timeDim="record_time")
        # self.idDiffs[self.config.follower] = self.idDiff
        # assert  self.idDiff == idDiff2, "estimateCaptureIdDiff did not come to same result for metaFrames and lv1detect"
        self.idDiffs[self.config.follower] = self.idDiff = int(self.lv1match.capture_id.diff("camera").values.flatten()[0])
        self.idDiffs[self.config.leader] = 0


        #remove non moving data
        minMovingPixels = np.array(self.config["minMovingPixels"])
        for camera in self.cameras:
            nChangedPixel = self.meta[camera].nMovingPixel.values
            passesThreshold = nChangedPixel >= minMovingPixels
            print(f"{passesThreshold.any(1).sum()/len(self.meta[camera].capture_time)*100}% frames are moving")
            self.meta[camera] = self.meta[camera].isel(capture_time = passesThreshold.any(1))


        self.uniqueCaptureIds = xr.DataArray(
            np.sort(
                np.unique(
                    np.concatenate(
                        (
                            self.meta[self.config.leader].capture_id, 
                            self.meta[self.config.follower].capture_id-self.idDiff
                            )
                        )
                    )
                ),
            dims=["merged_record_id"]
            )
        self.captureTimes = {}
        for camera in self.cameras:

            self.videos[camera] = av.VideoReaderMeta(
                fnamesLv0[camera], 
                self.meta[camera], 
                lv1detect=self.lv1detect[camera], 
                lv1match=self.lv1match.sel(camera=camera), 
                imagesL1detect=self.imagesL1detect[camera],
                config=self.config, 
                saveMode=False
            )
            self.captureTimes[camera] = xr.DataArray(
                self.meta[camera].capture_time, 
                coords=[self.meta[camera].capture_id], 
                dims=["capture_id"]
                )

    def get(self, rr):
        self.rr = rr
        thisID = self.uniqueCaptureIds[rr].values

        frame = list()
        metaFrames = []
        lv1detects = []
        lv1matches = []
        particles = {}
        for camera in self.cameras:
            particles[camera] = []
            if (thisID + self.idDiffs[camera]) in self.meta[camera].capture_id:
                captureTime = self.captureTimes[camera].sel(capture_id=thisID+ self.idDiffs[camera]).values
#                 captureTime = 0
                # print(f"found record {rr} in {camera} data at {captureTime}")
    
                # print(camera, captureTime)
                res, self.frame1, meta1, meta2, meta3, particle1 = self.videos[camera].getFrameByCaptureTimeWithParticles(captureTime, markParticles=self.markParticles, highlightPid="meta", increaseContrast=self.increaseContrast)
                if self.frame1 is not None:
                    frame.append(self.frame1)
                if particle1 is not None:
                    particles[camera] = particle1
                    # if meta2 is not None:
                        
                        # for fpid in meta2.fpid.values:
                            # thisMeta2 = meta2.sel(fpid=fpid)
                            # cnt = thisMeta2.cnt.values[thisMeta2.cnt.values[...,0]>=0]
                            # thisFrame = self.frame1[self.config.height_offset:,:,0]
                            # particleBoxMask, xOffset, yOffset = detection.extractRoi(thisMeta2.roi.values, cv2.fillPoly(np.zeros_like(thisFrame), pts =np.array([cnt], dtype=np.int32), color=255))
                            # particleBox, xOffset, yOffset = detection.extractRoi(thisMeta2.roi.values, thisFrame)
                            # particleBoxCropped = deepcopy(particleBox)
                            # particleBoxCropped[particleBoxMask == 0] = 255

                            # # particleBox = np.concatenate((particleBox, particleBoxMask),-1)
                            # particleBox = np.hstack((particleBox, particleBoxCropped ) )

                            # if np.prod(particleBoxCropped.shape) > 100:
                            #     factor = 2
                            # else:
                            #     factor = 4
                            # particleBox = skimage.transform.resize(particleBox,
                            #    np.array(particleBox.shape[:2])*factor,
                            #    mode='edge',
                            #    anti_aliasing=False,
                            #    anti_aliasing_sigma=None,
                            #    preserve_range=True,
                            #    order=0) 

                            # if meta3 is None:
                            #     color = (255,0,0)
                            # elif thisMeta2.pid.values in meta3.pid.values:
                            #     color = (0,0,0)
                            # else:
                            #     color = (255,0,0)

                            
                            # cv2.putText(particleBox, str(thisMeta2.pid.values), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                            

                else:
                    frame.append(
                        np.zeros((
                            self.config.frame_height+self.config.height_offset,
                            self.config.frame_width,
                            3
                        ), dtype=int) + 130
                    )
            else:
                # print(camera, "not found",(thisID + self.idDiffs[camera]) )
                # print(f"did not find record {rr} in {camera} data")
                frame.append(
                    np.zeros((
                        self.config.frame_height+self.config.height_offset,
                        self.config.frame_width,
                        3
                    ), dtype=int) 
                )
                meta1 =  meta2 = meta3 = None

            frameBorder = np.zeros((self.config.frame_height+self.config.height_offset, 10, 3), dtype=int)

            frame.append(frameBorder)
            metaFrames.append(meta1)
            lv1detects.append(meta2)
            lv1matches.append(meta3)

        try:
            lv1matches = xr.concat(lv1matches, dim="camera")
        except TypeError:
            lv1matches = None

        frame = np.concatenate(frame, axis=1)

        return frame, metaFrames, lv1detects, lv1matches, particles
    
    def next(self):
        newrr = self.rr+1
        if newrr >= len(self.uniqueCaptureIds):
            print("end of movie file")
            return None, None, None, None
        return self.get(newrr)
    
    def previous(self):
        newrr = self.rr-1
        if newrr < 0:
            print("beginning of movie file")
            return None, None, None, None
        return self.get(newrr)

    def nextCommon(self):
        while True:
            frame, metaFrames, lv1detects, lv1match, particles = self.next()
            if np.sum([m is None for m in metaFrames])==1:
                continue
            else:
                break

        return frame, metaFrames, lv1detects, lv1match, particles

    def previousCommon(self):
        while True:
            frame, metaFrames, lv1detects, lv1match, particles = self.previous()
            if np.sum([m is None for m in metaFrames])==1:
                continue
            else:
                break
        return frame, metaFrames, lv1detects, lv1match, particles
   

    def nextMatch(self):
        rr = self.rr
        rrs = range(rr+1,len(self.uniqueCaptureIds))
        if len(rrs) == 0:
            print("end of movie file")
            return None, None, None, None, None
        for newrr in rrs:
            thisID = self.uniqueCaptureIds[newrr].values
            if thisID in self.lv1match.capture_id.sel(camera=self.config.leader).values:
                break
        return self.get(newrr)

    def previousMatch(self):
        rr = self.rr
        rrs = range(rr-1,-1, -1)
        if len(rrs) == 0:
            print("start of movie file")
            return None, None, None, None, None
        for newrr in rrs:
            thisID = self.uniqueCaptureIds[newrr].values
            if thisID in self.lv1match.capture_id.sel(camera=self.config.leader).values:
                break
        return self.get(newrr)


    def close(self, ):
        for camera in self.cameras:
            self.meta[camera].close()
            self.lv1detect[camera].close()
            self.videos[camera].release()


class matchGUI():
    def __init__(self, case, config, markParticles=True, increaseContrast=True):
    
        self.sv = _stereoViewMatch(case, config, markParticles=markParticles, increaseContrast=increaseContrast)
        return 
        
    def updateHandles(self, frame, metaFrames, lv1detects, lv1matches, particles):

        self.metaFrames = metaFrames
        self.lv1detects = lv1detects
        self.lv1matches = lv1matches
        
        cc = 0
        mII = []
        if (self.lv1detects is not None) and (self.lv1detects[cc] is not None):
            
            for pid in self.lv1detects[cc].pid.values:
                mm = np.where(self.sv.lv1match.isel(camera=cc).pid == pid)[0]
                if len(mm) > 0:
                    mII.append(mm)
        if len(mII)>0:
            lv1match = self.sv.lv1match.isel(fpair_id=np.concatenate(mII))
        else:
            lv1match = None


        _, frame = cv2.imencode('.png', frame)
        try:
            self.display_handle.update(Image(data=frame.tobytes())) 
        except AttributeError:
            self.display_handle = display(Image(data=frame.tobytes()),display_id=True)
            
        self.setNN(self.sv.rr)
        with self.out:
            self.out.clear_output()

            if particles is not None:
                for k in particles:
                    for pid, part in particles[k].items():
                        print(k, pid)
                        tools.displayImage(part, rescale=4)

            if metaFrames[0] is None:
                c0, i0 = "n/a", "n/a"
            else:
                c0, i0 = metaFrames[0].capture_time.values, metaFrames[0].capture_id.values
            if metaFrames[1] is None:
                c1, i1 = "n/a", "n/a"
            else:
                c1, i1 = metaFrames[1].capture_time.values, metaFrames[1].capture_id.values
            
            print("leader:", c0,i0, "follower:", c1,i1)
            if lv1match is not None:
                Zdiff = lv1match.position3D.isel(position3D_elements=[2,3], drop=True).diff("position3D_elements").values.flatten()
                print(
                    lv1match.pair_id.values.flatten(),
                    ["%.7f score"%(l) for l in lv1match.matchScore.values],
                    ["%.i ms"%(l/1e6) for l in lv1match.capture_time.diff("camera").values.astype(int).flatten() ] ,
                    ["%.2f y"%(l) for l in lv1match.roi.diff("camera").sel(ROI_elements="y", drop=True).values.flatten()],
                    ["%.2f h"%(l) for l in lv1match.roi.diff("camera").sel(ROI_elements="h", drop=True).values.flatten()], 
                    ["%.2f Z"%z for z in Zdiff]
                    )
                print("blur", lv1match.blur.values)
                print("Dmax", lv1match.Dmax.values)
                print("#"*100)
            if (lv1detects[0] is not None) and (lv1detects[1] is not None) and (lv1detects[0]["pid"].shape == lv1detects[1]["pid"].shape):
                print(f"{'pid'.ljust(20)}: D {'X'.ljust(23)}, L {str(lv1detects[0]['pid'].values).ljust(23)}, F {str(lv1detects[1]['pid'].values).ljust(23)}") 
                for i in lv1detects[0].data_vars:
                    if i in ["touchesBorder", "pixPercentiles", "nThread", "record_id", "cnt"]: continue
                    print(f"{i.ljust(20)}: D {str(lv1detects[0][i].values - lv1detects[1][i].values).ljust(23)}, L {str(lv1detects[0][i].values).ljust(23)}, F {str(lv1detects[1][i].values).ljust(23)}") 




    def getNN(self):
        nn = int(self.texts[0].get_interact_value())
        return nn
    
    def setNN(self, nn):
        self.texts[0].value = str(nn)

    def createGUI(self, pid=0):
            
        self.out = widgets.Output()

        layout = widgets.Layout(width='auto', height='30px') #set width and height

        buttonNextMatch = widgets.Button(description='>>>', layout=layout)
        buttonPrevMatch = widgets.Button(description='<<<', layout=layout)
        buttonNextMatch.on_click(lambda x: self.updateHandles(*self.sv.nextMatch()))
        buttonPrevMatch.on_click(lambda x: self.updateHandles(*self.sv.previousMatch()))

        buttonNext = widgets.Button(description='>>', layout=layout)
        buttonPrev = widgets.Button(description='<<', layout=layout)
        buttonNext.on_click(lambda x: self.updateHandles(*self.sv.nextCommon()))
        buttonPrev.on_click(lambda x: self.updateHandles(*self.sv.previousCommon()))

        buttonNextFrame = widgets.Button(description='>', layout=layout)
        buttonPrevFrame = widgets.Button(description='<', layout=layout)
        buttonNextFrame.on_click(lambda x: self.updateHandles(*self.sv.next()))
        buttonPrevFrame.on_click(lambda x: self.updateHandles(*self.sv.previous()))

        
        buttons = [buttonPrevMatch,buttonNextMatch,buttonPrev,buttonNext,buttonPrevFrame,buttonNextFrame]
        self.texts = []

        self.texts.append(widgets.Text(
            value=str(pid),
            description=f"{len(self.sv.uniqueCaptureIds)} tot. ids",
            disabled=False,
            width='auto',
        )
                        )
        
        load = widgets.Button(description='Load', layout=layout)
        load.on_click(lambda x: self.updateHandles(*self.sv.get(self.getNN())))
        self.texts.append(load)



        display_handle=None
        self.updateHandles(*self.sv.get(0))

        self.statusP = widgets.HTML(
            value="-",
        )

        # displaying button and its output together
        buttonsH = widgets.HBox(buttons)
        statusH = widgets.HBox(self.texts)

        return widgets.VBox([statusH,buttonsH, self.out])



class _stereoViewDetect(object):
    def __init__(self, case, config, version=__version__, markParticles=True):
        
        self.case = case
        self.config = config
        self.version = version
        self.cameras = [config.leader, config.follower]
        self.version = version
        self.markParticles = markParticles
        
        self.this_capture_time = {}
        self.this_record_time = {}

        self.open()

    def open(self):

        fL = files.FindFiles(self.case, self.config.leader, self.config, self.version)
        assert(len(fL.listFiles("level0")) == 1), f"Please select case so that this {fL.fnamesPattern.level0} results only in one file"
        fL1 = files.Filenames(fL.listFiles("level0")[0], self.config)

        #  open all the files
        fnamesLv0 = {}
        self.meta = {}
        self.lv1detect = {}
        self.videos = {}
        self.idDiffs = {}

        fnamesLv0[self.config.leader] = fL.listFiles("level0")[0]
        fnames0F = fL1.filenamesOtherCamera(graceInterval=-1, level="level0")
        if len(fnames0F)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnames0F[1:])
            fnames0F = fnames0F[:1]
        fnamesLv0[self.config.follower] = fnames0F[0]

        self.meta[self.config.leader] = tools.open_mfmetaFrames(fL.listFiles("metaFrames"), self.config)
        self.lv1detect[self.config.leader] = tools.open_mflevel1detect(fL.listFiles("level1detect"), self.config)
        
        fnamesMF = fL1.filenamesOtherCamera(graceInterval=-1, level="metaFrames")
        if len(fnamesMF)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnamesMF[1:])
            fnamesMF = fnamesMF[:1]
        fnames1F = fL1.filenamesOtherCamera(graceInterval=-1, level="level1detect")
        if len(fnames1F)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnames[1:])
            fnames1F = fnames1F[:1]

        self.meta[self.config.follower] = tools.open_mfmetaFrames(fnamesMF, self.config)
        self.lv1detect[self.config.follower] = tools.open_mflevel1detect(fnames1F, self.config)
        

        self.index = {}
        for camera in self.cameras:
            self.videos[camera] = av.VideoReaderMeta(
                fnamesLv0[camera], 
                self.meta[camera], 
                lv1detect=self.lv1detect[camera], 
                lv1match=None, 
                config=self.config, 
                saveMode=False
            )

            self.index[camera] = range(len(self.meta[camera].capture_time))

        return

    def get(self, rrs):
        self.rrs = rrs

        frame = list()
        metaFrames = []
        lv1detects = []
        
        for camera in self.cameras:
            rr = rrs[camera]
            if rr in self.index[camera]:
#                 captureTime = 0
                # print(f"found record {rr} in {camera} data at {captureTime}")
    
    
                res, self.frame1, meta1, meta2, _ = self.videos[camera].getFrameByIndexWithParticles(rr, markParticles=self.markParticles)
                if self.frame1 is not None:
                    frame.append(self.frame1)
                else:
                    frame.append(
                        np.zeros((
                            self.config.frame_height+self.config.height_offset,
                            self.config.frame_width,
                            3
                        ), dtype=int) + 130
                    )
                self.this_capture_time[camera] = meta1.capture_time.values
                self.this_record_time[camera] = meta1.record_time.values
            else:
                print(f"did not find record {rr} in {camera} data")
                frame.append(
                    np.zeros((
                        self.config.frame_height+self.config.height_offset,
                        self.config.frame_width,
                        3
                    ), dtype=int) 
                )
                meta1 =  meta2 = None
                self.this_capture_time[camera] = None
                self.this_record_time[camera] = None
            frame.append(np.zeros((self.config.frame_height+self.config.height_offset, 10, 3), dtype=int))
            metaFrames.append(meta1)
            lv1detects.append(meta2)

        frame = np.concatenate(frame, axis=1)
        return frame, metaFrames, lv1detects
    
    def next(self, camera):
        if camera == "all":
            cameras = self.cameras
        else: 
            cameras = [camera]
        newrr = deepcopy(self.rrs)
        
        for camera in cameras:
            newrr[camera] += 1
            if newrr[camera] >= len(self.index[camera]):
                print("end of movie file", newrr)
                return None, None, None
        return self.get(newrr)
    
    def previous(self, camera):
        if camera == "all":
            cameras = self.cameras
        else: 
            cameras = [camera]
        newrr = deepcopy(self.rrs)
        
        for camera in cameras:
            newrr[camera] -= 1
            if newrr[camera] < 0:
                print("beginning of movie file")
                return None, None, None
        return self.get(newrr)

    def matchTime(self, time1, camera):
        refCam = tools.getOtherCamera(self.config, camera)
        nn = self.rrs 
        tDiff = np.abs(
            getattr(self, f"this_{time1}")[refCam] - self.meta[camera][time1].values)
        minII = np.argmin(tDiff)
        nn[camera] = minII
#         print('found time difference', time1, camera, nn)

        return self.get(nn)

    


    def close(self, ):
        for camera in self.cameras:
            self.meta[camera].close()
            self.lv1detect[camera].close()
            self.videos[camera].release()


class manualMatchGUI():
    def __init__(self, case, config, markParticles=True):
    
        self.sv = _stereoViewDetect(case, config, markParticles=markParticles)
        return 
        
    def updateHandles(self, frame, metaFrames, lv1detects):

        self.metaFrames = metaFrames
        self.lv1detects = lv1detects
        
        cc = 0
        mII = []

        if frame is None:
            return

        _, frame = cv2.imencode('.jpeg', frame)
        try:
            self.display_handle.update(Image(data=frame.tobytes())) 
        except AttributeError:
            self.display_handle = display(Image(data=frame.tobytes()),display_id=True)
            
        self.setNN(self.sv.rrs)
        with self.out:
            self.out.clear_output()
            if metaFrames[0] is None:
                c0, i0 = "n/a", "n/a"
            else:
                c0, i0 = metaFrames[0].capture_time.values, metaFrames[0].capture_id.values
            if metaFrames[1] is None:
                c1, i1 = "n/a", "n/a"
            else:
                c1, i1 = metaFrames[1].capture_time.values, metaFrames[1].capture_id.values
            
            print("leader:", c0,i0, "follower:", c1,i1)

            if (lv1detects[0] is not None) and (lv1detects[1] is not None):
                print(f"{'pid'.ljust(20)}: D {'X'.ljust(23)}, L {str(lv1detects[0]['pid'].values).ljust(23)}, F {str(lv1detects[1]['pid'].values).ljust(23)}") 
                for i in lv1detects[0].data_vars:
                    if i in ["touchesBorder", "pixPercentiles", "nThread", "record_id", ]: continue
                    print(f"{i.ljust(20)}: D {str(lv1detects[0][i].values - lv1detects[1][i].values).ljust(23)}, L {str(lv1detects[0][i].values).ljust(23)}, F {str(lv1detects[1][i].values).ljust(23)}") 

        return

    def getNN(self):
        nn = {}
        for ii, camera in enumerate(self.sv.config.instruments):
            nn[camera] = int(self.texts[ii].get_interact_value())
        return nn
    
    def setNN(self, nn):
        for ii, camera in enumerate(self.sv.config.instruments):
            self.texts[ii].value = str(nn[camera])

    def createGUI(self):
            
            
        self.out = widgets.Output()

        layout = widgets.Layout(width='auto', height='30px') #set width and height

        buttons = []

#             buttonNext = widgets.Button(description='>>', layout=layout)
#             buttonPrev = widgets.Button(description='<<', layout=layout)
#             buttonNext.on_click(lambda x: self.updateHandles(*self.sv.nextCommon()))
#             buttonPrev.on_click(lambda x: self.updateHandles(*self.sv.previousCommon()))

        buttonNextFrame = widgets.Button(description=f'>', layout=layout)
        buttonPrevFrame = widgets.Button(description=f'<', layout=layout)
        buttonNextFrame.on_click(lambda x: self.updateHandles(*self.sv.next("all")))
        buttonPrevFrame.on_click(lambda x: self.updateHandles(*self.sv.previous("all")))
        buttons += [buttonPrevFrame,buttonNextFrame]


        buttonNextFrame1 = widgets.Button(description=f'{self.sv.config.instruments[0]} >', layout=layout)
        buttonPrevFrame1 = widgets.Button(description=f'{self.sv.config.instruments[0]} <', layout=layout)
        buttonNextFrame1.on_click(lambda x: self.updateHandles(*self.sv.next(deepcopy(self.sv.config.instruments[0]))))
        buttonPrevFrame1.on_click(lambda x: self.updateHandles(*self.sv.previous(deepcopy(self.sv.config.instruments[0]))))
        buttons += [buttonPrevFrame1,buttonNextFrame1]
        
        buttonNextFrame2 = widgets.Button(description=f'{self.sv.config.instruments[1]} >', layout=layout)
        buttonPrevFrame2 = widgets.Button(description=f'{self.sv.config.instruments[1]} <', layout=layout)
        buttonNextFrame2.on_click(lambda x: self.updateHandles(*self.sv.next( self.sv.config.instruments[1])))
        buttonPrevFrame2.on_click(lambda x: self.updateHandles(*self.sv.previous( self.sv.config.instruments[1])))
        buttons += [buttonPrevFrame2,buttonNextFrame2]

        self.texts = []

        startId = {}
        for ii, camera in enumerate(self.sv.config.instruments):
            startId[camera] = 0
            
        for ii, camera in enumerate(self.sv.config.instruments):
            self.texts.append(widgets.Text(
                value=str(startId[camera]),
                description=f"{camera}: {len(self.sv.lv1detect[camera].capture_id)} tot. ids",
                disabled=False,
                width=500,
            ))

        load = widgets.Button(description='Load', layout=layout)
        load.on_click(lambda x: self.updateHandles(*self.sv.get(self.getNN())))
        self.texts.append(load)

        matching = []

        match1= widgets.Button(description='Match CT of %s'%self.sv.config.instruments[0], layout=layout)
        match1.on_click(lambda _: self.updateHandles(*self.sv.matchTime('capture_time', self.sv.config.instruments[0])))
        matching.append(match1)
        match1a= widgets.Button(description='Match CT of %s'%self.sv.config.instruments[1], layout=layout)
        match1a.on_click(lambda _: self.updateHandles(*self.sv.matchTime('capture_time', self.sv.config.instruments[1])))
        matching.append(match1a)

        match2= widgets.Button(description='Match RT of %s'%self.sv.config.instruments[0], layout=layout)
        match2.on_click(lambda _: self.updateHandles(*self.sv.matchTime('record_time', self.sv.config.instruments[0])))
        matching.append(match2)
        match2a= widgets.Button(description='Match RT of %s'%self.sv.config.instruments[1], layout=layout)
        match2a.on_click(lambda _: self.updateHandles(*self.sv.matchTime('record_time', self.sv.config.instruments[1])))
        matching.append(match2a)


        display_handle=None
        self.updateHandles(*self.sv.get(startId))

        self.statusP = widgets.HTML(
            value="-",
        )

        
        
        
        # displaying button and its output together
        buttonsH = widgets.HBox(buttons)
        statusH = widgets.HBox(self.texts)
        matchingH = widgets.HBox(matching)
        
        return widgets.VBox([statusH,buttonsH, matchingH, self.out])



