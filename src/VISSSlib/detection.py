# -*- coding: utf-8 -*-

from . import __version__
from . import av
from . import files
from . import metadata
from . import tools
from copy import deepcopy
import sys
import os
import itertools
import subprocess
import warnings

# import matplotlib.pyplot as plt
import IPython.display
import numpy as np
import xarray as xr
import scipy.stats
#import av
import bottleneck as bn
import warnings
try:
    import cv2
except ImportError:
    warnings.warn("opencv not available!")
import skimage
import zipfile

import logging
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


#for performance
logDebug = log.isEnabledFor(logging.DEBUG)

'''
changes in 11/23

blurSigma=1 isntead 1.5, #makes particle a bit sharper, no disadvantage found so far
minArea=0 instead 1, #changed!!
minDmax=0 instead 2, #changed!!
doubleDynamicRange=True, # does improve detection, problematic for wings
joinCannyEdges=False, # turned out to be not working
check4childCntLength = True changed! avoids erode filter of fgmask
bug fixes regarding handling of cnt children
'''


minCntSize = 3

class detectedParticles(object):
    def __init__(self,
                 config,
                 pidOffset=0,
                 trainingSize=500,
                 verbosity=0,
                 composite=True,
                 minContrast=20,
                 minDmax=0,
                 minBlur=10,
                 minArea=0,
                 erosionTestThreshold=0.06,
                 height_offset=64,
                 maskCorners=None,
                 cropImage=None,  # (offsetX, offsetY)
                backSubKW={"dist2Threshold": 400,
                               "detectShadows": False, "history": 100},
                backSub=cv2.createBackgroundSubtractorKNN,
                 applyCanny2Particle=True,  # much faster than 2 whole Frame!
                 dilateIterations=1,
                 blurSigma=1,
                 minAspectRatio=None,  # testing only
                 dilateErodeFgMask=False,            
                 joinCannyEdges=False, 
                 check4childCntLength = True,
                 doubleDynamicRange = True,
                 testing=[], 
                 ):

        self.version = __version__.split(".")[0]

        self.verbosity = verbosity
        self.composite = composite

        self.minDmax = minDmax
        self.minContrast = minContrast
        self.maxNParticle = config.level1detect.maxMovingObjects
        self.minBlur = minBlur
        self.minArea = minArea
        self.erosionTestThreshold = erosionTestThreshold
        self.height_offset = height_offset  # height of sttaus bar
        self.maskCorners = maskCorners
        self.cropImage = cropImage

        self.all = {}
        self.lastFrame = {}
        self.pp = pidOffset
        self.fgMask = None

        self.nMovingPix = 0
        self.nMovingPix2 = 0  # step2
        self.blur = 0
        self.capture_id = None
        self.record_id = None
        self.capture_time = None
        self.record_time = None
        self.nThread = None
        self.nParticle = 0

        # history 500, threshold=400
        self.backSub = backSub(**backSubKW)

        self.applyCanny2Particle = applyCanny2Particle
        self.dilateIterations = dilateIterations
        self.blurSigma = blurSigma
        self.minAspectRatio = minAspectRatio
        self.dilateErodeFgMask = dilateErodeFgMask
        self.doubleDynamicRange = doubleDynamicRange
        self.joinCannyEdges =  joinCannyEdges
        self.check4childCntLength = check4childCntLength
        self.testing = testing
        return

    def update(self, frame, pp, capture_id, record_id, capture_time, record_time, nThread, training=False, blockingThresh=None):

        self.capture_id = capture_id
        self.record_id = record_id
        self.capture_time = capture_time
        self.record_time = record_time
        self.nThread = nThread
        self.lastFrame = {}
        if (self.verbosity > 20):
            print("particles.update", "FRAME", pp,  'Start %s' % 'update')

        self.frame = frame[self.height_offset:]

        # convert to gray scale if required
        if len(self.frame.shape) == 3:
            self.frame = av.cvtColor(self.frame)

        if self.maskCorners is not None:
            self.frame[:self.maskCorners, :self.maskCorners] = 0
            self.frame[-self.maskCorners:, :self.maskCorners] = 0
            self.frame[-self.maskCorners:, -self.maskCorners:] = 0
            self.frame[:self.maskCorners, -self.maskCorners:] = 0

        if self.cropImage is not None:
            offsetX, offsetY = self.cropImage
            self.frame = self.frame[offsetY:-offsetY, offsetX:-offsetX]

        if (not training) and ("input" in self.testing):
            print("SHOWING", "input")
            tools.displayImage(self.frame)

        self.fgMask = self.backSub.apply(self.frame)
        # use only every 10thdata point for speed
        # using frame instead of background saves speed
        self.brightnessBackground = int(np.median(self.frame[::10, ::10]))

        if training:
            return True

        if blockingThresh is not None:
            nBlocked = np.sum(self.frame < blockingThresh)
            print("%%%%%%%%%%%%%%%%%%% pixels blocked", nBlocked, blockingThresh)
            if nBlocked > blockingThresh:
                return True

        # tools.displayImage(self.backSub.getBackgroundImage())
        # self.frame = av.doubleDynamicRange(cv2.bitwise_not(cv2.subtract(self.backSub.getBackgroundImage(), self.frame, )))
#

        # check whether anxthing is moving
        self.nMovingPix = self.fgMask.sum()//255
        if self.nMovingPix == 0:
            print("particles.update", "FRAME", pp, 'capture_time',
                  capture_time, 'nothing is moving')
            if "nonMovingFgMask" in self.testing:
                print("SHOWING", "nonMovingFgMask")
                tools.displayImage(self.fgMask)
            return True

        # it can happen that the background subtraction has little gaps
        # if self.fgMask is not None:
        #     self.fgMask = cv2.dilate(self.fgMask, None, iterations=1)
        #     self.fgMask = cv2.erode(self.fgMask, None, iterations=1)

        if "movingInput" in self.testing:
            print("SHOWING", "movingInput")
            tools.displayImage(self.frame)

        if "fgMaskWithHoles" in self.testing:
            print("SHOWING", "fgMaskWithHoles")
            tools.displayImage(self.fgMask)

        # if self.applyCanny2Frame:
        #     self.fgMask = self.applyCannyFilter(self.frame, self.fgMask)

        #     self.nMovingPix2 = self.fgMask.sum()//255
        #     if self.nMovingPix2 == 0:
        #         print("particles.update", "FRAME", pp,
        #               'nothing is moving after Canny filter')
        #         return True

        # if "fgMask" in self.testing:
        #     print("SHOWING", "fgMask")
        #     tools.displayImage(self.fgMask)

        if ("result" in self.testing) or ("resultAddedOnly" in self.testing):
            self.frame4drawing = av.cvtGray(self.frame)

        # sometimes, there is noise inside detected particles
        if self.dilateErodeFgMask:
            # turns out to be not so smart because it makes holes insides particles smaller
            self.fgMask = cv2.erode(cv2.dilate(self.fgMask, None, iterations=1), None, iterations=1)
        if "fgMask" in self.testing:
            print("SHOWING", "fgMaskWithHoles")
            tools.displayImage(self.fgMask)

        cnts, _ = cv2.findContours(self.fgMask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        self.cnts = list()
        for cc, cnt in enumerate(cnts):
            if cnt.shape[0] >= minCntSize:
                # skip very small particles
                self.cnts.append(cnt)

        self.nParticle = len(self.cnts)
        print(f"particles.update FRAME {pp} capture_time {capture_time} found {self.nParticle} particles")

        if self.nParticle > self.maxNParticle:
            print(f"particles.update FRAME {pp} SKIPPED. more than {self.maxNParticle} particles")
            return False
        added = False
        # loop over the contours
        for cnt in self.cnts:

            added1 = self.add(self.frame, self.fgMask, cnt, 
                             verbosity=self.verbosity, composite=self.composite)
            added = added or added1

        if "result" in self.testing:
            if len(self.cnts) > 0:
                print("SHOWING", "result")
                tools.displayImage(self.frame4drawing)
        if "resultAddedOnly" in self.testing:
            if len(self.cnts) > 0 and added:
                print("SHOWING", "resultAddedOnly")
                tools.displayImage(self.frame4drawing)

        return True

    def applyCannyFilter(self, frame, fgMask, threshold1=0, threshold2=25):
        if logDebug:
            log.debug(tools.concat("applyCannyFilter", threshold1, threshold2, self.doubleDynamicRange))
        if self.doubleDynamicRange:
            frame = av.doubleDynamicRange(frame)

        # if useSkimage:
        #     from skimage import feature
        #     from skimage.util import img_as_ubyte

        #     # Canny filter gets confused if the feature is directly at the edge, so make the moving
        #     # mask a little larger
        #     fgMask4Canny = (cv2.dilate(
        #         fgMask, None, iterations=dilateIterations)//255).astype(bool)
        #     # , mask = (self.fgMask//255).astype(bool)))
        #     fgMaskCanny = feature.canny(frame, sigma=2, mask=fgMask4Canny)
        #     fgMaskCanny = img_as_ubyte(fgMaskCanny)

        # else:

        # blur image, required to make algoprithm stable
        #print("self.blurSigma", self.blurSigma)
        if self.blurSigma != 0:
            frame = cv2.GaussianBlur(frame, (0, 0), self.blurSigma)
        # frame = cv2.bilateralFilter(frame, 5,70,2)

        # apply Canny filter, take low limits becuase we are reduced to moving parts
        fgMaskCanny = cv2.Canny(
            frame, threshold1, threshold2, L2gradient=True, apertureSize=3)

        if "debugCanny" in self.testing:
            print("background result fgMask")
            tools.displayImage(fgMask, rescale=4)
            print("canny input frame")
            tools.displayImage(frame, rescale=4)
            print("canny filter fgMaskCanny")
            tools.displayImage(fgMaskCanny, rescale=4)

        if self.joinCannyEdges:
            # note that contour is NOT filled in this case
            fgMaskCanny = joinEdges(fgMaskCanny)
            if "debugCanny" in self.testing:
                print("canny filter fgMaskCanny joinEdges2")
                tools.displayImage(fgMaskCanny, rescale=4)


        else:
            if self.dilateIterations > 0:
                # close gaps by finding contours, dillate, fill, and erode them
                fgMaskCanny = cv2.dilate(
                    fgMaskCanny, None, iterations=self.dilateIterations)
                if "debugCanny" in self.testing:
                    print("canny filter fgMaskCanny dilate")
                    tools.displayImage(fgMaskCanny, rescale=4)

                cnts = cv2.findContours(fgMaskCanny, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
                fgMaskCanny = cv2.fillPoly(fgMaskCanny, pts=cnts, color=255)
                fgMaskCanny = cv2.erode(
                    fgMaskCanny, None, iterations=self.dilateIterations)

            else:
                cnts = cv2.findContours(fgMaskCanny, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
                fgMaskCanny = cv2.fillPoly(fgMaskCanny, pts=cnts, color=255)


            # and condition, i.e. make sure both filters detected something
            if fgMask is not None:
                fgMaskCanny = (fgMask//255) * fgMaskCanny

        if "debugCanny" in self.testing:
            print("final result fgMaskCanny")
            tools.displayImage(fgMaskCanny, rescale=4)

        return fgMaskCanny

    def add(self, frame1, fgMask, cnt, **kwargs):

        added = False
        # check whether it touches border
        roi = tuple(int(b) for b in cv2.boundingRect(cnt))
        frameHeight, frameWidth = frame1.shape[:2]
        touchesBorder = [
            roi[0] == 0,
            (roi[0] + roi[2]) == frameWidth,
            roi[1] == 0,
            (roi[1] + roi[3]) == frameHeight
        ]
        if np.any(touchesBorder):
            print("particles.add", "PID", "n/a",
                  "touches border", touchesBorder)
            print("particles.update", "PID", "n/a", "Not added")
            return added

        # canny filter is expensive. so apply it of part of image taht is moving (plus a little extra otherwise edge detection does not work)
        # in case more than one particle is in one moving area, iterate over all of them
        if self.applyCanny2Particle:
            extra = 10
            particleBoxPlus, xOffset, yOffset, extraRoi = extractRoi(
                roi, frame1, extra=extra)

            fgBoxMaskPlus, xOffset, yOffset, extraRoi = extractRoi(
                roi, fgMask, extra=extra)
            particleBoxMaskPlus = self.applyCannyFilter(
                particleBoxPlus, fgBoxMaskPlus)
            
            if np.sum(particleBoxMaskPlus) == 0:
                print("canny filter did not detect anything")
                return False
            cnts, hierarchy = cv2.findContours(
                particleBoxMaskPlus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts, cntChildren = splitUpConours(cnts, hierarchy)

            frame4sp = particleBoxPlus
            mask4sp = particleBoxMaskPlus

        else:
            cnts = [cnt]   
            frame4sp = frame1
            mask4sp = fgMask
            xOffset = yOffset = 0
            cntChildren = [[]]

        # loop in case more than one particle is in single moving area
        for cnt, cntChild in zip(cnts, cntChildren):

            if len(cnt) < minCntSize:
                continue

            added = False
            newParticle = True

            # try:
            self.lastParticle = singleParticle(self, self.capture_id, self.record_id, self.capture_time, self.record_time,
                                               self.nThread, self.pp, frame4sp, mask4sp, cnt, cntChild, xOffset, yOffset, self.testing, 
                                               **kwargs)
            # except ZeroDivisionError:
            #     self.lastParticle = None
            #     return added, self.lastParticle
            if not self.lastParticle.success:
                if self.verbosity>2: print(f"particles.add PID {str(self.lastParticle.pid)} empty")
            else:

                ratio = self.lastParticle.perimeterEroded/self.lastParticle.perimeter
                if self.lastParticle.Dmax <= self.minDmax:
                    if self.verbosity>2: print(" ".join(["particles.add", "PID", "%i" %
                                       self.lastParticle.pid, "too small", "%.2f" % self.lastParticle.Dmax]))

                elif self.lastParticle.particleContrast < self.minContrast:
                    if self.verbosity>2: print(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid,
                                       "too small minContrast", "%.2f" % self.lastParticle.particleContrast]))

                elif self.lastParticle.blur < self.minBlur:
                    if self.verbosity>2: print(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid,
                                       "too small minBlur", "%.2f" % self.lastParticle.blur]))

                elif self.lastParticle.area <= self.minArea:
                    if self.verbosity>2: print(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid,
                                       "too small area", "%.2f" % self.lastParticle.area]))

                # erosionTestThreshold does not apply to needles or similalrly shaped particles, also does not work to very small particles
                elif (self.lastParticle.aspectRatio[-1] > 0.4) and (self.lastParticle.Dmax > 5) and (ratio < self.erosionTestThreshold):
                    if self.verbosity>2: print(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid, "particle not properly detected",
                                       "%.2f" %ratio, "%.2f" %self.erosionTestThreshold, "%.2f" % self.lastParticle.aspectRatio[-1]]))

                # only for testing
                elif (self.minAspectRatio is not None) and (self.lastParticle.aspectRatio[-1] < self.minAspectRatio):
                    if self.verbosity>2: print(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid, "particle below arThresh",
                                       "%.2f" % self.lastParticle.aspectRatio[-1], "%.2f" % self.minAspectRatio]))

                elif np.prod(self.lastParticle.particleBox.shape) == 0:
                    if self.verbosity>2: print(" ".join(["particles.add", "PID", "%i" %
                                       self.lastParticle.pid, "flat", str(particleFrame.shape)]))

                else:

                    if self.verbosity>2: print(" ".join(["FILTER", "%.2f" % self.lastParticle.particleBox.mean(
                    ), "%.2f" % self.lastParticle.particleBox.max(), "%.2f" % self.lastParticle.particleBox.std(ddof=1)]))

                    self.all[self.pp] = self.lastParticle
                    self.lastFrame[self.pp] = self.lastParticle
                    if self.verbosity>2: print(
                        " ".join(["particles.add", "PID", "%i" % self.lastParticle.pid, "Added"]))
                    self.pp += 1
                    added = True

            if added:
                # print("particles.update")
                # print("TEST4Canny", self.lastParticle.perimeter, self.lastParticle.perimeterEroded, self.lastParticle.perimeterEroded/self.lastParticle.perimeter)

                if ("result" in self.testing) or ("resultAddedOnly" in self.testing) or ("report" in self.testing):
                    print(self.lastParticle)
                if ("result" in self.testing) or ("resultAddedOnly" in self.testing):
                    self.frame4drawing = self.lastParticle.drawContour(
                        self.frame4drawing)
                    self.frame4drawing = self.lastParticle.annotate(
                        self.frame4drawing, extra='%.1f' % self.lastParticle.Dmax)
            else:
                if self.verbosity>2: print("PID %i Not added" % self.lastParticle.pid)

        return added

    def collectResults(self, includeCnts=False):

        fitMethod = ["cv2.fitEllipseDirect", "cv2.fitEllipse",
                      "cv2.minAreaRect"]
        dim2D = ['x', 'y']
        dim3D = ['x', 'y', 'z']
        self.particleProps = xr.Dataset(
            coords={'pid': list(self.all.keys()), 'percentiles': range(10, 100, 10)})
        for key in [
            'Dmax', 'area', "position_fit", "position_upperLeft", "Droi",  "position_circle",
            "Dfit", 'angle', 'perimeter', 'perimeterEroded',
            'pixMin', 'pixMax', 'pixMean', 'pixStd', 'pixSkew', 'pixKurtosis',
            'blur', 'capture_id', 'record_id', 'capture_time',
            'record_time', 'nThread', "aspectRatio","position_centroid"
        ]:
            arr = []
            for i in self.all.values():
                arr.append(getattr(i, key))
            if key in ["position_centroid", "position_upperLeft", "position_circle", "Droi"]:
                self.particleProps[key] = xr.DataArray(
                    arr, coords=[self.particleProps.pid, dim2D], dims=('pid', 'dim2D'))
            # elif key == 'touchesBorder':
            #     self.particleProps[key] = xr.DataArray(arr, coords=[self.particleProps.pid, ['left', 'right','top','bottom']], dims=('pid','side'))
            elif key in ["position_fit", "Dfit"]:
                self.particleProps[key] = xr.DataArray(
                    arr, coords=[self.particleProps.pid, fitMethod, dim2D], dims=('pid', 'fitMethod', 'dim2D'))
            elif key in ["angle", "aspectRatio"]:
                self.particleProps[key] = xr.DataArray(
                    arr, coords=[self.particleProps.pid, fitMethod], dims=('pid', 'fitMethod'))
            else:
                self.particleProps[key] = xr.DataArray(
                    arr, coords=[self.particleProps.pid], dims=('pid'))

            # We do not need 64 bit accuracy here and can save storage space
            if self.particleProps[key].dtype == np.float64:
                self.particleProps[key] = self.particleProps[key].astype(
                    np.float32)
            elif key in ["position_upperLeft", 'pixMin', 'pixMax', 'nThread', "Droi", "position_centroid"]:
                self.particleProps[key] = self.particleProps[key].astype(
                    np.int16)
        if includeCnts:
            arrTmp = []
            arrTmp2 = []
            for i in self.all.values():
                arrTmp.append(getattr(i, "cnt").squeeze())
                cntChild = getattr(i, "cntChild")
                allChildren = []
                if cntChild is not None:
                    for cntChild1 in cntChild:
                        allChildren.append(cntChild1.squeeze())
                        allChildren.append(np.full((1, 2), -99))
                    allChildren = np.concatenate(allChildren, axis=0)
                else:
                    allChildren = np.full((1, 2), -99)
                arrTmp2.append(allChildren)

            arrayLengths = [a.shape[0] for a in arrTmp]
            arrayLengths2 = [a.shape[0] for a in arrTmp2]

            maxLen = max(arrayLengths)
            maxLen2 = max(arrayLengths2)
            arr = np.zeros([len(arrTmp), maxLen, 2], dtype=np.int16) - 99
            arr2 = np.zeros([len(arrTmp2), maxLen2, 2], dtype=np.int16) - 99
            for i, j in enumerate(arrTmp):
                arr[i, 0:len(j)] = j
            for i, j in enumerate(arrTmp2):
                arr2[i, 0:len(j)] = j
            self.particleProps["cnt"] = xr.DataArray(arr, coords=[self.particleProps.pid, np.arange(
                arr.shape[-2], dtype=np.int16), ['x', 'y']], dims=('pid', 'cnt_element', 'position'))
            self.particleProps["cntChildren"] = xr.DataArray(arr2, coords=[self.particleProps.pid, np.arange(
                arr2.shape[-2], dtype=np.int16), ['x', 'y']], dims=('pid', 'cntChildren_element', 'position'))

        percentiles = []
        for i in self.all.values():
            percentiles.append(getattr(i, 'pixPercentiles'))
        self.particleProps['pixPercentiles'] = xr.DataArray(np.array(percentiles).astype(
            np.float32), coords=[self.particleProps.pid, self.particleProps.percentiles])
        return self.particleProps

    @property
    def N(self):
        return len(self.all)

    @property
    def pids(self):
        return list(self.all.keys())


class singleParticle(object):
    def __init__(self, parent, capture_id, record_id, capture_time, record_time, nThread,  pp1, frame1, mask1, cnt, cntChild, xOffset, yOffset, testing, verbosity=0, composite=True):
        self.verbosity = verbosity
        self.pid = pp1  # np.random.randint(-999,-1)
        self.record_id = record_id
        self.capture_id = capture_id
        self.capture_time = capture_time
        self.record_time = record_time
        self.nThread = nThread
        self.cnt = cnt
        self.cntChild = []
        self.composite = composite
        self.version = __version__.split(".")[0]
        self.testing = testing

        self.xOffset = xOffset
        self.yOffset = yOffset

        self.roi = np.array([int(b) for b in cv2.boundingRect(self.cnt)])
        # self.Cx, self.Cy, self.Dx, self.Dy = self.roi

        if parent.joinCannyEdges: # edges are joined too randomly...
            # particleBoxMask1 has neighbouring particles removed, but internal holes closed
            particleBoxMask1 = cv2.fillPoly(
                np.zeros_like(frame1), pts=[cnt], color=255)
            print("complete contour")
            tools.displayImage(particleBoxMask1, rescale=4)

            # thre is always one child
            if len(cntChild) >1:

                for cc, cntChild1 in enumerate(cntChild):
                    #skip tiny ones
                    if len(cntChild1) < minCntSize:
                        continue

                    childMask = cv2.fillPoly(np.zeros_like(mask1), pts=[cntChild1], color=255)
                    childCnt = cv2.drawContours(np.zeros_like(mask1), [cntChild1], -1, np.array((255.,255.,255.)), 1)

                    meanBrightnessMask = np.mean(frame1[childMask==255])
                    meanBrightnessCnt = np.mean(frame1[childCnt==255])
  
                    if (meanBrightnessMask > (meanBrightnessCnt + 5) and (cv2.contourArea(cntChild1) > 2)):
                        print(cc, "hole!", meanBrightnessMask, meanBrightnessCnt, len(cntChild1), cv2.contourArea(cntChild1))
                        particleBoxMask1 = particleBoxMask1 & (255 - childMask)
                        tools.displayImage(childMask, rescale=4)
                        tools.displayImage(cv2.bitwise_and(frame1,frame1,mask = childMask), rescale=4)
                        tools.displayImage(cv2.bitwise_and(frame1,frame1,mask = ~childMask), rescale=4)

            self.particleBoxMask, xo, yo, _ = extractRoi(self.roi, particleBoxMask1)
            self.particleBox, xo, yo, _ = extractRoi(self.roi, frame1)


        else:
            particleBoxMask1 = cv2.fillPoly(np.zeros_like(frame1), pts=[cnt], color=255)

            for cc, cntChild1 in enumerate(cntChild):

                # dont be bothered by too small holes
                if parent.check4childCntLength and ((len(cntChild1) < minCntSize) or (cv2.contourArea(cntChild1) <= 4)):
                    continue
                     # particleBoxHoles is the opposite to particleBoxMask1, internal holes (from movement detection) are open, but neighbor particles are present
                    #use child cnts to make mask with children
                particleBoxHoles = cv2.fillPoly(
                    np.zeros_like(frame1)+255, pts=[cntChild1], color=0)

                if "debugCanny" in self.testing:
                    print("particleBoxHoles",cc, len(cntChild1), cv2.contourArea(cntChild1))
                    tools.displayImage(particleBoxHoles, rescale=4)

                # remove single pixel holes
                if not parent.check4childCntLength:
                    particleBoxHoles = cv2.erode(cv2.dilate(particleBoxHoles, None, iterations=1), None, iterations=1)

                # combine masks, this is the final mask 
                particleBoxMask1 = (
                    (particleBoxMask1/255)*particleBoxHoles).astype(np.uint8)

                self.cntChild.append(cntChild1)

            self.particleBoxMask, xo, yo, _ = extractRoi(self.roi, particleBoxMask1)
            self.particleBox, xo, yo, _ = extractRoi(self.roi, frame1)


        if "particleMask" in parent.testing:
            print("particleBoxMask")
            tools.displayImage(self.particleBoxMask, rescale=4)

        self.particleBoxAlpha = np.stack(
            (self.particleBox, self.particleBoxMask), -1)

        fill_color = 255   # any  color value to fill with
        self.particleBoxCropped = deepcopy(self.particleBox)
        self.particleBoxCropped[self.particleBoxMask == 0] = fill_color
        particleBoxData = self.particleBox[self.particleBoxMask == 255]

        # empty particle
        if len(particleBoxData) == 0:
            self.success = False
            return

        self.pixMin = particleBoxData.min()
        self.pixMax = particleBoxData.max()
        self.pixMean = particleBoxData.mean()
        self.pixPercentiles = np.percentile(
            particleBoxData, range(10, 100, 10))  # , interpolation='nearest'
        self.pixStd = np.std(particleBoxData, ddof=1)

        self.pixSkew = scipy.stats.skew(particleBoxData)
        self.pixKurtosis = scipy.stats.kurtosis(particleBoxData)
        self.particleContrast = parent.brightnessBackground - self.pixMin

        # figure out whether particle was properly detected
        # if not, contour describes only a line which can be detected by
        # looking into how perimeter changes during erosion
        erodeMask = cv2.erode(np.pad(self.particleBoxMask,1), None, 1)
        cntsAfter = cv2.findContours(erodeMask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(cntsAfter) > 0:
            self.perimeterEroded = np.max(
                [cv2.arcLength(c, True) for c in cntsAfter])
        else:
            self.perimeterEroded = 0

        # apply Offsets
        self.roi[0] += self.xOffset
        self.roi[1] += self.yOffset
        if self.verbosity > 1:
            print("singleParticle.__init__", "PID", self.pid,
                  'found singleParticle at' , list(self.roi))

        self.cnt[..., 0] += self.xOffset
        self.cnt[..., 1] += self.yOffset

        if len(self.cntChild)>0:
            for cc in range(len(self.cntChild)):
                self.cntChild[cc][..., 0] += self.xOffset
                self.cntChild[cc][..., 1] += self.yOffset

        self.position_upperLeft = (self.roi[0], self.roi[1])
        self.Droi = (self.roi[2], self.roi[3])

        # estimate Dmax
        self.position_circle, radius = cv2.minEnclosingCircle(self.cnt)
        self.Dmax = 2*radius

        # estimate properties that need shifted position
        self.rect = cv2.minAreaRect(self.cnt)
        center, dims, self.angle = self.rect
        # angle definition depends on opencv version https://github.com/opencv/opencv/issues/19472
        # for newer opencv versions where angle is postive, this makes sure it ranges form 0 to 180
        # angle definition will be tackled at late rprocessing stage becusae full rect information is now saved
        # if dims[1] > dims[0]:
        #     self.angle = self.angle  - 90
        self.position_rect = self.rect[0]
        self.Dfit_rect = self.rect[1]
        self.angle_rect = self.rect[2]

        if len(self.cnt) > 4:  # fitELlipse doesnt work for 4 or less points
            # according to @fitzgibbon_direct_1996 , the direct method is both robust and fast.
            try:
                self.ellipseDirect = cv2.fitEllipseDirect(self.cnt)
            except:
                self.ellipseDirect = (np.nan, np.nan), (np.nan, np.nan), np.nan
            # very poor performance!
            # try:
            #     self.ellipseAMS = cv2.fitEllipseAMS(self.cnt)
            # except:
            #     self.ellipseAMS = (np.nan, np.nan), (np.nan, np.nan), np.nan
            # method used by MASC, often doesn't work
            try:
                self.ellipse = cv2.fitEllipse(self.cnt)
            except:
                self.ellipse = (np.nan, np.nan), (np.nan, np.nan), np.nan

        else:
            self.ellipseDirect = (np.nan, np.nan), (np.nan, np.nan), np.nan
            self.ellipse = (np.nan, np.nan), (np.nan, np.nan), np.nan
            # self.ellipseAMS = (np.nan, np.nan), (np.nan, np.nan), np.nan

        self.position_ellipse = self.ellipse[0]
        self.position_ellipseDirect = self.ellipseDirect[0]
        # self.position_ellipseAMS = self.ellipseAMS[0]

        self.Dfit_ellipse = self.ellipse[1]
        self.Dfit_ellipseDirect = self.ellipseDirect[1]
        # self.Dfit_ellipseAMS = self.ellipseAMS[1]

        self.angle_ellipse = self.ellipse[2]
        self.angle_ellipseDirect = self.ellipseDirect[2]
        # self.angle_ellipseAMS = self.ellipseAMS[2]

        self.position_fit = [self.position_ellipseDirect,
                             self.position_ellipse, self.position_rect]
        self.Dfit = [self.Dfit_ellipseDirect, self.Dfit_ellipse,
                      self.Dfit_rect]
        self.angle = [self.angle_ellipseDirect, self.angle_ellipse,
                       self.angle_rect]

        try:
            AR_ellipse = self.Dfit_ellipse[0]/self.Dfit_ellipse[1]
        except ZeroDivisionError:
            AR_ellipse = np.nan
        try:
            AR_ellipseDirect = self.Dfit_ellipseDirect[0] / \
                self.Dfit_ellipseDirect[1]
        except ZeroDivisionError:
            AR_ellipseDirect = np.nan
        # try:
        #     AR_ellipseAMS = self.Dfit_ellipseAMS[0]/self.Dfit_ellipseAMS[1]
        # except ZeroDivisionError:
        #     AR_ellipseAMS = np.nan
        try:
            AR_rect = min(self.Dfit_rect)/max(self.Dfit_rect)
        except ZeroDivisionError:
            AR_rect = np.nan
        self.aspectRatio = (AR_ellipseDirect,
                            AR_ellipse, AR_rect)

        self.area = cv2.contourArea(self.cnt)
        M = cv2.moments(self.cnt)
        # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
        try:
            self.position_centroid = (
                int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        except ZeroDivisionError:
            self.position_centroid = 0, 0

        self.perimeter = cv2.arcLength(self.cnt, True)
        # data type cv2.CV_16S requried to avoid overflow
        # https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        self.blur = cv2.Laplacian(self.particleBox, cv2.CV_64F).var(ddof=1)
        self.success = True

        return

    def dropImages(self):
        '''
        Save memory
        '''
        self.particleBox = None
        self.particleBoxCropped = None
        self.particleBoxMask = None
        self.particleBoxAlpha = None
        self.frame = None
        self.frameMask = None

    def __repr__(self):

        props = "#"*30
        props += '\n'
        props += 'PID: %i\n' % (self.pid)
        props += 'record_id, capture_id, capture_time, record_time: %i %i %s %s\n' % (
            self.record_id, self.capture_id, self.capture_time, self.record_time)
        props += 'Dmax: %.1f\n' % (self.Dmax)
        props += 'aspectRatio: %.2f (elDirect) %.2f (ellipse) %.2f (rect)\n' % self.aspectRatio
        props += 'angle_ellipseDirect: %f\n' % (self.angle_ellipseDirect)
        props += 'ROI (x, y, w, h): %i %i %i %i\n' % tuple(self.roi)
        props += 'Area: %.1f\n' % self.area
        props += 'position_centroid: %.1f, %.1f\n' % self.position_centroid
        props += 'Perimeter, PerimeterEroded: %.1f %.1f\n' % (
            self.perimeter, self.perimeterEroded)
        props += 'pixMin/pixMax: %.1f %i %f\n' % (
            self.pixMin, self.pixMax, self.pixMin/self.pixMax)
        props += 'pixPercentiles: ' + str(self.pixPercentiles.tolist()) + '\n'
        props += 'pixMean, pixStd, pixSkew, pixKurtosis: %.1f %.1f %.1f %.1f \n' % (
            self.pixMean, self.pixStd, self.pixSkew, self.pixKurtosis)
        props += "max contrast: %.1f\n" % self.particleContrast
        props += 'Blur: %f\n' % self.blur
        # props += f'touches border: {self.touchesBorder} \n'
        props += "#"*30
        props += '\n'
        return props

    def drawContour(self, frame):
        (x, y, w, h) = self.roi
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), (0, 255., 0), 1)

        cv2.drawContours(frame, [self.cnt], 0, np.array((255., 255., 0)), 1)

        if self.cntChild is not None:
            for cntChild in self.cntChild:
                cv2.drawContours(frame, [cntChild], 0,
                             np.array((255., 255., 0)), 1)
        box = cv2.boxPoints(self.rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, np.array((0, 0, 255.)), 1)
        try:
            cv2.ellipse(frame, self.ellipse, np.array((255., 0, 0)), 1)
        except:
            pass
        try:
            cv2.ellipse(frame, self.ellipseDirect,
                        np.array((255., 255., 255.)), 1)
        except:
            pass
        # try:
        #     cv2.ellipse(frame, self.ellipseAMS, np.array((0., 0., 0.)), 1)
        # except:
        #     pass
        cv2.circle(frame, np.ceil(self.position_circle).astype(
            int), int(np.ceil(self.Dmax/2)), (255, 0, 255), 1)

        return frame

    def annotate(self, frame, color=(0, 255, 0), extra=''):
        (x, y, w, h) = self.roi
        cv2.putText(frame, '%i %s' % (self.pid, extra),
                    (x+w+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return frame

    def getAnnotatedParticle(self, frame, extra=20):

        if frame.shape[-1] != 3:
            annotatedParticle = av.cvtGray(frame)
        else:
            annotatedParticle = frame.copy()
        annotatedParticle = self.drawContour(annotatedParticle)
        annotatedParticleCropped, x1, y1, _ = extractRoi(
            self.roi, annotatedParticle, extra=extra)

        return annotatedParticleCropped


def extractRoi(roi, frame, extra=0):

    if extra == 0:
        x, y, w, h = roi
        newRoi = (0, 0, w, h)
        return frame[y:y+h, x:x+w], x, y, newRoi
    else:
        xAdded = extra
        yAdded = extra
        x, y, w, h = roi
        x1 = x - extra
        x2 = x + w + extra
        y1 = y - extra
        y2 = y + h + extra

        if x1 < 0:
            x1 = 0
            xAdded = x
        if y1 < 0:
            y1 = 0
            yAdded = y
        try:
            ny, nx = frame.shape
        except ValueError:
            ny, nx, nz = frame.shape

        if x2 > nx:
            x2 = nx
        if y2 > ny:
            y2 = ny

        newRoi = (xAdded, yAdded, w, h)

        return frame[y1:y2, x1:x2], x1, y1, newRoi


def checkMotion(subFrame, oldFrame, threshs):
    '''
    Check whether something is moving - identical to VISSS C code
    '''

    if oldFrame is None:
        oldFrame = np.zeros(subFrame.shape, dtype=np.uint8)

    nChangedPixel = np.zeros(
        (len(threshs)), dtype=int)

    absdiff = cv2.absdiff(subFrame, oldFrame)

    for tt, thresh in enumerate(threshs):
        nChangedPixel[tt] = ((absdiff >= thresh)).sum()

    return nChangedPixel


# get trainign data"
def _getTrainingFrames(fnamesV, trainingSize, config):

    inVidTraining = {}
    for nThread, fnameV in fnamesV.items():
        assert fnameV.endswith(config.movieExtension)
        vid = cv2.VideoCapture(fnameV)
        if int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
            inVidTraining[nThread] = vid

    trainingFrames = []
    if len(inVidTraining) == 0:
        # print("empty training file")
        return trainingFrames

    inVidTrainingIter = itertools.cycle(list(inVidTraining.values()))
    for tt in range(trainingSize):
        ret, frame = next(inVidTrainingIter).read()
        if frame is not None:
            trainingFrames.append(frame)

    for nThread in inVidTraining.keys():
        inVidTraining[nThread].release()
    return trainingFrames


def detectParticles(fname,
                    config,
                    testing=[],
                    writeImg=True,
                    writeNc=True,
                    trainingSize=100,
                    testMovieFile=True,
                      backSubKW={"dist2Threshold": 400, 
                               "detectShadows": False, "history": 100},
                    backSub=cv2.createBackgroundSubtractorKNN,
                    applyCanny2Particle=True, 
                    dilateIterations=1, 
                    blurSigma=1,
                    minBlur=10,
                    erosionTestThreshold=0.06,
                    minArea=0,
                    minDmax=0,
                    dilateErodeFgMask=False,
                    doubleDynamicRange=True,
                    joinCannyEdges=False,
                    check4childCntLength = True,
                    stopAfter=None,
                    version=__version__,
                    verbosity=0,
                    ):
    '''
    detect particles

    Parameters
    ----------
    fname : str
        filename
    config : dict
        VISSS config
    testing : list, optional
        [description] (the default is [])
    writeImg : bool, optional
        [description] (the default is True)
    writeNc : bool, optional
        [description] (the default is True)
    trainingSize : number, optional
        [description] (the default is 100)
    testMovieFile : bool, optional
        [description] (the default is True)
    backSubKW : dict, optional
        dist2Threshold of 100 was extensively tested, but this makes small particles larger even though it helps with wings etc. 
     this is compensated by the canny filter, but not for holes in the particles (the default is {"dist2Threshold": 400, 
                                   "detectShadows": False, "history": 100})
    backSub : [type], optional
        [description] (the default is cv2.createBackgroundSubtractorKNN)
    applyCanny2Particle : bool, optional
        this is a must, canny to the whole image is too expensive (the default is True)
    dilateIterations : number, optional
        to close gaps in canny edges (the default is 1 whic is sufficient)
    blurSigma : number, optional
        [description] (the default is 1.5)
    minBlur : number, optional
        [description] (the default is 10)
    erosionTestThreshold : number, optional
        [description] (the default is 0.06)
    minArea : number, optional
        [description] (the default is 1)
    minDmax : number, optional
        [description] (the default is 2)
    dilateErodeFgMask : bool, optional
         turns out to be not so smart because it makes holes insides particles smaller (the default is False)
    doubleDynamicRange : bool, optional
        [description] (the default is True)
    joinCannyEdges : bool, optional
        novel way to close canny edges, nice idea but closes them to often in teh wrong way (the default is False)
    check4childCntLength : bool, optional
        discard short child contours instead of dilate/erose (the default is False)
    stopAfter : [type], optional
        [description] (the default is None)
    version : [type], optional
        [description] (the default is __version__)
    verbosity : [type], optional
        avoid expensive debugging module (the default is 0)
    '''





    #logging.config.dictConfig(tools.get_logging_config('detection_run.log'))
    #log = logging.getLogger("detection")

    assert os.path.isfile(fname)

    if type(config) is str:
        config = tools.readSettings(config)

    path = config["path"]
    pathTmp = config["pathTmp"]
    threshs = np.array(config["threshs"])
    instruments = config["instruments"]
    computers = config["computers"]
    visssGen = config["visssGen"]
    movieExtension = config["movieExtension"]
    frame_width = config["frame_width"]
    frame_height = config["frame_height"]
    height_offset = config["height_offset"]
    fps = config["fps"]
    minMovingPixels = np.array(config["minMovingPixels"])
    nThreads = config["nThreads"]
    cropImage = config["cropImage"]
    site = config["site"]
    goodFiles = config["goodFiles"]
    minBlur4write = config["level1detect"]["minBlur"]
    minDmax4write = config["level1detect"]["minSize"]

    fn = files.Filenames(fname, config, version=version)
    log.info(f"running {fn.fname.level1detect}")
    camera = fn.camera

    Dbins = np.arange(0, 201, 10).tolist()

    if nThreads is None:
        nThreads2 = 1
    else:
        nThreads2 = nThreads

    computerDict = {}
    goodFilesDict = {}
    for computer1, camera1, goodFile1 in zip(computers, instruments, goodFiles):
        computerDict[camera1] = computer1
        goodFilesDict[camera1] = goodFile1
    computer = computerDict[camera]
    goodFile = goodFilesDict[camera]

    fnamesV = fn.fnameMovAllThreads
    fnamesT = fn.fnameTxtAllThreads
    
    # txt data is transmitted first
    if len(fnamesT) == 0:
        with open('%s.nodata' % fn.fname.metaDetection, 'w') as f:
            f.write('no data in %s' % fn.fname.metaFrames)
        with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
            f.write('no data in %s' % fn.fname.metaFrames)
        log.warning('no movie files: ' + fname)
        return 0

    # mov data is sometimes transmtted later, so do not write nodata files in 
    # case data is missing yet
    if len(fnamesV) < int(nThreads2):
        if config.end == "today": 
            log.warning('movie files not found (yet?) ' + fname)
        else:
            log.warning('movie files not found, no data ' + fname)
            with open('%s.nodata' % fn.fname.metaDetection, 'w') as f:
                f.write('no data')
            with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
                f.write('no data')
        return 0

    # fnameM = [f.replace(config["movieExtension"], "txt") for f in fnamesV.values()]

    # metaData, nDroppedFrames = metadata.getMetaData(
    #     fnameM, camera, config, testMovieFile=True)
    try:
        metaData = xr.open_dataset(fn.fname.metaFrames)
    except FileNotFoundError:
        if os.path.isfile(f"{fn.fname.metaFrames}.nodata"):
            with open('%s.nodata' % fn.fname.metaDetection, 'w') as f:
                f.write('no data in %s' % fn.fname.metaFrames)
            with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
                f.write('no data in %s' % fn.fname.metaFrames)
            log.warning('metaFrames contains no data: ' + fn.fname.metaFrames)
        else:
            log.warning('metaFrames data not found: ' + fn.fname.metaFrames)
        return 0
    # if metaData is None:
    #     log.error('ERROR Unable to get meta data: ' + fname)
    #     raise RuntimeError('ERROR Unable to get meta data: ' + fname)
    if len(metaData.capture_time) == 0:
        log.info('nothing moves: ' + fname)

        with open('%s.nodata' % fn.fname.metaDetection, 'w') as f:
            f.write('no data')
        with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
            f.write('no data')
        return 0

    if testMovieFile and (goodFile is not None) and (goodFile != "None"):
        for fnameV in fnamesV.values():
            # check for broken files:
            process = subprocess.Popen(['ffprobe', fnameV],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True)
            while True:
                output = process.stdout.readline()
                return_code = process.poll()
                if return_code is not None:
                    # Process has finished, read rest of the output
                    break
            if return_code == 0:
                log.info('OK ' + fnameV)
                pass
            else:
                log.info('BROKEN ' + fnameV)
                brokenFile = '%s.broken' % fnameV
                repairedFile = '%s_fixed.%s' % (
                    ".".join(fnameV.split(".")[:-1]), config["movieExtension"])
                process = subprocess.Popen(['/home/mmaahn/bin/untrunc', goodFile, fnameV],
                                           stdout=subprocess.PIPE,
                                           universal_newlines=True)
                while True:
                    output = process.stdout.readline()
                    log.info(output.strip())
                    return_code = process.poll()
                    if return_code is not None:
                        log.info('RETURN CODE %i' % return_code)
                        # Process has finished, read rest of the output
                        for output in process.stdout.readlines():
                            log.info(output.strip())
                        break
                if return_code == 0:
                    os.rename(fnameV, brokenFile)
                    try:
                        os.rename(repairedFile, fnameV)
                    except FileNotFoundError:
                        # sometime utrunc names the files mp4?!
                        os.rename(repairedFile.replace(f'.{config["movieExtension"]}', ".mp4"), fnameV)
                else:
                    log.error('WAS NOT ABLE TO FIX %s' % fnameV)
                    raise RuntimeError('WAS NOT ABLE TO FIX %s' % fnameV)
                log.info('REPAIRED ' + fnameV)

    log.info(f"{fn.year}, {fn.month}, {fn.day}, {fname}")
    fn.createDirs()

    log.info('Processing %s' % fn.fname.level1detect)

    hasData = False

    particleProperties = []

    nFrames = len(metaData.capture_time)

    trainingFrames = _getTrainingFrames(fnamesV, trainingSize, config)
    log.info(f"found {len(trainingFrames)} for training.")

    # to few data in trainign data, look into previous files
    fname1 = deepcopy(fname)
    ii = 0
    while len(trainingFrames) < trainingSize:
        fname11 = files.Filenames(fname1, config, version=version).prevFile()

        if (ii > 20) or (fname11 is None):
            with open('%s.notenoughframes' % fn.fname.level1detect, 'w') as f:
                f.write('too few frames %i %i %s \r' %
                        (len(trainingFrames), ii, fname11))
                f.write(str(fnamesV))
                f.write(str(fnamesT))
            log.error('%s too few frames %i ' %
                      (fn.fname.level1detect, len(trainingFrames)))
            return None

        fnamesV1 = files.Filenames(
            fname11, config, version=version).fnameMovAllThreads
        trainingFrames1 = _getTrainingFrames(fnamesV1, trainingSize, config)
        log.info(f"added {len(trainingFrames1)} from {ii} previous file {fname11}.")
        trainingFrames = trainingFrames1 + trainingFrames

        fname1 = fname11
        ii += 1
    # just in case we have now too many frames by adding older files
    trainingFrames = trainingFrames[-trainingSize:]


    snowParticles = detectedParticles(
        config,
        verbosity=verbosity,
        trainingSize=trainingSize,
        backSubKW=backSubKW,
        cropImage=cropImage,
        applyCanny2Particle=applyCanny2Particle,
        dilateIterations=dilateIterations,
        blurSigma=blurSigma,
        minBlur=minBlur,
        backSub=backSub,
        erosionTestThreshold=erosionTestThreshold,
        minArea=minArea,
        minDmax=minDmax,
        minAspectRatio=config.level1detect.minAspectRatio,
        dilateErodeFgMask=dilateErodeFgMask,
        doubleDynamicRange = doubleDynamicRange,
        joinCannyEdges=joinCannyEdges,
        check4childCntLength = check4childCntLength,
        testing=testing,

    )

    # do training
    for ff, frame in enumerate(trainingFrames):
        # meta data does not matter fro training
        res = snowParticles.update(frame,
                                   ff,
                                   -99,
                                   -99,
                                   -99,
                                   -99,
                                   -99,
                                   training=True)

        log.debug('training %i' % ff)

    # trainign files not needed any more
    del trainingFrames

    # write background to PNG file
    cv2.imwrite(
        fn.fname.metaDetection.replace(".nc", ".png"),
        snowParticles.backSub.getBackgroundImage(),
        [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )

    # test motion
    # nMovingPixel missing in some mosaic data
    if np.all(metaData.nMovingPixel.values == -9999):
        log.info(f"nMovingPixel not found")
        motionChecked = False
    else:
        nChangedPixel = metaData.nMovingPixel.values
        passesThreshold = nChangedPixel >= minMovingPixels
        log.info(f"{passesThreshold.any(1).sum()/len(metaData.capture_time)*100}% frames are moving")
        metaData = metaData.isel(capture_time=passesThreshold.any(1))
        motionChecked = True

    if (stopAfter is not None):
        try:
            startAt, stopAfter = stopAfter
        except TypeError:
            startAt = 0
        metaData = metaData.isel(capture_time=slice(startAt, stopAfter))

    nFrames = len(metaData.capture_time)
    pps = range(nFrames)

    foundParticles = np.zeros((nFrames, len(Dbins)))
    movingObjects = np.zeros((nFrames))

    inVid = {}
    for nThread, fnameV in fnamesV.items():
        assert fnameV.endswith(config.movieExtension)
        inVid[nThread] = cv2.VideoCapture(fnameV)
        nFrames = int(inVid[nThread].get(cv2.CAP_PROP_FRAME_COUNT))
        log.info(f'opened {fnameV} with {nFrames} frames.')
        assert nFrames > 0
    frame = None

    if writeImg:
        imagesL1detect = tools.imageZipFile(
            fn.fname.imagesL1detect, 'w', compresslevel=9)

    tarRoot = fn.fname.imagesL1detect.split("/")[-1].replace(".tar.bz2", "")

    for pp in pps:
        if len(testing)>0:
            print("#"*80)
            print(f"frame {pp} of {len(pps)}")
            print("#"*80)

        metaData1 = metaData.isel(capture_time=pp)

        nThread = int(metaData1.nThread.values)

        rr = int(metaData1.record_id.values)
        log.info('Image %i from thread %i, frame %i, %s' %
                 (pp, nThread, rr, str(metaData1.capture_time.values)))

        if not motionChecked:
            oldFrame = deepcopy(frame)

        # sometime we have to skip frames due to missing meta data
        # we dont want to take the risk to jump to an index and get the wrong frame
        if int(inVid[nThread].get(cv2.CAP_PROP_POS_FRAMES)) < rr:
            while(int(inVid[nThread].get(cv2.CAP_PROP_POS_FRAMES)) < rr):
                log.debug('fast forwarding', int(
                    inVid[nThread].get(cv2.CAP_PROP_POS_FRAMES)), rr, )
                _, _ = inVid[nThread].read()
        elif int(inVid[nThread].get(cv2.CAP_PROP_POS_FRAMES)) > rr:
            raise RuntimeError('Cannot go back!')

        ret, frame = inVid[nThread].read()
        if frame is None:
            # there is a known problem that a movie file is broken if it contains
            # only a single frame. this happens sometimes for a new measurement
            # because everything is moving for the first frame. just skip it
            # because it is only a single frame
            firstRecordedFrame = metaData1.nMovingPixel.values[0] == (
                config.frame_height * config.frame_width)
            singleFrameInFile = (np.sum(metaData.nThread == nThread) == 1)

            if firstRecordedFrame and singleFrameInFile:
                log.warning("detected single frame issue %s thread %i" %
                            (fname, nThread))
                with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
                    f.write('no data (single frame problem)')
                continue
            else:
                raise ValueError('TOO FEW FRAMES???? %i of %i, %s thread %i' % (
                    pp, nFrames, fname, nThread))

        if not motionChecked:
            nChangedPixel = checkMotion(frame, oldFrame, config.threshs)
            passesThreshold = nChangedPixel >= minMovingPixels
            if not passesThreshold.any():
                log.debug('%s NOT moving %i' % str(metaData1.capture_time.values), (pp))
                continue
            else:
                log.debug('%s IS moving %i' % (str(metaData1.capture_time.values), pp))

        res = snowParticles.update(frame,
                                   pp,
                                   int(metaData1.capture_id.values),
                                   int(metaData1.record_id.values),
                                   metaData1.capture_time.values,
                                   metaData1.record_time.values,
                                   nThread,
                                   )

        if res:
            movingObjects[pp] = snowParticles.nParticle
            Dmax = []
            for part in snowParticles.lastFrame.values():
                Dmax.append(part.Dmax)
            foundParticles[pp] = np.histogram(
                Dmax, bins=Dbins + [10000])[0]
        else:
            foundParticles[pp, :] = -99

        timestamp = str(snowParticles.capture_time).replace(
            ':', '-')[:-6]

        if not res:
            if "skipped" in testing:
                plt.figure(figsize=(10, 10))
                plt.imshow(snowParticles.frame4drawing,
                           cmap='gray', vmin=0, vmax=255)
                plt.title('SKIPPED %s %i %s' %
                          (camera, pp, timestamp))
                plt.show()
                plt.savefig('testOutput/skipped_%s_%i_%s.png' %
                            (camera, pp, timestamp))

        for part in snowParticles.lastFrame.values():
            hasData = True

            if writeImg:
                if (
                    (part.Dmax >= minDmax4write) and
                    (part.blur >= minBlur4write)
                ):
                    pidStr = '%07i' % part.pid
                    # imName = '%s/%s/%s.npy' % (tarRoot, pidStr[:4], pidStr)
                    log.info('writing %s %s' %
                             (fn.fname.imagesL1detect, pidStr))

                    # imagesL1detect.addimage(imName, part.particleBoxAlpha)
                    imagesL1detect.addnpy(pidStr, part.particleBoxAlpha)
            if "particle" in testing:
                print("final particle and mask")
                img = np.hstack((part.particleBox, part.particleBoxCropped))
                tools.displayImage(img, rescale=4)
            if "annotatedParticle" in testing:
                annotatedParticle = part.getAnnotatedParticle(
                    frame[height_offset:])
                tools.displayImage(annotatedParticle, rescale=4)

            part.dropImages()

    if writeImg:
        nFiles = len(imagesL1detect.namelist())
        imagesL1detect.close()
        log.info("closing %s"%fn.fname.imagesL1detect)
        if nFiles == 0:
            os.remove(fn.fname.imagesL1detect)

    for nThread in inVid.keys():
        inVid[nThread].release()

    if writeNc:

        foundParticles = xr.DataArray(foundParticles,
                                      coords=[metaData.capture_time,
                                              xr.DataArray(Dbins, dims=['Dmax'])])
        movingObjects = xr.DataArray(movingObjects,
                                     coords=[metaData.capture_time])

        metaData['foundParticles'] = foundParticles
        metaData['movingObjects'] = movingObjects

        metaData["foundParticles"] = metaData["foundParticles"].astype(
            np.uint32)
        metaData["movingObjects"] = metaData["movingObjects"].astype(np.uint32)

        metaData = tools.finishNc(metaData, config.site, config.visssGen)

        metaData.to_netcdf(fn.fname.metaDetection, engine="netcdf4")
        metaData.close()

        if hasData:

            snowParticlesXR = snowParticles.collectResults()
            snowParticlesXR = tools.finishNc(snowParticlesXR, config.site, config.visssGen)

            snowParticlesXR.to_netcdf(fn.fname.level1detect, engine="netcdf4")
            log.info("written %s"%fn.fname.level1detect)
            snowParticlesXR.close()
            return snowParticlesXR
        else:
            with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
                f.write('no data')
            log.info("no data %s"%fn.fname.level1detect)
            return None
    else:
        return None

import numba
@numba.njit
def joinEdges(mask):
    mask = np.copy(mask)
    height, width = mask.shape
    dx = np.array([1, 1, 0, -1, -1, -1, 0, +1, 1 ])
    dy = np.array([0, 1, 1,  1,  0, -1,-1, -1, 0 ])
    d2x = np.array([2, 2, 2, 1, 0,-1,-2,-2,-2,-2,-2,-1, 0, 1, 2, 2 ])
    d2y = np.array([0, 1, 2, 2, 2, 2, 2, 1, 0,-1,-2,-2,-2,-2,-2,-1 ])
    d3x = np.array([3,3,3,3,2,1,0,-1,-2,-3,-3,-3,-3,-3,-3,-3,-2,-1,0,1,2,3,3,3 ])
    d3y = np.array([0,1,2,3,3,3,3,3,3,3,2,1,0,-1,-2,-3,-3,-3,-3,-3,-3,-3,-2,-1 ])


    for i in range(height):
        for j in range(width):
            if mask[i,j] == 0: continue
            one_step = -99
            two_step = -99
            three_step = -99
            connection = list()

            for dir1 in range(8):
                tmp_x = j + dx[dir1]
                tmp_y = i + dy[dir1]

                if ((tmp_x < 0) | (tmp_x >= width) | (tmp_y < 0) | (tmp_y >= height) ):
                    one_step=0 #treat outside image as 0
                else:
                    one_step=mask[tmp_y, tmp_x]

                if (one_step!=0): connection.append(dir1)

            if((len(connection)<3 )& (len(connection)!=0)):
                direction1 = connection[0]
                direction2 = connection[-1]                
                if ((direction2-direction1<=1) |(direction2-direction1==7)):
                    start_direction =(direction2+2)*3
                    end_direction =(direction1+6)*3
                    for dir2 in range(start_direction, end_direction):
                        tmp_1x = j + dx [dir2%24//3]
                        tmp_1y = i + dy [dir2%24//3]
                        tmp_2x = j + d2x[dir2%24*2//3]
                        tmp_2y = i + d2y[dir2%24*2//3]
                        tmp_3x = j + d3x[dir2%24]
                        tmp_3y = i + d3y[dir2%24]
                        if((tmp_2x < 0) | (tmp_2x >= width) | (tmp_2y < 0) | (tmp_2y >= height) ):
                            two_step=0 #treat outside image as 0
                        else:
                            two_step=mask[tmp_2y, tmp_2x]
                        if((tmp_3x < 0) | (tmp_3x >= width) | (tmp_3y < 0) | (tmp_3y >= height) ):
                            three_step=0 #treat outside image as 0
                        else:
                            three_step=mask[tmp_3y, tmp_3x]
                        if (two_step!=0):
                            mask[tmp_1y, tmp_1x] = 255
                            break
                        elif (three_step!=0):
                            mask[tmp_1y, tmp_1x]=255
                            mask[tmp_2y, tmp_2x]=255
                            break
    return mask


def splitUpConours(cntsTmp, hierarchy):
    '''
    reorder contours based on hierarchy
    numba acceleration tested: it is slower!
    '''
    parentIndex = hierarchy[0][:,3]
    # list where children are replaced by None (otehrwise indices are messed up)
    cnts = [cnt if (pIndex == -1) else None for cnt, pIndex in zip(cntsTmp, parentIndex)]
    # index of children
    childII = np.where(parentIndex != -1)[0]
    cntChildren = [list() for i in range(len(cnts))]
    # find a parent fo each child
    for cc in childII:
        thisParentIndex = parentIndex[cc]
        # iterate to find top most parent
        for ii in range(10): # read 'while True' with limit
            newParentIndex = parentIndex[thisParentIndex]
            if newParentIndex == -1:
                break
            thisParentIndex = newParentIndex
        cntChildren[thisParentIndex].append(cntsTmp[cc])
    #discard empty spots were children were before
    cnts = [cnt for cnt, pIndex in zip(cnts, parentIndex)  if (pIndex == -1)]
    cntChildren = [cnt for cnt, pIndex in zip(cntChildren, parentIndex) if (pIndex == -1)]
    
    return cnts, cntChildren

# def autoCanny(image, sigma=0.33):
#     # compute the median of the single channel pixel intensities
#     v = np.median(image)
#     # apply automatic Canny edge detection using the computed median
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = cv2.Canny(image, lower, upper, L2gradient=True)
#     print("autoCanny", lower, upper)
#     # return the edged image
#     return edged
