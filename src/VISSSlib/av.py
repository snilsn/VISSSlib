# -*- coding: utf-8 -*-
import functools
import os
import sys
import warnings

import numpy as np
try:
    import cv2
except ImportError:
    warnings.warn("opencv not available!")
import xarray as xr

from .tools import readSettings
from .files import Filenames

__all__ = ["VideoReader", "VideoReaderMeta"]


class VideoReader(cv2.VideoCapture):


#     @functools.lru_cache(maxsize=100, typed=False)
#     def getNextFrame(self, ii):
#         '''
#         like read, but output is cversionached. 
#         ii has to advance by +1
#         cache allows to access old frames
#         '''
#         assert self.position == ii
#         res, frame = self.read()
#         frame = cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         return res, frame

    @functools.lru_cache(maxsize=100, typed=False)
    def getFrameByIndex(self, ii, saveMode=False):
        '''
        like read, but for a specific index
        output is cached.
        '''
        if int(self.get(cv2.CAP_PROP_POS_FRAMES)) != ii:
            if saveMode:
                if int(self.get(cv2.CAP_PROP_POS_FRAMES)) < ii:
                    while(int(self.get(cv2.CAP_PROP_POS_FRAMES)) < ii):
                        # print('fast forwarding', int(self.get(cv2.CAP_PROP_POS_FRAMES)), ii, )
                        _, _ = self.read()
                elif int(self.get(cv2.CAP_PROP_POS_FRAMES)) > ii:
                    raise RuntimeError('Cannot go back in save mode')
            else:
                self.set(cv2.CAP_PROP_POS_FRAMES, ii)
        res, frame = self.read()
        if frame is not None:
            frame = cvtColor(frame)
        return res, frame

    @property
    def position(self):
        return int(self.get(cv2.CAP_PROP_POS_FRAMES))
    
    @property
    def total_frames(self):
        return int(self.get(cv2.CAP_PROP_FRAME_COUNT))


# can cause segfaults!

class VideoReaderMeta(object):
    def __init__(self, movFilePattern, metaFrames, lv1detect=None, lv1match=None, saveMode=False, config=None):
        if type(metaFrames) is xr.Dataset:
            self.metaFrames = metaFrames
        else:
            self.metaFrames = xr.open_dataset(metaFrames)
        if (lv1detect is None) or (type(lv1detect) is xr.Dataset):
            self.lv1detect = lv1detect
        else:
            self.lv1detect = xr.open_dataset(lv1detect)
        if (lv1match is None) or (type(lv1match) is xr.Dataset):
            self.lv1match = lv1match
        else:
            raise ValurError("provide level1match as Dataset with data selected for corresponding camera")
        self.saveMode = saveMode
        self.config = config

        self.movFilePattern = movFilePattern
        self.threads = np.unique(self.metaFrames.nThread)
        if len(self.threads)>1:
            self.movFilePattern = self.movFilePattern.replace("_0.", "_{thread}.")
        if config:
            assert self.movFilePattern.endswith(config.movieExtension)
        self.video = {}
        self.position = 0
        self.positions = {}
        self.currentThread = None
        
        self.currentFrame = None
        self.currentMetaFrames = None
        self.currentlv1detect = None
        self.currentPids = None
        
        self._openVideo()
        
    def _openVideo(self):
        for tt in self.threads:
            fname = self.movFilePattern.format(thread=tt)
            assert os.path.isfile(fname)
            self.video[tt] = VideoReader(fname)
            self.positions[tt] = 0
        assert len(self.video)>0
        
    def resetVideo(self):
        for tt in self.threads:
            self.video[tt].release()
        self._openVideo()
        
    def getNextFrame(self, markParticles=False):
        ii = self.position + 1
        if self.lv1detect:
            return self.getFrameByIndexWithParticles(ii, markParticles=markParticles)
        else:
            return self.getFrameByIndex(ii)
        
    def getPrevFrame(self, markParticles=False):
        ii = self.position - 1
        if self.lv1detect:
            return self.getFrameByIndexWithParticles(ii, markParticles=markParticles)
        else:
            return self.getFrameByIndex(ii)
         
    def getFrameByIndex(self, ii, increaseContrast=False):
        '''
        like read, but with meta data and appropriate thread
        '''
        if ii < 0:
            return False, None, None
        
        try:
            captureTime = self.metaFrames.capture_time[ii].values
        except IndexError:
            return False, None, None

        return self.getFrameByCaptureTime(captureTime, increaseContrast=increaseContrast)

    @functools.lru_cache(maxsize=100, typed=False)
    def getFrameByCaptureTime(self, captureTime, increaseContrast=False):
        '''
        like read, but with capturetime, meta data and appropriate thread
        '''
        try:
            self.currentMetaFrames = self.metaFrames.sel(capture_time=captureTime)
        except IndexError:
            self.res = None
            self.curentFrame = None
            self.currentMetaFrames = None
            self.position = None
        else:
            self.currentThread = int(self.currentMetaFrames.nThread.values)
            rr = int(self.currentMetaFrames.record_id.values)
            self.res, self.curentFrame = self.video[self.currentThread].getFrameByIndex(rr, saveMode=self.saveMode)
            if increaseContrast:
                self.curentFrame = doubleDynamicRange(self.curentFrame)
            self.position = np.where(self.metaFrames.capture_time==captureTime)[0][0]
            self.positions[self.currentThread] = self.video[self.currentThread].position

        return self.res, self.curentFrame, self.currentMetaFrames

    @functools.lru_cache(maxsize=100, typed=False)
    def getFrameByCaptureTimeWithParticles(self, captureTime, pad = 4, markParticles=False, highlightPid=None, increaseContrast=False):

        assert self.lv1detect is not None
        
        res, _, _ = self.getFrameByCaptureTime(captureTime, increaseContrast=increaseContrast)
        if (res is None) or (res == False):
            return None, None, None, None, None
    
        self.curentFrameC = cv2.cvtColor(self.curentFrame, cv2.COLOR_GRAY2BGR)

        if self.config.cropImage is not None:
            color = (255,255,255)
            cv2.rectangle(self.curentFrameC, (self.config['cropImage'][0], self.config['cropImage'][1] + self.config['height_offset']), (self.config['cropImage'][0]+ (self.config.frame_width - 2*self.config['cropImage'][0]), self.config['cropImage'][1] + self.config['height_offset'] + (self.config.frame_height - 2*self.config['cropImage'][1])), color, 2)

        ct = self.currentCaptureTime
        try: 
            self.currentlv1detect = self.lv1detect.isel(pid=(self.lv1detect.capture_time==ct))
        except:
            self.currentlv1detect = self.lv1detect.isel(fpid=(self.lv1detect.capture_time==ct))

        self.currentPids = self.currentlv1detect.pid

        assert np.all(self.currentMetaFrames.capture_time == self.currentlv1detect.capture_time)
            
        colors=[(0, 255, 0), (255, 0, 0), (0, 0 , 255), (255, 255, 0), (0, 255, 255)] * 30

        matchedDats = []

        if markParticles:
            for jj, pid in enumerate(self.currentlv1detect.pid.values):
                try: 
                    partic1 = self.currentlv1detect.sel(pid=pid)
                except KeyError:
                    partic1 = self.currentlv1detect.sel(fpid=(self.currentlv1detect.pid==pid)).squeeze()
                
                (x, y, w, h) = partic1.roi.values.astype(int)
                y = y + self.config['height_offset']

                if self.config.cropImage is not None:
                    y = y + self.config['cropImage'][1]
                    x = x + self.config['cropImage'][0]
                if (self.lv1match is not None) and (highlightPid == "meta") and pid in self.lv1match.pid:
                    thisMatch = self.lv1match.isel(fpair_id = self.lv1match.pid == pid)
                    matchedDats.append(thisMatch)
                    color = (0,0,0)
                    extraInfo = "%.2g"%thisMatch.matchScore.values
                elif highlightPid == pid:
                    color = (0,0,0)
                    extraInfo = ""
                else:
                    color = colors[jj]
                    extraInfo = ""

                x1, y1, x2, y2 = x - pad, y - pad, x + w + pad, y + h+ pad

                if x1 < 0:
                    x1= 0
                if y1 < 0:
                    y1= 0
                if x2 >= self.curentFrameC.shape[1]:
                    x2= self.curentFrameC.shape[1]-1
                if y2 >= self.curentFrameC.shape[0]:
                    y2= self.curentFrameC.shape[0]-1

                cv2.rectangle(self.curentFrameC, (x1, y1), (x2, y2), color, 2)

                # cnt = partic1.cnt.values[partic1.cnt.values[...,0]>=0]
                # cv2.drawContours(self.curentFrameC, np.array([cnt], dtype=np.int32),0,color,1)

                extra1 = str(partic1.capture_time.values)[:-6].split('T')[-1]

                posY = int(partic1.roi[1]+self.config['height_offset'] -10)
                posX = int(partic1.roi[0])
                if self.config.cropImage is not None:
                    posY = posY + self.config['cropImage'][1]
                    posX = posX + self.config['cropImage'][0]

                cv2.putText(self.curentFrameC, '%i %s %s' % (partic1.pid, extra1, extraInfo),
                            (posX, posY), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            if len(matchedDats) > 0:
                matchedDats = xr.concat(matchedDats, dim="fpair_id")
            else:
                matchedDats = None

        return self.res, self.curentFrameC, self.currentMetaFrames, self.currentlv1detect, matchedDats


    def getFrameByIndexWithParticles(self, ii, markParticles=False, highlightPid=None, increaseContrast=False):
        '''
        like read, but with even more meta data and appropriate thread
        '''
   
        if ii < 0:
            return False, None, None, None, None
        
        try:
            captureTime = self.metaFrames.capture_time[ii].values
        except IndexError:
            return False, None, None, None, None

        return self.getFrameByCaptureTimeWithParticles(captureTime, markParticles=markParticles, highlightPid=highlightPid, increaseContrast=increaseContrast)

    
    @property
    def currentCaptureTime(self):
        if self.currentMetaFrames is None:
            return None
        else:
            return np.datetime64(self.currentMetaFrames.capture_time.values)
    
            
    @property
    def total_frames(self):
        nFrames = 0
        for tt in self.threads:
            nFrames += self.video[tt].total_frames
        return nFrames
    
    def release(self):
        for tt in self.threads:
            self.video[tt].release()

    def getParticle(self, pid,heightOffset=64):
        particle = self.lv1detect.sel(pid=pid)
        kk = int(particle.record_id.values)
        _, frame1, _ = self.getFrameByIndex(kk)
        x, y, w, h = particle.roi.values.astype(int)
        if len(frame1.shape) == 3:
            frame1 = frame1[:,:,0]
        return frame1[y+heightOffset:y+heightOffset+h, x:x+w], frame1

def doubleDynamicRange(frame, offset="estimate", factor = 2):
    '''
    dynamic range can be typically doubled which helps with feature detection
    the factor of 2 makes sure all gradients scale with the same factor even for integers

    the offset is chosen such that the brightest point is max. 255 after multiplication of 2
    if this means the darkest point becomes negative, the offset is adjusted accordingly

    '''

    

    if offset == "estimate":
        # offset so that brightest spot is 255 even if doubled
        offset1 = frame.max()-(254//factor)
        # offset so that darkest point is zero.
        offset2 = frame.min()
        # take smaller one so that in doubt information is lost for brighter pixels
        offset = min(offset1, offset2)
        #make sure offset is >= 0
        offset = max(0,offset)
        # apply to frame. cv2.multiply handles overflows properly

    frame = cv2.subtract(frame, int(offset))
    frame = cv2.multiply(frame,factor)

    return frame

def main():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    '''
    python -m VISSSlib.av doubleImage fname1 index1 pid1 fname2 index2 pid2 confFile version  outfile 
    '''

    assert sys.argv[1] == "doubleImage"


    fname1 = sys.argv[2]
    video1ii = int(sys.argv[3])
    pid1 = int(sys.argv[4])
    fname2 = sys.argv[5]
    video2ii = int(sys.argv[6])
    pid2 = int(sys.argv[7])
    confFile = sys.argv[8]
    version = sys.argv[9]
    outFile = sys.argv[10]

    try:
        config = readSettings(confFile)

        f1 = Filenames(fname1, config, version)
        f2 = Filenames(fname2, config, version)

        frames = []
        video1 = VideoReaderMeta(
                                    f1.fnameLevel0, f1.fnameLevel1, f1.fnameLevel2, config=config)

        res, frame1, meta11, meta12 = video1.getFrameByIndexWithParticles(
                                    video1ii, markParticles=True, highlightPid=pid1)  # frame number

        frames.append(frame1)
        frames.append(np.zeros((frame1.shape[0], 10, 3), dtype=int))

        video2 = VideoReaderMeta(
                                    f2.fnameLevel0, f2.fnameLevel1, f2.fnameLevel2, config=config)

        res, frame2, meta21, meta22 = video2.getFrameByIndexWithParticles(
                                    video2ii, markParticles=True, highlightPid=pid2)  # frame number
        frames.append(frame2)

        frame = np.concatenate(frames, axis=1)

        plt.figure(figsize=(20, 10))
        plt.imshow(frame, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{pid1} {pid2}")

        plt.savefig(outFile)

        print(outFile)
    except:
        print(outFile, "FAILED")
    return 0

def cvtColor(frame):
    #faster than cv2.cvtColor but works only for gray images 
    return frame[:,:,0]

if __name__ == '__main__':
    main()

