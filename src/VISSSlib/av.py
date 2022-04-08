# -*- coding: utf-8 -*-
import functools
import os
import warnings

import numpy as np
try:
    import cv2
except ImportError:
    warnings.warn("opencv not available!")
import xarray as xr


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
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                        print('fast forwarding', int(self.get(cv2.CAP_PROP_POS_FRAMES)), ii, )
                        _, _ = self.read()
                elif int(self.get(cv2.CAP_PROP_POS_FRAMES)) > ii:
                    raise RuntimeError('Cannot go back in save mode')
            else:
                self.set(cv2.CAP_PROP_POS_FRAMES, ii)
        res, frame = self.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return res, frame

    @property
    def position(self):
        return int(self.get(cv2.CAP_PROP_POS_FRAMES))
    
    @property
    def total_frames(self):
        return int(self.get(cv2.CAP_PROP_FRAME_COUNT))


# can cause segfaults!

class VideoReaderMeta(object):
    def __init__(self, movFilePattern, metaL1, metaL2=None, saveMode=False, config=None):
        if type(metaL1) is xr.Dataset:
            self.metaL1 = metaL1
        else:
            self.metaL1 = xr.open_dataset(metaL1)
        if (metaL2 is None) or (type(metaL2) is xr.Dataset):
            self.metaL2 = metaL2
        else:
            self.metaL2 = xr.open_dataset(metaL2)
        self.saveMode = saveMode
        self.config = config

        self.movFilePattern = movFilePattern
        self.threads = np.unique(self.metaL1.nThread)

        self.video = {}
        self.position = 0
        self.positions = {}
        self.currentThread = None
        
        self.currentFrame = None
        self.currentMetaL1 = None
        self.currentMetaL2 = None
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
        
    def getNextFrame(self):
        ii = self.position + 1
        if self.metaL2:
            return self.getFrameByIndexWithParticles(ii)
        else:
            return self.getFrameByIndex(ii)
        
    def getPrevFrame(self):
        ii = self.position - 1
        if self.metaL2:
            return self.getFrameByIndexWithParticles(ii)
        else:
            return self.getFrameByIndex(ii)
         
    def getFrameByIndex(self, ii):
        '''
        like read, but with meta data and appropriate thread
        '''
        if ii < 0:
            return False, None, None
        
        try:
            self.currentMetaL1 = self.metaL1.isel(capture_time=ii)
        except IndexError:
            return False, None, None
        
        self.currentThread = int(self.currentMetaL1.nThread.values)
        rr = int(self.currentMetaL1.record_id.values)
        self.res, self.curentFrame = self.video[self.currentThread].getFrameByIndex(rr, saveMode=self.saveMode)
        self.position = ii
        self.positions[self.currentThread] = self.video[self.currentThread].position

        return self.res, self.curentFrame, self.currentMetaL1

    
    def getFrameByIndexWithParticles(self, ii, markParticles=False):
        '''
        like read, but with even more meta data and appropriate thread
        '''
   
        assert self.metaL2 is not None
        
        self.getFrameByIndex(ii)
    
        ct = self.currentCaptureTime
        self.currentMetaL2 = self.metaL2.where(self.metaL2.capture_time==ct).dropna('pid')
        self.currentPids = self.currentMetaL2.pid

        assert np.all(self.currentMetaL1.capture_time == self.currentMetaL2.capture_time)
            
        if markParticles:
            for jj, pid in enumerate(self.currentMetaL2.pid.values):
                partic1 = self.currentMetaL2.sel(pid=pid)
                (x, y, w, h) = partic1.roi.values.astype(int)
                y = y + self.config['height_offset']
                colors=[(0, 255, 0), (255, 0, 0), (0, 0 , 255), (255, 255, 0), (0, 255, 255)] * 30
                color = colors[jj]
                cv2.rectangle(self.curentFrame, (x, y),
                          (x + w, y + h), color, 2)
                extra1 = str(partic1.record_time.values)[:-6].split('T')[-1]
                extra2 = '%i'%partic1.Dmax.values
                cv2.putText(self.curentFrame, '%i %s %s' % (partic1.pid, extra1, extra2),
                            (int(partic1.roi[0]+w+5), int(partic1.roi[1]+self.config['height_offset'])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return self.res, self.curentFrame, self.currentMetaL1, self.currentMetaL2

    
    @property
    def currentCaptureTime(self):
        if self.currentMetaL1 is None:
            return None
        else:
            return np.datetime64(self.currentMetaL1.capture_time.values)
    
            
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
        particle = self.metaL2.sel(pid=pid)
        kk = int(particle.record_id.values)
        _, frame1, _ = self.getFrameByIndex(kk)
        x, y, w, h = particle.roi.values.astype(int)
        if len(frame1.shape) == 3:
            frame1 = frame1[:,:,0]
        return frame1[y+heightOffset:y+heightOffset+h, x:x+w], frame1
