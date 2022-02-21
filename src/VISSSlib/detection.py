# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats
#import av
import bottleneck as bn
import cv2

import logging
log = logging.getLogger()

from copy import deepcopy


from . import __version__



class movementDetection(object):
    def __init__(self, VideoReader, window=21, indices = None, threshold = 20, height_offset = 64):
        
        assert window%2 == 1, 'not tested for even windiow sizes'
        
        self.video = VideoReader
        self.window = window
        self.threshold = threshold
        self.height_offset = height_offset

        
        if indices is None:
            self.futureIndices = iter(range(self.video.total_frames))
            self.currentIndices = iter(range(self.video.total_frames))
        else:
            self.futureIndices = iter(indices)
            self.currentIndices = iter(indices)
        self.background = None
        
        #we cache the require dimages in a big array
        self.frameCache = np.empty((self.window, self.video.height, self.video.width),dtype='float32') * np.nan
        self.diffCache = np.empty((self.window, self.video.height-self.height_offset, self.video.width),dtype='float32') * np.nan
        #the index cache is only for double checkign that we did not mess anything up
        self.indexCache = np.zeros(self.window ,dtype='int') - 99
        # the window is centered about the current frame, ie.e. extends into the future
        self.w1, self.w2 = int(np.floor(self.window/2)), int(np.ceil(self.window/2))
        # get future frames and add them to the cache
        for ii in range(self.w2):
            self._nextFutureFrame()
        #get mean backgournd for the first time
        self._aveBackground()
        
    def _nextFutureFrame(self):

        '''
        update background by adding future frame to it
        roll cache backwards and put newest image at the end
        '''
        try:
            index = next(self.futureIndices)
        except StopIteration:
            logging.info(f'background estimate at end of video')
            index = -99
            frame = np.nan
        else:
            _, frame = self.video.get(index)


        #idea: roll index instad of array... itertools.cycle(range(window))
        self.frameCache = np.roll(self.frameCache, -1, axis=0)
        self.diffCache = np.roll(self.diffCache, -1, axis=0)
        self.indexCache = np.roll(self.indexCache, -1, axis=0)
        self.frameCache[-1] = frame
        self.diffCache[-1] = self.frameCache[-2, self.height_offset:] - self.frameCache[-1, self.height_offset:]
        self.indexCache[-1] = index
        return
    
    def _getCurrentFrame(self):
        #just to check whether we are sane
        self.currentIndex = next(self.currentIndices)
        assert self.indexCache[self.w1] == self.currentIndex
        
        return self.frameCache[self.w1]
    
    def _aveBackground(self):
        ### avergae backgorund considering only non movign parts
        frameCache1 = deepcopy(self.frameCache)[:, self.height_offset:]
        frameCache1[np.abs(self.diffCache) > (2/3 * self.threshold)] = np.nan
        assert np.min(bn.nansum(np.isfinite(frameCache1), axis=0))> (1/4 * self.threshold)
        self.background = bn.nanmean(frameCache1, axis=0)    

    def _extractForeground(self):
        # decide what is moving based on precalculated differences
        frame = self._getCurrentFrame()
        print('next line not tested')
        diff = self.background - frame 
        return (diff > self.threshold).astype('uint8'), frame
        
        
    def updateBackground(self):
        ### user routine to do everything in the rigth order
        try:
            foreground, frame = self._extractForeground()
        except StopIteration:
            logging.warning(f'reached end of video')
            return None, None
        self._nextFutureFrame()
        self._aveBackground()  

        return foreground, frame.astype('uint8')


#                 """2. use difference to curent frame to detect foreground"""



class detectedParticles(object):
    def __init__(self,
                 pidOffset=0,
                 history=500,
                 dist2Threshold=400,
                 detectShadows=False,
                 showResult=True,
                 verbosity=10,
                 composite = True,
                 maxDarkestPoint = 20,
                 minDmax = 5,
                 maxNParticle = 25,
                 minBlur=20,
                 minArea=3,
                 height_offset=64,
                 maskCorners=None,
                 cropImage = None, #(offsetX, offsetY)
                 function=cv2.createBackgroundSubtractorKNN,
                 ):

        self.version = __version__.split(".")[0]

        self.verbosity = verbosity
        self.showResult = showResult
        self.composite = composite

        self.minDmax = minDmax
        self.maxDarkestPoint = maxDarkestPoint
        self.maxNParticle = maxNParticle
        self.minBlur = minBlur
        self.minArea = minArea
        self.height_offset = height_offset #height of sttaus bar
        self.maskCorners = maskCorners
        self.cropImage = cropImage

        self.all = {}
        self.lastFrame = {}
        self.pp = pidOffset
        self.fgMask = None

        self.nMovingPix = 0
        self.blur = 0
        self.capture_id = None
        self.record_id = None
        self.capture_time = None
        self.record_time = None
        self.nParticle = 0

        #history 500, threshold=400
        self.backSub = function(
            history=history,
            dist2Threshold=dist2Threshold,
            detectShadows=detectShadows
        )



        return

    def update(self, frame, pp, capture_id, record_id, capture_time, record_time, training=False):

        self.capture_id = capture_id
        self.record_id = record_id
        self.capture_time = capture_time
        self.record_time = record_time
        self.lastFrame = {}
        if (self.verbosity > 2):
            print("particles.update", "FRAME", pp,  'Start %s' % 'update')

        self.frame = frame[self.height_offset:]
        if self.maskCorners is not None:
            self.frame[:self.maskCorners, :self.maskCorners] = 0
            self.frame[-self.maskCorners:, :self.maskCorners] = 0
            self.frame[-self.maskCorners:, -self.maskCorners:] = 0
            self.frame[:self.maskCorners, -self.maskCorners:] = 0
        
        if self.cropImage is not None:
            offsetX, offsetY = self.cropImage
            self.frame = self.frame[offsetY:-offsetY, offsetX:-offsetX]


        self.fgMask = self.backSub.apply(self.frame)

        if training :
            return True

        if self.fgMask.max() == 0:
            print("particles.update", "FRAME", pp, 'nothing is moving')
            return True


        self.frame4drawing = self.frame.copy()

        self.nMovingPix = self.fgMask.sum()/255

        # thresh = cv2.dilate(self.fgMask, None, iterations=2)
        cnts = cv2.findContours(self.fgMask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        self.cnts = list()
        for cnt in cnts[0]:
            if cnt.shape[0] >2:
                #skip very small particles
                self.cnts.append(cnt)

        self.nParticle = len(self.cnts)
        print("particles.update", "FRAME", pp, 'found', self.nParticle, 'particles')

        if self.nParticle > self.maxNParticle:
            print("particles.update", "FRAME", pp, 'SKIPPED. more than', self.maxNParticle, 'particles')
            return False


        # loop over the contours
        for cnt in self.cnts:

            added, particle1 = self.add(self.frame, cnt, verbosity=self.verbosity, composite = self.composite)
            if added:
                print("particles.update", "PID", pp, "Added with area =%i"%(particle1.area))
                if self.showResult:
                    particle1.drawContour(self.frame4drawing)
                    particle1.annotate(self.frame4drawing, extra='added')
            else:
                print("particles.update", "PID", pp, "Not added")
        return True


    def add(self, *args, **kwargs):
        newParticle = True
        added = False

        # try:
        self.lastParticle = singleParticle(self.capture_id, self.record_id, self.capture_time, self.record_time, self.pp, *args, **kwargs)
        # except ZeroDivisionError:
        #     self.lastParticle = None
        #     return added, self.lastParticle


        if self.lastParticle.Dmax < self.minDmax:
            print("particles.add", "PID", self.lastParticle.pid, "too small", self.lastParticle.Dmax)
            return added, self.lastParticle

        if self.lastParticle.pixMin > self.maxDarkestPoint:
            print("particles.add", "PID", self.lastParticle.pid, "too large maxDarkestPoint", self.lastParticle.pixMin)
            return added, self.lastParticle

        if self.lastParticle.blur < self.minBlur:
            print("particles.add", "PID", self.lastParticle.pid, "too small minBlur", self.lastParticle.blur)
            return added, self.lastParticle

        if self.lastParticle.area < self.minArea:
            print("particles.add", "PID", self.lastParticle.pid, "too small area", self.lastParticle.area)
            return added, self.lastParticle


        particleFrame = self.lastParticle.particleBox
        if np.prod(particleFrame.shape) == 0:
            print("particles.add", "PID", self.lastParticle.pid, "flat", particleFrame.shape)
            return added, self.lastParticle

        print("FILTER", particleFrame.mean(), particleFrame.max()   , particleFrame.std(ddof=1)   )

        # plt.figure()
        # plt.title(self.lastParticle.pid)
        # plt.imshow(particleFrame)
        # plt.show()
        self.all[self.pp] = self.lastParticle
        self.lastFrame[self.pp] = self.lastParticle
        print("particles.add", "PID", self.lastParticle.pid, "Added")
        self.pp += 1
        added = True

        return added, self.lastParticle

    def collectResults(self):
        self.particleProps = xr.Dataset(coords = {'pid':list(self.all.keys()), 'percentiles': range(10,100,10)}) 
        for key in ['Dmax', 'Dmin', 'area', 'aspectRatio', 'angle', 'roi', 'x', 'y', 'perimeter', 'pixMin', 'pixMax', 'pixMean', 'pixStd', 'pixSkew', 'pixKurtosis', 'blur', 'capture_id', 'record_id', 'capture_time', 'record_time', 'touchesBorder']:
            arr = []
            for i in self.all.values():
                arr.append(getattr(i, key))
            if key == 'roi':
                self.particleProps[key] = xr.DataArray(arr,coords=[self.particleProps.pid, ['x','y','w','h']], dims=('pid','ROI_elements'))
            elif key == 'touchesBorder':
                self.particleProps[key] = xr.DataArray(arr, coords=[self.particleProps.pid, ['left', 'right','top','bottom']], dims=('pid','side'))
            else:
                self.particleProps[key] = xr.DataArray(arr,coords=[self.particleProps.pid], dims=('pid'))

            # We do not need 64 bit accuracy here and can save storage space
            if self.particleProps[key].dtype == np.float64:
                self.particleProps[key] = self.particleProps[key].astype(np.float32)
            elif key in [ 'roi','x', 'y','pixMin', 'pixMax']:
                self.particleProps[key] = self.particleProps[key].astype(np.int16)
        percentiles = []
        for i in self.all.values():
            percentiles.append(getattr(i, 'pixPercentiles'))
        self.particleProps['pixPercentiles'] = xr.DataArray(np.array(percentiles).astype(np.float32),coords=[self.particleProps.pid, self.particleProps.percentiles])
        return self.particleProps


    @property
    def N(self):
        return len(self.all)

    @property
    def pids(self):
        return list(self.all.keys())



class singleParticle(object):
# TrackerKCF_create
    def __init__(self, capture_id, record_id, capture_time, record_time,  pp1, frame1, cnt, verbosity=0, composite = True):
        self.verbosity = verbosity
        #start with negative random id
        self.pid = pp1#np.random.randint(-999,-1) 
        self.record_id = record_id
        self.capture_id = capture_id
        self.capture_time = capture_time
        self.record_time = record_time
        self.cnt = cnt
        self.composite = composite
        self.version = __version__.split(".")[0]

        self.first = True
        self.lost = False

        self.roi = tuple(int(b) for b in cv2.boundingRect(self.cnt))
        self.Cx, self.Cy, self.Dx, self.Dy = self.roi
        if self.verbosity > 1:
            print("singleParticle.__init__", "PID", self.pid, 'found singleParticle at %i,%i, %i, %i' % self.roi)

        self.frame = frame1
        self.particleBox = self.extractRoi(frame1)
        self.frameHeight, self.frameWidth = self.frame.shape[:2]

        self.touchesBorder = [
            self.roi[0] == 0, 
            (self.roi[0] + self.roi[2]) == self.frameWidth, 
            self.roi[1] == 0, 
            (self.roi[1] + self.roi[3]) == self.frameHeight
            ]

        fill_color = 255   # any  color value to fill with
        mask_value = 1     # 1 channel  (can be any non-zero uint8 value)

        # our  - some `mask_value` contours on black (zeros) background, 
        stencil  = np.zeros_like(frame1)
        cv2.fillPoly(stencil, [self.cnt], mask_value)

        sel      = stencil != mask_value # select everything that is not mask_value

        self.particleBoxCropped = deepcopy(frame1)
        self.particleBoxCropped[stencil != mask_value] = fill_color
        self.particleBoxCropped = self.extractRoi(self.particleBoxCropped)
        particleBoxData = frame1[stencil == mask_value]

        self.pixMin = particleBoxData.min()
        self.pixMax = particleBoxData.max()
        self.pixMean = particleBoxData.mean()
        self.pixPercentiles = np.percentile(particleBoxData, range(10,100,10))#, interpolation='nearest'
        self.pixStd = np.std(particleBoxData, ddof=1)
        self.pixSkew = scipy.stats.skew(particleBoxData)
        self.pixKurtosis  = scipy.stats.kurtosis(particleBoxData)

        self.rect = cv2.minAreaRect(self.cnt)
        #todo: consider https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
        center, dims, self.angle = self.rect
        self.angle = self.angle
        self.Dmax = max(dims)
        self.Dmin = min(dims)
        self.aspectRatio = self.Dmin / self.Dmax

        self.area = cv2.contourArea(self.cnt)
        M = cv2.moments(self.cnt)
        #https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
        try:
           self.centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        except ZeroDivisionError:
            self.centroid = 0,0
        self.x, self.y = self.centroid
        self.perimeter = cv2.arcLength(self.cnt,True)
        self.blur = cv2.Laplacian(self.particleBox, cv2.CV_8U).var()


        return

    def dropImages(self):
        '''
        Save memory
        '''
        self.particleBox = None
        self.particleBoxCropped = None
        self.frame = None


    def __repr__(self):
        props = "#"*30
        props += '\n'
        props += 'PID: %i\n'%(self.pid)
        props += 'record_id, capture_id, capture_time, record_time: %i %i %s %s\n'%(self.record_id, self.capture_id, self.capture_time, self.record_time)
        props += 'Dmax, Dmin: %i %i\n'%(self.Dmax, self.Dmin)
        props += 'aspectRatio: %f\n'%(self.aspectRatio)
        props += 'angle: %f\n'%(self.angle)
        props += 'ROI (x, y, w, h): %i %i %i %i\n'%self.roi
        props += 'Area: %i\n'%self.area
        props += 'Centroid: %i, %i\n'%self.centroid
        props += 'Perimeter: %i\n'%self.perimeter
        props += 'pixMin/pixMax: %i %i %f\n'%(self.pixMin, self.pixMax, self.pixMin/self.pixMax)
        props += 'pixPercentiles: ' + str(self.pixPercentiles.tolist()) + '\n'
        props += 'pixMean, pixStd, pixSkew, pixKurtosis: %.1f %.1f %.1f %.1f \n'%(self.pixMean, self.pixStd, self.pixSkew, self.pixKurtosis)
        props += 'Blur: %f\n'%self.blur
        props += f'touches border: {self.touchesBorder} \n'
        props += "#"*30
        props += '\n'
        return props

    def extractRoi(self, frame):
        x, y, w, h = self.roi
        if len(frame.shape) == 3:
            frame = frame[:,:,0]
        return frame[y:y+h, x:x+w]

    def drawContour(self, frame, color=(0, 255, 0)):
        assert not self.lost
        (x, y, w, h) = self.roi
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), color, 2)


        cv2.drawContours(frame, [self.cnt],0,np.array(color) * 2/3,2)
        box = cv2.boxPoints(self.rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,np.array(color) * 1/3,2)

        return frame

    def annotate(self, frame, color=(0, 255, 0), extra=''):
        cv2.putText(frame, '%i %s' % (self.pid, extra),
                    (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)



