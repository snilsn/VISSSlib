# -*- coding: utf-8 -*-

import sys
import os
import itertools
import subprocess

# import matplotlib.pyplot as plt
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

import logging
import logging.config
log = logging.getLogger()

from copy import deepcopy

from . import tools
from . import metadata
from . import files


from . import __version__

print("next time look at background and store it!")

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

def checkMotion(subFrame, oldFrame, threshs):
    '''
    Check whether something is moving - identical to C code
    '''

    if oldFrame is None:
        oldFrame = np.zeros(subFrame.shape, dtype=np.uint8)

    nChangedPixel = np.zeros(
                ( len(threshs)), dtype=int)

    absdiff = cv2.absdiff(subFrame, oldFrame)

    for tt, thresh in enumerate(threshs):
            nChangedPixel[tt] = ((absdiff >= thresh)).sum()

    return nChangedPixel

def detectParticles(fname,
                    settings,
                    testing=False,
                    writeImg=True,
                    historySize=20,
                    minDmax4Plotting=17,
                    minBlur4Plotting=100,
                    testMovieFile = True,
                    ):

    logging.config.dictConfig(tools.get_logging_config('detection_run.log'))
    log = logging.getLogger()

    config = tools.readSettings(settings)

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

    fn = files.Filenames(fname, config, __version__)
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

    if not os.path.isfile(fname):
        log.error('ERROR Unable to open: ' + fname)
        raise RuntimeError('ERROR Unable to open: ' + fname)

    fnamesV = fn.fnameAllThreads
    if len(fnamesV) == 0:
        raise RuntimeError("len(fnamesV) == 0")

    fnameM = [f.replace(config["movieExtension"], "txt") for f in fnamesV.values()]

    # metaData, nDroppedFrames = metadata.getMetaData(
    #     fnameM, camera, config, testMovieFile=True)
    metaData = xr.open_dataset(fn.fname.metaFrames)

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
        for fnameV in fnamesV:
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
                log.info('OK ', fnameV)
                pass
            else:
                log.info('BROKEN ', fnameV)
                brokenFile = '%s.broken' % fnameV
                repairedFile = '%s_fixed.%s' % (".".join(fnameV.split(".")[:-1]), config["movieExtension"])
                process = subprocess.Popen(['/home/mmaahn/bin/untrunc', goodFile, fnameV],
                                           stdout=subprocess.PIPE,
                                           universal_newlines=True)
                while True:
                    output = process.stdout.readline()
                    log.info(output.strip())
                    return_code = process.poll()
                    if return_code is not None:
                        log.info('RETURN CODE', return_code)
                        # Process has finished, read rest of the output
                        for output in process.stdout.readlines():
                            log.info(output.strip())
                        break
                if return_code == 0:
                    os.rename(fnameV, brokenFile)
                    os.rename(repairedFile, fnameV)
                else:
                    log.error('WAS NOT ABLE TO FIX %s'%fnameV)
                    raise RuntimeError
                log.info('REPAIRED ', fnameV)


    log.info(f"{fn.year}, {fn.month}, {fn.day}, {fname}")
    fn.createDirs()

    log.info('Processing %s' % fn.fname.level1detect)

    hasData = False

    particleProperties = []

    nFrames = len(metaData.capture_time)
    pps = range(nFrames)

    inVid = {}
    nFramesVid = 0
    for nThread, fnameV in fnamesV.items():
        inVid[nThread] = cv2.VideoCapture(fnameV)
        nFramesVid += int(inVid[nThread].get(cv2.CAP_PROP_FRAME_COUNT))

    # lets assume that dropped rames are all at the end now and we figured all the other issues out

    # if ((site == 'mosaic') and nFrames+nDroppedFrames-1 == nFramesVid): 
    #     log.warning('lengths do not match %i %i (bug of MOSAiC C code) ' %
    #                 (nFrames+nDroppedFrames-1, nFramesVid)) 
    # elif ((site == 'mosaic') and nFrames+nDroppedFrames == nFramesVid-1):
    #     log.warning('lengths do not match %i %i (bug of MOSAiC C code) ' %
    #                 (nFrames+nDroppedFrames, nFramesVid-1))

    # elif (nFrames+nDroppedFrames != nFramesVid): 
    #     log.error('lengths do not match %i %i ' %
    #               (nFrames+nDroppedFrames, nFramesVid)) 

    #     raise RuntimeError('lengths do not match %i %i ' %
    #                        (nFrames, nFramesVid)) 

    if (nFrames < historySize):
        log.info('too few frames %i ' %
                 (nFrames))

        with open('%s.nodata' % fn.fname.metaDetect, 'w') as f:
            f.write('too few frames %i ' %
                    (nFrames))
        with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
            f.write('too few frames %i ' %
                    (nFrames))

        sys.exit(0)

    snowParticles = detectedParticles(
        maxDarkestPoint=130,  # 100 # 40,
        minDmax=3,
        maxNParticle=60,
        minBlur=0,  # 2, #10,  # 30,
        function=cv2.createBackgroundSubtractorKNN,
        history=historySize,
        cropImage=cropImage
    )

    inVidIter = itertools.cycle(list(inVid.values()))
    for tt in pps[:historySize]:
        print('training', tt)
        ret, frame = next(inVidIter).read()
        res = snowParticles.update(frame,
                                   tt,
                                   int(metaData.capture_id[tt].values),
                                   int(metaData.record_id[tt].values),
                                   metaData.capture_time[tt].values,
                                   metaData.record_time[tt].values,
                                   training=True)

    for nThread in inVid.keys():
        inVid[nThread].release()
        inVid[nThread] = cv2.VideoCapture(fnamesV[nThread])

    foundParticles = np.zeros((nFrames, len(Dbins)))
    movingObjects = np.zeros((nFrames))

    frame = None
    for pp in pps:

        #     if pp > 200:
        #         break

        metaData1 = metaData.isel(capture_time=pp)

        nThread = int(metaData1.nThread.values)

        rr = int(metaData1.record_id.values)
        log.info('Image %i from thread %i, frame %i ' % (pp, nThread, rr))

        oldFrame = deepcopy(frame)

        # sometime we have to skip frames due to missing meta data
        # we dont want to take the risk to jump to an index and get the wrong frame
        if int(inVid[nThread].get(cv2.CAP_PROP_POS_FRAMES)) < rr:
            while(int(inVid[nThread].get(cv2.CAP_PROP_POS_FRAMES)) < rr):
                print('fast forwarding', int(
                    inVid[nThread].get(cv2.CAP_PROP_POS_FRAMES)), rr, )
                _, _ = inVid[nThread].read()
        elif int(inVid[nThread].get(cv2.CAP_PROP_POS_FRAMES)) > rr:
            raise RuntimeError('Cannot go back!')

        ret, frame = inVid[nThread].read()
        if frame is None:
            raise ValueError('TOO FEW FRAMES????')

        threshs = np.array([20, 30, 40, 60, 80, 100, 120])
        nChangedPixel = checkMotion(frame, oldFrame, threshs)

        passesThreshold = nChangedPixel >= minMovingPixels
        if not passesThreshold.any():
            print('NOT moving %i' % pp)
            continue
        else:
            print('IS moving %i' % pp)

        res = snowParticles.update(frame,
                                   pp,
                                   int(metaData1.capture_id.values),
                                   int(metaData1.record_id.values),
                                   metaData1.capture_time.values,
                                   metaData1.record_time.values)
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
            if testing:
                plt.figure(figsize=(10, 10))
                plt.imshow(snowParticles.frame4drawing,
                           cmap='gray', vmin=0, vmax=255)
                plt.title('SKIPPED %s %i %s' %
                          (camera, pp, timestamp))
                plt.show()
                plt.savefig('skiped_%s_%i_%s.png' %
                            (camera, pp, timestamp))

        for part in snowParticles.lastFrame.values():
            hasData = True

            if writeImg:

                if not (
                    (part.Dmax < minDmax4Plotting) or
                    (part.blur < minBlur4Plotting) or
                    np.any(part.touchesBorder)
                ):
                    pidStr = '%07i' % part.pid
                    imFolder = fn.imagepath.imagesL1detect.format(ppid=pidStr[:4])
                    imName = '%s.png' % (pidStr)
                    log.info('writing %s/%s' % (imFolder, imName))
                    os.system('mkdir -p %s' % (imFolder))
                    cv2.imwrite('%s/%s' %
                                (imFolder, imName),
                                part.particleBox)

            part.dropImages()

    for nThread in inVid.keys():
        inVid[nThread].release()
    foundParticles = xr.DataArray(foundParticles,
                                  coords=[metaData.capture_time,
                                          xr.DataArray(Dbins, dims=['Dmax'])])
    movingObjects = xr.DataArray(movingObjects,
                                 coords=[metaData.capture_time])

    metaData['foundParticles'] = foundParticles
    metaData['movingObjects'] = movingObjects

    metaData["foundParticles"] = metaData["foundParticles"].astype(np.uint32)
    metaData["movingObjects"] = metaData["movingObjects"].astype(np.uint32)

    comp = dict(zlib=True, complevel=5)
    encoding = {
        var: comp for var in metaData.data_vars}
    metaData.to_netcdf(fn.fname.metaDetection, engine="netcdf4")

    if hasData and not testing:
        snowParticlesXR = snowParticles.collectResults()

        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in snowParticlesXR.data_vars}

        snowParticlesXR.to_netcdf(fn.fname.level1detect, encoding=encoding, engine="netcdf4")

    else:
        with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
            f.write('no data')

    return 0
