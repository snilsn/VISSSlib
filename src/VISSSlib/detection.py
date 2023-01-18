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
import logging.config
log = logging.getLogger("dtection")


# class movementDetection(object):
#     def __init__(self, VideoReader, window=21, indices = None, threshold = 20, height_offset = 64):
#         '''
#         Depreciated!
#         '''

#         assert window%2 == 1, 'not tested for even windiow sizes'

#         self.video = VideoReader
#         self.window = window
#         self.threshold = threshold
#         self.height_offset = height_offset


#         if indices is None:
#             self.futureIndices = iter(range(self.video.total_frames))
#             self.currentIndices = iter(range(self.video.total_frames))
#         else:
#             self.futureIndices = iter(indices)
#             self.currentIndices = iter(indices)
#         self.background = None

#         #we cache the require dimages in a big array
#         self.frameCache = np.empty((self.window, self.video.height, self.video.width),dtype='float32') * np.nan
#         self.diffCache = np.empty((self.window, self.video.height-self.height_offset, self.video.width),dtype='float32') * np.nan
#         #the index cache is only for double checkign that we did not mess anything up
#         self.indexCache = np.zeros(self.window ,dtype='int') - 99
#         # the window is centered about the current frame, ie.e. extends into the future
#         self.w1, self.w2 = int(np.floor(self.window/2)), int(np.ceil(self.window/2))
#         # get future frames and add them to the cache
#         for ii in range(self.w2):
#             self._nextFutureFrame()
#         #get mean backgournd for the first time
#         self._aveBackground()

#     def _nextFutureFrame(self):

#         '''
#         update background by adding future frame to it
#         roll cache backwards and put newest image at the end
#         '''
#         try:
#             index = next(self.futureIndices)
#         except StopIteration:
#             logging.info(f'background estimate at end of video')
#             index = -99
#             frame = np.nan
#         else:
#             _, frame = self.video.get(index)


#         #idea: roll index instad of array... itertools.cycle(range(window))
#         self.frameCache = np.roll(self.frameCache, -1, axis=0)
#         self.diffCache = np.roll(self.diffCache, -1, axis=0)
#         self.indexCache = np.roll(self.indexCache, -1, axis=0)
#         self.frameCache[-1] = frame
#         self.diffCache[-1] = self.frameCache[-2, self.height_offset:] - self.frameCache[-1, self.height_offset:]
#         self.indexCache[-1] = index
#         return

#     def _getCurrentFrame(self):
#         #just to check whether we are sane
#         self.currentIndex = next(self.currentIndices)
#         assert self.indexCache[self.w1] == self.currentIndex

#         return self.frameCache[self.w1]

#     def _aveBackground(self):
#         ### avergae backgorund considering only non movign parts
#         frameCache1 = deepcopy(self.frameCache)[:, self.height_offset:]
#         frameCache1[np.abs(self.diffCache) > (2/3 * self.threshold)] = np.nan
#         assert np.min(bn.nansum(np.isfinite(frameCache1), axis=0))> (1/4 * self.threshold)
#         self.background = bn.nanmean(frameCache1, axis=0)

#     def _extractForeground(self):
#         # decide what is moving based on precalculated differences
#         frame = self._getCurrentFrame()
#         print('next line not tested')
#         diff = self.background - frame
#         return (diff > self.threshold).astype('uint8'), frame


#     def updateBackground(self):
#         ### user routine to do everything in the rigth order
#         try:
#             foreground, frame = self._extractForeground()
#         except StopIteration:
#             logging.warning(f'reached end of video')
#             return None, None
#         self._nextFutureFrame()
#         self._aveBackground()

#         return foreground, frame.astype('uint8')


# #                 """2. use difference to curent frame to detect foreground"""

'''
on single CPU:

mog
24.8 ms ± 61.4 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
mog2
17.2 ms ± 113 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
knn
12.6 ms ± 99.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
gmg
64.7 ms ± 197 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
cnt
4.85 ms ± 14.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
gsoc
158 ms ± 814 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
lsbp
473 ms ± 8.09 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
VISSSlib wo training
1.07 ms ± 26.9 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

'''


# decided not to use own class due to 1) perfoamnce reasons and doubts hwethe rthe
# simple background model can handle different light situations properly
class movementDetection(object):
    def __init__(self, threshold=20, history=100, trainingInterval=140):
        self.backgrounds = None
        self.history = history
        self.trainingInterval = trainingInterval
        self.threshold = threshold
        self.ii = 0
        self.background = 0

    def apply(self, frame):
        if (self.ii % self.history == 0):
            try:
                self.background = np.mean(
                    self.backgrounds, axis=0,).astype(np.uint8)
            except np.AxisError:
                self.background = 0

        diff = cv2.absdiff(frame, self.background)
        mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]
        if (self.ii < self.history) or (self.ii % self.trainingInterval == 0):
            self.setBackground(frame)
        self.ii += 1
        return mask

    def getBackgroundImage(self):
        return self.background

    def setBackground(self, frame):
        if self.backgrounds is None:
            y, x = frame.shape
            self.backgrounds = np.zeros(
                (self.history, y, x), dtype=np.uint8) + frame
        else:
            self.backgrounds = np.roll(self.backgrounds, 1, axis=0)
            self.backgrounds[0] = frame


class detectedParticles(object):
    def __init__(self,
                 config,
                 pidOffset=0,
                 trainingSize=500,
                 verbosity=0,
                 composite=True,
                 minContrast=20,
                 minDmax=2,
                 minBlur=10,
                 minArea=2,
                 erosionTestThreshold=0.06,
                 height_offset=64,
                 maskCorners=None,
                 cropImage=None,  # (offsetX, offsetY)
                 backSub=cv2.bgsegm.createBackgroundSubtractorCNT,
                 backSubKW={"minPixelStability": 10, "maxPixelStability": 100},
                 applyCanny2Frame=False,
                 applyCanny2Particle=True,  # much faster than 2 whole Frame!
                 dilateIterations=1,
                 blurSigma=1.5,
                 minAspectRatio=None,  # testing only
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

        self.applyCanny2Frame = applyCanny2Frame
        self.applyCanny2Particle = applyCanny2Particle
        self.dilateIterations = dilateIterations
        self.blurSigma = blurSigma
        self.minAspectRatio = minAspectRatio

        return

    def update(self, frame, pp, capture_id, record_id, capture_time, record_time, nThread, training=False, testing=[], blockingThresh=None):

        self.testing = testing
        self.capture_id = capture_id
        self.record_id = record_id
        self.capture_time = capture_time
        self.record_time = record_time
        self.nThread = nThread
        self.lastFrame = {}
        if (self.verbosity > 2):
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

        if "input" in testing:
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

        self.nMovingPix = self.fgMask.sum()//255
        if self.nMovingPix == 0:
            print("particles.update", "FRAME", pp, 'capture_time',
                  capture_time, 'nothing is moving')
            if "nonMovingFgMask" in testing:
                print("SHOWING", "nonMovingFgMask")
                tools.displayImage(self.fgMask)
            return True

        # it can happen that the background subtraction has little gaps
        # if self.fgMask is not None:
        #     self.fgMask = cv2.dilate(self.fgMask, None, iterations=1)
        #     self.fgMask = cv2.erode(self.fgMask, None, iterations=1)

        if "movingInput" in testing:
            print("SHOWING", "movingInput")
            tools.displayImage(self.frame)

        if "fgMaskWithHoles" in testing:
            print("SHOWING", "fgMaskWithHoles")
            tools.displayImage(self.fgMask)

        if self.applyCanny2Frame:
            self.fgMask = self.applyCannyFilter(self.frame, self.fgMask)

            self.nMovingPix2 = self.fgMask.sum()//255
            if self.nMovingPix2 == 0:
                print("particles.update", "FRAME", pp,
                      'nothing is moving after Canny filter')
                return True

        if "fgMask" in testing:
            print("SHOWING", "fgMask")
            tools.displayImage(self.fgMask)

        if ("result" in testing) or ("resultAddedOnly" in testing):
            self.frame4drawing = av.cvtGray(self.frame)

        # thresh = cv2.dilate(self.fgMask, None, iterations=2)

        cnts, _ = cv2.findContours(self.fgMask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        self.cnts = list()
        for cc, cnt in enumerate(cnts):
            if cnt.shape[0] > 2:
                # skip very small particles
                self.cnts.append(cnt)

        self.nParticle = len(self.cnts)
        log.info(f"particles.update FRAME {pp} capture_time {capture_time} found {self.nParticle} particles")

        if self.nParticle > self.maxNParticle:
            log.info(f"particles.update FRAME {pp} SKIPPED. more than {self.maxNParticle} particles")
            return False

        added = False
        # loop over the contours
        for cnt in self.cnts:

            added = self.add(self.frame, self.fgMask, cnt, testing,
                             verbosity=self.verbosity, composite=self.composite)

        if "result" in testing:
            if len(self.cnts) > 0:
                print("SHOWING", "result")
                tools.displayImage(self.frame4drawing)
        if "resultAddedOnly" in testing:
            if len(self.cnts) > 0 and added:
                print("SHOWING", "resultAddedOnly")
                tools.displayImage(self.frame4drawing)

        return True

    def applyCannyFilter(self, frame, fgMask, useSkimage=False, threshold1=0, threshold2=25, doubleDynamicRange=True):

        if doubleDynamicRange:
            frame = av.doubleDynamicRange(frame)

        if useSkimage:
            from skimage import feature
            from skimage.util import img_as_ubyte

            # Canny filter gets confused if the feature is directly at the edge, so make the moving
            # mask a little larger
            fgMask4Canny = (cv2.dilate(
                fgMask, None, iterations=dilateIterations)//255).astype(bool)
            # , mask = (self.fgMask//255).astype(bool)))
            fgMaskCanny = feature.canny(frame, sigma=2, mask=fgMask4Canny)
            fgMaskCanny = img_as_ubyte(fgMaskCanny)

        else:

            # blur image, required to make algoprithm stable
            if self.blurSigma != 0:
                frame = cv2.GaussianBlur(frame, (0, 0), self.blurSigma)
            # frame = cv2.bilateralFilter(frame, 5,70,2)

            # apply Canny filter, take low limits becuase we are reduced to moving parts
            fgMaskCanny = cv2.Canny(
                frame, threshold1, threshold2, L2gradient=True, apertureSize=3)

        if self.dilateIterations > 0:
            # close gaps by finding contours, dillate, fill, and erode them
            fgMaskCanny = cv2.dilate(
                fgMaskCanny, None, iterations=self.dilateIterations)
            cnts = cv2.findContours(fgMaskCanny, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
            fgMaskCanny = cv2.fillPoly(fgMaskCanny, pts=cnts, color=255)
            fgMaskCanny = cv2.erode(
                fgMaskCanny, None, iterations=self.dilateIterations)

        # and condition, i.e. make sure both filters detected something
        if fgMask is not None:
            fgMaskCanny = (fgMask//255) * fgMaskCanny

        return fgMaskCanny

    def add(self, frame1, fgMask, cnt, testing, **kwargs):

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
            particleBoxMaskPlus, xOffset, yOffset = extractRoi(
                roi, fgMask, extra=extra)
            particleBoxPlus, xOffset, yOffset = extractRoi(
                roi, frame1, extra=extra)
            particleBoxMaskPlus = self.applyCannyFilter(
                particleBoxPlus, particleBoxMaskPlus)
            cntsTmp, hierarchy = cv2.findContours(
                particleBoxMaskPlus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cnts = list()
            cntChildren = list()
            for cc, cnt in enumerate(cntsTmp):
                if cnt.shape[0] > 2:  # skip very small particles
                    if hierarchy[0][cc][3] == -1:  # i.e. no parents!
                        cnts.append(cnt)

                        iiChildren = np.where(hierarchy[0][:, 3] == cc)[0]
                        log.debug(f"found children {iiChildren} of {cc}")
                        if len(iiChildren) == 0:
                            cntChildren.append(None)
                        else:
                            cntChildren.append(list())
                        for ii in iiChildren:
                            thisChild = cntsTmp[ii]
                            if thisChild.shape[0] > 2:
                                cntChildren[-1].append(thisChild)
            frame4sp = particleBoxPlus
            mask4sp = particleBoxMaskPlus

            log.debug(f"found {len(cnts)} instead of 1 particle after canny filter")
        else:
            cnts = [cnt]
            frame4sp = frame1
            mask4sp = fgMask
            xOffset = yOffset = 0
            cntChildren = [None]

        # loop in case more than one particle is in single moving area
        for cnt, cntChild in zip(cnts, cntChildren):
            added = False
            newParticle = True

            # try:
            self.lastParticle = singleParticle(self, self.capture_id, self.record_id, self.capture_time, self.record_time,
                                               self.nThread, self.pp, frame4sp, mask4sp, cnt, cntChild, xOffset, yOffset, **kwargs)
            # except ZeroDivisionError:
            #     self.lastParticle = None
            #     return added, self.lastParticle
            if not self.lastParticle.success:
                log.info(" ".join(["particles.add", "PID",
                                   self.lastParticle.pid, "empty"]))
            else:

                ratio = self.lastParticle.perimeterEroded/self.lastParticle.perimeter
                if self.lastParticle.Dmax < self.minDmax:
                    log.info(" ".join(["particles.add", "PID", "%i" %
                                       self.lastParticle.pid, "too small", "%.2f" % self.lastParticle.Dmax]))

                elif self.lastParticle.particleContrast < self.minContrast:
                    log.info(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid,
                                       "too small minContrast", "%.2f" % self.lastParticle.particleContrast]))

                elif self.lastParticle.blur < self.minBlur:
                    log.info(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid,
                                       "too small minBlur", "%.2f" % self.lastParticle.blur]))

                elif self.lastParticle.area < self.minArea:
                    log.info(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid,
                                       "too small area", "%.2f" % self.lastParticle.area]))

                # erosionTestThreshold does not apply to needles or similalrly shaped particles, also does not work to very small particles
                elif (self.lastParticle.aspectRatio[-1] > 0.4) and (self.lastParticle.Dmax > 5) and (ratio < self.erosionTestThreshold):
                    log.info(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid, "particle not properly detected",
                                       "%.2f" %ratio, "%.2f" %self.erosionTestThreshold, "%.2f" % self.lastParticle.aspectRatio[-1]]))

                # only for testing
                elif (self.minAspectRatio is not None) and (self.lastParticle.aspectRatio[-1] < self.minAspectRatio):
                    log.info(" ".join(["particles.add", "PID", "%i" % self.lastParticle.pid, "particle below arThresh",
                                       "%.2f" % self.lastParticle.aspectRatio[-1], "%.2f" % self.minAspectRatio]))

                elif np.prod(self.lastParticle.particleBox.shape) == 0:
                    log.info(" ".join(["particles.add", "PID", "%i" %
                                       self.lastParticle.pid, "flat", str(particleFrame.shape)]))

                else:

                    log.info(" ".join(["FILTER", "%.2f" % self.lastParticle.particleBox.mean(
                    ), "%.2f" % self.lastParticle.particleBox.max(), "%.2f" % self.lastParticle.particleBox.std(ddof=1)]))

                    self.all[self.pp] = self.lastParticle
                    self.lastFrame[self.pp] = self.lastParticle
                    log.info(
                        " ".join(["particles.add", "PID", "%i" % self.lastParticle.pid, "Added"]))
                    self.pp += 1
                    added = True

            if added:
                # log.info("particles.update")
                # print("TEST4Canny", self.lastParticle.perimeter, self.lastParticle.perimeterEroded, self.lastParticle.perimeterEroded/self.lastParticle.perimeter)

                if ("result" in testing) or ("resultAddedOnly" in testing):
                    print(self.lastParticle)
                    self.frame4drawing = self.lastParticle.drawContour(
                        self.frame4drawing)
                    self.frame4drawing = self.lastParticle.annotate(
                        self.frame4drawing, extra='%.1f' % self.lastParticle.Dmax)
            else:
                log.info("PID %i Not added" % self.lastParticle.pid)

        return added

    def collectResults(self):

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
            'record_time', 'nThread', "aspectRatio"
        ]:
            arr = []
            for i in self.all.values():
                arr.append(getattr(i, key))
            if key in ["position_centroid", "position_upperLeft", "position_circle", "Droi"]:
                self.particleProps[key] = xr.DataArray(
                    arr, coords=[self.particleProps.pid, dim2D], dims=('pid', 'dim2D'))
            # elif key == 'touchesBorder':
            #     self.particleProps[key] = xr.DataArray(arr, coords=[self.particleProps.pid, ['left', 'right','top','bottom']], dims=('pid','side'))
            if key in ["position_fit", "Dfit"]:
                self.particleProps[key] = xr.DataArray(
                    arr, coords=[self.particleProps.pid, fitMethod, dim2D], dims=('pid', 'fitMethod', 'dim2D'))
            if key in ["angle", "aspectRatio"]:
                self.particleProps[key] = xr.DataArray(
                    arr, coords=[self.particleProps.pid, fitMethod], dims=('pid', 'fitMethod'))
            else:
                self.particleProps[key] = xr.DataArray(
                    arr, coords=[self.particleProps.pid], dims=('pid'))

            # We do not need 64 bit accuracy here and can save storage space
            if self.particleProps[key].dtype == np.float64:
                self.particleProps[key] = self.particleProps[key].astype(
                    np.float32)
            elif key in ["position_upperLeft", 'pixMin', 'pixMax', 'nThread', "Droi"]:
                self.particleProps[key] = self.particleProps[key].astype(
                    np.int16)

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
    def __init__(self, parent, capture_id, record_id, capture_time, record_time, nThread,  pp1, frame1, mask1, cnt, cntChild, xOffset, yOffset, verbosity=0, composite=True):
        self.verbosity = verbosity
        self.pid = pp1  # np.random.randint(-999,-1)
        self.record_id = record_id
        self.capture_id = capture_id
        self.capture_time = capture_time
        self.record_time = record_time
        self.nThread = nThread
        self.cnt = cnt
        self.cntChild = cntChild
        self.composite = composite
        self.version = __version__.split(".")[0]

        self.xOffset = xOffset
        self.yOffset = yOffset

        self.roi = np.array([int(b) for b in cv2.boundingRect(self.cnt)])
        # self.Cx, self.Cy, self.Dx, self.Dy = self.roi

        # mask1 has neighbouring particles removed, but internal holes closed
        particleBoxMask1, xo, yo = extractRoi(self.roi, cv2.fillPoly(
            np.zeros_like(frame1), pts=[cnt], color=255))
        # mask 2 is the opposite, internal holes (from movement detection) are open, but neighbor pixels are present
        # particleBoxMask2, xo, yo = extractRoi(self.roi, mask1)
        # combine both masks
        # self.particleBoxMask = ((particleBoxMask1/255)*particleBoxMask2).astype(np.uint8)
        self.particleBox, xo, yo = extractRoi(self.roi, frame1)

        # tools.displayImage(particleBoxMask2, rescale=4)

        # mask shows only internal holes from cnt children if present
        if cntChild is not None:
            particleBoxHoles, xo, yo = extractRoi(self.roi, cv2.fillPoly(
                np.zeros_like(frame1)+255, pts=cntChild, color=0))

            # remove single pixel holes
            # kernel = cv2.getStructuringElement(   cv2.MORPH_CROSS, (3,3) )
            particleBoxHoles = cv2.dilate(particleBoxHoles, None, iterations=1)
            particleBoxHoles = cv2.erode(particleBoxHoles, None, iterations=1)

            # combine masks
            self.particleBoxMask = (
                (particleBoxMask1/255)*particleBoxHoles).astype(np.uint8)

        else:  # no holes detected
            self.particleBoxMask = particleBoxMask1
            particleBoxHoles = None

        if "particleMask" in parent.testing:
            tools.displayImage(particleBoxMask1, rescale=4)
            if particleBoxHoles is not None:
                tools.displayImage(particleBoxHoles, rescale=4)
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
        # if not, contour describes only a line whoch can be detected by
        # looking into how perimeter changes during erosion
        erodeMask = cv2.erode(self.particleBoxMask, None, 1)
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
                  'found singleParticle at %i,%i, %i, %i' % self.roi)

        self.cnt[..., 0] += self.xOffset
        self.cnt[..., 1] += self.yOffset

        if self.cntChild is not None:
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
        annotatedParticleCropped, x1, y1 = extractRoi(
            self.roi, annotatedParticle, extra=extra)

        return annotatedParticleCropped


def extractRoi(roi, frame, extra=0):

    if extra == 0:
        x, y, w, h = roi
        return frame[y:y+h, x:x+w], x, y
    else:
        x, y, w, h = roi
        x1 = x - extra
        x2 = x + w + extra
        y1 = y - extra
        y2 = y + h + extra

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        try:
            ny, nx = frame.shape
        except ValueError:
            ny, nx, nz = frame.shape

        if x2 > nx:
            x2 = nx
        if y2 > ny:
            y2 = ny

        return frame[y1:y2, x1:x2], x1, y1


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


# get trainign data
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
                    applyCanny2Frame=False,
                    applyCanny2Particle=True,
                    dilateIterations=1,
                    blurSigma=1.5,
                    minBlur=10,
                    erosionTestThreshold=0.06,
                    minArea=1,
                    minDmax=2,
                    stopAfter=None,
                    version=__version__
                    ):

    logging.config.dictConfig(tools.get_logging_config('detection_run.log'))
    log = logging.getLogger()

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
    minBlur4Plotting = config["level1detectQuicklook"]["minBlur"]
    minDmax4Plotting = config["level1detectQuicklook"]["minSize"]

    fn = files.Filenames(fname, config, version=version)
    print("running", fn.fname.level1detect)
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

    if len(fnamesV) == 0:
        with open('%s.nodata' % fn.fname.metaDetection, 'w') as f:
            f.write('no data in %s' % fn.fname.metaFrames)
        with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
            f.write('no data in %s' % fn.fname.metaFrames)
        log.warning('no movie files: ' + fname)

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

    for t, fna in fnamesV.items():
        if not os.path.isfile(fna):
            log.info('movie file not found (yet?) ' + fna)
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
                f.write('too few frames %i ' %
                        (nFrames))
            log.error('%s too few frames %i ' %
                      (fn.fname.level1detect, nFrames))
            sys.exit(0)

        fnamesV1 = files.Filenames(
            fname11, config, version=version).fnameMovAllThreads
        trainingFrames1 = _getTrainingFrames(fnamesV1, trainingSize, config)
        log.warning(f"added {len(trainingFrames1)} from {ii} previous file {fname11}.")
        trainingFrames = trainingFrames1 + trainingFrames

        fname1 = fname11
        ii += 1
    # just in case we have now too many frames by adding older files
    trainingFrames = trainingFrames[-trainingSize:]

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

    snowParticles = detectedParticles(
        config,
        trainingSize=trainingSize,
        backSubKW=backSubKW,
        cropImage=cropImage,
        applyCanny2Frame=applyCanny2Frame,
        applyCanny2Particle=applyCanny2Particle,
        dilateIterations=dilateIterations,
        blurSigma=blurSigma,
        minBlur=minBlur,
        backSub=backSub,
        erosionTestThreshold=erosionTestThreshold,
        minArea=minArea,
        minDmax=minDmax,
        minAspectRatio=config.level1detect.minAspectRatio,
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
        print(f"nMovingPixel not found")
        motionChecked = False
    else:
        nChangedPixel = metaData.nMovingPixel.values
        passesThreshold = nChangedPixel >= minMovingPixels
        print(f"{passesThreshold.any(1).sum()/len(metaData.capture_time)*100}% frames are moving")
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

    frame = None

    if writeImg:
        imagesL1detect = tools.imageZipFile(
            fn.fname.imagesL1detect, 'w', compresslevel=9)

    tarRoot = fn.fname.imagesL1detect.split("/")[-1].replace(".tar.bz2", "")

    for pp in pps:

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
                print(metaData1.capture_time.values, 'NOT moving %i' % pp)
                continue
            else:
                print(metaData1.capture_time.values, 'IS moving %i' % pp)

        res = snowParticles.update(frame,
                                   pp,
                                   int(metaData1.capture_id.values),
                                   int(metaData1.record_id.values),
                                   metaData1.capture_time.values,
                                   metaData1.record_time.values,
                                   nThread,
                                   testing=testing,
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
                    (part.Dmax >= minDmax4Plotting) and
                    (part.blur >= minBlur4Plotting)
                ):
                    pidStr = '%07i' % part.pid
                    # imName = '%s/%s/%s.npy' % (tarRoot, pidStr[:4], pidStr)
                    log.info('writing %s %s' %
                             (fn.fname.imagesL1detect, pidStr))

                    # imagesL1detect.addimage(imName, part.particleBoxAlpha)
                    imagesL1detect.addnpy(pidStr, part.particleBoxAlpha)
            if "particle" in testing:
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
        print("closing", fn.fname.imagesL1detect)
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

        metaData = tools.finishNc(metaData)

        metaData.to_netcdf(fn.fname.metaDetection, engine="netcdf4")
        metaData.close()

        if hasData:

            snowParticlesXR = snowParticles.collectResults()
            snowParticlesXR = tools.finishNc(snowParticlesXR)

            snowParticlesXR.to_netcdf(fn.fname.level1detect, engine="netcdf4")
            print("written", fn.fname.level1detect)
            snowParticlesXR.close()
            return snowParticlesXR
        else:
            with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
                f.write('no data')
            print("no data", fn.fname.level1detect)
            return None
    else:
        return None


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
