# -*- coding: utf-8 -*-

# import matplotlib.pyplot as plt
import warnings
import os
import sys

from copy import deepcopy
import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment
from scipy.linalg import block_diag
import scipy.stats
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from tqdm import tqdm


from . import __version__
from . import tools
from . import files
from . import matching

import logging

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#for performance
logDebug = log.isEnabledFor(logging.DEBUG)

_reference_slopes = {
    "visss": 0.089,
    "visss2":0.089,
    "visss3":0.089,
}

_reference_intercepts = {
    "visss": 1.605,
    "visss2":1.463,
    "visss3":1.441,
}
 

# x: x, xvel, y, yvel, z, zvel
#z = x,y,z

def myKF(FirstPos3D, velocityGuess=[0, 0, 50]):
    '''
    define KalmannFilter
    '''

    assert len(velocityGuess) == 3

    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = 1   # time step, we are in frame units!

    kf.F = np.array([[1, dt, 0,  0, 0, 0],
                     [0,  1, 0,  0, 0, 0],
                     [0,  0, 1, dt, 0, 0],
                     [0,  0, 0,  1, 0, 0],
                     [0,  0, 0,  0, 1, dt],
                     [0,  0, 0,  0, 0, 1],
                     ])
    kf.u = 0.

    # measurement function
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     ])

    # measurement noise
    R_std = 2  # px
    kf.R = np.eye(3) * R_std**2

    # process noise
    q = Q_discrete_white_noise(dim=3, dt=dt, var=1)
    kf.Q = block_diag(q, q)
#     #print(kf.Q)

    # prior
    kf.x = np.array([[FirstPos3D[0], velocityGuess[0], FirstPos3D[1],
                      velocityGuess[1], FirstPos3D[2], velocityGuess[2]]]).T
    kf.P = np.eye(6) * 100**2.
    return kf


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(
        self, 
        position, 
        feature, 
        size, 
        trackIdCount, 
        startTime, 
        velocityGuess=[0, 0, 50]
        ):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        assert len(velocityGuess) == 3

        self.track_id = trackIdCount  # identification of each track object
        # KF instance to track this object
        self.KF = myKF(position, velocityGuess=velocityGuess)
        self.predictedPos = position
        self.skipped_frames = 0  # number of frames skipped undetected
        self._trace = [position]  # trace path
        self._features = [feature]  # trace path
        self._sizes = [size]  # size path
        self.startTime = startTime
        #print("track created at ", position)
        self.predictedVel = np.array([np.nan]*3)
        self.predictedPos = np.array([np.nan]*3)

    def __repr__(self):
        return "Track %i %s" % (self.track_id, self.trace)

    def __len__(self):
        return len(self._trace)

    @property
    def length(self):
        '''real lenght without nans
        '''
        t = np.array(self._trace)
        return len(t[np.any(~np.isnan(t), axis=1)])

    @property
    def trace(self):
        return np.array(self._trace)

    @property
    def meanSize(self):
        return np.nanmean(self._sizes)
    
    def updateTrack(self, position, feature, size):
        if position is not None:
            self._trace.append(position)
            self._features.append(feature)
            self._sizes.append(size)
            self.KF.update(position)
        else:
            self._trace.append([np.nan, np.nan, np.nan])
            #recycle last features
            self._features.append(self._features[-1])
            self._sizes.append(np.nan)
            self.KF.update(self.predictedPos)

    def predict(self):
        self.KF.predict()
        self.predictedPos = self.KF.x[::2].squeeze()
        self.predictedVel = self.KF.x[1::2].squeeze()
        return self.predictedPos, self.predictedVel


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, lv1match, config, dist_thresh=1, max_frames_to_skip=2, max_trace_length=None,
                 velocityGuessXY=[0, 0], maxIter= 1e30, fig=None,#, 50
                featureVariance = {"distance":200**2, "area":20
                 # , "pixMean": 7
                 }):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self._lv1matchgp = lv1match
        self.config = config
        self.lv1track = deepcopy(lv1match)
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.velocityGuessXY = velocityGuessXY
        self.defaultVelocityGuessXY = velocityGuessXY
        self.velocityGuessFactor = 4 #double variance when only defualt velocity is available
        self.maxIter = maxIter
        self.featureVariance = xr.Dataset(featureVariance).to_array()
        self.featureKeys = list(featureVariance.keys())
        self.featureKeys.remove("distance")
        assert len(velocityGuessXY) == 2
        assert self.featureVariance.coords["variable"].values[0] == "distance"

        #intitalize
        self.lastTime = np.datetime64("2010-01-01T00:00:00")
        self.lastFrame = 0

        # we do not need to load all variables
        self._lv1matchgp = self._lv1matchgp[set(self.featureKeys + ["capture_time", "position3D_centroid", "pair_id", "area"])].load()

        self.activeTracks = []
        self.archiveTrackNSamples = []
        self.archiveTrackTimes = []
        self.archiveTrackSize = []
        self.archiveTrackVelocities = []

        self.trackIdCount = 0
        #print("Tracker created", dist_thresh, max_frames_to_skip)
        self.fig = fig
        if fig is not None:
            self.ax = self.fig.add_subplot(projection='3d')
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")

        self.iiCapture = -99

        #make a frame id used for tracking
        self._lv1matchgp["frameid4tracking"] = self._lv1matchgp.capture_time.isel(
            camera=0).astype(int)//np.around(1e9/config.fps, -3).astype(int)
        self._lv1matchgp["frameid4tracking"] -= self._lv1matchgp["frameid4tracking"][0]
        self._lv1matchgp = self._lv1matchgp.groupby("frameid4tracking")
        self.nFrames = len(self._lv1matchgp)
        self._lv1matchgp = iter(self._lv1matchgp)
        self._frameid = -1  # id of current frame

        # results will be written here
        nParts = len(self.lv1track.pair_id)
        self.lv1track["track_id"] = xr.DataArray(
            np.zeros(nParts, dtype=int)-99, coords=[self.lv1track.pair_id])
        self.lv1track["track_step"] = xr.DataArray(
            np.zeros(nParts, dtype=np.int16)-99, coords=[self.lv1track.pair_id])
        self.lv1track["track_velociytGuess"] = xr.DataArray(
            np.zeros((nParts, 4))*np.nan, {"pair_id":self.lv1track.pair_id, "dim3D":["x","y","z", "z_rotated"]})

        # init velocity first guess

        self.backSteps = 200 #max number of data point to look back
        self.backStepsMin = self.backSteps//10 # max number of default values to fill up backSteps

        self.velGuess_slope = _reference_slopes[config.visssGen]
        self.velGuess_intercept = _reference_intercepts[config.visssGen]

        # use only a fraction of the backSteps for reference dtaa ppoints
        Dlog = np.linspace(0,2,self.backStepsMin)
        vLog = self.velGuess_slope * Dlog + self.velGuess_intercept
        #shuffle and apply log
        rng = np.random.default_rng(1)
        ind = np.arange(self.backStepsMin)
        rng.shuffle(ind)
        self.Dref = 10**Dlog[ind]
        self.vref = 10**vLog[ind]




        log.info(f"processing {self.nFrames} frames of {lv1match.encoding['source']}")

    def updateAll(self, stopAfter=None):

        if stopAfter is None:
            stopAfter = self.nFrames
        else:
            stopAfter = min(self.nFrames, stopAfter)

        for ff in tqdm(range(stopAfter), file=sys.stdout):
            self.update(ff)
            #break after maxIter frames
            if self._frameid >= self.maxIter:
                self.lv1track = self.lv1track.isel(pair_id=(self.lv1track.track_id != -99))
                break

        return self.lv1track



    def update(self,ff):
        """Update tracks vector using following steps:
            - extract data from one frame
            - Create tracks if no tracks vector found
            - Calculate self.cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # extractData from lv1match iterator over frames
        _, self._thisDat = next(self._lv1matchgp)

        # identify jumps in time - reset everything even if a single frame is missing
        frameDiff = (self._thisDat.frameid4tracking.values[0] - self._frameid)
        if frameDiff > 2:
            #print("#"*10, f"resetting due to jump of {frameDiff} frames!", "#"*10)
            self.reset()
        # if only one frame is missing, update all tracks using predictions
        elif (frameDiff == 2):
            #print("#"*10, f"one frame is missing {frameDiff} update particles using predictions", "#"*10)
            for i in range(len(self.activeTracks)):
                self.activeTracks[i].updateTrack(None, None, None)
                self.activeTracks[i].skipped_frames += 1
            stop = True

        self._frameid = int(self._thisDat.frameid4tracking.values[0])

        # get particle position and id
        detections = self._thisDat.position3D_centroid.isel(dim3D=range(3)).values.T
        if len(self.featureVariance) > 1:
            features = self._thisDat[self.featureKeys].max("camera").to_array().T
        else:
            features = None
        capture_times = self._thisDat.capture_time.isel(camera=0).values
        pair_ids = self._thisDat.pair_id.values

        #print("#"*10, "update", self._frameid, "#"*10)

        # see whether we need to update the velocity first guess:

        # for ii in range(-min(10, len(self.archiveTracks)), 0):
        #     oldTrack = self.archiveTracks[ii]
        #     # we want only relatively recent observations with at least 4 data points
        #     if (capture_times[0] - oldTrack.startTime) < np.timedelta64(1, "s"):
        #         if len(oldTrack) > 3:
        #             self.velocityGuess = oldTrack.predictedVel
        #             #print(f"velocity first guess updated with {self.velocityGuess}")
        #             break
        # else:  # break not encountered, reset to default
        #     self.velocityGuess = self.defaultVelocityGuess
        #     #print(f"velocity first guess reset to {self.velocityGuess}")


        if (
            #(ff%10 == 0) or #update only every 10th frame
            ((capture_times[0] - self.lastTime) > np.timedelta64(500,"ms")) or # or if older than X s
            (len(self.archiveTrackNSamples) < (self.backSteps*10)) # or at the beginning
               ) :
            self.updateVelocityFirstGuess(capture_times,ff)

            # print("velocityGuess", self.velocityGuess)

        # Create tracks if no track vector found
        if (len(self.activeTracks) == 0):
            #print("created tracks")
            for i in range(detections.shape[0]):
                if features is not None:
                    feat = features.isel(pair_id=i)
                else:
                    feat = None

                size = self._thisDat.area.isel(pair_id=i).max("camera").values
                velocityGuess = self.getVelocityFirstGuess(size)
                track = Track(detections[i], feat, size, self.trackIdCount,
                              capture_times[i], velocityGuess=velocityGuess)
                self.trackIdCount += 1
                self.activeTracks.append(track)
                # save "result"
                pp = np.where(self.lv1track.pair_id == pair_ids[i])[0][0]
                self.lv1track["track_id"].values[pp] = track.track_id
                self.lv1track["track_step"].values[pp] = len(track)
                self.lv1track["track_velociytGuess"].values[pp, :3] = velocityGuess

                #print(f"assigned particle {pair_ids[i]} to ALL NEW track id {track.track_id}")

            return 0

        # # predict particles using the Kalman filter
        for i in range(len(self.activeTracks)):
            self.activeTracks[i].predict()

        # Calculate self.cost using sum of square distance between
        # predicted vs detected centroids
        predictions = np.array([a.predictedPos for a in self.activeTracks])
        diffs = predictions[:, np.newaxis]- detections[np.newaxis]
        # ralrely predictions is nan:
        diffs[np.isnan(diffs)] = 1e30

        distancesSq = np.sum(diffs**2, axis=-1)
        #make detection easier in case velocity field is not available
        distancesSq = distancesSq/self.velocityGuessFactor
        #print(self._frameid, predictions, detections)

        if len(self.featureVariance) > 1:
            trackFeatures = np.array([a._features[-1].values for a in self.activeTracks])
            featureDiff = (trackFeatures[:, np.newaxis] - features.values[np.newaxis])**2
            joinedDiffs = np.concatenate((distancesSq[:,:,np.newaxis], featureDiff),axis=-1)
        else:
            joinedDiffs = distancesSq[:,:,np.newaxis]
        #weigh squared difference with assumed variance and sum up
        #note: oder of self.featureVariance.values is fixed becuas eit is an DataArray
        self.cost = np.mean(joinedDiffs/self.featureVariance.values, axis=-1)

        #print(self._frameid, joinedDiffs/self.featureVariance.values)
        #print(self._frameid, self.cost)

        N = len(self.activeTracks)
        M = len(detections)



        # inflate teh cost of values exceeding the threshold
        # othewise the hungarian alogorithm can sometimes make the wrong decision, e.g.
        # array([[ 70.51900006, 149.70650454,  28.18374428],
        #    [ 75.02965046, 109.13251285,   0.20302552],
        #    [ 38.74350741,  92.42695696,  56.25212234],
        #    [  0.14997953,  46.60156128,  72.25367643],
        #    [ 26.85835339,  80.47296492,  50.26078078]])
        # results in 26.85835339, 46.60156128,  0.20302552

        self.cost[self.cost>self.dist_thresh] = 1e30

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        self.assignment = np.array([-1]*N)
#        for _ in range(N):
#            self.assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(self.cost)
        self.assignment[row_ind] = col_ind
        self.assignment = list(self.assignment)
        # print("ddists", (joinedDiffs)[row_ind, col_ind])
        # print("costs", (joinedDiffs/self.featureVariance.values)[row_ind, col_ind])


        #if 52 in [a.track_id for a in  self.activeTracks]:
        #     import pdb;pdb.set_trace()

        # for i in range(len(row_ind)):
        #     self.assignment[row_ind[i]] = col_ind[i]
        #print("assignment", self.assignment)

        # Identify tracks with no assignment, if any
        for i in range(len(self.assignment)):
            if (self.assignment[i] != -1):
                # check for self.cost distance threshold.
                # If self.cost is very high then un_assign (delete) the track
                if (self.cost[i][self.assignment[i]] > self.dist_thresh):
                    self.assignment[i] = -1
                    self.activeTracks[i].skipped_frames += 1
            else:
                self.activeTracks[i].skipped_frames += 1


        #print("un_assigned_tracks due to high costs", un_assigned_tracks)

        # If tracks are not detected for long time, remove them
        # del_ii = []
        # for i in range(len(self.activeTracks)):
        #     if (self.activeTracks[i].skipped_frames > self.max_frames_to_skip):
        #         del_ii.append(i)


        del_ii = [f.skipped_frames > self.max_frames_to_skip for f in self.activeTracks]
        if np.sum(del_ii) > 0:  # only when skipped frame exceeds max
            # for id in del_ii:
            #     if id < len(self.activeTracks):
            #         self.archiveTracks.append(self.activeTracks[id])
            #         if self.fig is not None:
            #             self.ax.scatter(
            #                 xs=self.activeTracks[id].trace[:, 0], ys=self.activeTracks[id].trace[:, 1], zs=self.activeTracks[id].trace[:, 2], alpha=1)
            #         # del self.activeTracks[id]
            #         # del self.assignment[id]
            #     else:
            #         print("ERROR: id is greater than length of tracks")

            if self.fig is not None:
                for ii in np.where(del_ii)[ii]:
                    self.ax.scatter(
                        xs=self.activeTracks[ii].trace[:, 0], ys=self.activeTracks[ii].trace[:, 1], zs=self.activeTracks[ii].trace[:, 2], alpha=1)
            self.removeTracks(del_ii)
            #print("deleted tracks becuase not seen any more", del_ii)

        # Now look for un_assigned detects
        un_assigned_detects = [i for i in range(M) if i not in self.assignment]
        # for i in range(M):
        #     if i not in self.assignment:
        #         un_assigned_detects.append(i)
        # if len(un_assigned_detects) > 0:
            # print("identify unassigned detects", un_assigned_detects)

        # Start new tracks
        for i in range(len(un_assigned_detects)):

            if features is not None:
                feat = features.isel(pair_id=un_assigned_detects[i])
            else:
                feat = None
            size = self._thisDat.area.isel(pair_id=un_assigned_detects[i]).max("camera").values
            velocityGuess = self.getVelocityFirstGuess(size)
            track = Track(detections[un_assigned_detects[i]], feat, size,
                          self.trackIdCount, capture_times[un_assigned_detects[i]],
                          velocityGuess=velocityGuess)
            self.trackIdCount += 1
            self.activeTracks.append(track)
            #print("started", track)
            # save "result"
            pp = np.where(self.lv1track.pair_id ==
                          pair_ids[un_assigned_detects[i]])[0][0]
            self.lv1track["track_id"].values[pp] = track.track_id
            self.lv1track["track_step"].values[pp] = len(track)
            self.lv1track["track_velociytGuess"].values[pp, :3] = velocityGuess

                #print(f"assigned particle {pair_ids[un_assigned_detects[i]]} to NEW track id {track.track_id}")

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(self.assignment)):

            if(self.assignment[i] != -1):
                self.activeTracks[i].skipped_frames = 0
                if features is not None:
                    feat = features.isel(pair_id=self.assignment[i])
                else:
                    feat = None
                size = self._thisDat.area.isel(pair_id=self.assignment[i]).max("camera").values
                velocityGuess = self.getVelocityFirstGuess(size)
                self.activeTracks[i].updateTrack(
                    detections[self.assignment[i]], feat, size)
                # save result
                pp = np.where(self.lv1track.pair_id ==
                              pair_ids[self.assignment[i]])[0][0]
                self.lv1track["track_id"].values[pp] = self.activeTracks[i].track_id
                self.lv1track["track_step"].values[pp] = len(self.activeTracks[i])
                self.lv1track["track_velociytGuess"].values[pp, :3] = velocityGuess

                #print(f"assigned particle {pair_ids[self.assignment[i]]} to track id {self.activeTracks[i].track_id}")
            else:
                # track not found in current frame, use predicted position to identify particle potentially again
                self.activeTracks[i].updateTrack(None, None, None)

            if self.max_trace_length is not None:
                if(len(self.activeTracks[i].trace) > self.max_trace_length):
                    for j in range(len(self.activeTracks[i].trace) -
                                   self.max_trace_length):
                        del self.activeTracks[i].trace[j]

            #print(i, "done")

    def updateVelocityFirstGuess(self, capture_times, ff):
        '''
        update the first guess based on the previous observations
        '''

        minTrackLen = 4
        maxAge = 100 #seconds

        self.velocityGuessXY = self.defaultVelocityGuessXY
        self.velocityGuessFactor = 4

        # first resteto standard values
        self.velGuess_slope = _reference_slopes[self.config.visssGen]
        self.velGuess_intercept = _reference_intercepts[self.config.visssGen]

        if len(self.archiveTrackTimes) > 0:

            nSamples = np.array(self.archiveTrackNSamples[-(self.backSteps*10):])
            times = np.array(self.archiveTrackTimes[-(self.backSteps*10):])
            zVels = np.array(self.archiveTrackVelocities)[-self.backSteps*10:,2]
            sizes = np.array(self.archiveTrackSize[-self.backSteps*10:])

            cond = (
                (nSamples >= minTrackLen) & 
                ((capture_times[0]  - times) < np.timedelta64(maxAge, "s")) &
                (zVels > 0) &  # due to the log scale we can deal only with positive velocities
                np.isfinite(zVels) &
                np.isfinite(sizes)
                )
            if np.any(cond):
                zVels = zVels[cond][-self.backSteps:]
                sizes = sizes[cond][-self.backSteps:]

                if logDebug: log.debug(f"using {len(sizes)} particle tracks")

                # add default values in case too few data points
                if len(zVels) < self.backStepsMin:
                    zVels = np.concatenate((zVels,self.vref))[:self.backStepsMin]
                    sizes = np.concatenate((sizes,self.Dref))[:self.backStepsMin]

                # lr = scipy.stats.linregress(np.log10(sizes), np.log10(zVels))
                # self.velGuess_slope = lr.slope
                # self.velGuess_intercept = lr.intercept
                self.velGuess_slope, self.velGuess_intercept = tools.linreg(np.log10(sizes), np.log10(zVels))

                if  np.isnan(self.velGuess_slope):
                    log.error("nan result of velocity size fit!")
                    raise ValueError
                if logDebug: log.debug(f"fit results in slope {self.velGuess_slope} and intercept {self.velGuess_intercept}")

                #print(repr(sizes))
                #print(repr(zVels))

                # if ff%10 == 0:
                #     import matplotlib.pylab as plt
                #     plt.figure()
                #     plt.plot(np.log10(sizes), np.log10(zVels),".")
                #     plt.plot(np.log10(np.arange(100)), self.velGuess_slope*np.log10(np.arange(100))+self.velGuess_intercept)
                #     from IPython import display
                #     display.display(plt.gcf())

                # implement size dependency here!
                xVel, yVel = np.nanmean(np.array(self.archiveTrackVelocities)[:,:2], axis=0)
                self.velocityGuessXY = [xVel, yVel] #, np.mean(zVels, axis=0)
                if logDebug: log.debug(self.velocityGuessXY)
                self.velocityGuessFactor = 1

        self.lastTime = capture_times[0]
        self.lastFrame = ff

        return 

    def getVelocityFirstGuess(self, size):

        '''get first guess of particle velocity based on the size
        '''

        velocityGuessZ = self.velGuess_slope * np.log10(size) + self.velGuess_intercept
        velocityGuessZ = 10**velocityGuessZ
        velocityGuess = self.velocityGuessXY + [velocityGuessZ]
        assert len(velocityGuess) == 3
        assert not np.any(np.isnan(velocityGuess))
        return velocityGuess


    def reset(self):
        'reset everything'

        if self.fig is not None:
            for ii in range(len(self.activeTracks)):
                # #print(self.activeTracks[ii].trace)
                self.ax.scatter(
                    xs=self.activeTracks[ii].trace[:, 0], ys=self.activeTracks[ii].trace[:, 1], zs=self.activeTracks[ii].trace[:, 2], alpha=1)

        # self.archiveTracks += self.activeTracks
        self.archiveTrackTimes += [t.startTime for t in self.activeTracks]
        self.archiveTrackNSamples += [t.length for t in self.activeTracks]
        self.archiveTrackSize += [t.meanSize for t in self.activeTracks]
        self.archiveTrackVelocities += [t.predictedVel for t in self.activeTracks]

        self.archiveTrackTimes = self.archiveTrackTimes[-(self.backSteps*10):]
        self.archiveTrackNSamples = self.archiveTrackNSamples[-(self.backSteps*10):]
        self.archiveTrackSize = self.archiveTrackSize[-(self.backSteps*10):]
        self.archiveTrackVelocities = self.archiveTrackVelocities[-(self.backSteps*10):]


        self.activeTracks = []
        self.assignment = []

    def removeTracks(self, del_ii):
        del_ii = np.where(del_ii)[0]
        self.archiveTrackTimes += [i.startTime for j, i in enumerate(self.activeTracks) if j in del_ii]
        self.archiveTrackNSamples += [i.length for j, i in enumerate(self.activeTracks) if j in del_ii]
        self.archiveTrackSize += [i.meanSize for j, i in enumerate(self.activeTracks) if j in del_ii]
        self.archiveTrackVelocities += [i.predictedVel for j, i in enumerate(self.activeTracks) if j in del_ii]

        self.archiveTrackTimes = self.archiveTrackTimes[-(self.backSteps*10):]
        self.archiveTrackNSamples = self.archiveTrackNSamples[-(self.backSteps*10):]
        self.archiveTrackSize = self.archiveTrackSize[-(self.backSteps*10):]
        self.archiveTrackVelocities = self.archiveTrackVelocities[-(self.backSteps*10):]


        self.activeTracks = [i for j, i in enumerate(self.activeTracks) if j not in del_ii]
        self.assignment = [i for j, i in enumerate(self.assignment) if j not in del_ii]
        return


def trackParticles(fnameLv1Detect, 
                   config,
                   version=__version__, 
                   dist_thresh=2, 
                   max_frames_to_skip=1,
                   max_trace_length=None,
                   velocityGuessXY=[0, 0],#, 50],
                   maxIter = 1e30,
                   featureVariance = {"distance":100**2, 'Dmax': 100  },
                   minMatchScore=1e-3,
                   doMatchIfRequired=False,
                   writeNc=True,
                   showFits=False,
                   ):

    if type(config) is str:
        config = tools.readSettings(config)

    ffl1 = files.FilenamesFromLevel(fnameLv1Detect, config)

    fnameLv1Match = ffl1.fname["level1match"]
    fnameTracking = ffl1.fname["level1track"]

    if os.path.isfile(fnameLv1Match):
        lv1match = xr.open_dataset(fnameLv1Match)
    elif os.path.isfile('%s.nodata' % fnameLv1Match) or os.path.isfile('%s.broken.txt' % fnameLv1Match):
        with tools.open2(f"{fnameTracking}.nodata", "w") as f:
            f.write("no data, lv1match nodata or broken")
        log.error(f"NO DATA {fnameTracking}")
        return None, fnameTracking
    elif ( doMtachIfRequired):
        log.info("need to create lv1match data")
        _, lv1match, _, _ = matching.matchParticles(fnameLv1Detect, config, writeNc=False)

        if lv1match is None:
            with tools.open2(f"{fnameTracking}.broken.txt", "w") as f:
                f.write("no data, lv1match processing failed")
            log.error(f"NO DATA {fnameTracking}")
            return None, fnameTracking
    else:
        log.error(f"NO DATA lv1match yet {fnameTracking}")
        return None, fnameTracking


    matchCond = (lv1match.matchScore >= minMatchScore).values

    if matchCond.sum() == 0:
        log.error("matchCond applies to ALL data")
        with tools.open2(f"{fnameTracking}.nodata", "w") as f:
            f.write("no data, matchCond applies to ALL data")
        log.error(f"NO DATA {fnameTracking}")
        return None, fnameTracking

    log.info(tools.concat("matchCond applies to", (matchCond.sum()/len(matchCond))*100, "% of data"))
    lv1match = lv1match.isel(pair_id=matchCond)

    track = Tracker(lv1match, 
        config,
        fig=None, 
        dist_thresh=dist_thresh, 
        max_frames_to_skip=max_frames_to_skip, 
        max_trace_length=max_trace_length,
        velocityGuessXY = velocityGuessXY,
        maxIter = maxIter,
        featureVariance = featureVariance
        )
    
    lv1track = track.updateAll()



    lv1track = tools.finishNc(lv1track, config.site, config.visssGen)
    if writeNc:
        tools.to_netcdf2(lv1track, fnameTracking)
    print("DONE", fnameTracking)

    return lv1track, fnameTracking


