# -*- coding: utf-8 -*-

# import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


from . import __version__
from . import tools
from . import files

import logging
log = logging.getLogger()


log = logging.getLogger(__name__)


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

    def __init__(self, position, feature, trackIdCount, startTime, velocityGuess=[0, 0, 50]):
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

    def updateTrack(self, position, feature):
        if position is not None:
            self._trace.append(position)
            self._features.append(feature)
            self.KF.update(position)
        else:
            self._trace.append([np.nan, np.nan, np.nan])
            #recycle last features
            self._features.append(self._features[-1])
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
                 velocityGuess=[0, 0, 50], maxIter= 1e30, fig=None, featureVariance = {"distance":200**2, "area":20
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
        self.lv1track = deepcopy(lv1match)
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.velocityGuess = velocityGuess
        self.defaultVelocityGuess = velocityGuess
        self.velocityGuessFactor = 4 #double variance when only defualt velocity is available
        self.maxIter = maxIter
        self.featureVariance = xr.Dataset(featureVariance).to_array()
        self.featureKeys = list(featureVariance.keys())
        self.featureKeys.remove("distance")
        assert len(velocityGuess) == 3
        assert self.featureVariance.coords["variable"].values[0] == "distance"

        self.activeTracks = []
        self.oldTracks = []
        self.trackIdCount = 0
        #print("Tracker created", dist_thresh, max_frames_to_skip)
        self.fig = fig
        if fig is not None:
            self.ax = self.fig.add_subplot(projection='3d')
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")

        self.iiCapture = -99
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
        self.lv1track["track_velociytGuess"] = xr.DataArray(
            np.zeros((nParts, 3))*np.nan, {"pair_id":self.lv1track.pair_id, "dim3Dt":["x","y","z"]})


        logging.info(f"processing {self.nFrames} frames")

    def updateAll(self):
        while True:
            try:
                self.update()
            except StopIteration:
                break
            if self._frameid >= self.maxIter:
                self.lv1track = self.lv1track.isel(pair_id=(self.lv1track.track_id != -99))
                break

        return self.lv1track



    def update(self):
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
        _, thisDat = next(self._lv1matchgp)

        # identify jumps in time - reset everything even if a single frame is missing
        frameDiff = (thisDat.frameid4tracking.values[0] - self._frameid)
        if frameDiff > 2:
            #print("#"*10, f"resetting due to jump of {frameDiff} frames!", "#"*10)
            self.reset()
        # if only one frame is missing, update all tracks using predictions
        elif (frameDiff == 2):
            #print("#"*10, f"one frame is missing {frameDiff} update particles using predictions", "#"*10)
            for i in range(len(self.activeTracks)):
                self.activeTracks[i].updateTrack(None, None)
                self.activeTracks[i].skipped_frames += 1
            stop = True

        self._frameid = int(thisDat.frameid4tracking.values[0])

        # get particle positoion and id
        detections = thisDat.position_3D.isel(dim3D=range(3)).values.T
        if len(self.featureVariance) > 1:
            features = thisDat[self.featureKeys].max("camera").to_array().T
        else:
            features = None
        capture_times = thisDat.capture_time.isel(camera=0).values
        pair_ids = thisDat.pair_id.values

        #print("#"*10, "update", self._frameid, "#"*10)

        # see whether we need to update the velocity first guess:

        # for ii in range(-min(10, len(self.oldTracks)), 0):
        #     oldTrack = self.oldTracks[ii]
        #     # we want only relatively recent observations with at least 4 data points
        #     if (capture_times[0] - oldTrack.startTime) < np.timedelta64(1, "s"):
        #         if len(oldTrack) > 3:
        #             self.velocityGuess = oldTrack.predictedVel
        #             #print(f"velocity first guess updated with {self.velocityGuess}")
        #             break
        # else:  # break not encountered, reset to default
        #     self.velocityGuess = self.defaultVelocityGuess
        #     #print(f"velocity first guess reset to {self.velocityGuess}")

        self.velocityGuess = self.defaultVelocityGuess
        self.velocityGuessFactor = 4
        if len(self.oldTracks) > 0:
            backSteps = 100
            minTrackLen = 4
            maxAge = 2 #seconds
            nSamples = np.array([t.length for t in self.oldTracks[-backSteps:]])
            times = np.array([t.startTime for t in self.oldTracks[-backSteps:]])
            cond = (nSamples >= minTrackLen) & ((capture_times[0]  - times) < np.timedelta64(maxAge, "s"))
            if np.any(cond):
                vels = np.array([t.predictedVel for t in self.oldTracks[-backSteps:]])
                self.velocityGuess = np.mean(vels[cond], axis=0)
                self.velocityGuessFactor = 1
            print("velocityGuess", self.velocityGuess)

        # Create tracks if no track vector found
        if (len(self.activeTracks) == 0):
            #print("created tracks")
            for i in range(detections.shape[0]):
                if features is not None:
                    feat = features.isel(pair_id=i)
                else:
                    feat = None
                track = Track(detections[i], feat, self.trackIdCount,
                              capture_times[i], velocityGuess=self.velocityGuess)
                self.trackIdCount += 1
                self.activeTracks.append(track)
                # save "result"
                pp = np.where(self.lv1track.pair_id == pair_ids[i])[0][0]
                self.lv1track["track_id"].values[pp] = track.track_id
                self.lv1track["track_velociytGuess"].values[pp] = self.velocityGuess

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
        self.cost = np.mean(joinedDiffs/self.featureVariance.values, axis=-1)

        #print(self._frameid, joinedDiffs/self.featureVariance.values)
        #print(self._frameid, self.cost)

        N = len(self.activeTracks)
        M = len(detections)
#         #print("N,M", (N, M))
#         self.cost = np.zeros(shape=(N, M))   # Cost matrix
#         for i in range(len(self.activeTracks)):
#             for j in range(len(detections)):
#                 #                 try:
#                 predicition = self.activeTracks[i].predictedPos
#                 diff = predicition - detections[j]
#                 distanceSq = np.sum(diff**2)

#                 #
#                 self.cost[i][j] = distanceSq
#                 ##print("measure distance", i, j,
#                 #      predicition.T,  detections[j], distance)
# #                 except:
# #                     pass
#         #print("self.cost calculated", self.cost)


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
        print("ddists", (joinedDiffs)[row_ind, col_ind])
        print("costs", (joinedDiffs/self.featureVariance.values)[row_ind, col_ind])


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
            #         self.oldTracks.append(self.activeTracks[id])
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
            track = Track(detections[un_assigned_detects[i]], feat,
                          self.trackIdCount, capture_times[un_assigned_detects[i]],
                          velocityGuess=self.velocityGuess)
            self.trackIdCount += 1
            self.activeTracks.append(track)
            #print("started", track)
            # save "result"
            pp = np.where(self.lv1track.pair_id ==
                          pair_ids[un_assigned_detects[i]])[0][0]
            self.lv1track["track_id"].values[pp] = track.track_id
            self.lv1track["track_velociytGuess"].values[pp] = self.velocityGuess

                #print(f"assigned particle {pair_ids[un_assigned_detects[i]]} to NEW track id {track.track_id}")

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(self.assignment)):

            if(self.assignment[i] != -1):
                self.activeTracks[i].skipped_frames = 0
                if features is not None:
                    feat = features.isel(pair_id=self.assignment[i])
                else:
                    feat = None
                self.activeTracks[i].updateTrack(
                    detections[self.assignment[i]], feat)
                # save result
                pp = np.where(self.lv1track.pair_id ==
                              pair_ids[self.assignment[i]])[0][0]
                self.lv1track["track_id"].values[pp] = self.activeTracks[i].track_id
                self.lv1track["track_velociytGuess"].values[pp] = self.velocityGuess

                #print(f"assigned particle {pair_ids[self.assignment[i]]} to track id {self.activeTracks[i].track_id}")
            else:
                # track not found in current frame, use predicted position to identify particle potentially again
                self.activeTracks[i].updateTrack(None, None)

            if self.max_trace_length is not None:
                if(len(self.activeTracks[i].trace) > self.max_trace_length):
                    for j in range(len(self.activeTracks[i].trace) -
                                   self.max_trace_length):
                        del self.activeTracks[i].trace[j]

            #print(i, "done")

    def reset(self):
        'reset everything'

        if self.fig is not None:
            for ii in range(len(self.activeTracks)):
                # #print(self.activeTracks[ii].trace)
                self.ax.scatter(
                    xs=self.activeTracks[ii].trace[:, 0], ys=self.activeTracks[ii].trace[:, 1], zs=self.activeTracks[ii].trace[:, 2], alpha=1)

        self.oldTracks += self.activeTracks
        self.activeTracks = []
        self.assignment = []

    def removeTracks(self, del_ii):
        del_ii = np.where(del_ii)[0]
        self.oldTracks += [i for j, i in enumerate(self.activeTracks) if j in del_ii]
        self.activeTracks = [i for j, i in enumerate(self.activeTracks) if j not in del_ii]
        self.assignment = [i for j, i in enumerate(self.assignment) if j not in del_ii]
        return


def trackParticles(fnameLv1Match, 
                   config,
                   version=__version__, 
                   dist_thresh=2, 
                   max_frames_to_skip=1,
                   max_trace_length=None,
                   velocityGuess=[0, 0, 50],
                   maxIter = 1e30,
                   featureVariance = {"distance":100**2, 'Dmax': 100  }
                   ):


    if type(config) is str:
        config = tools.readSettings(config)


    lv1match = xr.open_dataset(fnameLv1Match)
    ffl1 = files.FilenamesFromLevel(fnameLv1Match, config)
    fnameTracking = ffl1.fname["level1track"]


    lv1match["Dequiv"] = np.sqrt(4*lv1match["area"]/np.pi)
    #based on Garrett, T. J., and S. E. Yuter, 2014: Observed influence of riming, temperature, and turbulence on the fallspeed of solid precipitation. Geophys. Res. Lett., 41, 6515–6522, doi:10.1002/2014GL061016.
    lv1match["complexityBW"] = lv1match["perimeter"]/(np.pi * lv1match["Dequiv"])


    track = Tracker(lv1match, 
        config,
        fig=None, 
        dist_thresh=dist_thresh, 
        max_frames_to_skip=max_frames_to_skip, 
        max_trace_length=max_trace_length,
        velocityGuess = velocityGuess,
        maxIter = maxIter,
        featureVariance = featureVariance
        )
    lv1track = track.updateAll()



    lv1track = tools.finishNc(lv1track)
    tools.to_netcdf2(lv1track, fnameTracking)
    print("DONE", fnameTracking)

    return lv1track, fnameTracking


