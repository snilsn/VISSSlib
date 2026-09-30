# -*- coding: utf-8 -*-

import glob
import logging
import os
import sys
import warnings

import numpy as np
import xarray as xr
from loguru import logger as log

from . import __version__, files, matching, tools

warnings.filterwarnings("ignore", category=RuntimeWarning)


_reference_slopes = {
    "area": {
        "visss": 0.089,
        "visss2": 0.089,
        "visss3": 0.089,
    },
    "pixSum": {  # yes, these values should be visss specific but the effect of particle type is much larger than visss resolution
        "visss": 0.6983828315318985,
        "visss2": 0.6983828315318985,
        "visss3": 0.6983828315318985,
    },
}


_reference_intercepts = {
    "area": {
        "visss": 1.605,
        "visss2": 1.463,
        "visss3": 1.441,
    },
    "pixSum": {  # yes, these values should be visss specific but the effect of particle type is much larger than visss resolution
        "visss": -0.5487112497963409,
        "visss2": -0.5487112497963409,
        "visss3": -0.5487112497963409,
    },
}

# x: x, xvel, y, yvel, z, zvel
# z = x,y,z


def _buildKFConstants(R_std=2, q_var=1, reduced_q_var=0.5):
    """
    Precompute the Kalman-filter matrices shared by every Track.

    F, H, R, Q and P only depend on R_std/q_var, which are fixed for an entire
    Tracker run, not on the individual particle. Building them once and copying
    them into each new KalmanFilter avoids calling into scipy
    (Q_discrete_white_noise/block_diag) for every single track.

    Parameters
    ----------
    R_std : float, optional
        Standard deviation for measurement noise, default 2.
    q_var : float, optional
        Variance for process noise, default 1.
    reduced_q_var : float, optional
        Variance for the reduced process noise applied after a track's first
        update, default 0.5. Pass None to skip computing it.

    Returns
    -------
    dict
        Keys "F", "H", "R", "Q", "P", "reducedQ" (None if reduced_q_var is None).
    """
    from filterpy.common import Q_discrete_white_noise
    from scipy.linalg import block_diag

    dt = 1  # time step, we are in frame units!

    F = np.array(
        [
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    R = np.eye(3) * R_std**2
    q = Q_discrete_white_noise(dim=3, dt=dt, var=q_var)
    Q = block_diag(q, q)
    P = np.eye(6) * 100**2.0

    reducedQ = None
    if reduced_q_var is not None:
        qr = Q_discrete_white_noise(dim=3, dt=dt, var=reduced_q_var)
        reducedQ = block_diag(qr, qr)

    return {"F": F, "H": H, "R": R, "Q": Q, "P": P, "reducedQ": reducedQ}


def myKF(FirstPos3D, velocityGuess=[0, 0, 50], R_std=2, q_var=1, constants=None):
    """
    Initialize a Kalman Filter for 3D particle tracking.

    Parameters
    ----------
    FirstPos3D : array_like
        Initial 3D position [x, y, z] of the particle.
    velocityGuess : array_like, optional
        Initial velocity guess [vx, vy, vz], default [0, 0, 50].
    R_std : float, optional
        Standard deviation for measurement noise, default 2. Ignored if
        `constants` is given.
    q_var : float, optional
        Variance for process noise, default 1. Ignored if `constants` is given.
    constants : dict, optional
        Precomputed F/H/R/Q/P matrices from `_buildKFConstants`, reused instead
        of rebuilding them via scipy. Default None (build them here).

    Returns
    -------
    kf : filterpy.kalman.KalmanFilter
        Configured Kalman Filter instance.

    Notes
    -----
    The filter uses a constant velocity model with state vector [x, vx, y, vy, z, vz].
    """
    from filterpy.kalman import KalmanFilter

    assert len(velocityGuess) == 3

    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.dt = 1  # time step, we are in frame units! stor it in kf for convenience

    if constants is None:
        constants = _buildKFConstants(R_std=R_std, q_var=q_var, reduced_q_var=None)

    kf.F = constants["F"].copy()
    kf.u = 0.0
    kf.H = constants["H"].copy()
    kf.R = constants["R"].copy()
    kf.Q = constants["Q"].copy()
    kf.P = constants["P"].copy()

    # prior
    kf.x = np.array(
        [
            [
                FirstPos3D[0],
                velocityGuess[0],
                FirstPos3D[1],
                velocityGuess[1],
                FirstPos3D[2],
                velocityGuess[2],
            ]
        ]
    ).T
    return kf


class _RollingArray:
    """
    Append-only history buffer, backed by a numpy array, that only ever
    exposes its most recent `maxlen` entries.

    Tracker keeps a short rolling history of recently finished tracks
    (`archiveTrack*`) to fit the size/velocity relationship. That history
    used to be a plain Python list, rebuilt into a numpy array (and
    re-trimmed to `maxlen`) on every read in updateVelocityFirstGuess() -
    which runs on nearly every frame. Here the storage is already numpy, so
    reads (`.values`) are O(1) views and appends are amortized O(1) instead
    of paying an O(maxlen) list->array conversion on every read.
    """

    def __init__(self, maxlen, dtype, shape=()):
        self.maxlen = maxlen
        self._dtype = dtype
        self._shape = shape
        self._cap = max(maxlen * 2, 1)
        self._buf = np.empty((self._cap,) + shape, dtype=dtype)
        self._n = 0  # valid entries live in self._buf[:self._n]

    def extend(self, values):
        if len(values) == 0:
            return
        values = np.asarray(values, dtype=self._dtype)
        k = len(values)
        if self._n + k > self._cap:
            # drop everything older than the last `maxlen` entries, growing
            # the backing buffer too if the incoming batch still won't fit
            start = max(0, self._n - self.maxlen)
            kept = self._buf[start : self._n].copy()
            self._n = len(kept)
            if self._n + k > self._cap:
                self._cap = max(self._cap * 2, self._n + k)
                newBuf = np.empty((self._cap,) + self._shape, dtype=self._dtype)
                newBuf[: self._n] = kept
                self._buf = newBuf
            else:
                self._buf[: self._n] = kept
        self._buf[self._n : self._n + k] = values
        self._n += k

    @property
    def values(self):
        """The last `maxlen` entries, oldest to newest."""
        start = max(0, self._n - self.maxlen)
        return self._buf[start : self._n]

    def __len__(self):
        return min(self._n, self.maxlen)


class Track(object):
    """
    Track class for every object to be tracked
    """

    def __init__(
        self,
        position,
        feature,
        size,
        trackIdCount,
        startTime,
        velocityGuess=[0, 0, 50],
        R_std=2,
        q_var=1,
        reduced_q_var=0.5,
        kfConstants=None,
    ):
        """
        Initialize a particle track.

        Parameters
        ----------
        position : array_like
            Initial 3D position [x, y, z] of the particle.
        feature : array_like
            Feature vector for the particle.
        size : float
            Particle size (e.g., area or pixel sum).
        trackIdCount : int
            Unique ID for the track.
        startTime : datetime64
            Initial detection time.
        velocityGuess : array_like, optional
            Initial velocity guess [vx, vy, vz], default [0, 0, 50].
        R_std : float, optional
            Measurement noise standard deviation, default 2.
        q_var : float, optional
            Process noise variance, default 1.
        reduced_q_var : float, optional
            Reduced process noise variance after first step, default 0.5.
        kfConstants : dict, optional
            Precomputed matrices from `_buildKFConstants`, reused instead of
            rebuilding them via scipy for every track. Default None.

        Notes
        -----
        Maintains particle position, velocity, and feature history.
        """
        import vg

        self._vg = vg
        assert len(velocityGuess) == 3
        self.velocityGuess = velocityGuess
        self.track_id = trackIdCount  # identification of each track object
        # process noise variance
        self.q_var = q_var
        # process noice can be reduce a lot after the first step becuase a track is now available. factor applies to log10(q_var)
        self.reduced_q_var = reduced_q_var
        # KF instance to track this object
        self.KF = myKF(
            position,
            velocityGuess=velocityGuess,
            R_std=R_std,
            q_var=q_var,
            constants=kfConstants,
        )
        self.predictedPos = position
        self.skipped_frames = 0  # number of frames skipped undetected
        self._trace = [position]  # trace path
        self._features = [feature]  # trace path
        self._sizes = [size]  # size path
        self.startTime = startTime
        # print("track created at ", position)
        self.predictedVel = np.array([np.nan] * 3)
        self.predictedPos = np.array([np.nan] * 3)
        self.predictedAng = self._vg.angle(np.array([0, 0, 1]), np.array(velocityGuess))

        if self.reduced_q_var is not None:
            if kfConstants is not None and kfConstants.get("reducedQ") is not None:
                self.reducedQ = kfConstants["reducedQ"].copy()
            else:
                from filterpy.common import Q_discrete_white_noise
                from scipy.linalg import block_diag

                q = Q_discrete_white_noise(dim=3, dt=self.KF.dt, var=self.reduced_q_var)
                self.reducedQ = block_diag(q, q)

    def __repr__(self):
        return "Track %i %s" % (self.track_id, self.trace)

    def __len__(self):
        return len(self._trace)

    @property
    def lastAngle(self):
        """
        Calculate the angle of the last movement vector relative to vertical.

        Returns
        -------
        float
            Angle in radians between the last movement vector and vertical axis.
            Returns np.nan if there's not enough data.
        """
        try:
            dist = np.diff(self.trace, axis=0)[-1]
        except IndexError:
            return np.nan
        else:
            return self._vg.angle(np.array([0, 0, 1]), dist)

    @property
    def meanVelocity(self):
        """
        Calculate the mean velocity over the track history.

        Returns
        -------
        array_like
            Mean velocity vector [vx, vy, vz].
        """
        return np.nanmean(np.diff(self.trace, axis=0), axis=0)

    @property
    def length(self):
        """real lenght without nans"""
        return len(self.trace[np.any(~np.isnan(self.trace), axis=1)])

    @property
    def trace(self):
        """
        Get the track's position history.

        Returns
        -------
        array_like
            Array of 3D positions [x, y, z] over time.
        """
        return np.array(self._trace)

    @property
    def meanSize(self):
        """
        Calculate the mean size over the track history.

        Returns
        -------
        float
            Mean particle size.
        """
        return np.nanmean(self._sizes)

    def updateTrack(self, position, feature, size):
        """
        Update track with new observation.

        Parameters
        ----------
        position : array_like or None
            New 3D position [x, y, z]. If None, uses prediction.
        feature : array_like
            New feature vector.
        size : float
            New particle size.

        Notes
        -----
        When position is None, the track is updated with NaN positions
        and the last known feature is reused.
        """
        if position is not None:
            self._trace.append(position)
            self._features.append(feature)
            self._sizes.append(size)
            self.KF.update(position)
        else:
            self._trace.append([np.nan, np.nan, np.nan])
            # recycle last features
            self._features.append(self._features[-1])
            self._sizes.append(np.nan)
            self.KF.update(self.predictedPos)
        if self.reduced_q_var is not None:
            self.KF.Q = self.reducedQ

    def trimHistory(self, max_length):
        """
        Drop the oldest history entries so at most `max_length` remain.

        Parameters
        ----------
        max_length : int
            Maximum number of trace/feature/size entries to keep.

        Notes
        -----
        `trace`/`meanVelocity`/etc. read `_trace` through a property that
        returns a fresh `np.array(self._trace)` on every access, so trimming
        must mutate `_trace` (and the parallel `_features`/`_sizes` lists)
        directly rather than the array a property call returns.
        """
        excess = len(self._trace) - max_length
        if excess > 0:
            del self._trace[:excess]
            del self._features[:excess]
            del self._sizes[:excess]

    def predict(self):
        """
        Predict next state using Kalman Filter.

        Returns
        -------
        predictedPos : array_like
            Predicted 3D position [x, y, z].
        predictedVel : array_like
            Predicted velocity [vx, vy, vz].
        predictedAng : float
            Predicted angle from vertical (radians).
        """
        self.KF.predict()
        self.predictedPos = self.KF.x[::2].squeeze()
        self.predictedVel = self.KF.x[1::2].squeeze()
        self.predictedAng = self._vg.angle(np.array([0, 0, 1]), self.predictedVel)
        return self.predictedPos, self.predictedVel, self.predictedAng


class Tracker(object):
    """Tracker class that updates track vectors of object tracked"""

    def __init__(
        self,
        lv1match,
        config,
        dist_thresh=4,
        max_trace_length=None,
        velocityGuessXY=[0, 0],
        maxIter=1e30,
        fig=None,  # , 50
        featureVariance={
            "distance": 200**2,
            # "area": 20
            "Dmax": 1,
            # "pixSum": 250000
        },
        minTrackLen4training=4,
        maxAge4training=300,
        costExperiencePenalty=np.array([1, 1, 6, 9, 9, 9] + [9] * 50),
        velSlope=None,
        velIntercept=None,
        R_std=2,  # meas. noise for KF
        q_var=1,  # variance process noise KF
        reduced_q_var=1,
        training=True,  # go back to start after coefficients for velocity size relation have been determined
        verbosity=0,
    ):
        """
        Initialize particle tracker.

        Parameters
        ----------
        lv1match : xarray.Dataset
            Level1 matched particle data.
        config : object
            Configuration settings.
        dist_thresh : float, optional
            Distance threshold for assignment, default 4.
        max_trace_length : int, optional
            Maximum history length, default None (unlimited).
        velocityGuessXY : array_like, optional
            Initial XY velocity guess [vx, vy], default [0, 0].
        maxIter : int, optional
            Maximum frames to process, default 1e30.
        fig : matplotlib.figure.Figure, optional
            Figure for debugging visualization, default None.
        featureVariance : dict, optional
            Variances for cost calculation, default {"distance":40000, "Dmax":1}.
        minTrackLen4training : int, optional
            Minimum track length for training, default 4.
        maxAge4training : float, optional
            Maximum age (seconds) for training data, default 300.
        costExperiencePenalty : array_like, optional
            Penalty factors based on track length.
        velSlope : float, optional
            Precomputed slope for size-velocity relation, default None.
        velIntercept : float, optional
            Precomputed intercept for size-velocity relation, default None.
        R_std : float, optional
            Measurement noise standard deviation, default 2.
        q_var : float, optional
            Process noise variance, default 1.
        reduced_q_var : float, optional
            Reduced process noise after first step, default 1.
        training : bool, optional
            Training mode flag, default True.
        verbosity : int, optional
            Verbosity level, default 0.

        Notes
        -----
        Uses Kalman Filters and Hungarian algorithm for particle tracking.
        """
        self.sizeVariable = "pixSum"
        # self.sizeVariable = "area"
        print("sizeVariable:", self.sizeVariable)
        lv1match["pixSum"] = (~lv1match.pixMean.astype(np.uint8)) * lv1match.area

        self.lv1track = lv1match.load()
        self.config = config
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = (
            0  # hard coded becuase gaps in data are not considered
        )
        self.max_trace_length = max_trace_length
        self.velocityGuessXY = velocityGuessXY
        self.defaultVelocityGuessXY = velocityGuessXY
        # double variance at beginnign of dtat file when KF did not learn yet
        self.costGuessFactor = 4
        # track length is alwys at least 1, so first item is ignored
        self.costExperiencePenalty = costExperiencePenalty
        self.R_std = R_std
        self.q_var = q_var
        self.maxIter = maxIter
        self.minTrackLen4training = minTrackLen4training
        self.maxAge4training = maxAge4training  # seconds
        self.training = training
        self.verbosity = verbosity

        self.featureKeys = list(featureVariance.keys())
        self.featureKeys.remove("distance")
        assert len(velocityGuessXY) == 2
        assert list(featureVariance.keys())[0] == "distance"
        # plain numpy array ordered ["distance"] + featureKeys, used for the
        # elementwise cost-matrix division in update()
        self.featureVariance = np.array(
            [featureVariance[k] for k in ["distance"] + self.featureKeys]
        )
        self._hasFeatures = len(self.featureVariance) > 1
        # Dmax's cost variance is overridden per track below (scaled to the
        # track's own size) instead of using the fixed value from
        # featureVariance, so large/tumbling particles aren't penalized for
        # their naturally larger frame-to-frame Dmax jitter.
        self._dmaxFeatureIdx = (
            self.featureKeys.index("Dmax") if "Dmax" in self.featureKeys else None
        )

        # intitalize
        self.lastTime = np.datetime64("2010-01-01T00:00:00")
        self.lastFrame = 0

        self.activeTracks = []

        self.trackIdCount = 0
        # print("Tracker created", dist_thresh, max_frames_to_skip)
        self.fig = fig
        if fig is not None:
            self.ax = self.fig.add_subplot(projection="3d")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")

        # Pull everything the per-frame loop needs into plain numpy arrays once,
        # up front. update() used to re-enter xarray on every frame (a
        # groupby("frameid4tracking") iterator) and every particle (per-particle
        # .isel() calls in getFeatures()); profiling showed that xarray call
        # overhead, not the tracking math, dominated runtime (~90% of wall time
        # on real level1match files). From here on the hot loop only touches
        # plain numpy.
        captureTime0 = self.lv1track.capture_time.isel(camera=0).values
        frameInterval = np.around(1e9 / config.fps, -3).astype(int)
        frameid = captureTime0.astype("int64") // frameInterval
        frameid = frameid - frameid[0]

        # groupby("frameid4tracking") iterates groups in ascending key order;
        # an explicit stable sort reproduces that ordering exactly even if
        # frameid were ever not already monotonic.
        order = np.argsort(frameid, kind="stable")
        frameidSorted = frameid[order]
        uniqFrameid, frameStart = np.unique(frameidSorted, return_index=True)
        self._frameBoundaries = np.append(frameStart, len(order))
        self._uniqFrameid = uniqFrameid
        self.nFrames = len(uniqFrameid)

        positionsAll = self.lv1track.position3D_centroid.isel(dim3D=range(3)).values.T
        # the old per-particle `.isel(pair_id=i).mean("camera")` scalar reduction
        # promoted to float64; match that dtype here so results are bit-identical
        # (a bulk `.mean("camera")` over the full array instead keeps float32).
        sizesAll = (
            self.lv1track[self.sizeVariable].mean("camera").values.astype(np.float64)
        )
        pairIdsAll = self.lv1track.pair_id.values

        self._positionsSorted = positionsAll[order]
        self._sizesSorted = sizesAll[order]
        self._captureTimesSorted = captureTime0[order]
        self._pairIdsSorted = pairIdsAll[order]

        if self._hasFeatures:
            featuresAll = np.stack(
                [self.lv1track[k].mean("camera").values for k in self.featureKeys],
                axis=-1,
            )
            self._featuresSorted = featuresAll[order]
        else:
            self._featuresSorted = None

        self._frameid = -1  # id of current frame

        # pair_id -> row index in self.lv1track, used by save(). pair_id is
        # unique but, once matchCond has filtered rows, not necessarily
        # contiguous 0..n-1 any more, so this dict lookup replaces the O(n)
        # np.where(...) scan save() used to do for every particle (O(n^2) over
        # a whole file).
        self._pairIdToRow = {
            int(pid): i for i, pid in enumerate(self.lv1track.pair_id.values)
        }

        # F/H/R/Q/P only depend on R_std/q_var, fixed for the whole run, so
        # build them once instead of re-deriving via scipy for every new track.
        # Track's own reduced_q_var default (0.5) is used here, not the
        # reduced_q_var passed into Tracker/trackParticles - Track never
        # forwards that argument on, matching prior behavior.
        self._kfConstants = _buildKFConstants(R_std=R_std, q_var=q_var)

        # results will be written here
        nParts = len(self.lv1track.pair_id)
        self.lv1track["track_id"] = xr.DataArray(
            np.zeros(nParts, dtype=int) - 99, coords=[self.lv1track.pair_id]
        )
        self.lv1track["track_step"] = xr.DataArray(
            np.zeros(nParts, dtype=np.int16) - 99, coords=[self.lv1track.pair_id]
        )
        self.lv1track["track_velocityGuess"] = xr.DataArray(
            np.zeros((nParts, 4)) * np.nan,
            {"pair_id": self.lv1track.pair_id, "dim3D": ["x", "y", "z", "z_rotated"]},
        )
        self.lv1track["track_angleGuess"] = xr.DataArray(
            np.zeros((nParts)) * np.nan, coords=[self.lv1track.pair_id]
        )
        # cache references to the underlying numpy arrays (no copy - same memory
        # backs self.lv1track) so save() writes directly into them instead of
        # doing a `self.lv1track["..."]` xarray lookup for every particle
        self._trackIdArr = self.lv1track["track_id"].values
        self._trackStepArr = self.lv1track["track_step"].values
        self._trackVelocityGuessArr = self.lv1track["track_velocityGuess"].values
        self._trackAngleGuessArr = self.lv1track["track_angleGuess"].values

        # init velocity first guess

        self.backSteps = 200  # max number of data point to look back
        # max number of default values to fill up backSteps
        self.backStepsMin = self.backSteps // 10

        # rolling history of recently finished tracks, used to fit the
        # size/velocity relation in updateVelocityFirstGuess()
        archiveMaxlen = self.backSteps * 10
        self._archiveTrackTimes = _RollingArray(
            archiveMaxlen, dtype=self._captureTimesSorted.dtype
        )
        self._archiveTrackNSamples = _RollingArray(archiveMaxlen, dtype=np.int64)
        self._archiveTrackSize = _RollingArray(archiveMaxlen, dtype=np.float64)
        self._archiveTrackVelocities = _RollingArray(
            archiveMaxlen, dtype=np.float64, shape=(3,)
        )

        if velSlope is None:
            self.velGuess_slope = _reference_slopes[self.sizeVariable][config.visssGen]
        else:
            self.velGuess_slope = velSlope
        if velIntercept is None:
            self.velGuess_intercept = _reference_intercepts[self.sizeVariable][
                config.visssGen
            ]
        else:
            self.velGuess_intercept = velIntercept

        # use only a fraction of the backSteps for reference data ppoints
        if self.sizeVariable == "pixSum":
            Dlog = np.linspace(1, 4, self.backStepsMin)
        else:
            Dlog = np.linspace(0, 2, self.backStepsMin)

        vLog = self.velGuess_slope * Dlog + self.velGuess_intercept
        # shuffle and apply log
        rng = np.random.default_rng(1)
        ind = np.arange(self.backStepsMin)
        rng.shuffle(ind)
        self.Dref = 10 ** Dlog[ind]
        self.vref = 10 ** vLog[ind]

        self.trainingComplete = False

        log.info(f"processing {self.nFrames} frames of {lv1match.encoding['source']}")

    def updateAll(self):
        """
        Process all frames for tracking.

        Returns
        -------
        lv1track : xarray.Dataset
            Tracked particle data with track IDs.
        velGuess_slope : float
            Slope of size-velocity relationship.
        velGuess_intercept : float
            Intercept of size-velocity relationship.

        Notes
        -----
        Runs in two passes if in training mode:
        1. Training pass: learns size-velocity relationship
        2. Tracking pass: applies learned model to entire dataset
        """
        from tqdm import tqdm

        for ff in tqdm(range(self.nFrames), file=sys.stdout):
            self.update(ff)
            # break after maxIter frames
            #     break
            if self.training and self.trainingComplete:
                break

        if self.maxIter is None:
            stopAfter = self.nFrames
        else:
            stopAfter = min(self.nFrames, self.maxIter)

        if self.training:
            print(f"training complete after {ff} of {self.nFrames} frames")
            self.training = False
            # reset track index and all per-frame/per-track state so the real
            # pass starts clean at frame 0 instead of carrying over whatever
            # tracks were active wherever the training pass happened to stop.
            # The archive (archiveTrack*, used to fit the size/velocity
            # relation) is intentionally kept as a warm-start prior.
            self.trackIdCount = 0
            self.activeTracks = []
            self.assignment = []
            self._frameid = -1
            self.lastTime = np.datetime64("2010-01-01T00:00:00")
            self.lastFrame = 0
            for ff in tqdm(range(stopAfter), file=sys.stdout):
                self.update(ff)

        if self.maxIter is not None:
            self.lv1track = self.lv1track.isel(pair_id=(self.lv1track.track_id != -99))

        return self.lv1track, self.velGuess_slope, self.velGuess_intercept

    @property
    def activeTrackLength(self):
        """
        Get lengths of all active tracks.

        Returns
        -------
        array_like
            Array containing the length of each active track.
        """
        return np.array([t.length for t in self.activeTracks])

    def update(self, ff):
        """
        Process one frame of particle detections.

        Parameters
        ----------
        ff : int
            Frame index.

        Notes
        -----
        Steps:
        1. Extract particle positions and features
        2. Predict next state for active tracks
        3. Calculate assignment cost matrix
        4. Apply Hungarian algorithm for assignment
        5. Handle unassigned detections (new tracks)
        6. Update tracks with new measurements
        7. Remove stale tracks
        """
        from scipy.optimize import linear_sum_assignment

        if not self.training and (self.verbosity > 5):
            print(ff, self.costGuessFactor)
        # slice this frame's rows out of the arrays precomputed once in __init__
        frameSlice = slice(self._frameBoundaries[ff], self._frameBoundaries[ff + 1])
        thisFrameid = int(self._uniqFrameid[ff])
        self._thisSizes = self._sizesSorted[frameSlice]

        # identify jumps in time - reset everything even if a single frame is missing
        frameDiff = thisFrameid - self._frameid
        if frameDiff > 2:
            if not self.training and (self.verbosity > 5):
                print(
                    "#" * 10, f"resetting due to jump of {frameDiff} frames!", "#" * 10
                )
            self.reset()
        # if only one frame is missing, update all tracks using predictions
        elif frameDiff == 2:
            if not self.training and (self.verbosity > 5):
                print(
                    "#" * 10,
                    f"one frame is missing {frameDiff} update particles using predictions",
                    "#" * 10,
                )
            for i in range(len(self.activeTracks)):
                self.activeTracks[i].updateTrack(None, None, None)
                self.activeTracks[i].skipped_frames += 1

        self._frameid = thisFrameid

        # get particle position and id
        detections = self._positionsSorted[frameSlice]
        if self._hasFeatures:
            features = self._featuresSorted[frameSlice]
        else:
            features = None
        capture_times = self._captureTimesSorted[frameSlice]
        pair_ids = self._pairIdsSorted[frameSlice]

        if not self.training and (self.verbosity > 5):
            print("#" * 10, "update", self._frameid, "#" * 10)

        # see whether we need to update the velocity first guess:
        # for ii in range(-min(10, len(self.archiveTracks)), 0):
        #     oldTrack = self.archiveTracks[ii]
        #     # we want only relatively recent observations with at least 4 data points
        #     if (capture_times[0] - oldTrack.startTime) < np.timedelta64(1, "s"):
        #         if len(oldTrack) > 3:
        #             self.velocityGuess = oldTrack.predictedVel
        #            if not self.training and (self.verbosity > 5): print(f"velocity first guess updated with {self.velocityGuess}")
        #             break
        # else:  # break not encountered, reset to default
        #     self.velocityGuess = self.defaultVelocityGuess
        #    if not self.training and (self.verbosity > 5): print(f"velocity first guess reset to {self.velocityGuess}")

        if (
            # (ff%10 == 0) or #update only every 10th frame
            # or if older than X s
            ((capture_times[0] - self.lastTime) > np.timedelta64(500, "ms"))
            or (
                len(self._archiveTrackNSamples) < (self.backSteps * 10)
            )  # or at the beginning
        ):
            self.updateVelocityFirstGuess(capture_times, ff)

            if not self.training and (self.verbosity > 5):
                print(
                    "velocityGuess",
                    self.velocityGuessXY,
                    self.velGuess_slope,
                    self.velGuess_intercept,
                )

        # Create tracks if no track vector found
        if len(self.activeTracks) == 0:
            if not self.training and (self.verbosity > 5):
                print("created tracks")
            for i in range(detections.shape[0]):
                feat, size, velocityGuess = self.getFeatures(features, i)
                track = Track(
                    detections[i],
                    feat,
                    size,
                    self.trackIdCount,
                    capture_times[i],
                    velocityGuess=velocityGuess,
                    R_std=self.R_std,
                    q_var=self.q_var,
                    kfConstants=self._kfConstants,
                )
                self.trackIdCount += 1
                self.activeTracks.append(track)
                # save "result"
                if not self.training:
                    self.save(pair_ids[i], track)
                if not self.training and (self.verbosity > 5):
                    print(
                        f"assigned particle {pair_ids[i]} to ALL NEW track id {track.track_id}"
                    )

            return 0

        # # predict particles using the Kalman filter
        for i in range(len(self.activeTracks)):
            self.activeTracks[i].predict()

        # Calculate self.cost using sum of square distance between
        # predicted vs detected centroids
        predictions = np.array([a.predictedPos for a in self.activeTracks])
        diffs = predictions[:, np.newaxis] - detections[np.newaxis]
        # ralrely predictions is nan:
        diffs[np.isnan(diffs)] = 1e30

        distancesSq = np.sum(diffs**2, axis=-1)
        # make detection easier in case velocity field is not available yet (costGuessFactor)
        # make detection stricter the longer the observed track is (costExperiencePenalty)

        try:
            costExperiencePenalty = self.costExperiencePenalty[
                self.activeTrackLength, np.newaxis
            ]
        except IndexError:
            costExperiencePenalty = self.costExperiencePenalty[-1, np.newaxis]
        distancesSq = distancesSq * costExperiencePenalty
        # if self.costGuessFactor == 1:
        #     if len(predictions) != len(detections):
        #         import pdb; pdb.set_trace()
        if not self.training and (self.verbosity > 5):
            print(self._frameid, predictions, detections)

        if self._hasFeatures:
            trackFeatures = np.array([a._features[-1] for a in self.activeTracks])
            featureDiff = (trackFeatures[:, np.newaxis] - features[np.newaxis]) ** 2
            joinedDiffs = np.concatenate(
                (distancesSq[:, :, np.newaxis], featureDiff), axis=-1
            )
        else:
            joinedDiffs = distancesSq[:, :, np.newaxis]

        # weigh squared difference with assumed variance and sum up
        if self._hasFeatures and self._dmaxFeatureIdx is not None:
            # per-track variance: distance (and any other feature) keeps the
            # fixed value, but Dmax's variance scales with the track's own
            # size instead of a fixed 1 px^2
            variance = np.tile(self.featureVariance, (len(self.activeTracks), 1))
            trackDmax = trackFeatures[:, self._dmaxFeatureIdx]
            variance[:, 1 + self._dmaxFeatureIdx] = np.maximum(1, (0.1 * trackDmax) ** 2)
            self.cost = (joinedDiffs / variance[:, np.newaxis, :]).mean(axis=-1)
        else:
            self.cost = (joinedDiffs / self.featureVariance).mean(axis=-1)
        if not self.training and (self.verbosity > 5):
            print(self._frameid, joinedDiffs / self.featureVariance)
        if not self.training and (self.verbosity > 5):
            print(self._frameid, self.cost)

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

        self.cost[self.cost > self.dist_thresh] = 1e30

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        self.assignment = np.array([-1] * N)
        #        for _ in range(N):
        #            self.assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(self.cost)
        self.assignment[row_ind] = col_ind
        self.assignment = list(self.assignment)
        if not self.training and (self.verbosity > 5):
            print("ddists", (joinedDiffs)[row_ind, col_ind])
        if not self.training and (self.verbosity > 5):
            print(
                "costs", (joinedDiffs / self.featureVariance)[row_ind, col_ind]
            )

        # if 52 in [a.track_id for a in  self.activeTracks]:
        #     import pdb;pdb.set_trace()

        # for i in range(len(row_ind)):
        #     self.assignment[row_ind[i]] = col_ind[i]
        if not self.training and (self.verbosity > 5):
            print("assignment", self.assignment)

        # Identify tracks with no assignment, if any
        for i in range(len(self.assignment)):
            if self.assignment[i] != -1:
                # check for self.cost distance threshold.
                # If self.cost is very high then un_assign (delete) the track
                if self.cost[i][self.assignment[i]] > self.dist_thresh:
                    self.assignment[i] = -1
                    self.activeTracks[i].skipped_frames += 1
            else:
                self.activeTracks[i].skipped_frames += 1

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
            #       if not self.training and (self.verbosity > 5): print("ERROR: id is greater than length of tracks")

            if (not self.training) and (self.fig is not None):
                for ii in np.where(del_ii)[0]:
                    self.ax.scatter(
                        xs=self.activeTracks[ii].trace[:, 0],
                        ys=self.activeTracks[ii].trace[:, 1],
                        zs=self.activeTracks[ii].trace[:, 2],
                        alpha=1,
                    )
            self.removeTracks(del_ii)
            if not self.training and (self.verbosity > 5):
                print("deleted tracks becuase not seen any more", del_ii)

        # Now look for un_assigned detects
        un_assigned_detects = [i for i in range(M) if i not in self.assignment]
        # for i in range(M):
        #     if i not in self.assignment:
        #         un_assigned_detects.append(i)
        # if len(un_assigned_detects) > 0:
        if not self.training and (self.verbosity > 5):
            print("identify unassigned detects", un_assigned_detects)

        # Start new tracks
        for i in range(len(un_assigned_detects)):
            feat, size, velocityGuess = self.getFeatures(
                features, un_assigned_detects[i]
            )
            track = Track(
                detections[un_assigned_detects[i]],
                feat,
                size,
                self.trackIdCount,
                capture_times[un_assigned_detects[i]],
                velocityGuess=velocityGuess,
                R_std=self.R_std,
                q_var=self.q_var,
                kfConstants=self._kfConstants,
            )
            self.trackIdCount += 1
            self.activeTracks.append(track)
            if not self.training and (self.verbosity > 5):
                print("started", track)
            # save "result"
            if not self.training:
                self.save(pair_ids[un_assigned_detects[i]], track)
                if not self.training and (self.verbosity > 5):
                    print(
                        f"assigned particle {pair_ids[un_assigned_detects[i]]} to NEW track id {track.track_id}"
                    )

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(self.assignment)):
            if self.assignment[i] != -1:
                self.activeTracks[i].skipped_frames = 0
                feat, size, velocityGuess = self.getFeatures(
                    features, self.assignment[i]
                )
                self.activeTracks[i].updateTrack(
                    detections[self.assignment[i]], feat, size
                )
                # save result
                if not self.training:
                    self.save(pair_ids[self.assignment[i]], self.activeTracks[i])
                if not self.training and (self.verbosity > 5):
                    print(
                        f"assigned particle {pair_ids[self.assignment[i]]} to track id {self.activeTracks[i].track_id}"
                    )
            else:
                # track not found in current frame, use predicted position to identify particle potentially again
                self.activeTracks[i].updateTrack(None, None, None)

            if self.max_trace_length is not None:
                self.activeTracks[i].trimHistory(self.max_trace_length)

            if not self.training and (self.verbosity > 5):
                print(i, "done")
        return

    def getFeatures(self, features, pair_id):
        """
        Get features for a specific particle.

        Parameters
        ----------
        features : array_like or None
            Feature array for all particles.
        pair_id : int
            Index of the particle to get features for.

        Returns
        -------
        tuple
            Tuple of (feature_vector, size, velocity_guess)
        """
        if features is not None:
            feat = features[pair_id]
        else:
            feat = None
        size = self._thisSizes[pair_id]
        velocityGuess = self.getVelocityFirstGuess(size)
        return feat, size, velocityGuess

    def save(self, pair_id, track):
        """save results"""

        pp = self._pairIdToRow[int(pair_id)]
        self._trackIdArr[pp] = track.track_id
        self._trackStepArr[pp] = len(track)
        if len(track) == 1:
            self._trackVelocityGuessArr[pp, :3] = track.velocityGuess
        else:
            self._trackVelocityGuessArr[pp, :3] = track.predictedVel
        self._trackAngleGuessArr[pp] = track.predictedAng
        if not self.training and (self.verbosity > 5):
            print(track.track_id, len(track), track.predictedAng, track.lastAngle)

    def updateVelocityFirstGuess(self, capture_times, ff):
        """
        Update velocity first guess using recent track data.

        Parameters
        ----------
        capture_times : array_like
            Current frame capture times.
        ff : int
            Frame index.

        Notes
        -----
        Fits a new size-velocity model when sufficient recent data exists.
        Resets to defaults when data is stale or insufficient.
        """
        self.velocityGuessXY = self.defaultVelocityGuessXY
        self.costGuessFactor = 4

        # first resteto standard values
        self.velGuess_slope = _reference_slopes[self.sizeVariable][self.config.visssGen]
        self.velGuess_intercept = _reference_intercepts[self.sizeVariable][
            self.config.visssGen
        ]

        if len(self._archiveTrackTimes) > 0:
            nSamples = self._archiveTrackNSamples.values
            times = self._archiveTrackTimes.values
            velocities = self._archiveTrackVelocities.values
            zVels = velocities[:, 2]
            sizes = self._archiveTrackSize.values

            cond = (
                (nSamples >= self.minTrackLen4training)
                & (
                    (capture_times[0] - times)
                    < np.timedelta64(self.maxAge4training, "s")
                )
                &
                # due to the log scale we can deal only with positive velocities
                (zVels > 0)
                & np.isfinite(zVels)
                & np.isfinite(sizes)
            )

            # in case we are training the size velocity relation, do we have enough data?
            if np.sum(cond) >= self.backSteps * 2:
                self.trainingComplete = True

            if np.any(cond):
                zVels = zVels[cond][-self.backSteps :]
                sizes = sizes[cond][-self.backSteps :]

                log.debug(f"using {len(sizes)} particle tracks")
                # print(f"using {len(sizes)} particle tracks")
                # add default values in case too few data points
                if len(zVels) < self.backStepsMin:
                    zVels = np.concatenate((zVels, self.vref))[: self.backStepsMin]
                    sizes = np.concatenate((sizes, self.Dref))[: self.backStepsMin]

                # lr = scipy.stats.linregress(np.log10(sizes), np.log10(zVels))
                # self.velGuess_slope = lr.slope
                # self.velGuess_intercept = lr.intercept
                self.velGuess_slope, self.velGuess_intercept = tools.linreg(
                    np.log10(sizes), np.log10(zVels)
                )

                if np.isnan(self.velGuess_slope):
                    log.error("nan result of velocity size fit!")
                    raise ValueError
                log.debug(
                    f"fit results in slope {self.velGuess_slope} and intercept {self.velGuess_intercept}"
                )
                # print(
                #     f"fit results in slope {self.velGuess_slope} and intercept {self.velGuess_intercept}"
                # )
                # print(repr(sizes))
                # print(repr(zVels))

                # if ff%10 == 0:
                #     import matplotlib.pylab as plt
                #     plt.figure()
                #     plt.plot(np.log10(sizes), np.log10(zVels),".")
                #     plt.plot(np.log10(np.arange(100)), self.velGuess_slope*np.log10(np.arange(100))+self.velGuess_intercept)
                #     from IPython import display
                #     display.display(plt.gcf())

                xVel, yVel = np.nanmean(velocities[:, :2], axis=0)
                self.velocityGuessXY = [xVel, yVel]  # , np.mean(zVels, axis=0)
                log.debug(self.velocityGuessXY)
                self.costGuessFactor = 1

        self.lastTime = capture_times[0]
        self.lastFrame = ff

        return

    def getVelocityFirstGuess(self, size):
        """
        Estimate initial velocity from particle size.

        Parameters
        ----------
        size : float
            Particle size.

        Returns
        -------
        velocityGuess : list
            Velocity vector [vx, vy, vz] in mm/s.

        Notes
        -----
        Uses logarithmic relationship: log10(v) = slope*log10(size) + intercept
        """
        velocityGuessZ = self.velGuess_slope * np.log10(size) + self.velGuess_intercept
        velocityGuessZ = 10**velocityGuessZ
        velocityGuess = self.velocityGuessXY + [velocityGuessZ]
        assert len(velocityGuess) == 3
        assert not np.any(np.isnan(velocityGuess))
        return velocityGuess

    def reset(self):
        """
        Reset active tracks and archive recent track data.

        Notes
        -----
        Moves active tracks to archive and clears active track list.
        Maintains only recent archive data (last backSteps*10 items).
        """
        if (not self.training) and (self.fig is not None):
            for ii in range(len(self.activeTracks)):
                # #print(self.activeTracks[ii].trace)
                self.ax.scatter(
                    xs=self.activeTracks[ii].trace[:, 0],
                    ys=self.activeTracks[ii].trace[:, 1],
                    zs=self.activeTracks[ii].trace[:, 2],
                    alpha=1,
                )

        # self.archiveTracks += self.activeTracks
        self._archiveTrackTimes.extend([t.startTime for t in self.activeTracks])
        self._archiveTrackNSamples.extend([t.length for t in self.activeTracks])
        self._archiveTrackSize.extend([t.meanSize for t in self.activeTracks])
        self._archiveTrackVelocities.extend(
            [t.meanVelocity for t in self.activeTracks]
        )

        self.activeTracks = []
        self.assignment = []

    def removeTracks(self, del_ii):
        """
        Remove inactive tracks and archive their data.

        Parameters
        ----------
        del_ii : array_like
            Indices of tracks to remove.

        Notes
        -----
        Archives track data before removing from active list.
        Maintains only recent archive data (last backSteps*10 items).
        """
        del_ii = np.where(del_ii)[0]
        self._archiveTrackTimes.extend(
            [i.startTime for j, i in enumerate(self.activeTracks) if j in del_ii]
        )
        self._archiveTrackNSamples.extend(
            [i.length for j, i in enumerate(self.activeTracks) if j in del_ii]
        )
        self._archiveTrackSize.extend(
            [i.meanSize for j, i in enumerate(self.activeTracks) if j in del_ii]
        )
        self._archiveTrackVelocities.extend(
            [i.meanVelocity for j, i in enumerate(self.activeTracks) if j in del_ii]
        )

        self.activeTracks = [
            i for j, i in enumerate(self.activeTracks) if j not in del_ii
        ]
        self.assignment = [i for j, i in enumerate(self.assignment) if j not in del_ii]
        return


@log.catch(reraise=True)
def trackParticles(
    fnameLv1Detect,
    config,
    version=__version__,
    dist_thresh=4,
    fig=None,
    max_trace_length=None,
    velocityGuessXY=[0, 0],  # , 50],
    maxIter=1e30,
    featureVariance={"distance": 200**2, "Dmax": 1},
    minMatchScore=1e-3,
    minTrackLen4training=2,
    maxAge4training=100,
    costExperiencePenalty=np.array([1, 1, 6, 9, 9, 9] + [9] * 50),
    R_std=2,
    q_var=1,
    reduced_q_var=1,
    doMatchIfRequired=False,
    writeNc=True,
    skipExisting=True,
    verbosity=0,
):
    """
    Main particle tracking workflow.

    Parameters
    ----------
    fnameLv1Detect : str
        Level1 detection filename.
    config : object
        Configuration settings.
    version : str, optional
        Processing version, default __version__.
    dist_thresh : float, optional
        Assignment distance threshold, default 4.
    fig : matplotlib.figure.Figure, optional
        Figure for visualization, default None.
    max_trace_length : int, optional
        Maximum track history length, default None.
    velocityGuessXY : list, optional
        Initial XY velocity guess [vx, vy], default [0,0].
    maxIter : int, optional
        Maximum frames to process, default 1e30.
    featureVariance : dict, optional
        Feature variances for cost calculation, default {"distance":40000, "Dmax":1}.
    minMatchScore : float, optional
        Minimum match score for particles, default 1e-3.
    minTrackLen4training : int, optional
        Minimum track length for training, default 2.
    maxAge4training : float, optional
        Maximum training data age (seconds), default 100.
    costExperiencePenalty : array_like, optional
        Assignment penalty factors by track length.
    R_std : float, optional
        Measurement noise standard deviation, default 2.
    q_var : float, optional
        Process noise variance, default 1.
    reduced_q_var : float, optional
        Reduced process noise after first step, default 1.
    doMatchIfRequired : bool, optional
        Run matching if missing, default False.
    writeNc : bool, optional
        Write netCDF output, default True.
    skipExisting : bool, optional
        Skip processing if output exists, default True.
    verbosity : int, optional
        Verbosity level, default 0.

    Returns
    -------
    lv1track : xarray.Dataset or None
        Tracked particle data, None if skipped/broken.
    fnameTracking : str
        Output filename.

    Notes
    -----
    Handles data loading, particle matching (if needed), and tracking.
    Creates output files and handles existing/skipped cases.
    """
    config = tools.readSettings(config)

    ffl1 = files.FilenamesFromLevel(fnameLv1Detect, config)

    fnameLv1Match = ffl1.fname["level1match"]
    fnameTracking = ffl1.fname["level1track"]

    # check whether output exists
    if skipExisting and tools.checkForExisting(
        fnameTracking,
        parents=glob.glob(f"{fnameLv1Match}*"),
        breakpointLevel="level1track",
    ):
        print("SKIPPING", fnameTracking)
        return None, None

    if os.path.isfile(fnameLv1Match):
        lv1match = xr.open_dataset(fnameLv1Match)
        lv1match.load()  # important to do that early, is much slower after applying filters with isel
    elif ffl1.isNoData("level1match"):
        ffl1.propagateNoData(
            "level1match", "level1track", message="no data, lv1match nodata "
        )
        log.error(f"NO DATA {fnameTracking}")
        return None, fnameTracking
    elif ffl1.isBroken("level1match"):
        ffl1.writeStatus("level1track", "broken.txt", "no data, lv1match  broken")
        log.error(f"NO DATA {fnameTracking}")
        return None, fnameTracking
    elif doMatchIfRequired:
        log.info("need to create lv1match data")
        _, lv1match, _, _, _, _, _, _ = matching.matchParticles(
            fnameLv1Detect, config, writeNc=False
        )

        if lv1match is None:
            ffl1.writeStatus(
                "level1track", "broken.txt", "no data, lv1match processing failed"
            )
            log.error(f"NO DATA {fnameTracking}")
            return None, fnameTracking
    else:
        log.error(f"NO DATA lv1match yet {fnameTracking}")
        return None, fnameTracking

    matchCond = (lv1match.matchScore >= minMatchScore).values

    if matchCond.sum() == 0:
        log.error("matchCond applies to ALL data")
        ffl1.writeStatus(
            "level1track", "nodata", "no data, matchCond applies to ALL data"
        )
        log.error(f"NO DATA {fnameTracking}")
        return None, fnameTracking

    log.info(
        tools.concat(
            "matchCond applies to",
            (matchCond.sum() / len(matchCond)) * 100,
            "% of data",
        )
    )
    lv1match = lv1match.isel(pair_id=matchCond)

    trackTrainer = Tracker(
        lv1match,
        config,
        fig=fig,
        dist_thresh=dist_thresh,
        max_trace_length=max_trace_length,
        velocityGuessXY=velocityGuessXY,
        maxIter=maxIter,
        featureVariance=featureVariance,
        minTrackLen4training=minTrackLen4training,
        maxAge4training=maxAge4training,
        costExperiencePenalty=costExperiencePenalty,
        R_std=R_std,
        q_var=q_var,
        reduced_q_var=reduced_q_var,
        training=True,
        verbosity=verbosity,
    )
    lv1track, velSlope, velIntercept = trackTrainer.updateAll()
    print("final slope and intercept", velSlope, velIntercept)

    lv1track = tools.finishNc(
        lv1track,
        config.site,
        config.visssGen,
        extra=tools.collectVersionAttrs(
            "level1track", {"level1match": [fnameLv1Match]}
        ),
    )
    lv1track.load()
    print(lv1track)
    if writeNc:
        tools.to_netcdf2(lv1track, config, fnameTracking)
    print("DONE", fnameTracking)

    return lv1track, fnameTracking
