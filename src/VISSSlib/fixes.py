# -*- coding: utf-8 -*-


import warnings

# import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import groupby

import numpy as np
import xarray as xr
from loguru import logger as log

# various tools to fix bugs in the data


def fixMosaicTimeL1(dat1, config):
    """
    Attempt to fix drift of capture time with record_time.

    This function attempts to correct timing drift between capture_time and
    record_time by estimating and interpolating drift patterns over time.

    Parameters
    ----------
    dat1 : xarray.Dataset
        Input dataset containing capture_time and record_time variables
    config : object
        Configuration object containing fps parameter for frame rate

    Returns
    -------
    xarray.Dataset
        Dataset with corrected capture_time values

    Notes
    -----
    This is a poor attempt at fixing drift and is not used anymore.
    The function groups data into time chunks and estimates drift patterns
    to interpolate and correct the timing issues.
    """
    datS = dat1[["capture_time", "record_time"]]
    datS = datS.isel(capture_time=slice(None, None, config["fps"]))
    diff = datS.capture_time - datS.record_time

    # no estiamte the drift
    drifts1 = []
    # group netcdf into 1 minute chunks
    index1min = (
        diff.capture_time.resample(capture_time="1T", label="right")
        .first()
        .capture_time.values
    )
    if len(index1min) <= 2:
        index1min = (
            diff.capture_time.resample(capture_time="30s", label="right")
            .first()
            .capture_time.values
        )
        if len(index1min) <= 2:
            index1min = (
                diff.capture_time.resample(capture_time="10s", label="right")
                .first()
                .capture_time.values
            )
            if len(index1min) <= 2:
                index1min = (
                    diff.capture_time.resample(capture_time="1s", label="right")
                    .first()
                    .capture_time.values
                )

    grps = diff.groupby_bins("capture_time", bins=index1min)

    # find max. difference in each chunk
    # this is the one were we assume it is the true dirft
    # also time stamp or max.  is needed, this is why resample cannot be used directly
    for ii, grp in grps:
        drifts1.append(grp.isel(capture_time=grp.argmax()))
    drifts = xr.concat(drifts1, dim="capture_time")

    # interpolate to original resolution
    # extrapolation required for beginning or end - works usually very good!
    driftsInt = (
        drifts.astype(int)
        .interp_like(dat1.capture_time, kwargs={"fill_value": "extrapolate"})
        .astype("timedelta64[ns]")
    )

    # get best time estimate
    bestestimate = dat1.capture_time.values - driftsInt.values

    #                 plt.figure()
    #                 driftsInt.plot(marker="x")
    #                 diff.plot()

    # replace time in nc file
    dat1["capture_time_orig"] = deepcopy(dat1["capture_time"])
    dat1 = dat1.assign_coords(capture_time=bestestimate)

    # the difference between bestestimate and capture time must jump more than 1% of the measurement interval
    timeDiff = np.abs(
        (
            (dat1.capture_time - dat1.capture_time_orig).diff("capture_time")
            / dat1.capture_time_orig.diff("capture_time")
        )
    )
    assert np.all(timeDiff < 0.01), timeDiff.max()

    return dat1


def captureIdOverflows(dat, config, storeOrig=True, idOffset=0, dim="pid"):
    """
    Fix capture_id overflows for M1280 devices.

    For M1280 devices, capture_id is a 16-bit integer that overflows every few minutes.
    This function detects and fixes overflow conditions by applying appropriate offsets.

    Parameters
    ----------
    dat : xarray.Dataset
        Input dataset containing capture_id and capture_time variables
    config : object
        Configuration object containing fps parameter for frame rate
    storeOrig : bool, optional
        Whether to store original capture_id values, default is True
    idOffset : int, optional
        Constant offset to add to capture_id, default is 0
    dim : str, optional
        Dimension name for diff operations, default is "pid"

    Returns
    -------
    xarray.Dataset
        Dataset with fixed capture_id values

    Notes
    -----
    This function handles the specific case where capture_id overflows due to
    being a 16-bit integer. It detects overflow points and applies corrections
    to maintain proper sequential numbering.
    """
    log.info("fixing captureIdOverflows")
    maxInt = 65535

    # if someone already messed with the data, revert it
    if "capture_id_orig" in dat.keys():
        dat["capture_id"] = deepcopy(dat["capture_id_orig"])

    if storeOrig:
        dat["capture_id_orig"] = deepcopy(dat["capture_id"])

    # constant offset
    if idOffset != 0:
        dat["capture_id"] += idOffset

    idDiffObserved = dat.capture_id.diff(dim)
    idDiffEstimated = np.round(
        dat.capture_time.diff(dim) / np.timedelta64(round(1 / config.fps * 1e6), "us")
    ).astype(int)

    stepsObserved = (idDiffObserved < 0) | (idDiffEstimated >= maxInt)
    nStepsObserved = stepsObserved.sum()

    # estimate expected steps
    firstII = dat.capture_id.values[0]
    firstCaptureT = dat.capture_time.values[0]
    lastCaptureT = dat.capture_time.values[-1]

    deltaT = (lastCaptureT - firstCaptureT) / np.timedelta64(1, "s")
    nFrames = np.ceil(deltaT * config.fps).astype(int)
    nStepsExpected = int((firstII + nFrames) / maxInt)

    if nStepsObserved == nStepsExpected == 0:
        # nothing to do
        return dat

    if (nStepsExpected == nStepsObserved) or ((nStepsExpected - 1) == nStepsObserved):
        jumpIIs = np.where(stepsObserved)[0] + 1

        for jumpII in jumpIIs:
            dat["capture_id"][jumpII:] += maxInt

    else:
        raise RuntimeError("was einfallen lassen...")

    assert np.all(dat.capture_id.diff(dim) >= 0)
    log.info(
        f"expecting {nStepsExpected} jumps, found and fixed {(stepsObserved).sum().values} jumps"
    )

    return dat


def revertIdOverflowFix(dat):
    """
    Revert capture_id overflow fix by restoring original values.

    This function restores the original capture_id values by renaming
    the fixed and original variables back to their original names.

    Parameters
    ----------
    dat : xarray.Dataset
        Input dataset with fixed capture_id and capture_id_orig variables

    Returns
    -------
    xarray.Dataset
        Dataset with original capture_id restored

    Notes
    -----
    This function is used to undo the effects of captureIdOverflows when
    needed for data recovery or analysis consistency.
    """
    log.info("reverting revertIdOverflowFix")
    dat = dat.rename({"capture_id": "capture_id_fixed"})
    dat = dat.rename({"capture_id_orig": "capture_id"})
    return dat


def removeGhostFrames(metaDat, config, intOverflow=True, idOffset=0, fixIteration=3):
    """
    Remove ghost frames from MOSAiC follower data.

    For MOSAiC follower devices, additional ghost frames are occasionally added
    to the dataset. These can be identified by their spacing being less than
    1/fps apart. This function identifies and removes such frames.

    Parameters
    ----------
    metaDat : xarray.Dataset
        Input dataset containing capture_time and capture_id variables
    config : object
        Configuration object containing fps parameter for frame rate
    intOverflow : bool, optional
        Whether to handle integer overflows, default is True
    idOffset : int, optional
        Offset to add to capture_id, default is 0
    fixIteration : int, optional
        Number of iterations to attempt ghost frame removal, default is 3

    Returns
    -------
    tuple
        A tuple containing (fixed_dataset, dropped_frames, beyond_repair_flag)
        where:
        - fixed_dataset is the dataset with ghost frames removed
        - dropped_frames is the count of removed frames
        - beyond_repair_flag indicates if data is beyond repair

    Notes
    -----
    Ghost frames are typically identified by their spacing being significantly
    different from the expected 1/fps interval. The function performs multiple
    iterations to handle complex cases where ghost frames might be in data gaps.
    """
    log.info("fixing removeGhostFrames")

    beyondRepair = False
    metaDat["capture_id_orig"] = deepcopy(metaDat["capture_id"])

    metaDat["capture_id"] = metaDat["capture_id"] + idOffset

    if intOverflow:
        metaDat = captureIdOverflows(
            metaDat, config, dim="capture_time", storeOrig=False
        )

    # ns are assumed
    assert metaDat["capture_time"].dtype == "<M8[ns]"

    droppedFrames = 0
    for nn in range(fixIteration + 1):
        slope = (
            (
                metaDat["capture_time"].diff("capture_time")
                / metaDat["capture_id"].diff("capture_time")
            )
        ).astype(int)
        configSlope = 1e9 / config.fps
        # we find them because dat is not 1/fps apart
        jumps = ((slope / configSlope).values > 1.03) | (
            (slope / configSlope).values < 0.97
        )
        jumpsII = np.where(jumps)[0]
        nGroups = sum(k for k, v in groupby(jumps))

        # the last loop is only for testng
        if nn == fixIteration:
            if nGroups != 0:
                log.error("FILE BROKEN BEYOND REPAIR")
                droppedFrames += len(metaDat.capture_time) - jumpsII[0]
                # remove fishy data and everything after
                metaDat = metaDat.isel(capture_time=slice(0, jumpsII[0]))
                beyondRepair = True
            break

        lastII = np.concatenate((jumpsII[:-1][np.diff(jumpsII) != 1], jumpsII[-1:])) + 1
        assert nGroups == len(lastII)

        for lastI in lastII:
            metaDat["capture_id"][lastI:] = metaDat["capture_id"][lastI:] - 1

        # remove all fishy frames
        metaDat = metaDat.drop_isel(capture_time=jumpsII)
        droppedFrames += len(jumpsII)

        if nGroups > 0:
            log.warn(
                f"ghost iteration {nn}: found {nGroups} ghost frames at {lastII.tolist()}"
            )
        else:
            break

    return metaDat, droppedFrames, beyondRepair


def delayedClockReset(metaDat, config):
    """
    Check for and fix delayed clock reset issues.

    This function detects delayed clock resets in the data and attempts to
    correct them by adjusting timestamps accordingly.

    Parameters
    ----------
    metaDat : xarray.Dataset
        Input dataset containing capture_time and capture_id variables
    config : object
        Configuration object containing fps parameter for frame rate

    Returns
    -------
    xarray.Dataset
        Dataset with corrected timestamps if reset was detected

    Notes
    -----
    Delayed clock resets are identified by large negative time differences
    (>10 seconds). The function handles both cases where integer overflows
    and timestamp issues coexist, and attempts to fix the timing problems
    by recalculating timestamps based on known good values.
    """
    if (metaDat.capture_time.diff() <= -10e6).any():
        log.info("fixing detected delayedClockReset")

        resetII = np.where((metaDat.capture_time.diff() < -10e6))[0]
        assert len(resetII) == 1, "len(resetII) %i" % len(resetII)
        resetII = resetII[0]  # +1 already applied by pandas!
        assert resetII < 20, (
            "time jump usually occures within first few frames %i" % resetII
        )

        if (metaDat.capture_id.diff()[1 : resetII + 1] < 0).any():
            # we cannot handle int overflows in capture id AND wrong timestamps,
            # cut data
            metaDat = metaDat.iloc[resetII:]
        else:
            # attempt to fix it!
            firstGoodTime = metaDat.capture_time.iat[resetII]
            firstGoodID = metaDat.capture_id.iat[resetII]
            deltaT = round(1 / config.fps * 1e6)
            offsets = (metaDat.capture_id.iloc[:resetII] - firstGoodID) * deltaT
            metaDat.iloc[:resetII, metaDat.columns.get_loc("capture_time")] = (
                firstGoodTime + offsets
            )

    return metaDat


def makeCaptureTimeEven(datF, config, dim="capture_time"):
    """
    Make capture time even for M1280 follower devices.

    For M1280 follower devices, significant drift can occur causing clocks to
    drift more than 1 frame apart within 10 minutes. This function creates
    a new time vector with even spacing based on a trusted capture_id.

    Parameters
    ----------
    datF : xarray.Dataset
        Input dataset containing capture_time and capture_id variables
    config : object
        Configuration object containing fps parameter for frame rate
    dim : str, optional
        Dimension name for operations, default is "capture_time"

    Returns
    -------
    xarray.Dataset
        Dataset with new evenly spaced capture_time_even variable

    Notes
    -----
    This function is specifically designed for capture_id offset estimation
    and creates a new time vector that maintains even spacing regardless
    of timing drift issues. It validates that the calculated slopes are
    within acceptable ranges.
    """
    log.info("making follower times even")

    if len(datF[dim]) <= 1:
        print("makeCaptureTimeEven: too short, nothing to do")
        return datF

    if dim in ["fpid", "pid"]:
        unqiue, uniqueII = np.unique(datF.capture_time, return_index=True)
        datF4slope = datF.isel(**{dim: uniqueII})
    else:
        datF4slope = datF

    assert len(datF4slope.capture_id) > 1, "need at least two samples to do derivative"

    assert np.all(
        datF4slope.capture_id.diff(dim) >= 0
    ), "capture_id must increase monotonically "
    assert np.all(
        datF4slope.capture_time.diff(dim).astype(int) > 0
    ), "capture_time must increase monotonically "

    slopeF = datF4slope["capture_time"].diff(dim).astype(int) // datF4slope[
        "capture_id"
    ].diff(dim).astype(int)

    configSlope = int(round(1e9 / config.fps, -3))
    deltaSlope = 1000  # =1us

    # make sure we do not have ghost frames in the data
    if dim == "pid":
        # we can have slope 0 in level1detect
        slopeF = slopeF.isel(pid=(datF["capture_id"].diff(dim) != 0))

    assert slopeF.min() >= (
        configSlope - deltaSlope
    ), f"min slope {slopeF.min()} too small {(configSlope+deltaSlope)}"
    assert slopeF.max() <= (
        configSlope + deltaSlope
    ), f"max slope {slopeF.max()} too large {(configSlope+deltaSlope)}"

    offset = datF.capture_time.values[0]
    fixedTime = ((datF.capture_id - datF.capture_id[0]) * configSlope) + offset

    # datF["capture_time_orig"] = deepcopy(datF["capture_time"])
    datF["capture_time_even"] = fixedTime

    return datF


def makeCaptureTimeEvenBothCameras(leaderDat, followerDat, config):
    """
    Reconstruct an evenly-spaced ``capture_time_even`` from ``capture_id``
    for *both* leader and follower, each anchored at its own first sample.

    Unlike :func:`makeCaptureTimeEven`, this does not validate the
    reconstructed slope against ``config.fps`` (no assertions) -- some
    M2050 deployments' actual frame interval differs from the nominal
    ``1/config.fps`` by a few microseconds, more than
    :func:`makeCaptureTimeEven`'s tolerance allows, which would otherwise
    reject perfectly usable segments. It also does not deduplicate by
    ``capture_time`` first, since callers here already pass 1D
    (``fpid``-indexed) arrays without the multiple-particles-per-frame
    complication :func:`makeCaptureTimeEven` guards against for
    ``pid``-indexed level1detect data.

    Grounded in the hardware sync: the follower is pulse-triggered by the
    leader at capture time, so both cameras' true capture instants are
    tied to a shared, evenly-spaced ``capture_id`` sequence -- only each
    camera's *own* onboard clock (``capture_time``) can drift from that
    sequence independently. Reconstructing time from ``capture_id``
    removes that per-camera clock drift from the offset estimate.

    Parameters
    ----------
    leaderDat, followerDat : xarray.Dataset
        Datasets with ``capture_time`` and ``capture_id`` variables along
        their (matching) leading dimension.
    config : object
        Configuration object containing the ``fps`` parameter.

    Returns
    -------
    tuple(xarray.Dataset, xarray.Dataset)
        (leaderDat, followerDat), each with a new ``capture_time_even``
        variable.
    """
    configSlope = int(round(1e9 / config.fps, -3))

    def _evenTime(dat):
        offset = dat.capture_time.values[0]
        return ((dat.capture_id - dat.capture_id[0]) * configSlope) + offset

    leaderDat = leaderDat.copy()
    leaderDat["capture_time_even"] = _evenTime(leaderDat)
    followerDat = followerDat.copy()
    followerDat["capture_time_even"] = _evenTime(followerDat)

    return leaderDat, followerDat


# def revertMakeCaptureTimeEven(dat):
#     dat = dat.rename({"capture_time": "capture_time_even"})
#     dat = dat.rename({"capture_time_orig": "capture_time"})
#     return dat


def detectCaptureIdDropTimes(
    leaderDat,
    followerDat,
    dim="fpid",
    nPoints=500,
    maxDiffMs=1,
    timeDim="capture_time",
    minRunLength=5,
    maxJump=2,
):
    """
    Find times within a window where the leader-follower capture_id
    offset makes a clean, sustained jump of a few frames -- the
    signature of one camera silently dropping (or duplicating) a
    frame index, as opposed to genuine timing ambiguity between the
    two cameras.

    Pre-PTP hardware occasionally lost a captured frame's index
    without otherwise disturbing the capture_id sequence. That leaves
    the true offset well-defined and constant on either side of the
    drop, but `tools.estimateCaptureIdDiffCore` averages the whole
    window into one offset and fails its >70% consistency check
    whenever a drop happens to fall inside it. The offset itself is
    not ambiguous -- only the single-offset-per-window assumption is
    wrong. Returns candidate leader `capture_time` values to split the
    window at (feed into `timeBlocks` alongside genuine follower
    restarts) so each side can be resolved independently; not meant
    as a replacement for `estimateCaptureIdDiffCore` itself.

    Parameters
    ----------
    leaderDat, followerDat : xarray.Dataset
        Same inputs as `tools.estimateCaptureIdDiffCore`.
    dim : str, optional
        Dimension to sample leader points from, by default "fpid".
    nPoints : int, optional
        Number of leader points to sample, by default 500.
    maxDiffMs : float, optional
        Matching window in ms, by default 1.
    timeDim : str, optional
        Time coordinate to use, by default "capture_time".
    minRunLength : int, optional
        Minimum number of samples for a run to be trusted as a real
        offset regime rather than noise, by default 5.
    maxJump : int, optional
        Only trust a jump between neighbouring runs up to this many
        frames (a dropped/duplicated frame is normally exactly 1), by
        default 2.

    Returns
    -------
    list of numpy.datetime64
        Leader capture_times to split the window at, in time order.
    """
    if (len(leaderDat[dim]) == 0) or (len(followerDat[dim]) == 0):
        return []

    # Both leader and follower have their own onboard clock that can drift
    # independently (see makeCaptureTimeEvenBothCameras's docstring) --
    # over a ~10-minute window that drift can accumulate past half a frame
    # period, at which point this function's own nearest-time matching
    # silently latches onto the *next* frame instead of the true
    # corresponding one, producing a spurious, sustained idDiff step that
    # looks exactly like a genuine dropped frame (confirmed on real data:
    # hyytiala2_v3 20240216-125000 falsely "detected" a drop at 12:57:47
    # this way -- V1.0 output, from before this function existed, used a
    # single constant offset for the whole file with no matchScore
    # degradation anywhere, proving no real drop occurred). Prefer
    # capture_time_even (drift-removed) for *both* sides symmetrically if
    # the caller has already reconstructed it -- previously only the
    # follower side was checked, which happened to be enough to hide this
    # exact false positive in testing but left the leader side vulnerable
    # to the same failure mode.
    timeDimLeader = timeDim
    if (timeDim == "capture_time") and ("capture_time_even" in leaderDat.data_vars):
        timeDimLeader = "capture_time_even"
    timeDimFollower = timeDim
    if (timeDim == "capture_time") and ("capture_time_even" in followerDat.data_vars):
        timeDimFollower = "capture_time_even"

    if len(leaderDat[dim]) > nPoints:
        points = np.linspace(0, len(leaderDat[dim]), nPoints, dtype=int, endpoint=False)
    else:
        points = range(len(leaderDat[dim]))

    times = []
    idDiffs = []
    for point in points:
        absDiff = np.abs(
            leaderDat[timeDimLeader].isel(**{dim: point}).values
            - followerDat[timeDimFollower]
        )
        pMin = np.min(absDiff).values
        if pMin < np.timedelta64(int(maxDiffMs), "ms"):
            pII = absDiff.argmin().values
            idDiffs.append(
                followerDat.capture_id.values[pII]
                - leaderDat.capture_id.isel(**{dim: point}).values
            )
            times.append(leaderDat[timeDimLeader].isel(**{dim: point}).values)

    if len(idDiffs) < 2 * minRunLength:
        return []

    times = np.array(times)
    idDiffs = np.array(idDiffs)
    order = np.argsort(times)
    times, idDiffs = times[order], idDiffs[order]

    changeAt = np.where(np.diff(idDiffs) != 0)[0] + 1
    runBounds = np.concatenate(([0], changeAt, [len(idDiffs)]))
    runs = [
        (idDiffs[a], a, b)
        for a, b in zip(runBounds[:-1], runBounds[1:])
        if (b - a) >= minRunLength
    ]
    if len(runs) < 2:
        return []

    breakTimes = []
    for (valA, _, _), (valB, sB, _) in zip(runs[:-1], runs[1:]):
        if 0 < abs(int(valB) - int(valA)) <= maxJump:
            breakTimes.append(times[sB])
    return breakTimes


def detectPhaseJumpTimes(
    leaderDat,
    followerDat,
    config,
    dim="fpid",
    binSeconds=10,
    minJumpFrac=0.5,
    recoverBins=3,
    maxOnsetsPerCamera=5,
):
    """
    Find times where one camera's raw ``capture_time`` makes a brief,
    self-correcting jump away from its own steady, ``capture_id``-implied
    schedule -- roughly one frame period, lasting minutes, then decaying
    back -- as opposed to a genuine ``capture_id``-level drop (handled by
    `detectCaptureIdDropTimes`) or ordinary independent per-camera clock
    drift (removed by `makeCaptureTimeEvenBothCameras`).

    Confirmed on real data (hyytiala2_v3 20240213-215000): the follower's
    raw capture_time jumped ~4.25ms (~1 frame period at this deployment's
    fps) relative to its own capture_id-implied schedule at 21:51:07,
    then decayed back over the following ~8.5 minutes -- capture_id
    numbering itself never skipped or repeated a value there (confirmed
    by inspection), so `detectCaptureIdDropTimes` (which works in
    capture_id space) cannot see it, and resolving the matching offset
    from `capture_time_even` (reconstructed purely from capture_id) is
    blind to it by construction, since that reconstruction cannot
    represent a discrepancy between capture_id and the camera's own
    recorded timestamp. The single segment-wide offset
    `_resolveMatchingOffset` picked for the rest of that file (correct
    for the bulk of it) was wrongly applied to the ~70s window right
    after the jump too, degrading Z-consistency there even though the
    file's aggregate quality looked fine.

    This is deliberately a different signal from `detectCaptureIdDropTimes`:
    ordinary drift accumulates *smoothly* -- exactly what
    `capture_time_even` already removes correctly -- and never produces a
    single large bin-to-bin jump the way this transient does. Watching
    the *rate of change* of (raw capture_time minus its own
    capture_id-reconstructed value), rather than its absolute size, is
    what tells the two apart without reintroducing the false-positive
    `detectCaptureIdDropTimes` had before it started preferring
    `capture_time_even` (see that function's docstring).

    Parameters
    ----------
    leaderDat, followerDat : xarray.Dataset
        RAW (not drift-corrected) leader/follower data with
        ``capture_time`` and ``capture_id``, as read from level1detect.
    config : dict
        Configuration settings (for ``config.fps``).
    dim : str, optional
        Dimension to operate along, by default "fpid".
    binSeconds : float, optional
        Bin width for the raw-vs-reconstructed deviation series, by
        default 10.
    minJumpFrac : float, optional
        Minimum bin-to-bin jump, as a fraction of one nominal frame
        period (``1000/config.fps`` ms), to flag as a glitch, by
        default 0.5.
    recoverBins : int, optional
        Number of consecutive bins the deviation must stay back within
        tolerance of its pre-jump level to mark the glitch as over, by
        default 3.
    maxOnsetsPerCamera : int, optional
        A genuine transient should be rare (at most a couple of times
        in a 10-minute file) -- confirmed on real data
        (hyytiala2_v3 20240112-030001, an extreme-density file with
        2.6M leader particles): the raw-vs-reconstructed deviation
        itself becomes noisy enough at that density to cross the
        threshold in nearly every bin, which produced 179 "onsets"
        spaced exactly one bin apart rather than a real repeating
        hardware event, and would have fragmented the file into ~180
        useless few-second segments. If a camera's raw onset count
        exceeds this, the signal for that camera is untrustworthy at
        this operating point -- discard all its onsets rather than
        acting on noise, by default 5.

    Returns
    -------
    list of numpy.datetime64
        capture_time values (paired onset/recovery per glitch found) to
        add as extra segment-split points, in time order. Each value is
        in whichever camera's own capture_time it was detected from --
        consistent with how `timeBlocks` already mixes leader- and
        follower-timeline boundaries as approximate wall-clock cut
        points.
    """
    import pandas as pd

    framePeriodMs = 1000 / config.fps
    threshold = minJumpFrac * framePeriodMs
    configSlope = int(round(1e9 / config.fps, -3))

    breakTimes = []
    for dat in (leaderDat, followerDat):
        if len(dat[dim]) < 2 * recoverBins:
            continue
        ct = dat.capture_time.values
        cid = dat.capture_id.values
        even = (
            (cid.astype("int64") - cid[0]) * configSlope + ct[0].astype("int64")
        ).astype("datetime64[ns]")
        diffMs = (ct - even) / np.timedelta64(1, "ms")

        s = (
            pd.Series(diffMs, index=pd.DatetimeIndex(ct))
            .resample(f"{int(binSeconds)}s")
            .median()
            .dropna()
        )
        if len(s) < 2 * recoverBins:
            continue

        vals = s.values
        times = s.index.values
        jumps = np.abs(np.diff(vals))
        onsets = np.where(jumps > threshold)[0] + 1

        if len(onsets) > maxOnsetsPerCamera:
            log.warning(
                f"detectPhaseJumpTimes found {len(onsets)} onsets in one "
                f"camera, exceeding maxOnsetsPerCamera={maxOnsetsPerCamera} "
                "-- treating the raw-vs-reconstructed deviation as too "
                "noisy to trust (e.g. extreme particle density) rather "
                "than a real repeating transient, discarding all of this "
                "camera's onsets"
            )
            continue

        for onsetIdx in onsets:
            baseline = vals[onsetIdx - 1]
            recovered = None
            for j in range(onsetIdx, len(vals) - recoverBins + 1):
                window = vals[j : j + recoverBins]
                if np.all(np.abs(window - baseline) <= threshold):
                    recovered = j
                    break
            breakTimes.append(times[onsetIdx])
            if recovered is not None:
                breakTimes.append(times[recovered])

    if len(breakTimes) == 0:
        return []
    return sorted(np.unique(np.array(breakTimes, dtype="datetime64[ns]")).tolist())


def removeFlippedCaptureTimeFrames(metaDat1, fname):
    """
    Drop frames around isolated backwards jumps in one source's capture_time.

    A camera's own onboard clock occasionally stamps two consecutive frames
    with flipped capture_time (frame k+1 gets a capture_time a few
    microseconds *earlier* than frame k, even though k was physically
    written first) -- see the "flipped capture_time" note in metadata.py's
    module docstring. Left alone, this is not just a cosmetic timestamp
    error: getMetaData() later concatenates and sorts all camera-thread
    data by capture_time to interleave threads chronologically, and sorting
    by a locally-flipped value reorders that one thread's own record_id
    sequence out of order at exactly that point. detection.py's frame
    reader walks each thread strictly by increasing record_id and has no
    way to seek backwards, so this then surfaces downstream as a hard
    "Cannot go back!" crash for the whole 10 minute file.

    This must be called on a single source's data (e.g. one camera
    thread's ascii file) while it is still in its own original, unsorted
    recording order -- calling it after data from multiple sources has
    already been concatenated and sorted by capture_time is a no-op, since
    sorted data cannot contain a backwards step by construction (that is
    exactly what makes the flip invisible again once threads are merged).

    Only rows whose own capture_time is already self-contradictory (stamped
    earlier than a frame recorded before it, in the same camera's write
    order) are dropped. No surviving frame's capture_time is ever changed,
    shifted, or interpolated -- this only removes evidence that was already
    unusable, it never invents a value that downstream stereo matching
    could be misled by.

    Parameters
    ----------
    metaDat1 : xarray.Dataset
        Single-source metadata in its original (not capture_time-sorted)
        recording order, as returned by _getMetaData1().
    fname : str
        Source filename, used only for the diagnostic print.

    Returns
    -------
    tuple
        (metaDat1, nDropped) with the frames around each backwards jump
        removed and the count of dropped frames.
    """
    jumps = np.diff(metaDat1.capture_time.astype(int)) < 0
    nJumps = np.sum(jumps)
    droppedIndices = []
    if nJumps > 0:
        ss = np.where(jumps)[0]
        assert nJumps < 20, "more than 20 is very fishy..."
        # Consecutive jump indices belong to one glitch and are repaired
        # together; separate (non-adjacent) glitches elsewhere in the same
        # file are independent and each get their own neighbours dropped.
        # This matters in practice: a handful of isolated single-sample
        # capture_time flips scattered through one 10 minute file is the
        # commonly observed pattern, not one contiguous bad patch.
        groups = np.split(ss, np.where(np.diff(ss) != 1)[0] + 1)
        for group in groups:
            log.warning(
                "%s: capture_time flip, DROPPING FRAMES around %i-%i"
                % (fname, group[0], group[-1])
            )
            droppedIndices.append(group[0] - 1)
            droppedIndices.extend(group.tolist())
            droppedIndices.append(group[-1] + 1)
        droppedIndices = np.unique(droppedIndices)
        metaDat1 = metaDat1.drop_isel(capture_time=droppedIndices)

    return metaDat1, len(droppedIndices)


# a movie file occasionally ends a handful of frames before the ascii log
# does (e.g. the video encoder's last buffered frames never got flushed
# before the file was closed/rotated). Once the video has genuinely run
# out of frames, treat up to this many orphaned trailing ascii rows as an
# acceptable, unrecoverable tail loss rather than aborting the whole file.
# Observed shortfalls in production (hyytiala2_v3, nyaalesund) are 1-6
# frames; this cap is kept well below that to avoid ever masking a real
# mid-file corruption instead.
_MAX_TRAILING_FRAMES_TO_DROP = 25


def isDroppableTrailingFrameShortfall(
    rowsRemainingForThread, maxFrames=_MAX_TRAILING_FRAMES_TO_DROP
):
    """
    Is an end-of-video frame shortfall small enough to treat as benign?

    detection.py calls this once a camera thread's video has genuinely run
    out of decodable frames while the ascii log still has more rows for
    that thread. A shortfall of a handful of frames right at the tail of a
    10 minute recording is expected/benign (e.g. the video encoder's last
    buffered frames never got flushed before the file was closed/
    rotated); a shortfall spanning a large fraction of the file instead
    indicates real, unrelated corruption that should still fail loudly
    rather than be silently swallowed.

    Parameters
    ----------
    rowsRemainingForThread : int
        Number of ascii rows for this thread, from the current position to
        the end of the file, that have no corresponding video frame.
    maxFrames : int, optional
        Upper bound on what counts as a benign trailing shortfall.

    Returns
    -------
    bool
    """
    return rowsRemainingForThread <= maxFrames
