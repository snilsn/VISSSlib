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


# ---------------------------------------------------------------------------
# MOSAiC (first VISSS, M1280 cameras, two computers, no PTP): leader/follower
# frame mapping via capture_id
#
# What the data show (Oct/Nov 2019, see project notes):
#
# * Both cameras are effectively hardware triggered: within one follower run
#   the follower-leader capture_id lag is constant for many hours while the
#   two camera clocks (capture_time) drift 25-30 ppm apart. Pairing frames by
#   capture_id is therefore exact, capture_time is not needed at all.
# * capture_id counts 1..65535 and wraps 65535 -> 1 (period 65535).
# * Every frame is recorded (also frames without moving particles), so
#   capture_id is gap free, even when the follower drops ~274 frame blocks.
# * Ghost frames (follower only): 6 consecutive frame intervals of only
#   ~0.81-0.93 frame periods (summing up to 5 periods) while capture_id
#   advances by 6, i.e. one extra capture_id is inserted. From then on, the
#   lag is permanently off by one. 0-31 events per day.
# * The follower is restarted every few hours, resetting capture_id.
# * record_time (computer clock, set in the processing queue) gives the lag
#   only to +-1 frame for a 5 min file, the median over a whole restart
#   segment is usually exact. The exact frame is determined from the
#   vertical position of single particles seen by both cameras, which is
#   only consistent for the correct lag (a one frame error corresponds to
#   ~120 px for a 1 m/s particle).
# ---------------------------------------------------------------------------

MOSAIC_CAPTURE_ID_PERIOD = 65535


def _mosaicLoadMetaFrames(case, camera, config):
    """Load capture_time/record_time (us), raw capture_id and moving pixel
    counts of all metaFrames files of a day, in recording order."""
    from . import files

    fnames = files.FindFiles(case, camera, config).listFiles("metaFrames")
    parts = []
    for fname in fnames:
        with xr.open_dataset(fname) as ds:
            if len(ds.capture_time) == 0:
                continue
            parts.append(
                {
                    "ct": ds.capture_time.values.astype("datetime64[us]").astype(
                        np.int64
                    ),
                    "rt": ds.record_time.values.astype("datetime64[us]").astype(
                        np.int64
                    ),
                    "cid": ds.capture_id.values.astype(np.int64),
                    "nmp": np.nan_to_num(
                        ds.nMovingPixel.isel(nMovingPixelThresh=0).values
                    ),
                }
            )
    if len(parts) == 0:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _mosaicFramePeriod(dat, config):
    """Frame period in us as measured by the camera's own clock."""
    dct = np.diff(dat["ct"])
    dc = np.diff(dat["cid"]) % MOSAIC_CAPTURE_ID_PERIOD
    good = (dc == 1) & (dct > 0)
    if good.sum() < 100:
        return 1e6 / config.fps
    return float(np.median(dct[good]))


def _mosaicSegmentsAndGhosts(dat, period, correctGhosts=True):
    """
    Split frames into camera runs (restart = capture_id step not consistent
    with capture_time step), unwrap capture_id per run, and detect/correct
    ghost frames.

    Returns
    -------
    seg : array of int
        run index per frame
    u : array of int
        unwrapped, ghost corrected capture_id per frame (congruent to the
        raw capture_id minus the number of preceding ghost frames of the run)
    drop : array of bool
        frames inside a ghost frame sequence (timing ambiguous)
    ghosts : list of tuples
        (index of first frame, index of first corrected frame, extra ids)
    """
    P = MOSAIC_CAPTURE_ID_PERIOD
    cid = dat["cid"]
    dc = np.diff(cid) % P
    dt = np.diff(dat["ct"]) / period
    reset = (dt <= 0) | (np.abs(dc - (np.round(dt) % P)) > 3)
    seg = np.concatenate(([0], np.cumsum(reset)))

    starts = np.concatenate(([0], np.flatnonzero(reset) + 1))
    ends = np.concatenate((starts[1:], [len(cid)]))
    u = np.empty(len(cid), dtype=np.int64)
    for a, b in zip(starts, ends):
        u[a:b] = cid[a] + np.concatenate(([0], np.cumsum(dc[a : b - 1])))

    # ghost frames: compressed frame intervals with regular capture_id steps
    short = (dt > 0.5) & (dt < 0.97) & (dc == 1)
    idx = np.flatnonzero(short)
    drop = np.zeros(len(cid), dtype=bool)
    ghosts = []
    if len(idx) > 0:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            extra = int(round(len(run) - dt[run].sum()))
            if extra < 1:
                continue
            i0, i1 = run[0], run[-1] + 1
            if (seg[i0] != seg[i1]) or not correctGhosts:
                continue
            segEnd = ends[seg[i1]]
            u[i1:segEnd] -= extra
            drop[i0 + 1 : i1] = True
            ghosts.append((i0, i1, extra))

    # ghost frames hidden in a block of dropped frames: capture_id advances
    # by 1-3 more than the elapsed frame periods (e.g. 275 ids for 274.007
    # periods, 20200106 07:54). Observed rarely, but each one shifts the lag
    # by one for the rest of the run.
    if correctGhosts:
        excess = dc - np.round(dt)
        hidden = np.flatnonzero((dt > 1.5) & (excess >= 1) & (excess <= 3))
        for i0 in hidden:
            i1 = i0 + 1
            if seg[i0] != seg[i1]:
                continue
            extra = int(excess[i0])
            u[i1 : ends[seg[i1]]] -= extra
            ghosts.append((i0, i1, extra))
        ghosts.sort()
    return seg, u, drop, ghosts


def _mosaicActivityLag(lu, lact, fu, fact, guess, searchRange=60):
    """
    Lag (follower u - leader u) maximizing the cross correlation of the
    per-frame "anything moving" signal of both cameras. Independent of any
    clock, but the peak is a few frames broad, so only good to +-1-2 frames.

    Returns (lag, z-score of peak) or (None, nan)
    """
    from scipy.signal import fftconvolve

    if (len(lu) < 1000) or (len(fu) < 1000):
        return None, np.nan
    if (lu.max() - lu.min() > 2e7) or (fu.max() - fu.min() > 2e7):
        return None, np.nan

    def series(u, act):
        s = np.full(u.max() - u.min() + 1, np.nan)
        s[u - u.min()] = act > 0
        valid = np.isfinite(s)
        s = np.where(valid, s - np.nanmean(s), 0.0)
        return s, valid.astype(float)

    a, va = series(lu, lact)
    b, vb = series(fu, fact)
    if (np.sum(a * a) == 0) or (np.sum(b * b) == 0):
        return None, np.nan
    cc = fftconvolve(b, a[::-1], "full")
    nn = fftconvolve(vb, va[::-1], "full")
    lags = np.arange(-(len(a) - 1), len(b)) + fu.min() - lu.min()
    sel = (np.abs(lags - guess) <= searchRange) & (nn > 1000)
    if sel.sum() < 10:
        return None, np.nan
    cc = cc[sel] / nn[sel]
    lags = lags[sel]
    k = np.argmax(cc)
    bg = np.median(cc)
    sd = 1.4826 * np.median(np.abs(cc - bg))
    if sd == 0:
        return None, np.nan
    return int(lags[k]), float((cc[k] - bg) / sd)


def _mosaicLoadParticles(case, camera, config):
    """capture_time (us), Dmax and vertical center position of all
    level1detect particles of a day"""
    from . import files

    fnames = files.FindFiles(case, camera, config).listFiles("level1detect")
    parts = []
    for fname in fnames:
        with xr.open_dataset(fname) as ds:
            if len(ds.pid) == 0:
                continue
            z = ds.position_upperLeft.sel(dim2D="y") + ds.Droi.sel(dim2D="y") / 2.0
            parts.append(
                {
                    "ct": ds.capture_time.values.astype("datetime64[us]").astype(
                        np.int64
                    ),
                    "D": ds.Dmax.values.astype(float),
                    "z": z.values.astype(float),
                }
            )
    if len(parts) == 0:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _mosaicParticleFrames(part, dat, seg, u, drop, period):
    """assign each particle the (seg, u) of its frame; -1 if unknown/dropped"""
    pSeg = np.full(len(part["ct"]), -1, dtype=np.int64)
    pU = np.full(len(part["ct"]), -1, dtype=np.int64)
    order = np.argsort(dat["ct"], kind="stable")
    ctSorted = dat["ct"][order]
    ii = np.searchsorted(ctSorted, part["ct"])
    ii = np.clip(ii, 1, len(ctSorted) - 1)
    left = np.abs(part["ct"] - ctSorted[ii - 1]) < np.abs(part["ct"] - ctSorted[ii])
    ii = np.where(left, ii - 1, ii)
    ok = np.abs(ctSorted[ii] - part["ct"]) < period / 4
    frame = order[ii]
    ok &= ~drop[frame]
    pSeg[ok] = seg[frame[ok]]
    pU[ok] = u[frame[ok]]
    return pSeg, pU


def _mosaicSingleParticles(part, pSeg, pU, segId, Dmin):
    """particles which are the only particle in their frame"""
    m = pSeg == segId
    uu, inv, cnt = np.unique(pU[m], return_inverse=True, return_counts=True)
    single = (cnt[inv] == 1) & (part["D"][m] >= Dmin)
    idx = np.flatnonzero(m)[single]
    order = np.argsort(pU[idx])
    return idx[order]


def _mosaicParticleLagTest(
    lPart, lSeg, lU, ls, fPart, fSeg, fU, fs, candidates, Dmin, minPairs=20
):
    """
    For each candidate lag, pair single-particle frames of both cameras and
    measure the spread (IQR) of the vertical position difference. Only the
    correct lag gives a narrow distribution.

    Returns dict(lag, iqr, iqr2nd, n, dz) of the best candidate or None.
    """
    li = _mosaicSingleParticles(lPart, lSeg, lU, ls, Dmin)
    fi = _mosaicSingleParticles(fPart, fSeg, fU, fs, Dmin)
    if (len(li) < minPairs) or (len(fi) < minPairs):
        return None
    fUs = fU[fi]
    res = []
    for lag in candidates:
        target = lU[li] + lag
        jj = np.clip(np.searchsorted(fUs, target), 0, len(fUs) - 1)
        hit = fUs[jj] == target
        if hit.sum() < minPairs:
            continue
        dz = lPart["z"][li[hit]] - fPart["z"][fi[jj[hit]]]
        q75, q25 = np.percentile(dz, [75, 25])
        # fraction of pairs in a narrow peak around the median: robust
        # during heavy snowfall, when many "single particle" frames pair
        # different particles and broaden the IQR even for the correct lag
        peakFrac = np.mean(np.abs(dz - np.median(dz)) < 10)
        res.append((q75 - q25, lag, hit.sum(), np.median(dz), peakFrac))
    if len(res) < 2:
        return None
    res.sort()
    return {
        "lag": int(res[0][1]),
        "iqr": float(res[0][0]),
        "iqr2nd": float(res[1][0]),
        "n": int(res[0][2]),
        "dz": float(res[0][3]),
        "peakFrac": float(res[0][4]),
        "peakFrac2nd": float(max(r[4] for r in res[1:])),
    }


def mosaicFrameMappingFname(case, config):
    """daily cache file next to metaRotation"""
    from . import files

    fl = files.FindFiles(case, config.leader, config)
    return fl.fnamesDaily["metaRotation"].replace("metaRotation", "metaFrameMapping")


def createMosaicFrameMapping(
    case,
    config,
    skipExisting=True,
    writeNc=True,
    minOverlapS=60,
    maxIqr=60.0,
    minIqrRatio=1.5,
    activitySearchRange=20000,
    minPeakFrac=0.15,
    minPeakFracRatio=3.0,
    minConsensusIqrRatio=1.1,
):
    """
    Determine, for one day, the exact follower -> leader capture_id mapping
    for every combination of leader and follower camera run.

    For every overlap of a leader run and a follower run:
    1. first guess of the lag from record_time (median over the overlap),
    2. clock-independent check by cross correlating the per-frame "anything
       moving" signal of both cameras,
    3. exact lag from the vertical position of single particles seen by
       both cameras (tested for small and for large particles, which are
       rarer and thus less ambiguous during heavy snowfall).

    Ghost frames are removed from the follower before (see module notes).
    The mapping is cached as metaFrameMapping netCDF next to metaRotation.

    Parameters
    ----------
    case : str
        Day YYYYMMDD
    config : dict or str
        Settings
    skipExisting : bool
        Read cached file if present
    writeNc : bool
        Write cache file
    minOverlapS : float
        Minimum overlap of leader and follower run in seconds
    maxIqr : float
        Maximum IQR (px) of the vertical position difference for the best lag
    minIqrRatio : float
        Minimum ratio IQR(2nd best lag)/IQR(best lag)
    minConsensusIqrRatio : float
        Alternative acceptance if the best lag agrees within 1 frame with
        both the activity (z >= 8) and the record_time estimate: minimum
        IQR ratio of 2nd best/best lag
    minPeakFrac, minPeakFracRatio : float
        Alternative acceptance when the IQR is broad (heavy snowfall): the
        fraction of pairs within +-10 px of the median must be at least
        minPeakFrac and minPeakFracRatio times that of every other lag
    activitySearchRange : int
        Frames around the record_time guess searched by the activity cross
        correlation. Must be wide because the clocks of the two computers
        were not always synchronized (e.g. ~5 s off end of Nov 2019).

    Returns
    -------
    xarray.Dataset or None
        Dataset with dimension "segment" (one per leader/follower run
        overlap; lag, quality flags, follower capture_time range) and
        "ghost" (follower ghost frame sequences)
    """
    import datetime
    import os
    import uuid

    from . import __version__, tools

    config = tools.readSettings(config)
    case = case.split("-")[0]
    fname = mosaicFrameMappingFname(case, config)
    if skipExisting and os.path.isfile(fname):
        with xr.open_dataset(fname) as ds:
            return ds.load()

    log.info(f"createMosaicFrameMapping {case}")
    P = MOSAIC_CAPTURE_ID_PERIOD
    L = _mosaicLoadMetaFrames(case, config.leader, config)
    F = _mosaicLoadMetaFrames(case, config.follower, config)

    rows = []
    ghostRows = []
    if (L is not None) and (F is not None):
        lPeriod = _mosaicFramePeriod(L, config)
        fPeriod = _mosaicFramePeriod(F, config)
        lSegF, lu, lDrop, lGhosts = _mosaicSegmentsAndGhosts(L, lPeriod)
        fSegF, fu, fDrop, fGhosts = _mosaicSegmentsAndGhosts(F, fPeriod)
        if len(lGhosts) > 0:
            # never observed so far; leader capture_id is used as is
            log.warning(f"{len(lGhosts)} ghost frame sequences in LEADER data")
            lSegF, lu, lDrop, _ = _mosaicSegmentsAndGhosts(
                L, lPeriod, correctGhosts=False
            )
        log.info(f"{len(fGhosts)} follower ghost frame sequences")
        for i0, i1, extra in fGhosts:
            ghostRows.append(
                {
                    "ghost_fsegment": fSegF[i0],
                    "ghost_ct_start": F["ct"][i0],
                    "ghost_ct_end": F["ct"][i1],
                    "ghost_extra": extra,
                }
            )

        lPart = _mosaicLoadParticles(case, config.leader, config)
        fPart = _mosaicLoadParticles(case, config.follower, config)
        if (lPart is not None) and (fPart is not None):
            lPSeg, lPU = _mosaicParticleFrames(lPart, L, lSegF, lu, lDrop, lPeriod)
            fPSeg, fPU = _mosaicParticleFrames(fPart, F, fSegF, fu, fDrop, fPeriod)

        fKeep = ~fDrop
        for ls in np.unique(lSegF):
            lm = lSegF == ls
            for fs in np.unique(fSegF):
                fm = (fSegF == fs) & fKeep
                if fm.sum() == 0:
                    continue
                t0 = max(L["rt"][lm].min(), F["rt"][fm].min())
                t1 = min(L["rt"][lm].max(), F["rt"][fm].max())
                if (t1 - t0) < minOverlapS * 1e6:
                    continue
                lmm = lm & (L["rt"] >= t0) & (L["rt"] <= t1)
                fmm = fm & (F["rt"] >= t0) & (F["rt"] <= t1)
                if (lmm.sum() < 1000) or (fmm.sum() < 1000):
                    continue
                row = {
                    "lsegment": ls,
                    "fsegment": fs,
                    "rt_start": t0,
                    "rt_end": t1,
                    "fct_start": F["ct"][fmm].min(),
                    "fct_end": F["ct"][fmm].max(),
                    "lag_recordTime": -1,
                    "lag_activity": -1,
                    "activity_z": np.nan,
                    "lag": -1,
                    "lag_mod": -1,
                    "dz_iqr": np.nan,
                    "dz_iqr_2nd": np.nan,
                    "dz_median": np.nan,
                    "n_pairs": 0,
                    "dmin": 0,
                    "peak_frac": np.nan,
                    "peak_frac_2nd": np.nan,
                    "resolved": False,
                }

                # 1. record_time first guess
                lIdx = np.flatnonzero(lmm)
                lIdx = lIdx[
                    np.linspace(0, len(lIdx) - 1, min(20000, len(lIdx))).astype(int)
                ]
                fIdx = np.flatnonzero(fmm)
                fOrder = np.argsort(F["rt"][fIdx], kind="stable")
                fRt = F["rt"][fIdx][fOrder]
                jj = np.clip(np.searchsorted(fRt, L["rt"][lIdx]), 1, len(fRt) - 1)
                jj = np.where(
                    np.abs(fRt[jj - 1] - L["rt"][lIdx])
                    < np.abs(fRt[jj] - L["rt"][lIdx]),
                    jj - 1,
                    jj,
                )
                close = np.abs(fRt[jj] - L["rt"][lIdx]) < 20e3
                if close.sum() < 100:
                    rows.append(row)
                    continue
                guess = int(np.median(fu[fIdx][fOrder][jj[close]] - lu[lIdx[close]]))
                row["lag_recordTime"] = guess

                # 2. activity cross correlation
                lagAct, zAct = _mosaicActivityLag(
                    lu[lmm],
                    L["nmp"][lmm],
                    fu[fmm],
                    F["nmp"][fmm],
                    guess,
                    searchRange=activitySearchRange,
                )
                candidates = set(range(guess - 3, guess + 4))
                if lagAct is not None:
                    row["lag_activity"] = lagAct
                    row["activity_z"] = zAct
                    if zAct > 8:
                        candidates |= set(range(lagAct - 3, lagAct + 4))

                # 3. exact lag from particles
                if (lPart is None) or (fPart is None):
                    rows.append(row)
                    continue
                best = None
                for Dmin in [8, 20]:
                    res = _mosaicParticleLagTest(
                        lPart,
                        lPSeg,
                        lPU,
                        ls,
                        fPart,
                        fPSeg,
                        fPU,
                        fs,
                        sorted(candidates),
                        Dmin,
                    )
                    if res is None:
                        continue
                    res["dmin"] = Dmin
                    res["ratio"] = res["iqr2nd"] / max(res["iqr"], 1e-3)
                    if (best is None) or (res["ratio"] > best["ratio"]):
                        if (best is not None) and (best["lag"] != res["lag"]):
                            # both particle sizes are decisive but disagree
                            if min(best["ratio"], res["ratio"]) >= minIqrRatio:
                                best = None
                                break
                        best = res
                if best is not None:
                    row.update(
                        {
                            "lag": best["lag"],
                            "lag_mod": best["lag"] % P,
                            "dz_iqr": best["iqr"],
                            "dz_iqr_2nd": best["iqr2nd"],
                            "dz_median": best["dz"],
                            "n_pairs": best["n"],
                            "dmin": best["dmin"],
                            "resolved": (best["iqr"] <= maxIqr)
                            and (best["ratio"] >= minIqrRatio),
                            "resolved_peak": (
                                (best["peakFrac"] >= minPeakFrac)
                                and (
                                    best["peakFrac"]
                                    >= minPeakFracRatio * best["peakFrac2nd"]
                                )
                            )
                            # consensus: three independent estimates agree
                            # (only possible if the computer clocks are in
                            # sync); the particle test is then just not
                            # decisive, typically because particles move
                            # little within one frame
                            or (
                                (best["ratio"] >= minConsensusIqrRatio)
                                and (row["activity_z"] >= 8)
                                and (abs(best["lag"] - row["lag_activity"]) <= 1)
                                and (abs(best["lag"] - row["lag_recordTime"]) <= 1)
                            ),
                            "peak_frac": best["peakFrac"],
                            "peak_frac_2nd": best["peakFrac2nd"],
                        }
                    )
                rows.append(row)

    # alternative acceptance (peak fraction or consensus), but only if the vertical
    # offset agrees with the segments of the same day resolved by the IQR
    # criterion (camera geometry does not change within a day)
    # (if none, the peak-qualified segments themselves are the reference)
    dzRef = [r["dz_median"] for r in rows if r["resolved"]]
    if len(dzRef) == 0:
        dzRef = [r["dz_median"] for r in rows if r.get("resolved_peak", False)]
    for r in rows:
        peakOk = r.pop("resolved_peak", False)
        if (not r["resolved"]) and peakOk and (len(dzRef) > 0):
            r["resolved"] = bool(abs(r["dz_median"] - np.median(dzRef)) < 10)

    def asArray(key, rr, dim, dtype=None):
        return xr.DataArray(np.array([r[key] for r in rr], dtype=dtype), dims=[dim])

    ds = xr.Dataset()
    if len(rows) > 0:
        for key in rows[0].keys():
            if key in ["rt_start", "rt_end", "fct_start", "fct_end"]:
                ds[key] = (
                    asArray(key, rows, "segment", np.int64)
                    .astype("datetime64[us]")
                    .astype("datetime64[ns]")
                )
            elif key == "resolved":
                ds[key] = asArray(key, rows, "segment", np.int8)
            else:
                ds[key] = asArray(key, rows, "segment")
    if len(ghostRows) > 0:
        for key in ghostRows[0].keys():
            if key in ["ghost_ct_start", "ghost_ct_end"]:
                ds[key] = (
                    asArray(key, ghostRows, "ghost", np.int64)
                    .astype("datetime64[us]")
                    .astype("datetime64[ns]")
                )
            else:
                ds[key] = asArray(key, ghostRows, "ghost", np.int64)
    ds.attrs["description"] = (
        "MOSAiC leader/follower frame mapping: leader capture_id = "
        "((follower capture_id - ghost_extra of preceding ghosts in the same "
        "fsegment - lag_mod - 1) mod 65535) + 1 for follower frames with "
        "fct_start <= capture_time <= fct_end of a resolved segment"
    )
    ds.attrs[
        "history"
    ] = f"{datetime.datetime.utcnow()}: created with VISSSlib {__version__}"
    nRes = int(ds.resolved.sum()) if "resolved" in ds else 0
    log.info(
        f"createMosaicFrameMapping {case}: {nRes} of {len(rows)} segments resolved"
    )

    if writeNc:
        tools.createParentDir(fname, mode=config.dirMode)
        tmpFile = f"{fname}.{os.getpid()}.{uuid.uuid4().hex}.tmp.cdf"
        ds.to_netcdf(tmpFile)
        os.chmod(tmpFile, config.fileMode)
        os.replace(tmpFile, fname)
        log.info(f"saved {fname}")
    return ds


def mosaicFollowerCaptureIdToLeader(follower1D, config, dim="fpid"):
    """
    Replace the (raw, not yet overflow corrected) follower capture_id by the
    capture_id the leader assigned to the same trigger, using the daily
    metaFrameMapping. If missing (e.g. the next day around midnight), it is
    computed on the fly but not cached, because level1detect of that day
    might not be complete yet; the cache is written by createMetaRotation.
    Particles in ghost frame sequences or in periods without reliable
    mapping are removed.

    Parameters
    ----------
    follower1D : xarray.Dataset
        follower level1detect data with raw capture_id
    config : dict
        Settings
    dim : str
        particle dimension

    Returns
    -------
    xarray.Dataset or None
    """
    import pandas as pd

    P = MOSAIC_CAPTURE_ID_PERIOD
    ct = follower1D.capture_time.values.astype("datetime64[ns]")
    raw = follower1D.capture_id.values.astype(np.int64)
    days = np.unique(
        np.concatenate(
            [
                pd.to_datetime(ct - np.timedelta64(10, "s")).strftime("%Y%m%d"),
                pd.to_datetime(ct + np.timedelta64(10, "s")).strftime("%Y%m%d"),
            ]
        )
    )
    newId = np.full(len(raw), -1, dtype=np.int64)
    ghostDrop = np.zeros(len(raw), dtype=bool)
    for day in days:
        mapping = createMosaicFrameMapping(
            day,
            config,
            skipExisting=True,
            writeNc=False,
        )
        if (mapping is None) or ("segment" not in mapping.dims):
            continue
        for ss in range(len(mapping.segment)):
            seg = mapping.isel(segment=ss)
            if not bool(seg.resolved):
                continue
            m = (ct >= seg.fct_start.values) & (ct <= seg.fct_end.values) & (newId < 0)
            if m.sum() == 0:
                continue
            nBefore = np.zeros(m.sum(), dtype=np.int64)
            if "ghost" in mapping.dims:
                g = mapping.isel(ghost=(mapping.ghost_fsegment == seg.fsegment).values)
                for gg in range(len(g.ghost)):
                    gStart = g.ghost_ct_start.values[gg]
                    gEnd = g.ghost_ct_end.values[gg]
                    ghostDrop[m] |= (ct[m] > gStart) & (ct[m] < gEnd)
                    nBefore += (ct[m] >= gEnd) * int(g.ghost_extra.values[gg])
            newId[m] = ((raw[m] - nBefore - int(seg.lag_mod) - 1) % P) + 1

    keep = (newId > 0) & ~ghostDrop
    log.info(
        f"mosaicFollowerCaptureIdToLeader: kept {keep.sum()} of {len(keep)} "
        f"follower particles ({ghostDrop.sum()} in ghost frames, "
        f"{(newId < 0).sum()} without reliable mapping)"
    )
    if keep.sum() == 0:
        return None
    follower1D = follower1D.isel({dim: keep})
    follower1D["capture_id"] = xr.DataArray(
        newId[keep].astype(follower1D.capture_id.dtype), dims=[dim]
    )
    return follower1D


def mosaicCaptureIdOffset(leader1D, follower1D, dim="fpid"):
    """
    Offset between follower and leader capture_id after
    mosaicFollowerCaptureIdToLeader and captureIdOverflows. Both cameras then
    use the same ids, but captureIdOverflows unwraps each data set relative
    to its own start, so the offset is a multiple of 65535, resolved with
    record_time (good to a few seconds, i.e. << 65535 frames).
    """
    P = MOSAIC_CAPTURE_ID_PERIOD
    lRt = leader1D.record_time.values.astype("datetime64[ns]").astype(np.int64)
    fRt = follower1D.record_time.values.astype("datetime64[ns]").astype(np.int64)
    order = np.argsort(fRt)
    fRt = fRt[order]
    fId = follower1D.capture_id.values.astype(np.int64)[order]
    jj = np.clip(np.searchsorted(fRt, lRt), 0, len(fRt) - 1)
    diff = fId[jj] - leader1D.capture_id.values.astype(np.int64)
    return int(np.round(np.median(diff) / P)) * P


def mosaicUnmappedTimes(case, config, timeIndex, maxUnmappedS=10):
    """
    Flag time bins in which follower frames could not be mapped to leader
    frames (no resolved metaFrameMapping segment, e.g. around follower
    restarts or for unresolved segments). Follower particles of such periods
    are dropped by mosaicFollowerCaptureIdToLeader, so level2match/track
    would otherwise silently report too few particles.

    Parameters
    ----------
    case : str
        Day YYYYMMDD
    config : dict
        Settings
    timeIndex : pandas.DatetimeIndex
        Start times of the level2 time bins (with freq)
    maxUnmappedS : float
        Bins with more than this many seconds not covered by a resolved
        segment are flagged

    Returns
    -------
    xarray.DataArray of bool with dimension time
    """
    import os

    import pandas as pd

    freqS = timeIndex.freq.nanos * 1e-9
    starts = []
    ends = []
    day = pd.Timestamp(case[:8])
    for dd in [day - pd.Timedelta("1D"), day, day + pd.Timedelta("1D")]:
        fname = mosaicFrameMappingFname(dd.strftime("%Y%m%d"), config)
        if not os.path.isfile(fname):
            continue
        with xr.open_dataset(fname) as ds:
            if "segment" not in ds.dims:
                continue
            ok = ds.resolved.values.astype(bool)
            starts.append(ds.fct_start.values[ok])
            ends.append(ds.fct_end.values[ok])
    if len(starts) > 0:
        starts = np.concatenate(starts)
        ends = np.concatenate(ends)
    binStart = timeIndex.values.astype("datetime64[ns]")
    binEnd = binStart + np.timedelta64(int(freqS * 1e9), "ns")
    covered = np.zeros(len(binStart))
    for s, e in zip(starts, ends):
        overlap = (np.minimum(binEnd, e) - np.maximum(binStart, s)) / np.timedelta64(
            1, "s"
        )
        covered += np.clip(overlap, 0, None)
    unmapped = (freqS - covered) > maxUnmappedS
    log.info(f"mosaicUnmappedTimes: {unmapped.sum()} of {len(unmapped)} bins flagged")
    return xr.DataArray(unmapped, dims=["time"], coords=[timeIndex])
