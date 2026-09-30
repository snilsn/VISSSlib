# -*- coding: utf-8 -*-
import datetime
import os
import sys
import warnings
from copy import deepcopy

import numpy as np
import xarray as xr

# import av
from loguru import logger as log

from . import __version__, files, fixes, metadata, quicklooks, tools

# log = logger.bind(name=__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


deltaY = deltaH = deltaI = 1.0


def rotate_L2F(L_x, L_y, L_z, phi, theta, psi):
    """
    Rotate from leader to follower coordinate system.

    Parameters
    ----------
    L_x : float
        Leader x coordinate (in common xyz)
    L_y : float
        Leader y coordinate (in common xyz)
    L_z : float
        Leader z coordinate (in common xyz)
    phi : float
        Follower roll in degrees
    theta : float
        Follower pitch in degrees
    psi : float
        Follower yaw in degrees

    Returns
    -------
    tuple
        Follower x, y, z coordinates
    """
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)

    F_xp = (
        np.cos(theta) * np.cos(psi) * L_x
        + np.cos(theta) * np.sin(psi) * L_y
        - np.sin(theta) * L_z
    )
    F_yp = (
        (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * L_x
        + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * L_y
        + np.sin(phi) * np.cos(theta) * L_z
    )
    F_zp = (
        (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * L_x
        + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * L_y
        + np.cos(phi) * np.cos(theta) * L_z
    )

    return F_xp, F_yp, F_zp


def shiftRotate_L2F(L_x, L_y, L_z, phi, theta, psi, Olx, Ofy, Ofz):
    """
    Shift and rotate from leader to follower coordinate system.

    Parameters
    ----------
    L_x : float
        Leader x coordinate (in common xyz)
    L_y : float
        Leader y coordinate (in common xyz)
    L_z : float
        Leader z coordinate (in common xyz)
    phi : float
        Follower roll in degrees
    theta : float
        Follower pitch in degrees
    psi : float
        Follower yaw in degrees
    Olx : float
        Leader shift in x direction
    Ofy : float
        Follower shift in y direction
    Ofz : float
        Follower shift in z direction

    Returns
    -------
    tuple
        Follower x, y, z coordinates
    """
    L_xp = L_x + Olx

    F_x, F_yp, F_zp = rotate_L2F(L_xp, L_y, L_z, phi, theta, psi)

    F_y = F_yp - Ofy
    F_z = F_zp - Ofz

    return F_x, F_y, F_z


def rotate_F2L(F_xp, F_yp, F_zp, phi, theta, psi):
    """
    Rotate from follower to leader coordinate system.

    Parameters
    ----------
    F_xp : float
        Follower x coordinate (in common xyz)
    F_yp : float
        Follower y coordinate (in common xyz)
    F_zp : float
        Follower z coordinate (in common xyz)
    phi : float
        Follower roll in degrees
    theta : float
        Follower pitch in degrees
    psi : float
        Follower yaw in degrees

    Returns
    -------
    tuple
        Leader x, y, z coordinates
    """
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)

    L_x = (
        np.cos(theta) * np.cos(psi) * F_xp
        + (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * F_yp
        + (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * F_zp
    )
    L_y = (
        np.cos(theta) * np.sin(psi) * F_xp
        + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * F_yp
        + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * F_zp
    )
    L_z = (
        -np.sin(theta) * F_xp
        + np.sin(phi) * np.cos(theta) * F_yp
        + np.cos(phi) * np.cos(theta) * F_zp
    )

    return L_x, L_y, L_z


def shiftRotate_F2L(F_x, F_y, F_z, phi, theta, psi, Olx, Ofy, Ofz):
    """
    Shift and rotate from follower to leader coordinate system.

    Parameters
    ----------
    F_x : float
        Follower x coordinate (in common xyz)
    F_y : float
        Follower y coordinate (in common xyz)
    F_z : float
        Follower z coordinate (in common xyz)
    phi : float
        Follower roll in degrees
    theta : float
        Follower pitch in degrees
    psi : float
        Follower yaw in degrees
    Olx : float
        Leader shift in x direction
    Ofy : float
        Follower shift in y direction
    Ofz : float
        Follower shift in z direction

    Returns
    -------
    tuple
        Leader x, y, z coordinates
    """
    F_yp = F_y + Ofy
    F_zp = F_z + Ofz

    L_xp, L_y, L_z = rotate_F2L(F_x, F_yp, F_zp, phi, theta, psi)

    L_x = L_xp - Olx

    return L_x, L_y, L_z


def calc_L_z(L_x, F_yp, F_zp, phi, theta, psi):
    """
    Estimate z coordinate for the leader based on combined leader and follower measurements.

    Parameters
    ----------
    L_x : float
        x measurement of leader
    F_yp : float
        y measurement of follower without shift
    F_zp : float
        z measurement of follower without shift
    phi : float
        Follower roll in degrees
    theta : float
        Follower pitch in degrees
    psi : float
        Follower yaw in degrees

    Returns
    -------
    float
        z coordinate as seen by leader ignoring offsets
    """
    # with wolfram simplification

    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)

    L_z = (
        -(np.sin(theta)) / (np.cos(theta) * np.cos(psi)) * L_x
        - (np.sin(theta) * np.sin(psi) * np.cos(phi) - np.cos(psi) * np.sin(phi))
        / (np.cos(theta) * np.cos(psi))
        * F_yp
        + (np.sin(theta) * np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(phi))
        / (np.cos(theta) * np.cos(psi))
        * F_zp
    )
    return L_z


def calc_L_z_withOffsets(
    L_x,
    F_y,
    F_z,
    camera_phi=0,
    camera_theta=0,
    camera_psi=0,
    camera_Ofy=0,
    camera_Ofz=0,
    camera_Olx=0,
):
    """
    Estimate z coordinate for the leader based on combined leader and follower measurements.

    Parameters
    ----------
    L_x : float
        x measurement of leader
    F_y : float
        y measurement of follower
    F_z : float
        z measurement of follower
    camera_phi : float, optional
        Follower roll in degrees (default is 0)
    camera_theta : float, optional
        Follower pitch in degrees (default is 0)
    camera_psi : float, optional
        Follower yaw in degrees (default is 0)
    camera_Ofy : float, optional
        Follower shift in y direction (default is 0)
    camera_Ofz : float, optional
        Follower shift in z direction (default is 0)
    camera_Olx : float, optional
        Leader shift in x direction (default is 0)

    Returns
    -------
    float
        z coordinate as seen by leader
    """
    for k in camera_phi, camera_theta, camera_psi, camera_Ofy, camera_Ofz, camera_Olx:
        assert not np.any(np.isnan(k)), k

    F_yp = F_y + camera_Ofy
    F_zp = F_z + camera_Ofz
    L_xp = L_x + camera_Olx

    return calc_L_z(L_xp, F_yp, F_zp, camera_phi, camera_theta, camera_psi)


def forward(x, L_x=None, F_y=None, F_z=None):
    """
    Forward model for pyOptimalEstimation.

    Parameters
    ----------
    x : pandas Series
        State vector "phi", "theta", "psi", "Ofy", "Ofz", "Olx"
    L_x : array, optional
        x coordinate as seen by the leader (default is None)
    F_y : array, optional
        y coordinate as seen by the follower (default is None)
    F_z : array, optional
        z coordinate as seen by the follower (default is None)

    Returns
    -------
    pandas Series
        z coordinate as seen by leader
    """
    import pandas as pd

    y = calc_L_z_withOffsets(L_x, F_y, F_z, **x.to_dict())
    y = pd.Series(y, index=np.array(range(len(y))))
    return y


def retrieveRotation(
    dat3, x_ap, x_cov_diag, y_cov_diag, config, verbose=False, maxIter=30
):
    """
    Apply Optimal Estimation to retrieve rotation of cameras.

    Parameters
    ----------
    dat3 : xarray Dataset
        Input data containing particle information
    x_ap : dict
        A priori values for state variables
    x_cov_diag : array-like
        Diagonal elements of a priori covariance matrix
    y_cov_diag : array-like
        Diagonal elements of observation covariance matrix
    config : object
        Configuration object with camera settings
    verbose : bool, optional
        Whether to print verbose output (default is False)
    maxIter : int, optional
        Maximum number of iterations (default is 30)

    Returns
    -------
    tuple
        (x_op, x_op_err, dgf_x) - Optimal state vector, errors, and goodness of fit
    """
    import pandas as pd
    import pyOptimalEstimation as pyOE

    nPart = len(dat3.pair_id)
    allVars = [
        "camera_phi",
        "camera_theta",
        "camera_psi",
        "camera_Ofy",
        "camera_Ofz",
        "camera_Olx",
    ]
    assert dat3.camera[0].values == config.leader

    L_x, L_z, F_y, F_z = get3DPosition(
        dat3.sel(camera=config.leader), dat3.sel(camera=config.follower), config
    )

    x_vars = list(x_ap.keys())
    b_vars = [k for k in allVars if k not in x_vars]

    b_p = pd.Series([0] * len(allVars), index=allVars)[b_vars]
    S_b = np.identity(len(b_vars)) * 0.1
    y_vars = np.array(range(nPart))

    x_cov = np.identity(len(x_vars)) * np.array(x_cov_diag)
    y_cov = np.identity(nPart) * np.array(y_cov_diag)

    y_obs = L_z

    forwardKwArgs = {"L_x": L_x, "F_y": F_y, "F_z": F_z}

    # create optimal estimation object
    oe = pyOE.optimalEstimation(
        x_vars,
        x_ap,
        x_cov,
        y_vars,
        y_obs,
        y_cov,
        forward,
        b_vars=b_vars,
        b_p=b_p,
        S_b=S_b,
        forwardKwArgs=forwardKwArgs,
        verbose=verbose,
    )

    oe.doRetrieval(maxIter=maxIter)

    assert not np.any(np.isnan(oe.x_op))

    return oe.x_op, oe.x_op_err, oe.dgf_x


def _selectRefitSample(matchedDat, config, nSamples4rot, minPairs):
    """
    Pick which pairs `_refitRotationWindow` should fit the OE retrieval
    against.

    The obvious choice -- the `nSamples4rot` pairs with the highest
    *current* matchScore -- breaks down as the pair count grows: with
    hundreds of thousands of pairs, there is essentially always a small
    tail whose noise happens to cancel a real systematic rotation bias by
    pure chance, so "top by current score" preferentially selects exactly
    those already-lucky pairs and the OE fit just confirms the bias
    instead of correcting it (confirmed on a real file: the "top 300"
    subset had matchScore ~0.038 -- already far above the 0.001 threshold
    -- while the file's true median was ~9e-5, and refitting against it
    left the rotation and the file's real median matchScore essentially
    unchanged).

    Prefer pairs from single-particle frames instead: frames where the
    leader detected exactly one particle have no correspondence ambiguity
    at all (the same principle `manualRotationEstimate`'s
    `singleParticleFramesOnly` already relies on elsewhere in this
    module), so their Z-residual reflects the rotation error directly
    rather than either a lucky coincidence or a possible mismatch. Within
    that pool, prefer the largest particles (Dmax), since a bigger
    particle's centroid is measured more precisely, further reducing
    noise unrelated to the rotation itself. Falls back to the previous
    top-by-matchScore behavior when fewer than `minPairs` single-particle-
    frame pairs are available (e.g. small/sparse files, where that was
    never broken in the first place).

    Parameters
    ----------
    matchedDat : xarray.Dataset
        The matched-pairs dataset to select from.
    config : dict
        Configuration settings (for `config.leader`).
    nSamples4rot : int
        Maximum number of pairs to return.
    minPairs : int
        Minimum number of single-particle-frame pairs required before
        preferring them over the matchScore-based fallback.

    Returns
    -------
    xarray.Dataset
        Up to `nSamples4rot` pairs from `matchedDat`.
    """
    n = min(nSamples4rot, len(matchedDat.pair_id))

    captureId = matchedDat.sel(camera=config.leader).capture_id.values
    _, inverse, counts = np.unique(captureId, return_inverse=True, return_counts=True)
    singleParticleMask = counts[inverse] == 1
    nSingle = int(np.sum(singleParticleMask))

    if nSingle >= minPairs:
        idx = np.where(singleParticleMask)[0]
        dmax = matchedDat.sel(camera=config.leader).Dmax.values[idx]
        idx = idx[np.argsort(dmax)[::-1][:n]]
        return matchedDat.isel(pair_id=sorted(idx))

    return matchedDat.isel(pair_id=sorted(np.argsort(matchedDat.matchScore.values)[-n:]))


def _refitRotationWindow(
    matchedDat,
    rotate,
    rotate_err,
    config,
    nSamples4rot,
    y_cov_diag,
    minPairs,
):
    """
    Single-window implementation behind `refitRotationFromMatches`: fit
    one rotation against every pair in `matchedDat` and rescale
    matchScore accordingly. Factored out so `refitRotationFromMatches`
    can also apply it per time-segment (see `nSegments`) for files whose
    true rotation drifts smoothly within the file itself, rather than
    being one constant offset -- see that function's docstring.

    Returns
    -------
    tuple or None
        (newMatchedDat, newRotate, newRotateErr), or None if there were
        too few pairs or the OE fit itself failed.
    """
    import pandas as pd

    if len(matchedDat.pair_id) < minPairs:
        return None

    rotate = pd.Series(dict(rotate))
    rotate_err = pd.Series(dict(rotate_err))

    top = _selectRefitSample(matchedDat, config, nSamples4rot, minPairs)
    try:
        newRotate, newRotateErr, _ = retrieveRotation(
            top,
            rotate,
            (rotate_err * 10) ** 2,
            y_cov_diag,
            config,
        )
    except AssertionError as e:
        log.warning(tools.concat("refitRotationFromMatches: OE refit failed", str(e)))
        return None

    # matchScore is a product of independent terms and only the Z term
    # depends on rotation (see docstring), so it can be rescaled by the
    # ratio of the old vs. new Z-probability without re-running doMatch --
    # but position3D_center/_centroid (the actual calibrated positions,
    # the real point of this product) and the per-pair rotation columns
    # must be properly recomputed via addPosition, not left stale.
    #
    # matchedDat may or may not already have position3D_center -- the
    # normal (non-rotationOnly) matchParticles path calls addPosition
    # per segment before this is ever reached, but createMetaRotation's
    # rotationOnly=True path returns matchedDat4Rot straight out of
    # _refineRotationIteration, which never calls addPosition at all.
    # Compute L_z/L_z_estimated directly instead of relying on it having
    # been added already, so this works either way.
    zSigma, zDelta = 1.7, 0.5
    L_x, L_z, F_y, F_z = get3DPosition(
        matchedDat.sel(camera=config.leader),
        matchedDat.sel(camera=config.follower),
        config,
    )
    L_z_est_old = calc_L_z_withOffsets(L_x, F_y, F_z, **rotate.to_dict())
    propZ_old = probability(L_z - L_z_est_old, 0, zSigma, zDelta)

    newMatchedDat = addPosition(
        matchedDat.copy(deep=True), newRotate.to_dict(), newRotateErr.to_dict(), config
    )
    newPos = newMatchedDat.position3D_center
    propZ_new = probability(
        newPos.sel(dim3D="z") - newPos.sel(dim3D="z_rotated"), 0, zSigma, zDelta
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rescale = xr.where(propZ_old > 0, propZ_new / propZ_old, 0.0)
    newMatchedDat["matchScore"] = matchedDat.matchScore * rescale

    if not np.all(np.isfinite(newMatchedDat.matchScore.values)):
        return None

    log.warning(
        tools.concat(
            "refitRotationFromMatches: old rotate",
            rotate.to_dict(),
            "new rotate",
            newRotate.to_dict(),
            "old median matchScore",
            float(matchedDat.matchScore.median()),
            "new median matchScore",
            float(newMatchedDat.matchScore.median()),
        )
    )
    return newMatchedDat, newRotate, newRotateErr


def zResidualSigma(matchedDat, rotate, config):
    """
    Standard deviation of the per-pair Z-consistency residual (diffZ =
    L_z - L_z_estimated) for the given matched pairs and rotation -- the
    same quantity `_refitRotationWindow`/`addPosition` compute internally
    to rescale matchScore's Z term, exposed standalone so it can be
    checked *before* attempting a refit (see `matchParticles`) or against
    already-written output independent of the aggregate matchScore (see
    `scripts/qc_report.py`'s `--matchscore-check` and
    `distributions.addVariables`'s per-timestep quality flag).

    A rotation refit is fundamentally a location-shift operation: it can
    correct a *biased-but-tight* residual (a systematic mis-calibration
    for this file/window -- refit finds a better rotation and the
    residual collapses toward zero) but cannot shrink an intrinsically
    *wide* one (refit can only move the center of an already-scattered
    distribution, not tighten it). See `config.quality.maxZSigma` for how
    this is used as a threshold, and its comment for the empirical basis.

    Parameters
    ----------
    matchedDat : xarray.Dataset
        Matched-pairs dataset (as produced by doMatch/doMatchSlicer, or
        read back from a level1match/level1track file). If it already has
        `position3D_center` (`addPosition` has run -- true for real
        output and for matchedDat at the point matchParticles' quality
        gate runs), its dim3D "z"/"z_rotated" entries are used directly,
        which is both cheaper and guaranteed consistent with whatever
        rotation actually produced this exact matchedDat. Otherwise
        (matchedDat is a raw pre-addPosition doMatch result) falls back
        to recomputing via `get3DPosition`/`calc_L_z_withOffsets` with the
        given `rotate`.
    rotate : dict or pandas.Series
        The rotation used to produce matchedDat (camera_Ofz/camera_phi/
        camera_theta, ...) -- only used in the fallback path.
    config : dict
        Configuration settings (for config.leader/config.follower).

    Returns
    -------
    float
        np.nanstd of the per-pair Z residual, in the same units as
        camera_Ofz -- pixels, level1's native unit (position3D_center is
        built directly from level1 pixel measurements; only level2
        aggregation converts to SI/meters via config.calibration.slope).
    """
    if "position3D_center" in matchedDat:
        pos = matchedDat.position3D_center
        diffZ = pos.sel(dim3D="z") - pos.sel(dim3D="z_rotated")
    else:
        rotate = dict(rotate)
        L_x, L_z, F_y, F_z = get3DPosition(
            matchedDat.sel(camera=config.leader),
            matchedDat.sel(camera=config.follower),
            config,
        )
        L_z_est = calc_L_z_withOffsets(L_x, F_y, F_z, **rotate)
        diffZ = L_z - L_z_est
    return float(np.nanstd(diffZ))


def _worstLocalZSigma(matchedDat, config, binSeconds=60):
    """
    Worst (maximum) per-time-bin Z-residual sigma within a single
    matchedDat, or None if no bin has enough pairs to judge.

    matchParticles' quality gate only ever looks at the file's AGGREGATE
    matchScore/Z-residual sigma -- a file with one genuinely bad few-
    minute stretch can pass comfortably on aggregate if the rest of the
    10-minute file is fine, exactly the gap
    `distributions.getZResidualQuality`'s level2 `zResidualTooWide` flag
    was built to catch after the fact, on already-written output. This
    computes the same per-bin check *before* writing the file, so a
    localized problem gets a chance at the same refit self-heal an
    aggregate failure already gets, rather than only ever being
    reported downstream with nothing attempted.

    Parameters
    ----------
    matchedDat : xarray.Dataset
        Matched-pairs dataset with `position3D_center` (i.e. already
        through `addPosition`) and a leader/follower "camera" dimension.
    config : dict
        Configuration settings (for config.leader, config.quality.*).
    binSeconds : float, optional
        Width of the time bins to check, default 60s -- matches level2's
        own per-minute granularity.

    Returns
    -------
    float or None
        The worst per-bin sigma among bins with enough pairs to trust
        (the same "~1 pair/second" floor `getZResidualQuality` uses), or
        None if no bin qualifies (too few pairs everywhere, or the file
        spans less than one full bin).
    """
    import pandas as pd

    time = matchedDat.capture_time.sel(camera=config.leader).values
    pos = matchedDat.position3D_center
    diffZ = (pos.sel(dim3D="z") - pos.sel(dim3D="z_rotated")).values

    start = pd.Timestamp(np.min(time)).floor(f"{int(binSeconds)}s")
    end = pd.Timestamp(np.max(time)) + pd.Timedelta(seconds=binSeconds)
    bins = pd.date_range(start, end, freq=f"{int(binSeconds)}s")
    if len(bins) < 2:
        return None

    binCode = pd.cut(pd.DatetimeIndex(time), bins=bins, right=False, labels=False)
    binned = pd.Series(diffZ, index=binCode).dropna()
    grouped = binned.groupby(level=0)
    sigmaPerBin = grouped.std()
    countPerBin = grouped.count()

    minPairsPerBin = max(1, round(binSeconds))
    valid = sigmaPerBin[countPerBin >= minPairsPerBin]
    if len(valid) == 0:
        return None
    return float(valid.max())


def refitRotationFromMatches(
    matchedDat,
    rotate,
    rotate_err,
    config,
    sigma="default",
    nSamples4rot=300,
    y_cov_diag=1.65**2,
    minPairs=10,
    nSegments=1,
):
    """
    Try to recover a low-matchScore result by refitting the rotation
    directly from the already-matched pairs, instead of re-matching from
    scratch.

    Rationale (see docs/source/metaRotation.rst and the investigation
    that motivated this): matchScore is the product of independent
    per-dimension probability terms (Y, T or I, H, Z), and only the Z
    term (particle height via the stereo geometry, calc_L_z_withOffsets)
    depends on the rotation. Y/H/T/I terms only depend on frame position,
    height-in-frame, and capture_id/time -- none of which change when the
    rotation is refit. So a file where correspondence is trustworthy
    (Y/H/T/I already look fine) but matchScore is dragged down by a
    tight, systematic (not noisy) Z offset is a sign the rotation
    calibration has drifted for this window, not that the matches are
    wrong -- and can be fixed with a single Optimal Estimation refit
    against the existing pairs (retrieveRotation), which is fast and
    independent of file size, rather than the much slower alternative of
    re-matching from a blind prior (manualRotationEstimate).

    This only recomputes the Z-dependent term of matchScore for the
    existing pairs -- it does not re-run doMatch, so it cannot find new
    correspondences the original (wrong-rotation) match missed, only
    rescue/rescore the ones it already found. It is only valid for the
    "default" sigma/delta used everywhere in this codebase (Z: sigma=1.7,
    delta=0.5, the same constant in both the ptpTime and non-ptpTime
    default branches of doMatch) -- skipped for a custom sigma, since
    this function has no way to know what Z sigma/delta that used.

    With `nSegments > 1`, instead of fitting one rotation for the whole
    file, the pairs are split into `nSegments` contiguous, time-ordered
    chunks (by the leader's capture_time) and each chunk gets its own
    independent refit -- for files where the *true* rotation drifts
    smoothly within the file itself (e.g. progressive snow loading,
    thermal settling during a long, heavy-precipitation file), a single
    global rotation is structurally the wrong model: it necessarily lands
    on a compromise that fits neither the start nor the end well, however
    good the OE fit machinery is. See the investigation notes this
    followed up on (self-heal work, matchScore failures with slow
    within-file drift) for the empirical basis. Each segment that has too
    few pairs or fails its own refit falls back to its original
    (unrefit) scores/positions rather than failing the whole file --
    partial recovery from the segments that do refit successfully is
    still better than none.

    Parameters
    ----------
    matchedDat : xarray.Dataset
        The (low-scoring) matched-pairs dataset from doMatch/doMatchSlicer,
        with a "camera" dimension holding the leader/follower rows and a
        pair_id-indexed "matchScore".
    rotate : dict or pandas.Series
        The rotation that was used to produce matchedDat.
    rotate_err : dict or pandas.Series
        Its uncertainty.
    config : dict
        Configuration settings.
    sigma : str or dict, optional
        The sigma the original match used (default "default"); refit is
        skipped for anything else, see Notes above.
    nSamples4rot : int, optional
        Number of best-scoring pairs to refit from per window, same
        default as metaRotation's own refinement (default 300).
    y_cov_diag : float, optional
        Observation covariance diagonal for the OE fit (default 1.65**2,
        matchParticles' own default).
    minPairs : int, optional
        Minimum number of pairs required to even attempt a refit, per
        window (default 10).
    nSegments : int, optional
        Number of contiguous time-ordered chunks to split `matchedDat`
        into and refit independently (default 1, i.e. today's
        single-window behavior).

    Returns
    -------
    tuple or None
        (newMatchedDat, newRotate, newRotateErr) -- matchedDat with
        "matchScore", "position3D_center"/"position3D_centroid" (via
        addPosition), and the per-pair rotation columns all recomputed
        for the refit rotation(s) -- if at least one window's refit ran
        and produced a finite result; None if there were too few pairs
        overall, sigma wasn't "default", or every window's OE fit
        failed. With `nSegments > 1`, newRotate/newRotateErr are the
        last successfully-refit segment's values (there is no single
        rotation to report -- only per-pair positions/scores are
        meaningful in that case, and those are all correctly set on
        newMatchedDat regardless of segment).
    """
    if sigma != "default":
        return None
    if len(matchedDat.pair_id) < minPairs:
        return None

    if nSegments <= 1:
        return _refitRotationWindow(
            matchedDat, rotate, rotate_err, config, nSamples4rot, y_cov_diag, minPairs
        )

    leaderTime = matchedDat.capture_time.sel(camera=config.leader).values
    order = np.argsort(leaderTime, kind="stable")
    n = len(matchedDat.pair_id)
    edges = np.linspace(0, n, nSegments + 1, dtype=int)

    segments = []
    lastRotate, lastRotateErr = rotate, rotate_err
    anyRefit = False
    for i in range(nSegments):
        idx = order[edges[i] : edges[i + 1]]
        if len(idx) == 0:
            continue
        segDat = matchedDat.isel(pair_id=idx)
        healed = _refitRotationWindow(
            segDat,
            rotate,
            rotate_err,
            config,
            min(nSamples4rot, len(idx)),
            y_cov_diag,
            minPairs,
        )
        if healed is not None:
            segDat, lastRotate, lastRotateErr = healed
            anyRefit = True
        segments.append(segDat)

    if not anyRefit:
        return None

    newMatchedDat = xr.concat(segments, dim="pair_id").sortby("pair_id")
    if not np.all(np.isfinite(newMatchedDat.matchScore.values)):
        return None

    log.warning(
        tools.concat(
            "refitRotationFromMatches: segmented refit",
            f"nSegments={nSegments}",
            "old median matchScore",
            float(matchedDat.matchScore.median()),
            "new median matchScore",
            float(newMatchedDat.matchScore.median()),
        )
    )
    return newMatchedDat, lastRotate, lastRotateErr


def probability(x, mu, sigma, delta):
    """
    Calculate probability using normal distribution.

    Parameters
    ----------
    x : array-like
        Values to calculate probability for
    mu : float
        Mean of the distribution
    sigma : float
        Standard deviation of the distribution
    delta : float
        Width of integration interval

    Returns
    -------
    array-like
        Probability values
    """
    from scipy.special import ndtr

    x = x.astype(float)
    mu = float(mu)
    sigma = float(sigma)
    delta = float(delta)

    # standardized bounds of the integration interval; ndtr(z) is the
    # standard normal CDF. Equivalent to but much faster than
    # scipy.stats.norm.cdf(), which pays for generic rv_continuous
    # argument checking/masking machinery on every call.
    x1 = (x - (delta / 2) - mu) / sigma
    x2 = (x + (delta / 2) - mu) / sigma

    # integrated over delta x region
    return ndtr(x2) - ndtr(x1)


def step(x, mu, sigma):
    """
    Step function for probability calculation.

    Parameters
    ----------
    x : array-like
        Values to calculate probability for
    mu : float
        Mean value for comparison
    sigma : float
        Threshold for comparison

    Returns
    -------
    array-like
        Binary values (0 or 1) based on comparison
    """
    x = x.astype(float)
    mu = float(mu)
    sigma = float(sigma)

    # normalize with mean value
    x = x - mu
    # step function
    prob = np.abs(x) < sigma
    return prob.astype(int)


def removeDoubleCounts(mPart, mProp, doubleCounts):
    """
    Remove duplicate particle matches.

    Parameters
    ----------
    mPart : numpy.ndarray
        Particle match indices, shape (n, maxMatches)
    mProp : numpy.ndarray
        Match probabilities, shape (n, maxMatches)
    doubleCounts : array-like
        Indices of particles that appear multiple times

    Returns
    -------
    tuple
        Updated mPart and mProp arrays (mutated in place) with duplicates
        removed
    """
    for doubleCount in doubleCounts:
        ii = np.where(mPart[:, 0] == doubleCount)[0]
        bestProp = mProp[ii, 0].argmax()
        #         print(doubleCount, ii, bestProp)
        for jj, i1 in enumerate(ii):
            if jj == bestProp:
                continue
            mPart[i1, :-1] = mPart[i1, 1:]
            mProp[i1, :-1] = mProp[i1, 1:]
            mPart[i1, -1] = np.nan
            mProp[i1, -1] = np.nan

    return mPart, mProp


def doMatch(
    leader1D,
    follower1D,
    sigmaIn,
    mu,
    delta,
    config,
    rotate,
    ptpTime,
    minProp=1e-10,
    minNumber4Stats=10,
    maxMatches=100,
    indexOffset=0,
    testing=False,
):
    """
    Match particles between leader and follower cameras.

    Parameters
    ----------
    leader1D : xarray Dataset
        Leader camera particle data
    follower1D : xarray Dataset
        Follower camera particle data
    sigmaIn : dict or str
        Sigma values for matching criteria or 'default'
    mu : dict
        Mean values for matching criteria
    delta : dict
        Delta values for matching criteria
    config : object
        Configuration object with camera settings
    rotate : dict
        Rotation parameters
    ptpTime : bool
        Whether to use PTP time synchronization
    minProp : float, optional
        Minimum required probability (default is 1e-10)
    minNumber4Stats : int, optional
        Minimum number of samples for statistics (default is 10)
    maxMatches : int, optional
        Maximum number of matches to consider (default is 100)
    indexOffset : int, optional
        Offset for pairing indices (default is 0)
    testing : bool, optional
        Whether to generate test plots (default is False)

    Returns
    -------
    tuple
        (matchedDat, disputedPairs, new_sigma, new_mu) - Matched data, disputed pairs, updated sigma and mu values
    """
    import bottleneck as bn
    import pandas as pd

    # print("using", sigma, mu, delta)
    # print("doMatch", len(leader1D.fpid), len(follower1D.fpid))
    prop = {}

    if sigmaIn == "default":
        if ptpTime:
            sigma = {
                "Z": 1.7,  # estimated from OE results
                "H": 1.2,  # estimated from OE results
                "T": 1e-4,  # pratical experience
            }
        else:
            sigma = {
                "Z": 1.7,  # estimated from OE results
                "H": 1.2,  # estimated from OE results
                "I": 0.01,
            }
    else:
        sigma = sigmaIn

    log.info(f"match with rotate={str(rotate)}")
    # particle Z position difference in joint coordinate system
    if "Z" in sigma.keys():
        L_x, L_z, F_y, F_z = get3DPosition(leader1D, follower1D, config)
        F_z = F_z.T
        F_y = F_y.T

        F_z = F_z.reshape((1, len(F_z)))
        L_z = L_z.reshape((len(L_z), 1))
        F_y = F_y.reshape((1, len(F_y)))
        L_x = L_x.reshape((len(L_x), 1))

        L_z_estimated = calc_L_z_withOffsets(L_x, F_y, F_z, **rotate)
        # Fz_estimated = calc_Fz(rotate["phi"], rotate["theta"], rotate["Ofz"], Lx, Lz, Fy)

        diffZ = L_z - L_z_estimated
        if testing:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.title("diffZ")
            plt.imshow(diffZ, vmin=-20, vmax=20, cmap="bwr")
            plt.colorbar()
            plt.xticks(follower1D.pid.values)
            plt.yticks(leader1D.pid.values)
            plt.xlabel("follower")
            plt.ylabel("leader")

        prop["Z"] = probability(diffZ, mu["Z"], sigma["Z"], delta["Z"])
    else:
        prop["Z"] = 1.0

    # particle camera Y position difference
    if "Y" in sigma.keys():
        fyCenter = follower1D.position_upperLeft.sel(dim2D="y") + (
            follower1D.Droi.sel(dim2D="y") / 2
        )
        lyCenter = leader1D.position_upperLeft.sel(dim2D="y") + (
            leader1D.Droi.sel(dim2D="y") / 2
        )
        diffY = np.array([fyCenter.values]) - np.array([lyCenter.values]).T
        prop["Y"] = probability(diffY, mu["Y"], sigma["Y"], delta["Y"])
    else:
        prop["Y"] = 1.0

    # particle height difference
    if "H" in sigma.keys():
        diffH = (
            np.array([follower1D.Droi.sel(dim2D="y").values])
            - np.array([leader1D.Droi.sel(dim2D="y").values]).T
        )
        prop["H"] = probability(diffH, mu["H"], sigma["H"], delta["H"])
    else:
        prop["H"] = 1.0

    # capture_time difference
    if "T" in sigma.keys():
        diffT = (
            np.array([follower1D.capture_time.values])
            - np.array([leader1D.capture_time.values]).T
        ).astype(int) * 1e-9
        # use step instead of normal distribution
        prop["T"] = step(diffT, mu["T"], sigma["T"])
    else:
        prop["T"] = 1.0

    # capture_id difference
    if "I" in sigma.keys():
        diffI = (
            np.array([follower1D.capture_id.values])
            - np.array([leader1D.capture_id.values]).T
        )
        prop["I"] = probability(diffI, mu["I"], sigma["I"], delta["I"])
    else:
        prop["I"] = 1.0

    # estimate joint probability
    propJoint = prop["Y"] * prop["T"] * prop["H"] * prop["I"] * prop["Z"]

    if testing:
        for k in prop.keys():
            if type(prop[k]) is not float:
                plt.figure()
                plt.title(k)
                plt.imshow(prop[k])
                plt.xticks(leader1D.pid.values)
                plt.yticks(follower1D.pid.values)
                plt.xlabel("follower")
                plt.ylabel("leader")
        plt.figure()
        plt.title("joined")
        plt.imshow(propJoint)
        plt.xticks(leader1D.pid.values)
        plt.yticks(follower1D.pid.values)
        plt.xlabel("follower")
        plt.ylabel("leader")

    # matchedParticles/matchedProbabilities/matchedOwnIdx are kept as
    # plain numpy arrays throughout the matching/dedup logic below (only
    # wrapped into xarray once, at the very end, to build the output
    # Dataset). xarray's per-element label-based indexing overhead was
    # showing up all over this hot loop even outside removeDoubleCounts
    # -- every argsort/where/dropna call on an xr.DataArray pays for
    # dimension/coordinate bookkeeping that's pure waste here, since the
    # "fpidII" coordinate never carries anything but its own position
    # (range(nOwn)), which matchedOwnIdx tracks directly instead.
    matchedParticles = {}
    matchedProbabilities = {}
    matchedOwnIdx = {}

    # try to solve this from both perspectives
    for camera, prop1, dat2 in zip(
        [config["leader"], config["follower"]],
        [propJoint, propJoint.T],
        [leader1D, follower1D],
    ):
        nCandidates = prop1.shape[1]
        if nCandidates > maxMatches:
            # Partial sort: only the top `maxMatches` candidates per row
            # are needed, not a full row sort. argpartition is O(n)
            # instead of argsort's O(n log n), which matters a lot when
            # maxMatches is much smaller than the number of candidates
            # (busy follower time windows). Ties are resolved arbitrarily
            # among equally-probable candidates, same as the previous
            # (also non-stable) full argsort; any candidate that matters
            # is far above the minProp cutoff applied further down, so
            # this can't change which matches are kept.
            topIdx = np.argpartition(prop1, -maxMatches, axis=1)[:, -maxMatches:]
            topProp = np.take_along_axis(prop1, topIdx, axis=1)
            order = np.argsort(topProp, axis=1)[:, ::-1]
            matchedParticles[camera] = np.take_along_axis(topIdx, order, axis=1)
            matchedProbabilities[camera] = np.take_along_axis(topProp, order, axis=1)
        else:
            order = np.argsort(prop1, axis=1)[:, ::-1]
            matchedParticles[camera] = order
            matchedProbabilities[camera] = np.take_along_axis(prop1, order, axis=1)

        matchedOwnIdx[camera] = np.arange(len(dat2.fpid))

    del propJoint, prop

    for reverseFactor in [1, -1]:
        cam1, cam2 = [config["leader"], config["follower"]][::reverseFactor]

        belowMinProp = matchedProbabilities[cam1] <= minProp
        mPart = np.where(belowMinProp, np.nan, matchedParticles[cam1])
        mProp = np.where(belowMinProp, np.nan, matchedProbabilities[cam1])

        for kk in range(maxMatches):
            u, c = np.unique(mPart[:, 0], return_counts=True)
            doubleCounts = u[np.where(c > 1)[0]]
            doubleCounts = doubleCounts[np.isfinite(doubleCounts)]
            if len(doubleCounts) != 0:
                # print(
                # cam1, "particles have been matched twice, fixing", kk)
                mPart, mProp = removeDoubleCounts(mPart, mProp, doubleCounts)
            else:
                break

        u, c = np.unique(mPart[:, 0], return_counts=True)
        doubleCounts = u[np.where(c > 1)[0]]
        doubleCounts = doubleCounts[np.isfinite(doubleCounts)]

        assert len(doubleCounts) == 0, (
            "%s particles have still been matched twice" % cam1
        )

        matchedParticles[cam1] = mPart
        matchedProbabilities[cam1] = mProp

    for reverseFactor in [1, -1]:
        cam1, cam2 = [config["leader"], config["follower"]][::reverseFactor]
        matchedParticles[cam1] = matchedParticles[cam1][:, 0]
        matchedProbabilities[cam1] = matchedProbabilities[cam1][:, 0]

        # equivalent of the previous .dropna("fpidII"): matchedParticles
        # and matchedProbabilities are NaN at exactly the same positions
        # by construction (set together above and in removeDoubleCounts),
        # so one mask suffices for both, and for matchedOwnIdx.
        valid = np.isfinite(matchedParticles[cam1])
        matchedParticles[cam1] = matchedParticles[cam1][valid]
        matchedProbabilities[cam1] = matchedProbabilities[cam1][valid]
        matchedOwnIdx[cam1] = matchedOwnIdx[cam1][valid]

    if np.all([len(v) == 0 for v in matchedParticles.values()]):
        noMatches = True
        log.warning(tools.concat("no matched particles"))
        nMatched = 0
        return None, len(leader1D.fpid), len(follower1D.fpid), nMatched

    cam1, cam2 = [config["leader"], config["follower"]]

    pairs1 = set(
        zip(
            matchedOwnIdx[cam1],
            matchedParticles[cam1].astype(int),
        )
    )
    pairs2 = set(
        zip(
            matchedParticles[cam2].astype(int),
            matchedOwnIdx[cam2],
        )
    )

    disputedPairs = pairs1 - pairs2

    # sort pairs together
    dats = []
    dats.append(leader1D.isel(fpid=matchedOwnIdx[config["leader"]].astype(int)))
    dats.append(follower1D.isel(fpid=matchedParticles[config["leader"]].astype(int)))

    for dd, d1 in enumerate(dats):
        pid = deepcopy(d1.pid.values)
        file_starttime = deepcopy(d1.file_starttime.values)
        d1 = d1.rename(fpid="pair_id")
        d1 = d1.assign_coords(
            pair_id=np.arange(len(matchedOwnIdx[config["leader"]])) + indexOffset
        )

        d1["pid"] = xr.DataArray(pid, coords=[d1.pair_id])
        d1["file_starttime"] = xr.DataArray(file_starttime, coords=[d1.pair_id])
        dats[dd] = d1

    matchedDat = xr.concat(dats, dim="camera")
    matchedDat = matchedDat.assign_coords(camera=[config["leader"], config["follower"]])
    # add propabilities
    matchedDat["matchScore"] = xr.DataArray(
        matchedProbabilities[config["leader"]].astype(np.float32),
        coords=[matchedDat.pair_id],
    )

    # clean up
    del (
        dats,
        matchedParticles,
        matchedProbabilities,
        matchedOwnIdx,
        leader1D,
        follower1D,
        pairs1,
        pairs2,
    )

    # estimate new offsets, potentially for the next file

    new_mu = {}
    new_sigma = {}

    if len(matchedDat.pair_id) >= minNumber4Stats:
        yCenter = matchedDat.position_upperLeft.sel(dim2D="y") + (
            matchedDat.Droi.sel(dim2D="y") / 2
        )
        di = yCenter.diff("camera").values
        new_sigma["Y"] = bn.nanstd(di)
        new_mu["Y"] = bn.nanmedian(di)

        di = matchedDat.Droi.sel(dim2D="y").diff("camera").values
        new_sigma["H"] = bn.nanstd(di)
        new_mu["H"] = bn.nanmedian(di)

        di = matchedDat.capture_time.diff("camera").values
        di = di[np.isfinite(di)].astype(int) * 1e-9
        new_sigma["T"] = bn.nanstd(di)
        new_mu["T"] = bn.nanmedian(di)

        di = matchedDat.capture_id.diff("camera").values
        new_sigma["I"] = bn.nanstd(di)
        new_mu["I"] = bn.nanmedian(di)

        # print(f"{len(matchedDat.pair_id)} matches found. ")
        # print(" match coefficients, ",new_mu)
    else:
        log.warning(
            tools.concat(
                f"{len(matchedDat.pair_id)} matches found. Setting match coefficients to NAN"
            )
        )
        new_sigma["Y"] = new_mu["Y"] = new_sigma["H"] = new_mu["H"] = np.nan
        new_sigma["T"] = new_mu["T"] = new_sigma["T"] = new_mu["T"] = np.nan

    new_sigma = pd.Series(new_sigma)
    new_mu = pd.Series(new_mu)

    return matchedDat, disputedPairs, new_sigma, new_mu


def get3DPosition(leaderDat, followerDat, config):
    """
    Get 3D positions from leader and follower data.

    Parameters
    ----------
    leaderDat : xarray Dataset
        Leader camera particle data
    followerDat : xarray Dataset
        Follower camera particle data
    config : object
        Configuration object with camera settings

    Returns
    -------
    tuple
        (L_x, L_z, F_y, F_z) - Position coordinates
    """
    F_z = (
        followerDat.position_upperLeft.sel(dim2D="y")
        + (followerDat.Droi.sel(dim2D="y") / 2)
    ).values
    F_y = (
        followerDat.position_upperLeft.sel(dim2D="x")
        + (followerDat.Droi.sel(dim2D="x") / 2)
    ).values
    L_x = (
        leaderDat.position_upperLeft.sel(dim2D="x")
        + (leaderDat.Droi.sel(dim2D="x") / 2)
    ).values
    L_z = (
        leaderDat.position_upperLeft.sel(dim2D="y")
        + (leaderDat.Droi.sel(dim2D="y") / 2)
    ).values

    # watch out, right hand coordinate system!
    F_y = config.frame_width - F_y

    return L_x, L_z, F_y, F_z


def get3DCentroid(leaderDat, followerDat, config):
    """
    Get 3D centroids from leader and follower data.

    Parameters
    ----------
    leaderDat : xarray Dataset
        Leader camera particle data
    followerDat : xarray Dataset
        Follower camera particle data
    config : object
        Configuration object with camera settings

    Returns
    -------
    tuple
        (L_x, L_z, F_y, F_z) - Centroid coordinates
    """
    F_z = followerDat.position_centroid.sel(dim2D="y").values
    F_y = followerDat.position_centroid.sel(dim2D="x").values
    L_x = leaderDat.position_centroid.sel(dim2D="x").values
    L_z = leaderDat.position_centroid.sel(dim2D="y").values

    # watch out, right hand coordinate system!
    F_y = config.frame_width - F_y

    return L_x, L_z, F_y, F_z


def addPosition(matchedDat, rotate, rotate_err, config):
    """
    Add position variable to match dataset based on retrieved rotation parameters.

    Parameters
    ----------
    matchedDat : xarray Dataset
        Matched particle data
    rotate : dict
        Rotation parameters
    rotate_err : dict
        Rotation parameter errors
    config : object
        Configuration object with camera settings

    Returns
    -------
    xarray Dataset
        Updated dataset with position information added
    """
    matchedDat["dim3D"] = ["x", "y", "z", "z_rotated"]

    L_x, L_z, F_y, F_z = get3DPosition(
        matchedDat.sel(camera=config.leader),
        matchedDat.sel(camera=config.follower),
        config,
    )
    # Fz_estimated = calc_Fz(rotate["phi"], rotate["theta"], rotate["Ofz"], Lx, Lz, Fy)
    L_z_estimated = calc_L_z_withOffsets(L_x, F_y, F_z, **rotate)
    matchedDat["position3D_center"] = xr.DataArray(
        [L_x, F_y, L_z, L_z_estimated], coords=[matchedDat.dim3D, matchedDat.pair_id]
    )

    L_x, L_z, F_y, F_z = get3DCentroid(
        matchedDat.sel(camera=config.leader),
        matchedDat.sel(camera=config.follower),
        config,
    )
    # Fz_estimated = calc_Fz(rotate["phi"], rotate["theta"], rotate["Ofz"], Lx, Lz, Fy)
    L_z_estimated = calc_L_z_withOffsets(L_x, F_y, F_z, **rotate)
    matchedDat["position3D_centroid"] = xr.DataArray(
        [L_x, F_y, L_z, L_z_estimated], coords=[matchedDat.dim3D, matchedDat.pair_id]
    )

    nid = len(matchedDat.pair_id)
    matchedDat["camera_rotation"] = np.array(["mean", "err"])
    for k in rotate.keys():
        matchedDat[k] = xr.DataArray(
            np.ones((nid, 2)) * np.array([rotate[k], rotate_err[k]]),
            coords=[matchedDat.pair_id, matchedDat["camera_rotation"]],
        )

    return matchedDat


def doMatchSlicer(
    leader1D,
    follower1D,
    sigma,
    mu,
    delta,
    config,
    rotate,
    ptpTime,
    minProp=1e-10,
    maxMatches=100,
    minNumber4Stats=10,
    chunckSize=700,
    testing=False,
):
    """
    Do matching with slicing to handle memory constraints.

    Parameters
    ----------
    leader1D : xarray Dataset
        Leader camera particle data
    follower1D : xarray Dataset
        Follower camera particle data
    sigma : dict
        Sigma values for matching criteria
    mu : dict
        Mean values for matching criteria
    delta : dict
        Delta values for matching criteria
    config : object
        Configuration object with camera settings
    rotate : dict
        Rotation parameters
    ptpTime : bool
        Whether to use PTP time synchronization
    minProp : float, optional
        Minimum required probability (default is 1e-10)
    maxMatches : int, optional
        Maximum number of matches to consider (default is 100)
    minNumber4Stats : int, optional
        Minimum number of samples for statistics (default is 10)
    chunckSize : int, optional
        Size of data chunks (default is 700)
    testing : bool, optional
        Whether to generate test plots (default is False)

    Returns
    -------
    tuple
        (matchedDat, disputedPairs, new_sigma, new_mu) - Matched data, disputed pairs, updated sigma and mu values
    """
    import pandas as pd
    from tqdm import tqdm

    # short cut for small data sets
    if (len(leader1D.fpid) < chunckSize) or (len(follower1D.fpid) < chunckSize):
        if testing:
            follower1D = tools.cutFollowerToLeader(
                leader1D, follower1D, gracePeriod=0.01
            )

        return doMatch(
            leader1D,
            follower1D,
            sigma,
            mu,
            delta,
            config,
            rotate,
            ptpTime,
            minProp=minProp,
            maxMatches=maxMatches,
            minNumber4Stats=minNumber4Stats,
            testing=testing,
        )

    # ok it is too long...
    matchedDat = []
    new_sigma = []
    new_mu = []
    disputedPairs = []

    indexOffset = 0

    JJs = np.linspace(
        0, len(leader1D.fpid), len(leader1D.fpid) // chunckSize + 1, dtype=int
    )

    log.info(tools.concat(f"slicing data into {len(JJs)-1} pieces"))
    for ii, jj in tqdm(zip(JJs[:-1], JJs[1:]), total=len(JJs) - 1, file=sys.stdout):
        leader1DSlice = leader1D.isel(fpid=slice(ii, jj))
        follower1DSlice = tools.cutFollowerToLeader(
            leader1DSlice, follower1D, gracePeriod=0.01
        )

        if len(follower1DSlice.fpid) == 0:
            res = None
            log.warning(tools.concat("No follower data remains"))
            continue

        res = doMatch(
            leader1DSlice,
            follower1DSlice,
            sigma,
            mu,
            delta,
            config,
            rotate,
            ptpTime,
            minProp=minProp,
            maxMatches=maxMatches,
            minNumber4Stats=minNumber4Stats,
            indexOffset=indexOffset,
        )
        matchedDat1, disputedPairs1, new_sigma1, new_mu1 = res

        if matchedDat1 is not None:
            matchedDat.append(matchedDat1)
            indexOffset = matchedDat1.pair_id[-1].values + 1
            disputedPairs += list(disputedPairs1)
            new_sigma.append(new_sigma1)
            new_mu.append(new_mu1)

    if len(matchedDat) > 0:
        new_sigma = pd.concat(new_sigma, axis=1).mean(axis=1)
        new_mu = pd.concat(new_mu, axis=1).mean(axis=1)
        matchedDat = xr.concat(matchedDat, dim="pair_id")
        return matchedDat, disputedPairs, new_sigma, new_mu
    else:
        log.warning(tools.concat("doMatchSlicer: nothing matched"))
        nMatched = 0
        return None, len(leader1D.fpid), len(follower1D.fpid), nMatched


class _MatchEarlyReturn(Exception):
    """Internal control-flow signal: matchParticles should return `result`
    immediately. Used for the offsetsOnly short-circuit inside
    _matchSegments, which historically returned a different tuple shape
    than the rest of the function."""

    def __init__(self, result):
        self.result = result


def _sliceFollowerSegment(
    FR1, FR2, follower1DAll, leaderMinTime, leaderMaxTime, tt, nSegments
):
    """
    Select the follower1DAll subset covering one follower-restart-to-
    restart segment [FR1, FR2] and decide whether it overlaps the leader's
    time range enough to bother with. Always logs its own outcome.

    Returns
    -------
    xarray.Dataset or None
        The follower1D slice, or None if this segment should be skipped
        entirely (too early, too late, too short, or too little
        overlapping follower data).
    """
    log.info(
        tools.concat(tt + 1, "of", nSegments, "slice for follower restart", FR1, FR2)
    )

    if (FR1 < leaderMinTime) and (FR2 < leaderMinTime):
        log.info(
            tools.concat(
                "CONTINUE, slice for follower restart",
                tt,
                FR1,
                FR2,
                "before leader time range",
                leaderMinTime.values,
            )
        )
        return None
    if (FR1 > leaderMaxTime) and (FR2 > leaderMaxTime):
        log.info(
            tools.concat(
                "CONTINUE, slice for follower restart",
                tt,
                FR1,
                FR2,
                "after leader time range",
                leaderMaxTime.values,
            )
        )
        return None
    if (FR2 - FR1) < np.timedelta64(1, "s"):
        log.info(
            tools.concat(
                "CONTINUE, slice for follower restart",
                tt,
                FR1,
                FR2,
                "less than one second",
                (FR2 - FR1) / 1e9,
            )
        )
        return None

    # the 2nd <= is on purpose because it is required if there is no restart. if there is a restart, there is anyway no data exactly at that time
    TIMES = (FR1 <= follower1DAll.capture_time.values) & (
        follower1DAll.capture_time.values <= FR2
    )
    if np.sum(TIMES) <= 3:
        log.warning(
            f"CONTINUE, too little follower data (#{np.sum(TIMES)}) overlapping with leader period"
        )
        return None

    return follower1DAll.isel(fpid=TIMES)


def _applyCaptureTimeEvenFix(follower1D, config, rotationOnly):
    """
    Apply the optional per-config makeCaptureTimeEven fix to a follower
    segment.

    Returns
    -------
    tuple(xarray.Dataset, str or None, bool)
        (follower1D, errorMessage, skip). If skip is True, the caller
        should move on to the next segment without using follower1D or
        errorMessage. errorMessage, when not None, is a soft failure the
        caller should record for this segment (processing continues with
        the *original*, unfixed follower1D) rather than a reason to skip.
    """
    if "makeCaptureTimeEven" not in config.dataFixes:
        return follower1D, None, False

    try:
        return fixes.makeCaptureTimeEven(follower1D, config, dim="fpid"), None, False
    except AssertionError as e:
        log.error("fixes.makeCaptureTimeEven FAILED")
        log.error(str(e))
        nTimes = len(follower1D.fpid)
        if rotationOnly:
            return follower1D, None, True
        if nTimes <= 20:
            log.error(tools.concat(f"so little data {nTimes} ignore it!"))
            return follower1D, None, True
        return follower1D, f"fixes.makeCaptureTimeEven FAILED {str(e)}", False


def _followerCameraWasReset(follower1D):
    """True if the follower's capture_id goes backward within this
    segment -- the camera physically reset mid-recording, and there is
    nothing to salvage here."""
    return not np.all(np.diff(follower1D.capture_id) >= 0)


def _resolveMatchingOffset(
    leader1D,
    follower1D,
    lEvents,
    fEvents,
    config,
    offsetsOnly,
    rotationOnly,
    maxDiffMs,
    nPoints,
    fname1Match,
    tt,
    FR1,
    FR2,
    errorStrs,
    errors,
):
    """
    Determine how to time-align leader and follower for this segment:
    either directly from synchronized PTP timestamps, or by estimating
    the leader-follower capture_id offset (trying both capture_time- and
    record_time-based estimates and taking whichever matched more).

    Mutates errorStrs[-1] and errors in place on soft failures (unless
    rotationOnly, matching the historical behavior of not bothering to
    record errors when only a rotation estimate was requested).

    Returns
    -------
    tuple(dict, dict, bool, float) or None
        (mu, delta, ptpTime, maxDiffMs) for doMatchSlicer, with maxDiffMs
        resolved from "config" to a number if it wasn't already (the
        caller should carry the returned value forward to the next
        segment, mirroring the original code resolving it once and
        reusing it for the rest of the day). None if this segment should
        be skipped entirely (already logged).

    Raises
    ------
    _MatchEarlyReturn
        If offsetsOnly is set and an offset was successfully found --
        matchParticles' historic contract for that mode.
    """
    ptpDisabled = (
        offsetsOnly
        or ("ptpStatus" not in lEvents.data_vars)
        or ("ptpStatus" not in fEvents.data_vars)
        or np.any(
            lEvents.ptpStatus.where(lEvents.event == "newfile", drop=True) == "Disabled"
        ).values
        or np.any(
            fEvents.ptpStatus.where(fEvents.event == "newfile", drop=True) == "Disabled"
        ).values
    )

    if not ptpDisabled:
        mu = {"Z": 0, "H": 0, "T": 0}
        delta = {"Z": 0.5, "Y": 0.5, "H": 1, "T": 1 / config.fps}
        return mu, delta, True, maxDiffMs

    if maxDiffMs == "config":
        maxDiffMs = 1000 / config.fps / 2

    try:
        captureIdOffset1, nMatched1 = tools.estimateCaptureIdDiffCore(
            leader1D,
            follower1D,
            "fpid",
            maxDiffMs=maxDiffMs,
            nPoints=nPoints,
            timeDim="capture_time",
        )
    except Exception as e:
        captureIdOffset1 = nMatched1 = -99
        error1 = str(e)
        # self-heal: the plain capture_time-based estimate failed (e.g.
        # "capture_id varies too much") -- if this deployment is known to
        # need it, retry once with both cameras' capture_time rebuilt
        # from capture_id, which removes independent per-camera clock
        # drift. Only ever runs on an already-failed estimate, so a file
        # that succeeds on the first try is completely unaffected.
        if "makeCaptureTimeEvenBothCameras" in config.dataFixes:
            try:
                leader1DEven, follower1DEven = fixes.makeCaptureTimeEvenBothCameras(
                    leader1D, follower1D, config
                )
                captureIdOffset1, nMatched1 = tools.estimateCaptureIdDiffCore(
                    leader1DEven,
                    follower1DEven,
                    "fpid",
                    maxDiffMs=maxDiffMs,
                    nPoints=nPoints,
                    timeDim="capture_time",
                )
                error1 = None
            except Exception as e2:
                captureIdOffset1 = nMatched1 = -99
                error1 = f"{error1}\r(makeCaptureTimeEvenBothCameras retry also failed) {e2}"
    try:
        captureIdOffset2, nMatched2 = tools.estimateCaptureIdDiffCore(
            leader1D,
            follower1D,
            "fpid",
            maxDiffMs=maxDiffMs,
            nPoints=nPoints,
            timeDim="record_time",
        )
    except Exception as e:
        captureIdOffset2 = nMatched2 = -99
        error2 = str(e)

    if nMatched2 == nMatched1 == -99:
        log.error(tools.concat("tools.estimateCaptureIdDiff FAILED"))
        log.error(tools.concat(error1))
        log.error(tools.concat(error2))
        if not rotationOnly:
            errorStrs[-1].append(
                f"tools.estimateCaptureIdDiff(ffl1, config, graceInterval=2)\r{error1}\r{error2}"
            )
        return None

    if (nMatched2 <= 1) and (nMatched1 <= 1):
        log.error(tools.concat("NOT ENOUGH DATA", fname1Match, tt, FR1, FR2))
        return None

    # In theory, capture time is much better, but there are cases were it is off. Try to identify them by chgecking whether record_time yielded more matches.
    # for mosaic, capture time is pretty much useless!
    if (nMatched2 > nMatched1) or (config.site == "mosaic"):
        if nMatched2 == -99:
            log.error(
                tools.concat(
                    "record_id based diff estiamtion failed", fname1Match, tt, FR1, FR2
                )
            )
            errors["offsetEstimation"] = True
            return None

        captureIdOffset = captureIdOffset2
        nMatched = nMatched2
        log.info(
            tools.concat(
                f"Taking offset from record_time {(captureIdOffset2, nMatched2)} intead of capture_time {(captureIdOffset1, nMatched1)}"
            )
        )
    else:
        captureIdOffset = captureIdOffset1
        nMatched = nMatched1

    if offsetsOnly:
        raise _MatchEarlyReturn((captureIdOffset, nMatched))

    mu = {"Z": 0, "H": 0, "T": 0, "I": captureIdOffset}
    delta = {"Z": 0.5, "Y": 0.5, "H": 1, "T": 1 / config.fps, "I": 1}
    return mu, delta, False, maxDiffMs


def _prepareDataForRotation(
    leader1D,
    follower1D,
    leader1D4rot,
    follower1D4rot,
    doRot,
    minDMax4rot,
    singleParticleFramesOnly,
    nSamples4rot,
    minSamples4rot,
):
    """
    Subset this segment's leader/follower data down to what's worth using
    for rotation refinement: filter to large, sharp (blur) particles if
    requested, optionally keep only frames with a single particle
    (removes matching ambiguity), and cap the sample size for speed.
    Turns doRot off if what's left is too little to bother refining
    rotation with at all.

    If doRot is already False on entry, leader1D4rot/follower1D4rot are
    returned unchanged -- mirroring the original inline code, which
    simply never touched them in that case, leaving whatever they held
    from the last segment that *did* run this (they are not reset to
    this segment's data just because doRot happens to be off here).

    Returns
    -------
    tuple(xarray.Dataset, xarray.Dataset, bool, bool)
        (leader1D4rot, follower1D4rot, dataTruncated4rot, doRot)
    """
    if not doRot:
        return leader1D4rot, follower1D4rot, False, doRot

    dataTruncated4rot = False
    minBlur4rot = 100

    if minDMax4rot > 0:
        filt = (leader1D.Dmax > minDMax4rot).values & (
            leader1D.blur > minBlur4rot
        ).values
        log.info(
            tools.concat(
                "DMax&blur filter leader:",
                minDMax4rot,
                np.sum(filt) / len(leader1D.fpid) * 100,
                "%",
            )
        )
        leader1D4rot = leader1D.isel(fpid=filt)
    else:
        leader1D4rot = leader1D.copy()

    if minDMax4rot > 0:
        filt = (follower1D.Dmax > minDMax4rot).values & (
            follower1D.blur > minBlur4rot
        ).values
        log.info(
            tools.concat(
                "DMax&blur filter follower:",
                minDMax4rot,
                np.sum(filt) / len(follower1D.fpid) * 100,
                "%",
            )
        )
        follower1D4rot = follower1D.isel(fpid=filt)
    else:
        follower1D4rot = follower1D.copy()

    # to get rotation coefficients, using frames with only a single particle is helpful!
    if singleParticleFramesOnly:
        un, ii, counts = np.unique(
            leader1D4rot.capture_time, return_index=True, return_counts=True
        )
        leader1D4rot = leader1D4rot.isel(fpid=ii[counts == 1])

        un, ii, counts = np.unique(
            follower1D4rot.capture_time, return_index=True, return_counts=True
        )
        follower1D4rot = follower1D4rot.isel(fpid=ii[counts == 1])

    if (
        len(leader1D4rot.fpid) > nSamples4rot * 10
    ):  # assuming we have about 10 times more particles outside the obs volume
        leader1D4rot = leader1D4rot.isel(fpid=slice(nSamples4rot * 10))
        dataTruncated4rot = True
    elif len(leader1D4rot.fpid) < minSamples4rot:
        log.error(
            "not enough leader data to estimate rotation %i" % len(leader1D4rot.fpid)
        )
        doRot = False

    if len(follower1D4rot.fpid) > nSamples4rot * 10:
        follower1D4rot = follower1D4rot.isel(fpid=slice(nSamples4rot * 10))
        dataTruncated4rot = True
    elif len(follower1D4rot.fpid) < minSamples4rot:
        log.error(
            "not enough follower data to estimate rotation %i"
            % len(follower1D4rot.fpid)
        )
        doRot = False

    return leader1D4rot, follower1D4rot, dataTruncated4rot, doRot


def _refineRotationIteration(
    leader1D4rot,
    follower1D4rot,
    sigma,
    mu,
    delta,
    config,
    rotate,
    rotate_err,
    ptpTime,
    testing,
    nSamples4rot,
    minSamples4rot,
    y_cov_diag,
    maxIter,
    errors,
    matchedDat,
    matchedDat4Rot,
    rotate_result,
    rotate_err_result,
):
    """
    Iteratively refine the camera rotation via optimal estimation (up to
    20 steps): match particles with the current rotation guess, refit
    the rotation from the best-matched pairs, and stop once the change
    is smaller than its own uncertainty (or data runs out, or the fit
    itself fails).

    matchedDat, matchedDat4Rot, rotate_result, and rotate_err_result are
    accepted as well as returned because -- matching the original inline
    loop exactly -- they are left *unchanged* if this loop breaks before
    ever landing a usable match, carrying over whatever value a previous
    segment last computed (for better or worse; not a decision this
    extraction should be changing).

    Mutates errors in place.

    Returns
    -------
    tuple
        (matchedDat, matchedDat4Rot, rotate_result, rotate_err_result)
    """
    rotates = []
    for ii in range(20):
        log.info(
            tools.concat(
                "rotation coefficients iteration",
                ii,
                "of 20 with",
                len(leader1D4rot.fpid),
                "and",
                len(follower1D4rot.fpid),
                "data points",
            )
        )
        # in here is all the magic
        res = doMatchSlicer(
            leader1D4rot,
            follower1D4rot,
            sigma,
            mu,
            delta,
            config,
            rotate,
            ptpTime,
            chunckSize=1e6,
            testing=testing,
        )
        if res[0] is None:
            log.error(
                "doMatchSlicer 4 rot failed %s"
                % str(leader1D4rot.capture_time.values[0])
            )
            if (len(leader1D4rot.fpid) > nSamples4rot) and (
                len(follower1D4rot.fpid) > nSamples4rot
            ):
                log.error(
                    f"reason for error unclear because number of samples is {len(leader1D4rot.fpid)} and {len(follower1D4rot.fpid)}"
                )
                errors["doMatchSlicer"] = True

            break
        matchedDat, disputedPairs, new_sigma, new_mu = res

        if len(matchedDat.pair_id) >= minSamples4rot:
            matchedDat4Rot = deepcopy(matchedDat)
            #                 matchedDat4Rot = matchedDat4Rot.isel(pair_id=(matchedDat4Rot.matchScore>minMatchScore4rot))
            matchedDat4Rot = matchedDat4Rot.isel(
                pair_id=sorted(np.argsort(matchedDat4Rot.matchScore)[-nSamples4rot:])
            )

            x_ap = rotate
            x_cov_diag = (rotate_err * 10) ** 2
            try:
                rotate_result, rotate_err_result, dgf_x = retrieveRotation(
                    matchedDat4Rot,
                    x_ap,
                    x_cov_diag,
                    y_cov_diag,
                    config,
                    verbose=True,
                    maxIter=maxIter,
                )
            except AssertionError as e:
                log.error(tools.concat(f"pyOE error, taking previous values."))
                log.error(tools.concat(str(e)))
                break

            log.debug(
                tools.concat(
                    "MATCH",
                    ii,
                    matchedDat.matchScore.mean().values,
                )
            )
            log.debug(
                tools.concat(
                    "ROTATE",
                    ii,
                    "\n",
                    rotate_result,
                    "\n",
                    "error",
                    "\n",
                    rotate_err_result,
                    "\n",
                    "dgf",
                    "\n",
                    dgf_x,
                )
            )
            rotates.append(rotate_result)

            if ii > 0:
                # if the change of the coefficients is smaller than their 1std errors for all of them, stop
                if np.all(np.abs(rotates[ii - 1] - rotate_result) < rotate_err_result):
                    log.info(tools.concat("interupting loop"))
                    log.info(tools.concat(rotate_result))
                    break
        else:
            log.warning(
                tools.concat(
                    f"{len(matchedDat.pair_id)} pairs is not enough data to estimate rotation, taking previous values."
                )
            )

            break

    return matchedDat, matchedDat4Rot, rotate_result, rotate_err_result


def _finalizeSegmentMatch(
    leader1D,
    follower1D,
    sigma,
    mu,
    delta,
    config,
    ptpTime,
    chunckSize,
    testing,
    dataTruncated4rot,
    doRot,
    rotate,
    rotate_err,
    rotate_result,
    rotate_err_result,
    rotate_final,
    rotate_err_final,
    matchedDat,
    errors,
):
    """
    Produce this segment's final matched-particle dataset and the
    rotation used to compute particle positions from it.

    If rotation refinement already ran against the segment's *full* data
    (not truncated for speed), its matchedDat is already final and is
    returned unchanged, along with rotate_final/rotate_err_final left
    untouched -- matching the original code, which in that case simply
    never updates them for this segment, leaving whatever a previous
    segment (or the caller) last set. Otherwise, re-run the match once
    more against the full segment data with whatever rotation ended up
    being used (refinement, if it ran at all, only ever saw a subset),
    and set rotate_final/rotate_err_final from that.

    Mutates errors in place.

    Returns
    -------
    tuple(xarray.Dataset or None, dict, dict, bool)
        (matchedDat, rotate_final, rotate_err_final, skip). skip is True
        if this segment produced nothing usable and the caller should
        move on without appending anything for it.
    """
    if not (dataTruncated4rot or (not doRot)):
        # matchDat is alread final because it was not truncated
        return matchedDat, rotate_final, rotate_err_final, False

    log.info(tools.concat("final doMatch"))

    if rotate_result is None:
        log.warning(f"falling back on default rotate {rotate}")
        rotate_final = rotate
        rotate_err_final = rotate_err
    else:
        rotate_final = rotate_result
        rotate_err_final = rotate_err_result

    # do it again because we did not consider everything before
    res = doMatchSlicer(
        leader1D,
        follower1D,
        sigma,
        mu,
        delta,
        config,
        rotate_final,
        ptpTime,
        chunckSize=chunckSize,
        testing=testing,
    )

    if res[0] is None:
        log.error(tools.concat("doMatchSlicer failed"))
        errors["doMatchSlicer"] = True
        return matchedDat, rotate_final, rotate_err_final, True

    matchedDat, disputedPairs, new_sigma, new_mu = res
    log.info(
        tools.concat(
            "doMatch ok, number of detections:",
            len(leader1D.fpid),
            len(follower1D.fpid),
            "number of matches:",
            len(matchedDat.pair_id),
        ),
    )
    return matchedDat, rotate_final, rotate_err_final, False


def _matchSegments(
    leader1D,
    follower1DAll,
    lEvents,
    fEvents,
    timeBlocks,
    leaderMinTime,
    leaderMaxTime,
    config,
    rotate,
    rotate_err,
    rotate_time,
    rotate_final,
    rotate_err_final,
    sigma,
    y_cov_diag,
    chunckSize,
    rotationOnly,
    nPoints,
    nSamples4rot,
    minSamples4rot,
    testing,
    minDMax4rot,
    singleParticleFramesOnly,
    doRot,
    offsetsOnly,
    maxIter,
    maxDiffMs,
    errors,
    fname1Match,
):
    """Match particles per follower-restart segment and refine the camera
    rotation via optimal estimation.

    Consumes already-opened leader/follower level1detect and metaEvents
    datasets and returns the matched pairs plus rotation estimate; does not
    touch the filesystem. `errors` is mutated in place. Raises
    _MatchEarlyReturn with (captureIdOffset, nMatched) if offsetsOnly is
    set, matching matchParticles' historic early-return contract.
    """
    matchedDat = None
    matchedDat4Rot = None
    matchedDats = []
    errorStrs = []
    nSamples = []
    rotate_result = None
    rotate_err_result = None
    # only required if it fails early
    leader1D4rot = leader1D
    follower1D4rot = follower1DAll

    nFollower = 0
    nLeader = 0

    lEvents.close()
    fEvents.close()

    # loop over all follower segments separated by camera restarts
    nSegments = len(timeBlocks) - 1
    for tt, (FR1, FR2) in enumerate(zip(timeBlocks[:-1], timeBlocks[1:])):
        follower1D = _sliceFollowerSegment(
            FR1, FR2, follower1DAll, leaderMinTime, leaderMaxTime, tt, nSegments
        )
        if follower1D is None:
            continue

        errorStrs.append([])
        nSamples.append(len(follower1D.fpid))

        follower1D, fixError, skip = _applyCaptureTimeEvenFix(
            follower1D, config, rotationOnly
        )
        if skip:
            continue
        if fixError:
            errorStrs[-1].append(fixError)

        if _followerCameraWasReset(follower1D):
            log.error(tools.concat("follower camera reset detected"))
            if not rotationOnly:
                errorStrs[-1].append("follower camera reset detected")
            continue

        # if (minDMax4rot > 0):
        #     filt = (leader1D.Dmax>minDMax4rot).values
        #     log.info(tools.concat("DMax capture id filter leader:", minDMax4rot, np.sum(filt)/len(leader1D.fpid) * 100,"%"))
        #     leader1D = leader1D.isel(fpid=filt)

        # if (minDMax4rot > 0):
        #     filt = (follower1D.Dmax>minDMax4rot).values
        #     log.info(tools.concat("DMax capture id filter follower:", minDMax4rot, np.sum(filt)/len(follower1D.fpid) * 100,"%"))
        #     follower1D = follower1D.isel(fpid=filt)

        offsetResult = _resolveMatchingOffset(
            leader1D,
            follower1D,
            lEvents,
            fEvents,
            config,
            offsetsOnly,
            rotationOnly,
            maxDiffMs,
            nPoints,
            fname1Match,
            tt,
            FR1,
            FR2,
            errorStrs,
            errors,
        )
        if offsetResult is None:
            continue
        mu, delta, ptpTime, maxDiffMs = offsetResult

        # figure out how cameras ae rotated, first prepare data
        (
            leader1D4rot,
            follower1D4rot,
            dataTruncated4rot,
            doRot,
        ) = _prepareDataForRotation(
            leader1D,
            follower1D,
            leader1D4rot,
            follower1D4rot,
            doRot,
            minDMax4rot,
            singleParticleFramesOnly,
            nSamples4rot,
            minSamples4rot,
        )
        # iterate to rotation coefficients in max. 20 steps
        if doRot:
            (
                matchedDat,
                matchedDat4Rot,
                rotate_result,
                rotate_err_result,
            ) = _refineRotationIteration(
                leader1D4rot,
                follower1D4rot,
                sigma,
                mu,
                delta,
                config,
                rotate,
                rotate_err,
                ptpTime,
                testing,
                nSamples4rot,
                minSamples4rot,
                y_cov_diag,
                maxIter,
                errors,
                matchedDat,
                matchedDat4Rot,
                rotate_result,
                rotate_err_result,
            )
        else:
            log.warning(
                tools.concat(f"taking provided data for rotation from {rotate_time}")
            )
            rotate_result = rotate
            rotate_err_result = rotate_err

        if rotationOnly:
            nLeader += len(leader1D4rot.fpid)
            nFollower += len(follower1D4rot.fpid)
            continue
            # return fname1Match, matchedDat4Rot, rotate, rotate_err

        nLeader += len(leader1D.fpid)
        nFollower += len(follower1D.fpid)

        matchedDat, rotate_final, rotate_err_final, skip = _finalizeSegmentMatch(
            leader1D,
            follower1D,
            sigma,
            mu,
            delta,
            config,
            ptpTime,
            chunckSize,
            testing,
            dataTruncated4rot,
            doRot,
            rotate,
            rotate_err,
            rotate_result,
            rotate_err_result,
            rotate_final,
            rotate_err_final,
            matchedDat,
            errors,
        )
        if skip:
            continue

        if (matchedDat is not None) and len(matchedDat.pair_id) > 0:
            # add position with final roation coeffs.
            matchedDat = addPosition(matchedDat, rotate_final, rotate_err_final, config)

            # fixed values would lead to confusion, so stay with original ones
            if "captureIdOverflows" in config.dataFixes:
                matchedDat = fixes.revertIdOverflowFix(matchedDat)

            matchedDats.append(matchedDat)

    # end loop camera restart FR

    return (
        matchedDats,
        matchedDat,
        errorStrs,
        nSamples,
        matchedDat4Rot,
        rotate_result,
        rotate_err_result,
        rotate_final,
        rotate_err_final,
        nLeader,
        nFollower,
    )


@log.catch(reraise=True)
def matchParticles(
    fnameLv1Detect,
    config,
    y_cov_diag=1.65**2,
    version=__version__,
    chunckSize=1000,
    rotate="config",
    rotate_err="config",
    maxDiffMs="config",
    rotationOnly=False,
    nPoints=500,
    sigma="default",
    nSamples4rot=300,
    minSamples4rot=100,
    testing=False,
    minDMax4rot=0,
    singleParticleFramesOnly=False,
    doRot=False,
    writeNc=True,
    offsetsOnly=False,
    subset=None,
    maxIter=30,
    skipExisting=True,
):
    """
    Match particles between leader and follower cameras.

    Parameters
    ----------
    fnameLv1Detect : str
        Path to level1 detect file
    config : object
        Configuration object with camera settings
    y_cov_diag : float, optional
        Observation covariance diagonal (default is 1.65**2)
    version : str, optional
        Version string (default is __version__)
    chunckSize : int, optional
        Size of data chunks (default is 1000)
    rotate : dict or str, optional
        Initial rotation parameters or "config" (default is "config")
    rotate_err : dict or str, optional
        Initial rotation parameter errors or "config" (default is "config")
    maxDiffMs : float or str, optional
        Maximum time difference in milliseconds or "config" (default is "config")
    rotationOnly : bool, optional
        Whether to only estimate rotation (default is False)
    nPoints : int, optional
        Number of points for estimation (default is 500)
    sigma : dict or str, optional
        Sigma values for matching criteria or "default" (default is "default")
    nSamples4rot : int, optional
        Number of samples for rotation estimation (default is 300)
    minSamples4rot : int, optional
        Minimum samples for rotation estimation (default is 100)
    testing : bool, optional
        Whether to generate test plots (default is False)
    minDMax4rot : float, optional
        Minimum DMax threshold for rotation estimation (default is 0)
    singleParticleFramesOnly : bool, optional
        Whether to use only single particle frames (default is False)
    doRot : bool, optional
        Whether to perform rotation estimation (default is False)
    writeNc : bool, optional
        Whether to write NetCDF files (default is True)
    offsetsOnly : bool, optional
        Whether to only estimate offsets (default is False)
    subset : tuple, optional
        Subset of particles to process (default is None)
    maxIter : int, optional
        Maximum iterations for optimization (default is 30)
    skipExisting : bool, optional
        Whether to skip existing files (default is True)

    Returns
    -------
    tuple
        (fname1Match, matchedDat, rotate_final, rotate_err_final, nLeader, nFollower, nMatched, errors) - Results of matching
    """
    import pandas as pd

    errors = pd.Series(
        {
            "openingData": False,
            "tooFewObs": False,
            "followerBlocked": False,
            "leaderBlocked": False,
            "offsetEstimation": False,
            "doMatchSlicer": False,
            "noMetaRot": False,
        }
    )

    if type(config) is str:
        config = tools.readSettings(config)

    ffl1 = files.FilenamesFromLevel(fnameLv1Detect, config)
    fname1Match = ffl1.fname["level1match"]
    fnames1F = ffl1.filenamesOtherCamera(graceInterval=-1, level="level1detect")
    fnames1FRAW = ffl1.filenamesOtherCamera(graceInterval=-1, level="level0txt")
    fnameMetaRotation = ffl1.fname["metaRotation"]

    rotate_time = None

    if not doRot:
        # check whether output exists
        if skipExisting and tools.checkForExisting(
            fname1Match,
            parents=[fnameLv1Detect, fnameMetaRotation] + fnames1F,
            breakpointLevel="level1match",
        ):
            return fname1Match, None, None, None, None, None, None, errors

        if ffl1.isBroken("metaRotation"):
            raise RuntimeError(f"{fnameMetaRotation}.broken.txt is broken")

        if ffl1.isNoData("metaRotation"):
            # metaRotation is a confirmed data gap (see
            # matching.createMetaRotation), not just "not created yet":
            # nothing will ever be available here, so propagate nodata
            # the same way as every other failure mode in this function
            if not rotationOnly:
                ffl1.propagateNoData("metaRotation", "level1match")
            log.error(f"metaRotation is nodata: {fnameMetaRotation}")
            errors["noMetaRot"] = True
            return fname1Match, None, None, None, None, None, None, errors

        # get rotation estimates and add to config instead of estimating them
        try:
            metaRotationDat = xr.open_dataset(fnameMetaRotation)
        except FileNotFoundError:
            log.error(f"did not find {fnameMetaRotation}")
            errors["noMetaRot"] = True

            return fname1Match, None, None, None, None, None, None, errors
        try:
            metaRotationDat = metaRotationDat.where(
                metaRotationDat.camera_Ofz.notnull(), drop=True
            )
        except ValueError as e:
            log.error(f"all camera_Ofz in {fnameMetaRotation} nan")
            error = str(e)
            log.error(error)
            metaRotationDat.close()
            if not rotationOnly:
                raise RuntimeError(error)
            errors["openingData"] = True
            return fname1Match, np.nan, None, None, None, None, None, errors

        config = tools.rotXr2dict(metaRotationDat, config)
        metaRotationDat.close()

    if np.any(rotate == "config"):
        rotate, rotate_err, rotate_time = tools.getPrevRotationEstimates(
            ffl1.datetime64, config
        )

    # in case everything else below fails
    rotate_final = rotate
    rotate_err_final = rotate_err

    log.info(
        f"opening {fnameLv1Detect} with rotation first guess {rotate} from {rotate_time}"
    )
    try:
        leader1D = tools.open_mflevel1detect(fnameLv1Detect, config)  # with fixes
    except AssertionError as e:
        log.error(tools.concat("tools.open_mflevel1detect leader FAILED"))
        error = str(e)
        log.error(tools.concat(error))

        if not rotationOnly:
            raise AssertionError(error)
        errors["openingData"] = True
        return fname1Match, np.nan, None, None, None, None, None, errors

    if leader1D is None:
        if not rotationOnly:
            ffl1.writeStatus(
                "level1match", "nodata", f"no leader data in {fnameLv1Detect}"
            )
        log.error(tools.concat(f"no leader data in {fnameLv1Detect}"))
        errors["tooFewObs"] = True
        return fname1Match, None, None, None, None, None, None, errors

    log.info(tools.concat(len(leader1D.pid)))

    if len(leader1D.pid) <= 1:
        if not rotationOnly:
            ffl1.writeStatus(
                "level1match", "nodata", f"only one particle in  {fnameLv1Detect}"
            )
        log.error(tools.concat(f"only one particle in {fnameLv1Detect}"))
        errors["tooFewObs"] = True
        return fname1Match, None, None, None, None, None, None, errors

    if subset is not None:
        leader1D = leader1D.isel(fpid=slice(*subset))

    file_starttime = leader1D.file_starttime[0].values

    if len(fnames1FRAW) != len(fnames1F):
        log.error(tools.concat(f"no follower data for {fnameLv1Detect} processed YET"))
        log.error(tools.concat(fnames1F))
        log.error(tools.concat(fnames1FRAW))
        errors["openingData"] = True
        return fname1Match, np.nan, None, None, None, None, None, errors
    if len(fnames1F) == 0:
        if not rotationOnly:
            ffl1.writeStatus(
                "level1match", "nodata", f"no follower data for {fnameLv1Detect}"
            )
        log.error(tools.concat(f"no follower data for {fnameLv1Detect}"))
        errors["openingData"] = True
        return fname1Match, None, None, None, None, None, None, errors

    fClass = [files.FilenamesFromLevel(f, config) for f in fnames1F]
    fCases = np.unique([f.case.split("-")[0] for f in fClass])
    # # just in case
    metadata.createEvent(
        ffl1.case, config.leader, config, quiet=True, skipExisting=True
    )
    for fCase in fCases:
        metadata.createEvent(
            fCase, config.follower, config, quiet=True, skipExisting=True
        )

    lEvents = ffl1.fname.metaEvents
    with xr.open_dataset(lEvents) as ds:
        lEvents = ds.load()

    fEvents = np.unique([f.fname.metaEvents for f in fClass])
    with xr.open_mfdataset(fEvents) as ds:
        fEvents = ds.load()

    start = leader1D.capture_time[0].values - np.timedelta64(2, "s")
    end = leader1D.capture_time[-1].values + np.timedelta64(2, "s")
    log.info(tools.concat(f"opening {fnames1F}"))
    try:
        follower1DAll = tools.open_mflevel1detect(
            fnames1F, config, start=start, end=end
        )  # with foxes
    except Exception as e:
        log.error(tools.concat("tools.open_mflevel1detect follower FAILED"))
        error = str(e)
        log.error(tools.concat(error))

        if not rotationOnly:
            raise RuntimeError(error)
        errors["openingData"] = True
        return fname1Match, np.nan, None, None, None, None, None, errors

    leader1D = tools.removeBlockedBlowingData(leader1D, lEvents, config)
    follower1DAll = tools.removeBlockedBlowingData(follower1DAll, fEvents, config)

    if follower1DAll is None:
        if not rotationOnly:
            ffl1.writeStatus(
                "level1match",
                "nodata",
                f"no follower data after removal of blocked data {fname1Match}",
            )
        log.error(f"no follower data after removal of blocked data {fname1Match}")
        errors["followerBlocked"] = True
        return fname1Match, None, None, None, None, None, None, errors

    if leader1D is None:
        if not rotationOnly:
            ffl1.writeStatus(
                "level1match",
                "nodata",
                f"no leader data after removal of blocked data {fname1Match}",
            )
        log.error(f"no leader data after removal of blocked data {fname1Match}")
        errors["leaderBlocked"] = True
        return fname1Match, None, None, None, None, None, None, errors

    if "ptpStatus" in lEvents.data_vars:
        lEventsInterpolated = lEvents.where(lEvents.event == "newfile", drop=True).sel(
            file_starttime=leader1D.capture_time, method="ffill"
        )
        if not np.all(np.isin(lEventsInterpolated.ptpStatus, ["Slave", "Disabled"])):
            brokenDat = lEventsInterpolated.ptpStatus.isel(
                fpid=~np.isin(lEventsInterpolated.ptpStatus, ["Slave", "Disabled"])
            )
            if not rotationOnly:
                ffl1.writeStatus(
                    "level1match",
                    "nodata",
                    f"Leader ptpStatus is not Slave: {brokenDat.values} at {brokenDat.file_starttime.values}",
                )
            log.error(
                f"Leader ptpStatus is not Slave: {brokenDat.values} at {brokenDat.file_starttime.values}"
            )
            errors["leaderPtpError"] = True
            return fname1Match, None, None, None, None, None, None, errors

    if "ptpStatus" in fEvents.data_vars:
        fEventsInterpolated = fEvents.where(fEvents.event == "newfile", drop=True).sel(
            file_starttime=follower1DAll.capture_time, method="ffill"
        )
        if not np.all(np.isin(fEventsInterpolated.ptpStatus, ["Slave", "Disabled"])):
            brokenDat = fEventsInterpolated.ptpStatus.isel(
                fpid=~np.isin(fEventsInterpolated.ptpStatus, ["Slave", "Disabled"])
            )
            if not rotationOnly:
                ffl1.writeStatus(
                    "level1match",
                    "nodata",
                    f"Follower ptpStatus is not Slave: {brokenDat.values} at {brokenDat.file_starttime.values}",
                )
            log.error(
                f"Follower ptpStatus is not Slave: {brokenDat.values} at {brokenDat.file_starttime.values}"
            )
            errors["leaderPtpError"] = True
            return fname1Match, None, None, None, None, None, None, errors

    # try to figure out when follower was restarted in leader time period
    followerRestartedII = np.where(
        [
            (str(e).startswith("start") or str(e).startswith("launch"))
            for e in fEvents.event.values
        ]
    )[0]
    followerRestarted = fEvents.file_starttime[followerRestartedII].values

    # Pre-PTP hardware could silently drop a captured frame's index,
    # which permanently offsets the leader-follower capture_id
    # correspondence from that point on within an otherwise unbroken
    # recording. _matchSegments estimates one offset per timeBlocks
    # segment and (correctly) refuses to guess when a window straddles
    # such a drop, since the true offset differs by a whole frame on
    # either side. Detect those drops here -- same condition _matchSegments
    # uses to decide it needs capture_id-based offset estimation at all
    # (ptpStatus missing/Disabled) -- and add them as extra segment
    # boundaries, exactly like a genuine follower restart, so each side
    # gets resolved independently instead of failing outright. PTP-active
    # data never takes this path, since ptpDisabled is False for it and
    # matching then relies on synchronized timestamps rather than
    # capture_id offsets in the first place.
    ptpDisabled = (
        ("ptpStatus" not in lEvents.data_vars)
        or ("ptpStatus" not in fEvents.data_vars)
        or np.any(
            lEvents.ptpStatus.where(lEvents.event == "newfile", drop=True) == "Disabled"
        ).values
        or np.any(
            fEvents.ptpStatus.where(fEvents.event == "newfile", drop=True) == "Disabled"
        ).values
    )
    captureIdDropTimes = []
    phaseJumpTimes = []
    if ptpDisabled:
        maxDiffMsForDropDetection = maxDiffMs
        if maxDiffMsForDropDetection == "config":
            maxDiffMsForDropDetection = 1000 / config.fps / 2
        # detectCaptureIdDropTimes runs on the whole (multi-segment) window
        # up front, before any per-segment capture_time_even reconstruction
        # has happened -- level1detect never stores capture_time_even on
        # disk, so without this, both leaderDat/followerDat only have raw,
        # drift-affected capture_time available, and each camera's own
        # onboard clock can drift enough within a ~10-minute window to make
        # this function's nearest-time matching latch onto the wrong frame
        # partway through, producing a spurious, sustained idDiff step that
        # is indistinguishable from a genuine dropped frame. Confirmed on
        # real data (hyytiala2_v3 20240216-125000): this falsely detected a
        # "drop" that split the window and matched the back ~130s with the
        # wrong offset, tanking matchScore there, even though a
        # constant-offset match (as an older code version without this
        # detection produced) scored well for the entire file. Proactively
        # reconstruct capture_time_even for both cameras first when this
        # deployment opts into the fix, exactly like the reactive fallback
        # `_resolveMatchingOffset` already uses per-segment, so the drop
        # detector isn't fooled by drift it doesn't otherwise see.
        leaderForDropDetection = leader1D
        followerForDropDetection = follower1DAll
        if "makeCaptureTimeEvenBothCameras" in config.dataFixes:
            try:
                leaderForDropDetection, followerForDropDetection = (
                    fixes.makeCaptureTimeEvenBothCameras(
                        leader1D, follower1DAll, config
                    )
                )
            except Exception as e:
                log.warning(
                    tools.concat(
                        "makeCaptureTimeEvenBothCameras FAILED ahead of "
                        "detectCaptureIdDropTimes, falling back to raw "
                        "capture_time",
                        str(e),
                    )
                )
        try:
            captureIdDropTimes = fixes.detectCaptureIdDropTimes(
                leaderForDropDetection,
                followerForDropDetection,
                dim="fpid",
                nPoints=nPoints,
                maxDiffMs=maxDiffMsForDropDetection,
                timeDim="capture_time",
            )
        except Exception as e:
            log.warning(tools.concat("detectCaptureIdDropTimes FAILED", str(e)))
        if len(captureIdDropTimes) > 0:
            log.warning(
                tools.concat(
                    "found likely dropped frame(s), splitting matching window at",
                    captureIdDropTimes,
                )
            )

        # detectCaptureIdDropTimes works in capture_id space and, since
        # this session's false-positive fix, deliberately prefers
        # capture_time_even -- which makes it structurally blind to a
        # brief raw-capture_time-only glitch on one camera (capture_id
        # numbering itself stays perfectly regular). Run the
        # complementary raw-time-vs-its-own-reconstruction check on the
        # untouched raw data for the same reason the drop detector needs
        # capture_time_even: two different failure modes need two
        # different signals. See detectPhaseJumpTimes's docstring for
        # the confirmed real-world case (hyytiala2_v3 20240213-215000)
        # this recovers.
        if "detectTimingGlitches" in config.dataFixes:
            try:
                phaseJumpTimes = fixes.detectPhaseJumpTimes(
                    leader1D, follower1DAll, config
                )
            except Exception as e:
                log.warning(tools.concat("detectPhaseJumpTimes FAILED", str(e)))
            if len(phaseJumpTimes) > 0:
                log.warning(
                    tools.concat(
                        "found likely timing glitch(es), splitting matching "
                        "window at",
                        phaseJumpTimes,
                    )
                )

    timeBlocks = np.concatenate(
        (
            follower1DAll.capture_time.values[:1],
            followerRestarted,
            np.array(captureIdDropTimes, dtype=follower1DAll.capture_time.values.dtype),
            np.array(phaseJumpTimes, dtype=follower1DAll.capture_time.values.dtype),
            follower1DAll.capture_time.values[-1:],
        )
    )
    timeBlocks = np.sort(timeBlocks)

    # the extra seconds makes it more robust with respect to time drifts
    leaderMinTime = leader1D.file_starttime.min() - np.timedelta64(1, "s")
    leaderMaxTime = max(
        leader1D.capture_time.max(), leader1D.record_time.max()
    ) + np.timedelta64(1, "s")

    leader1D.load()
    follower1DAll.load()
    leader1D.close()
    follower1DAll.close()

    lEvents.close()
    fEvents.close()

    try:
        (
            matchedDats,
            matchedDat,
            errorStrs,
            nSamples,
            matchedDat4Rot,
            rotate_result,
            rotate_err_result,
            rotate_final,
            rotate_err_final,
            nLeader,
            nFollower,
        ) = _matchSegments(
            leader1D,
            follower1DAll,
            lEvents,
            fEvents,
            timeBlocks,
            leaderMinTime,
            leaderMaxTime,
            config,
            rotate,
            rotate_err,
            rotate_time,
            rotate_final,
            rotate_err_final,
            sigma,
            y_cov_diag,
            chunckSize,
            rotationOnly,
            nPoints,
            nSamples4rot,
            minSamples4rot,
            testing,
            minDMax4rot,
            singleParticleFramesOnly,
            doRot,
            offsetsOnly,
            maxIter,
            maxDiffMs,
            errors,
            fname1Match,
        )
    except _MatchEarlyReturn as e:
        return e.result

    if rotationOnly:
        try:
            nMatched = len(matchedDat4Rot.pair_id)
        except AttributeError:  # i.e. matched is None
            nMatched = 0
        return (
            fname1Match,
            matchedDat4Rot,
            rotate_result,
            rotate_err_result,
            nLeader,
            nFollower,
            nMatched,
            errors,
        )

    # if an error occurred, figure out whether it affects a significant part of the data set
    # most errors are negligible because affecting only the period between
    # syncing both cameras affecting only few frames
    if len(nSamples) > 0:
        sumNsample = np.sum(nSamples)
        for nSample, error in zip(nSamples, errorStrs):
            if len(error) > 0:
                errRatio = nSample / sumNsample
                if errRatio > 0.1:
                    log.error(
                        f"error in {errRatio*100}%, i.e. more than 10% of the data"
                    )
                    for err in error:
                        log.error(err)
                    raise RuntimeError(tools.concat(error))
                if errRatio > 0.01:
                    log.warning(f"error in {errRatio*100}% of the data")

    if len(matchedDats) == 0:
        ffl1.writeStatus("level1match", "nodata", "no data")
        log.error(tools.concat("NO DATA", fname1Match))

        return (
            fname1Match,
            None,
            rotate_final,
            rotate_err_final,
            nLeader,
            nFollower,
            0,
            errors,
        )

    elif len(matchedDats) == 1:
        # easy case
        matchedDats = matchedDat
    else:
        for ii in range(len(matchedDats)):
            del matchedDats[ii]["pair_id"]
        matchedDats = xr.concat(matchedDats, dim="pair_id")
        matchedDats["pair_id"] = range(len(matchedDats["pair_id"]))

    nPairs = len(matchedDats["pair_id"])
    if nPairs > config.newFileInt:  # i.e at least one match per second
        matchScoreMedian = matchedDats.matchScore.median().values
        if matchScoreMedian < config.quality.minMatchScore:
            log.warning(
                tools.concat(
                    "median matchScore is only",
                    matchScoreMedian,
                    "smaller than minMatchScore",
                    config.quality.minMatchScore,
                    "even though we found",
                    nPairs,
                    "particles -- trying a "
                    "cheap rotation refit against the existing matches "
                    "before giving up",
                )
            )
            preRefitMatchedDats = matchedDats
            # A rotation refit can only correct a *biased* Z residual (shift
            # its mean) -- it cannot shrink an intrinsically *wide* one, so
            # spending a single-window AND a 15-segment OE refit on a file
            # whose pairs already disagree with each other more than
            # config.quality.maxZSigma is compute wasted on a fix that
            # cannot work by construction. This also produces a clearer
            # diagnostic (wide, likely correspondence ambiguity) than the
            # generic "matchScore too low" message below. See
            # zResidualSigma's docstring and config.quality.maxZSigma's
            # comment for the empirical basis; skipped for non-"default"
            # sigma, same restriction refitRotationFromMatches itself has.
            if sigma == "default":
                zSigmaCurrent = zResidualSigma(matchedDats, rotate_final, config)
                if zSigmaCurrent > config.quality.maxZSigma:
                    raise RuntimeError(
                        f"Z-residual spread is {zSigmaCurrent} which exceeds "
                        f"maxZSigma {config.quality.maxZSigma} -- matched pairs "
                        "disagree with each other more than a rotation refit "
                        "can fix (likely correspondence ambiguity, not a "
                        f"calibration drift); median matchScore is "
                        f"{matchScoreMedian}, nPairs={nPairs}"
                    )
            healed = refitRotationFromMatches(
                matchedDats,
                rotate_final,
                rotate_err_final,
                config,
                sigma=sigma,
                nSamples4rot=nSamples4rot,
                y_cov_diag=y_cov_diag,
            )
            if healed is not None:
                matchedDats, rotate_final, rotate_err_final = healed
                matchScoreMedian = matchedDats.matchScore.median().values
            if matchScoreMedian < config.quality.minMatchScore:
                # single global rotation still isn't enough -- for a
                # large file, that can mean the true rotation drifts
                # smoothly within the file itself (progressive snow
                # loading, thermal settling), which no single-rotation
                # refit can fit well. Escalate to a time-segmented refit
                # (independent rotation per time chunk) before giving
                # up; refit from the pre-refit pairs, since segmenting
                # is a different, not additive, strategy.
                log.warning(
                    tools.concat(
                        "single-window refit still below minMatchScore, "
                        "trying a time-segmented refit"
                    )
                )
                healedSeg = refitRotationFromMatches(
                    preRefitMatchedDats,
                    rotate_final,
                    rotate_err_final,
                    config,
                    sigma=sigma,
                    nSamples4rot=nSamples4rot,
                    y_cov_diag=y_cov_diag,
                    nSegments=15,
                )
                if healedSeg is not None:
                    matchedDats, rotate_final, rotate_err_final = healedSeg
                    matchScoreMedian = matchedDats.matchScore.median().values
            if matchScoreMedian < config.quality.minMatchScore:
                raise RuntimeError(
                    f"median matchScore is only {matchScoreMedian} and smaller than "
                    f"minMatchScore {config.quality.minMatchScore} even though we "
                    f"found {nPairs} particles"
                )

        # The gate above only ever looks at the file's AGGREGATE
        # matchScore/Z-residual, which a localized problem confined to a
        # few minutes of an otherwise-fine 10-minute file can hide from
        # entirely -- exactly the gap distributions.getZResidualQuality's
        # level2 zResidualTooWide flag exists to catch, but only after
        # the fact, with nothing ever attempted. Give a localized problem
        # the same refit chance an aggregate failure gets -- but unlike
        # an aggregate failure, never reject the whole file over it:
        # unlike an aggregate failure (which means this file would
        # otherwise be entirely discarded), an otherwise-passing file
        # with one bad patch is already useful data, so if the refit
        # doesn't actually help, silently keep the original result and
        # let the existing level2 flag do its job rather than discarding
        # a mostly-good file over a few minutes a refit couldn't fix.
        if sigma == "default":
            preLocalSigma = _worstLocalZSigma(matchedDats, config)
            if (preLocalSigma is not None) and (
                preLocalSigma > config.quality.maxZSigma
            ):
                overallSigma = zResidualSigma(matchedDats, rotate_final, config)
                if overallSigma > config.quality.maxZSigma:
                    # same reasoning as the early-exit above: a refit
                    # corrects bias, not spread, and the file-wide spread
                    # is already too wide, so a localized refit can't
                    # help either -- don't spend the compute.
                    log.warning(
                        tools.concat(
                            "localized Z-residual spread is",
                            preLocalSigma,
                            "(exceeds maxZSigma",
                            config.quality.maxZSigma,
                            ") but the file's overall Z-residual spread is",
                            overallSigma,
                            "(also too wide) -- a refit can't fix scatter, "
                            "leaving it for level2's zResidualTooWide flag "
                            "to report",
                        )
                    )
                else:
                    log.warning(
                        tools.concat(
                            "localized Z-residual spread is",
                            preLocalSigma,
                            "which exceeds maxZSigma",
                            config.quality.maxZSigma,
                            "in at least one time bin, even though the "
                            "file passed on aggregate -- trying a cheap "
                            "rotation refit before accepting the "
                            "localized problem as-is",
                        )
                    )
                    preLocalRefitMatchedDats = matchedDats
                    candidate = candidateRotate = candidateRotateErr = None
                    healed = refitRotationFromMatches(
                        matchedDats,
                        rotate_final,
                        rotate_err_final,
                        config,
                        sigma=sigma,
                        nSamples4rot=nSamples4rot,
                        y_cov_diag=y_cov_diag,
                    )
                    if healed is not None:
                        candidate, candidateRotate, candidateRotateErr = healed
                    if (candidate is None) or (
                        candidate.matchScore.median().values
                        < config.quality.minMatchScore
                    ):
                        # single-window refit either failed outright or
                        # dragged the aggregate score below the minimum --
                        # try the same time-segmented escalation the
                        # aggregate-failure path falls back to.
                        healedSeg = refitRotationFromMatches(
                            preLocalRefitMatchedDats,
                            rotate_final,
                            rotate_err_final,
                            config,
                            sigma=sigma,
                            nSamples4rot=nSamples4rot,
                            y_cov_diag=y_cov_diag,
                            nSegments=15,
                        )
                        if healedSeg is not None:
                            candidate, candidateRotate, candidateRotateErr = (
                                healedSeg
                            )

                    keepCandidate = False
                    candidateLocalSigma = None
                    if candidate is not None:
                        candidateAggregate = candidate.matchScore.median().values
                        candidateLocalSigma = _worstLocalZSigma(candidate, config)
                        # only accept the refit if it didn't drag the
                        # aggregate score below the minimum AND it
                        # actually shrank the localized problem -- a
                        # refit that "succeeds" by making things worse
                        # elsewhere is not an improvement.
                        keepCandidate = (
                            candidateAggregate >= config.quality.minMatchScore
                            and candidateLocalSigma is not None
                            and candidateLocalSigma < preLocalSigma
                        )
                    if keepCandidate:
                        log.info(
                            tools.concat(
                                "localized Z-residual refit improved worst-bin "
                                "sigma from",
                                preLocalSigma,
                                "to",
                                candidateLocalSigma,
                            )
                        )
                        matchedDats, rotate_final, rotate_err_final = (
                            candidate,
                            candidateRotate,
                            candidateRotateErr,
                        )
                    else:
                        log.warning(
                            "localized Z-residual refit did not help (or made "
                            "things worse) -- keeping the original result; "
                            "the affected minute(s) will still be flagged by "
                            "level2's zResidualTooWide quality bit"
                        )

    matchedDats = tools.finishNc(
        matchedDats,
        config.site,
        config.visssGen,
        extra=tools.collectVersionAttrs(
            "level1match", {"level1detect": [fnameLv1Detect] + fnames1F}
        ),
    )

    matchedDats["fitMethod"] = matchedDats.fitMethod.astype("U30")
    matchedDats["dim2D"] = matchedDats.dim2D.astype("U2")
    matchedDats["dim3D"] = matchedDats.dim3D.astype("U9")
    matchedDats["camera"] = matchedDats.camera.astype("U30")
    matchedDats["camera_rotation"] = matchedDats.camera_rotation.astype("U30")

    if writeNc:
        tools.to_netcdf2(matchedDats, config, fname1Match)

    log.info(
        tools.concat("DONE", fname1Match, "with", len(matchedDats.pair_id), "particles")
    )

    return (
        fname1Match,
        matchedDats,
        rotate_final,
        rotate_err_final,
        nLeader,
        nFollower,
        len(matchedDats.pair_id),
        errors,
    )


@tools.loopify
def createMetaRotation(
    case,
    config,
    skipExisting=True,
    version=__version__,
    y_cov_diag=1.65**2,
    chunckSize=1000,
    rotate="config",
    rotate_err="config",
    maxDiffMs="config",
    nPoints=500,
    sigma="default",
    minDMax4rot=10,
    nSamples4rot=300,
    minSamples4rot=50,
    testing=False,
    completeDaysOnly=True,
    writeNc=True,
    stopOnFailure=False,
    maxAgeDaysPrevFile=1,
    doPlots=True,
    tryAgain=True,
):
    """
    Create meta rotation data for camera alignment.

    Parameters
    ----------
    case : str
        Case identifier
    config : object
        Configuration object with camera settings
    skipExisting : bool, optional
        Whether to skip existing files (default is True)
    version : str, optional
        Version string (default is __version__)
    y_cov_diag : float, optional
        Observation covariance diagonal (default is 1.65**2)
    chunckSize : int, optional
        Size of data chunks (default is 1000)
    rotate : dict or str, optional
        Initial rotation parameters or "config" (default is "config")
    rotate_err : dict or str, optional
        Initial rotation parameter errors or "config" (default is "config")
    maxDiffMs : float or str, optional
        Maximum time difference in milliseconds or "config" (default is "config")
    nPoints : int, optional
        Number of points for estimation (default is 500)
    sigma : dict or str, optional
        Sigma values for matching criteria or "default" (default is "default")
    minDMax4rot : float, optional
        Minimum DMax threshold for rotation estimation (default is 10)
    nSamples4rot : int, optional
        Number of samples for rotation estimation (default is 300)
    minSamples4rot : int, optional
        Minimum samples for rotation estimation (default is 50)
    testing : bool, optional
        Whether to generate test plots (default is False)
    completeDaysOnly : bool, optional
        Whether to process only complete days (default is True)
    writeNc : bool, optional
        Whether to write NetCDF files (default is True)
    stopOnFailure : bool, optional
        Whether to stop on failure (default is False)
    maxAgeDaysPrevFile : int, optional
        Maximum age of previous file in days (default is 1)

    Returns
    -------
    tuple
        (metaRotation, fnameMetaRotation) - Meta rotation data and file name
    """
    import pandas as pd

    nError = 0
    nL = None
    nF = None
    nM = None

    # find files
    fl = files.FindFiles(case, config.leader, config, version)
    ff = files.FindFiles(case, config.follower, config, version)

    # get events
    eventFile, eventDat = fl.getEvents()
    # get all the other file names
    try:
        fflM = files.FilenamesFromLevel(eventFile, config)
    except IndexError:
        log.error("NO EVENT DATA %s" % case)
        return None, None

    # output file
    fnameMetaRotation = fflM.fname["metaRotation"]

    if eventFile.endswith("nodata") or eventFile.endswith("broken.txt"):
        # metaEvents itself being .broken.txt (confirmed raw-data gap) or
        # .nodata still means there is nothing here to compute a rotation
        # from -- metaRotation keeps writing its own .nodata for this
        # exact day either way; the existing copy-old-data mechanism
        # further down (rotate="config" seeding from the last known
        # rotation, or tools.copyLastMetaRotation for longer gaps) is what
        # makes sure processing resumes cleanly once real data comes back.
        log.warning(f"No data available for {case}: {eventFile}")
        fflM.writeStatus(
            "metaRotation", "nodata", f"No data available for {case}: {eventFile}"
        )
        return None, None

    isBad, reason = tools.isBadPeriod(case, config, product="metaRotation")

    if isBad:
        raise RuntimeError(f"metaRotation data marked as broken due to: {reason}")

    # check whether output exists
    # fL/level resolve metaRotation's declared LEVEL_REGISTRY parents
    # (both cameras' level1detect/metaEvents) via `fl` automatically --
    # metaRotation is leaderOnly so fl's own camera role is always right
    # here -- instead of a hand-written list that can drift out of sync.
    if skipExisting and tools.checkForExisting(
        fnameMetaRotation,
        fL=fl,
        level="metaRotation",
    ):
        return None, None

    # figure out whether all level1detect data has been processed
    # (requireL0Files=True so a day with zero level0 files is not
    # mistaken for "complete" just because 0 of 0 expected files exist)
    if completeDaysOnly and not fl.isComplete("level1detect", requireL0Files=True):
        if fl.isGenuineDataGap():
            log.warning(
                f"Newer leader L0 files have been found, likely data gap on {case}"
            )
            fflM.writeStatus(
                "metaRotation", "nodata", f"No leader level 0 data available for {case}"
            )
            return None, None
        log.warning(
            "L1 LEADER NOT COMPLETE YET %i of %i "
            % (len(fl.listFilesExt("level1detect")), len(fl.listFiles("level0txt")))
        )
        return None, None

    # figure out whether all level1detect data has been processed
    if completeDaysOnly and not ff.isComplete("level1detect", requireL0Files=True):
        if ff.isGenuineDataGap():
            log.warning(
                f"Newer follower L0 files have been found, likely data gap on {case}"
            )
            fflM.writeStatus(
                "metaRotation", "nodata", f"No follower level 0 data available for {case}"
            )
            return None, None
        log.warning(
            "L1 FOLLOWER NOT COMPLETE YET %i of %i "
            % (len(ff.listFilesExt("level1detect")), len(ff.listFiles("level0txt")))
        )
        return None, None

    log.info("running %s" % fnameMetaRotation)

    # collect results here later
    metaRotation = []

    # try to estimate first guess from previous data
    if isinstance(rotate, str) and (rotate == "config"):
        # get previous rotation filename
        prevFile = []
        flyesterday = fl.yesterdayObject
        for ii in range(maxAgeDaysPrevFile):
            prevFile = flyesterday.listFiles("metaRotation")
            if len(prevFile) == 0:
                flyesterday = flyesterday.yesterdayObject
            else:
                log.warning(f"Taking metaRotation start from {flyesterday.case}")
                prevFile = prevFile[0]
                break
        # prevFile = fflM.prevFile2(
        #     "metaRotation", maxOffset=np.timedelta64(maxAgeDaysPrevFile, "h")
        # )

        # handle case that there is no previous file, make sure time in config is not too old
        if (len(prevFile) == 0) and (
            datetime.datetime.strptime(config.start, "%Y-%m-%d") != fl.datetime.date()
        ):
            _, prevTime = tools.getPrevRotationEstimate(
                fflM.datetime64, "transformation", config
            )
            deltaT = fflM.datetime64 - prevTime

            if deltaT > np.timedelta64(2, "D"):
                log.warning(
                    f"no previous data found for {fnameMetaRotation}"
                    f"! data in config file "
                    f"{round(deltaT/np.timedelta64(1,'h'))}h old which is more "
                    f"than 48h. "
                )
                yesterdayEventFileMissing = not os.path.isfile(
                    fflM.yesterdayObject.fnamesDaily.metaEvents
                )

                (
                    foundLastFile,
                    lastCase,
                    lastFile,
                    lastFileTime,
                    lastFileCase,
                ) = files.findLastFile(
                    config, "metaRotation", config.leader, beforeCase=fflM.yesterday
                )
                if lastFileTime == "n/a":
                    log.error(
                        f"Did not find previous metaRotation. "
                        f"Try running '{sys.executable} -m VISSSlib tools.copyLastMetaRotation "
                        f"{config.filename} {lastFileCase} {fflM.yesterday}' if instrument was offline",
                    )
                    return None, None

                dataGapSmallEnough = (
                    fflM.datetime64 - np.datetime64(lastFileTime)
                ) < np.timedelta64(8, "D")
                if yesterdayEventFileMissing and dataGapSmallEnough:
                    log.error(
                        f"I cannot find {fflM.yesterdayObject.fnamesDaily.metaEvents}"
                        " and I assume that the instrument was offline."
                        " I try to fix it with copyLastMetaRotation",
                        lastFileCase=lastFileCase,
                        yesterday=fflM.yesterday,
                    )
                    tools.copyLastMetaRotation(config, lastFileCase, fflM.yesterday)
                    # try again
                    prevFile = fl.yesterdayObject.listFiles("metaRotation")[0]

                else:
                    log.error(
                        f"Try running '{sys.executable} -m VISSSlib tools.copyLastMetaRotation "
                        f"{config.filename} {lastFileCase} {fflM.yesterday}' if instrument was offline",
                    )
                    return None, None
                # raise RuntimeError(
                #     f"Skipping, no previous data found for {fnameMetaRotation}"
                #     "! data in config file "
                #     f"{round(deltaT/np.timedelta64(1,'h'))}h old which is more "
                #     f"than 48h. Try running '{sys.executable} -m VISSSlib tools.copyLastMetaRotation "
                #     f"{config.filename} {lastCase} {fflM.yesterday}' if instrument was offline",
                # )

        # add previous configuration to config file structure
        if len(prevFile) > 0:
            with xr.open_dataset(prevFile) as ds:
                prevDat = ds.where(ds.camera_Ofz.notnull(), drop=True).load()
            config = tools.rotXr2dict(prevDat, config)

        # get most recent rotation estimate from config object
        rotate_default, rotate_err_default, prevTime = tools.getPrevRotationEstimates(
            fflM.datetime64, config
        )
        log.info(f"got {rotate_default} from {prevTime} with getPrevRotationEstimates")

        # add most recent estimate to output so that there is always at least
        # one data point in a metaRoation even if it fails completely
        metaRotation.append(
            tools.rotDict2Xr(rotate_default, rotate_err_default, prevTime)
        )

        rotate_default = pd.Series(dict(rotate_default))
        rotate_err_default = pd.Series(dict(rotate_err_default))

    # do not use previous data but provided arguments
    else:
        log.info(f"got {rotate} from function key words")
        # use values provided by arguments
        rotate_default = pd.Series(dict(rotate))
        rotate_err_default = pd.Series(dict(rotate_err))

    maxConsecutiveFixFailures = 5
    consecutiveFixFailures = 0

    def tryFixFromScratch(ffl1, reason):
        """
        Last-resort recovery for a single file's rotation estimate: refit
        from scratch (blur>100, large-particle-only, wide-open prior --
        see manualRotationEstimate) instead of trusting the possibly-wrong
        carried-forward default. Used both when matchParticles raises and
        when it silently returns without updating the rotation (e.g. a
        real camera shift the current default no longer explains, which
        otherwise gets stuck repeating the same stale value for the rest
        of the day and into subsequent days). Only ever used in-memory for
        this run; does not write back to the config file. Returns
        (rot, rot_err) as pd.Series, or (None, None) if the fit also fails
        validation inside manualRotationEstimate -- giving up silently is
        expected some of the time.

        Backs off after maxConsecutiveFixFailures in a row (e.g. a
        genuinely blocked-camera stretch the from-scratch fit can never
        succeed on -- each attempt costs a real multi-second refit, not
        worth repeating file after file for the same doomed stretch) and
        resumes automatically the moment any success happens again
        (tracked in the caller, since a plain unaided matchParticles
        success is just as good evidence the stretch ended as a
        successful fix is).
        """
        nonlocal consecutiveFixFailures
        if consecutiveFixFailures >= maxConsecutiveFixFailures:
            log.warning(
                f"skipping from-scratch fix for {ffl1.case}: already failed "
                f"{consecutiveFixFailures} times in a row, assuming this "
                "stretch is unrecoverable (e.g. camera genuinely blocked) "
                "until something succeeds again"
            )
            return None, None
        log.warning(f"{reason} for {ffl1.case}, trying to fix it from scratch")
        try:
            fixed = manualRotationEstimate(
                ffl1.case, config, returnResultOnly=True
            ).get(ffl1.case)
        except (RuntimeError, AssertionError) as e:
            log.error(f"fixing attempt raised for {ffl1.case}")
            log.error(str(e))
            consecutiveFixFailures += 1
            return None, None
        if fixed is None:
            log.error(f"fixing attempt FAILED for {ffl1.case}")
            consecutiveFixFailures += 1
            return None, None
        log.warning(
            f"fixed rotation from scratch for {ffl1.case}: {fixed['transformation']}"
        )
        return (
            pd.Series(fixed["transformation"]),
            pd.Series(fixed["transformation_err"]),
        )

    # loop through all files
    fnames1L = fl.listFilesExt("level1detect")
    for fname1L in fnames1L:
        ffl1 = files.FilenamesFromLevel(fname1L, config)

        if (
            fname1L.endswith("broken.txt")
            or fname1L.endswith("nodata")
            or fname1L.endswith("notenoughframes")
        ):
            log.warning("NO leader DATA", fname1L)
            continue

        # check whether we can use a result from the config file
        rotate_config, rot_err, rotate_time_config = tools.getPrevRotationEstimates(
            ffl1.datetime64, config
        )
        if np.abs(rotate_time_config - ffl1.datetime64) < np.timedelta64(1, "s"):
            log.warning(
                "taking rotation estimate directly from config file instead of calculating %s"
                % rotate_time_config
            )
            rot = pd.Series(dict(rotate_config))
            rot_err = pd.Series(dict(rot_err))

        # otherwise try estimation
        else:
            rot = None
            try:
                _, matchedDat4Rot, rot, rot_err, nL, nF, nM, errors = matchParticles(
                    fname1L,
                    config,
                    y_cov_diag=y_cov_diag,
                    chunckSize=chunckSize,
                    rotate=rotate_default,
                    rotate_err=rotate_err_default,
                    maxDiffMs=maxDiffMs,
                    rotationOnly=True,
                    nPoints=nPoints,
                    sigma=sigma,
                    minDMax4rot=minDMax4rot,
                    nSamples4rot=nSamples4rot,
                    minSamples4rot=minSamples4rot,
                    testing=testing,
                    singleParticleFramesOnly=True,
                    doRot=True,
                )

                # rotationOnly's own convergence check (change smaller
                # than its own uncertainty, see _refineRotationIteration)
                # is a different, looser bar than what level1match will
                # later require of the full, unfiltered particle
                # population (config.quality.minMatchScore) -- a rotation
                # can converge on the restricted single-particle-frame
                # subset used here and still not be accurate enough for
                # production matching. Proactively check that now (cheap:
                # matchedDat4Rot and its matchScore already exist) and
                # refit if needed, so metaRotation doesn't carry forward
                # a converged-but-not-quite-right estimate that would
                # only surface as a level1match failure later.
                if (
                    (rot is not None)
                    and (matchedDat4Rot is not None)
                    and (len(matchedDat4Rot.pair_id) > 0)
                    and (
                        matchedDat4Rot.matchScore.median().values
                        < config.quality.minMatchScore
                    )
                ):
                    healed = refitRotationFromMatches(
                        matchedDat4Rot,
                        rot,
                        rot_err,
                        config,
                        sigma=sigma,
                        nSamples4rot=nSamples4rot,
                        y_cov_diag=y_cov_diag,
                    )
                    if healed is not None:
                        _, rot, rot_err = healed

                # metaRotation.append(xr.DataArray([rot], ))
                # metaRotationErr.append(xr.DataArray())

            except (RuntimeError, AssertionError) as e:
                log.error(
                    "matchParticles FAILED %s, we try to fix it" % fnameMetaRotation
                )
                log.error(str(e))
                ## as a last resort, try to fix it from scratch:
                rot, rot_err = tryFixFromScratch(ffl1, "matchParticles FAILED")
                if rot is None:
                    continue
                nL = nF = nM = None
                errors = {"doMatchSlicer": False}

        # avoid division by zero
        if (nL == 0) or (nL is None):
            nL = 1
        if (nF == 0) or (nF is None):
            nF = 1
        if nM is None:
            nM = 1

        log.debug(
            tools.concat(
                fname1L,
                rot,
                nL,
                nF,
                nM,
                (nL > nSamples4rot),
                (nF > nSamples4rot),
                ((nM // nL) < 0.01),
                ((nM // nF) < 0.01),
            )
        )

        # append result to metaRotation object
        if rot is not None:
            metaRotation.append(tools.rotDict2Xr(rot, rot_err, ffl1.datetime64))
            # update default
            rotate_default = rot
            rotate_err_default = rot_err
            # any success -- fix-based or not -- is evidence a bad stretch
            # (if we were in one) has ended, so give tryFixFromScratch
            # another chance on the next failure instead of staying
            # backed off for the rest of the day
            consecutiveFixFailures = 0
        # result failed, but dataset was in theory large enough, add explicit nans in this case
        elif (
            (nL > nSamples4rot)
            and (nF > nSamples4rot)
            and ((nM / nL) < 0.01)  # less than 1% leader matched
            and ((nM / nF) < 0.01)  # less than 1% follower matched
            and (errors["doMatchSlicer"] == True)
        ):
            log.error(f"only {nM} of {nL}+{nF} particles matched!")
            metaRotation.append(tools.rotDict2Xr(np.nan, np.nan, ffl1.datetime64))
            if stopOnFailure:
                raise RuntimeError
            nError += 1
        elif (nL > nSamples4rot) or (nF > nSamples4rot):
            # matchParticles returned cleanly but didn't update the
            # rotation (e.g. the current default no longer explains this
            # file at all) even though there were enough particles to work
            # with -- worth a from-scratch attempt before giving up
            fixedRot, fixedRotErr = tryFixFromScratch(
                ffl1, "rotation not updated despite enough particles"
            )
            if fixedRot is not None:
                metaRotation.append(
                    tools.rotDict2Xr(fixedRot, fixedRotErr, ffl1.datetime64)
                )
                rotate_default = fixedRot
                rotate_err_default = fixedRotErr
            else:
                metaRotation.append(
                    tools.rotDict2Xr(
                        rotate_default, rotate_err_default, ffl1.datetime64
                    )
                )
        else:
            # just use default values again
            metaRotation.append(
                tools.rotDict2Xr(rotate_default, rotate_err_default, ffl1.datetime64)
            )

    if tryAgain and (nError > 0):
        # lets simply try to run it again in case it failed
        # we assume the last rotate_default and rotate_err_default were ok
        return createMetaRotation(
            case,
            config,
            skipExisting=skipExisting,
            version=version,
            y_cov_diag=y_cov_diag,
            chunckSize=chunckSize,
            rotate=rotate_default,
            rotate_err=rotate_err_default,
            maxDiffMs=maxDiffMs,
            nPoints=nPoints,
            sigma=sigma,
            minDMax4rot=minDMax4rot,
            nSamples4rot=nSamples4rot,
            minSamples4rot=minSamples4rot,
            testing=testing,
            completeDaysOnly=completeDaysOnly,
            writeNc=writeNc,
            stopOnFailure=stopOnFailure,
            maxAgeDaysPrevFile=maxAgeDaysPrevFile,
            doPlots=doPlots,
            tryAgain=False,
        )
    else:
        # merge results
        if len(metaRotation) > 0:
            metaRotation = xr.concat(metaRotation, dim="file_starttime")

        if writeNc:
            metaRotation = tools.finishNc(metaRotation, config.site, config.visssGen)
            tools.to_netcdf2(metaRotation, config, fnameMetaRotation)
        log.debug("DONE", fnameMetaRotation)

        if doPlots:
            quicklooks.metaRotationQuicklook(case, config, skipExisting=skipExisting)

        return metaRotation, fnameMetaRotation


@log.catch(reraise=True)
@tools.loopify
def manualRotationEstimate(
    case,
    config,
    nPoints=1000,
    iterations=4,
    minSamples4rot=90,
    minDMax4rot=None,
    returnResultOnly=True,
):
    """
    Estimate camera rotation parameters through iterative particle matching.

    This function processes multiple cases to estimate optimal camera rotation parameters
    by iteratively refining the rotation estimate using particle matching. Each case undergoes
    up to 4 iterations of matching with progressively refined parameters.

    Parameters
    ----------
    cases : list of str
        List of case identifiers to process in format ["YYYYMMDD-HHMMSS"]
    config : dict or str
        Configuration settings or path to settings file
    nPoints : int, optional
        Number of points to use in matching (default=1000)
    iterations : int, optional
        Number of iterations (default=4)
    minDMax4rot : float, optional
        Minimum particle size (Dmax, pixels) to admit into the first
        iteration's fit. If None (default), auto-computed per case from
        the file's own size distribution (the size above which there are
        at least 5000 blur>100 particles, falling back to 15 if the file
        doesn't have that many). Pass this explicitly to retry a case
        that fails with the auto-computed size using a smaller, more
        permissive one -- e.g. when there simply aren't enough
        large particles in a file for the auto-computed threshold to
        yield a usable fit.

    Returns
    -------
    str
        yaml Dictionary with case names as keys and rotation parameters as values
        Format: {case: {"transformation": dict, "transformation_err": dict}}

    Notes
    -----
    The function performs these steps for each case: load level1 detection
    data; calculate a minimum particle size (minSize) for filtering; run up
    to 4 iterations of particle matching (the 1st with the full parameter
    set and strict filters, subsequent ones with relaxed parameters using
    the previous rotation estimate); validate results at each iteration;
    and store the final rotation parameters once all validations pass.
    """
    import pandas as pd
    import yaml

    results = {}  # Initialize results dictionary
    rotate_default = pd.Series(
        {"camera_phi": 0.0, "camera_theta": 0.0, "camera_Ofz": 0}
    )
    rotate_err_default = pd.Series(
        {"camera_phi": 1, "camera_theta": 1, "camera_Ofz": 50}
    )
    log.warning("#" * 80)
    log.warning(f"trying to fix {case}")
    log.warning("#" * 80)

    fl = files.FindFiles(case, config.leader, config)
    fname1L = fl.listFiles("level1detect")[0]

    if minDMax4rot is not None:
        minSize = minDMax4rot
    else:
        # Precompute minSize once per case
        with xr.open_dataset(fname1L) as l1dat:
            try:
                minSize = np.sort(l1dat.isel(pid=(l1dat.blur > 100)).Dmax)[-500 * 10]
            except IndexError:
                minSize = 15
    log.info("minSize %i" % minSize)

    # Initialize rotation parameters
    current_rot = rotate_default
    current_rot_err = rotate_err_default
    results[case] = None  # Default if case fails

    # Loop through matchParticles calls (up to 4 iterations)
    for i in range(iterations):
        # Configure parameters per iteration
        kwargs = {
            "doRot": True,
            "rotationOnly": True,
            "rotate": current_rot,
            "rotate_err": current_rot_err,
            "nPoints": nPoints,
        }

        if i == 0:  # First iteration has special parameters
            kwargs.update(
                {
                    "maxDiffMs": "config",
                    "chunckSize": 1000,
                    "minSamples4rot": minSamples4rot,
                    "minDMax4rot": minSize,
                    "singleParticleFramesOnly": True,
                    "nSamples4rot": 2000,
                    "sigma": {"H": 1.2, "T": 1e-4},
                }
            )
        else:  # Subsequent iterations
            kwargs.update({"minSamples4rot": 30})

        # Execute matching
        (
            fout,
            matchedDat,
            new_rot,
            new_rot_err,
            nL,
            nF,
            nM,
            errors,
        ) = matchParticles(fname1L, config, **kwargs)

        # Validation checks
        if not nL:
            log.warning("Too little leader data!")
            break
        if not nM:
            log.warning("NO MATCHED DATA!")
            break
        try:
            if nL / nM < 0.05:
                log.warning("Too little matched data!")
                break
        except ZeroDivisionError:
            log.warning("NO MATCHED DATA!")
            break
        if (new_rot is None) or (new_rot.get("camera_phi") == 0):
            log.warning(f"Rotation invalid at iteration {i+1}")
            break

        # Update rotation for next iteration
        current_rot, current_rot_err = new_rot, new_rot_err
        log.info(f"Iteration {i+1}: MATCHED {nM/nL:.2f}, Leader particles: {nL}")
        log.info(current_rot)

        # Final iteration processing
        if i == 3:
            matchScoreMedian = matchedDat.matchScore.median().values
            if matchScoreMedian < config.quality.minMatchScore:
                log.warning(f"Low match score: {matchScoreMedian}")
                break

            # Store successful result
            results[case] = {
                "transformation": current_rot.round(6).to_dict(),
                "transformation_err": current_rot_err.round(6).to_dict(),
            }
            log.info(results[case])

    if returnResultOnly:
        return results
    else:
        return (
            None,
            None,
            current_rot,
            current_rot_err,
            nL,
            nF,
            nM,
            errors,
        )
