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
    import scipy.stats

    x = x.astype(float)
    mu = float(mu)
    sigma = float(sigma)
    delta = float(delta)

    x1 = x - (delta / 2)
    x2 = x + (delta / 2)

    # integrated over delta x region
    return scipy.stats.norm.cdf(x2, loc=mu, scale=sigma) - scipy.stats.norm.cdf(
        x1, loc=mu, scale=sigma
    )


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
    mPart : array-like
        Particle match indices
    mProp : array-like
        Match probabilities
    doubleCounts : array-like
        Indices of particles that appear multiple times

    Returns
    -------
    tuple
        Updated mPart and mProp arrays with duplicates removed
    """
    for doubleCount in doubleCounts:
        ii = np.where(mPart[:, 0] == doubleCount)[0]
        bestProp = mProp[ii, 0].values.argmax()
        #         print(doubleCount, ii, bestProp)
        for jj, i1 in enumerate(ii):
            if jj == bestProp:
                continue
            mPart[i1, :-1] = mPart[i1, 1:].values
            mProp[i1, :-1] = mProp[i1, 1:].values
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

    matchedParticles = {}
    matchedProbabilities = {}

    # try to solve this from both perspectives
    for camera, prop1, dat2 in zip(
        [config["leader"], config["follower"]],
        [propJoint, propJoint.T],
        [leader1D, follower1D],
    ):
        matchedParticles[camera] = np.argsort(prop1, axis=1)[:, -maxMatches:][:, ::-1]
        matchedProbabilities[camera] = np.sort(prop1, axis=1)[:, -maxMatches:][:, ::-1]

        matchedParticles[camera] = xr.DataArray(
            matchedParticles[camera],
            coords=[range(len(dat2.fpid)), range(matchedParticles[camera].shape[1])],
            dims=["fpidII", "match"],
        )
        matchedProbabilities[camera] = xr.DataArray(
            matchedProbabilities[camera],
            coords=[range(len(dat2.fpid)), range(matchedParticles[camera].shape[1])],
            dims=["fpidII", "match"],
        )

    del propJoint, prop

    for reverseFactor in [1, -1]:
        cam1, cam2 = [config["leader"], config["follower"]][::reverseFactor]

        matchedParticles[cam1] = matchedParticles[cam1].where(
            matchedProbabilities[cam1] > minProp
        )
        matchedProbabilities[cam1] = matchedProbabilities[cam1].where(
            matchedProbabilities[cam1] > minProp
        )

        for kk in range(maxMatches):
            u, c = np.unique(matchedParticles[cam1][:, 0], return_counts=True)
            doubleCounts = u[np.where(c > 1)[0]]
            doubleCounts = doubleCounts[np.isfinite(doubleCounts)]
            if len(doubleCounts) != 0:
                # print(
                # cam1, "particles have been matched twice, fixing", kk)
                matchedParticles[cam1], matchedProbabilities[cam1] = removeDoubleCounts(
                    matchedParticles[cam1], matchedProbabilities[cam1], doubleCounts
                )
            else:
                break

        u, c = np.unique(matchedParticles[cam1][:, 0], return_counts=True)
        doubleCounts = u[np.where(c > 1)[0]]
        doubleCounts = doubleCounts[np.isfinite(doubleCounts)]

        assert len(doubleCounts) == 0, (
            "%s particles have still been matched twice" % cam1
        )

    for reverseFactor in [1, -1]:
        cam1, cam2 = [config["leader"], config["follower"]][::reverseFactor]
        matchedParticles[cam1] = matchedParticles[cam1][:, 0]
        matchedProbabilities[cam1] = matchedProbabilities[cam1][:, 0]

        matchedParticles[cam1] = matchedParticles[cam1].dropna("fpidII")
        matchedProbabilities[cam1] = matchedProbabilities[cam1].dropna("fpidII")

    if np.all([len(v) == 0 for v in matchedParticles.values()]):
        noMatches = True
        log.warning(tools.concat("no matched particles"))
        nMatched = 0
        return None, len(leader1D.fpid), len(follower1D.fpid), nMatched

    cam1, cam2 = [config["leader"], config["follower"]]

    pairs1 = set(
        zip(
            matchedParticles[cam1].fpidII.values,
            matchedParticles[cam1].values.astype(int),
        )
    )
    pairs2 = set(
        zip(
            matchedParticles[cam2].values.astype(int),
            matchedParticles[cam2].fpidII.values,
        )
    )

    disputedPairs = pairs1 - pairs2

    # sort pairs together
    dats = []
    dats.append(
        leader1D.isel(fpid=matchedParticles[config["leader"]].fpidII.values.astype(int))
    )
    dats.append(
        follower1D.isel(fpid=matchedParticles[config["leader"]].values.astype(int))
    )

    for dd, d1 in enumerate(dats):
        pid = deepcopy(d1.pid.values)
        file_starttime = deepcopy(d1.file_starttime.values)
        d1 = d1.rename(fpid="pair_id")
        d1 = d1.assign_coords(
            pair_id=np.arange(len(matchedParticles[config["leader"]].fpidII))
            + indexOffset
        )

        d1["pid"] = xr.DataArray(pid, coords=[d1.pair_id])
        d1["file_starttime"] = xr.DataArray(file_starttime, coords=[d1.pair_id])
        dats[dd] = d1

    matchedDat = xr.concat(dats, dim="camera")
    matchedDat = matchedDat.assign_coords(camera=[config["leader"], config["follower"]])
    # add propabilities
    matchedDat["matchScore"] = xr.DataArray(
        matchedProbabilities[config["leader"]].values.astype(np.float32),
        coords=[matchedDat.pair_id],
    )

    # clean up
    del (
        dats,
        matchedParticles,
        matchedProbabilities,
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

    matchedDat = None
    matchedDat4Rot = None
    rotate_time = None

    if not doRot:
        # check whether output exists
        if skipExisting and tools.checkForExisting(
            fname1Match,
            parents=[fnameLv1Detect] + fnames1F,
        ):
            return fname1Match, None, None, None, None, None, None, errors

        # get rotation estimates and add to config instead of estimating them
        fnameMetaRotation = ffl1.fname["metaRotation"]

        if os.path.isfile(f"{fnameMetaRotation}.broken.txt"):
            raise RuntimeError(f"{fnameMetaRotation}.broken.txt is broken")

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
            with tools.open2("%s.nodata" % fname1Match, config, "w") as f:
                f.write(f"no leader data in {fnameLv1Detect}")
        log.error(tools.concat(f"no leader data in {fnameLv1Detect}"))
        errors["tooFewObs"] = True
        return fname1Match, None, None, None, None, None, None, errors

    log.info(tools.concat(len(leader1D.pid)))

    if len(leader1D.pid) <= 1:
        if not rotationOnly:
            with tools.open2("%s.nodata" % fname1Match, config, "w") as f:
                f.write(f"only one particle in  {fnameLv1Detect}")
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
            with tools.open2("%s.nodata" % fname1Match, config, "w") as f:
                f.write(f"no follower data for {fnameLv1Detect}")
        log.error(tools.concat(f"no follower data for {fnameLv1Detect}"))
        errors["openingData"] = True
        return fname1Match, None, None, None, None, None, None, errors

    fClass = [files.FilenamesFromLevel(f, config) for f in fnames1F]
    fCases = np.unique([f.case.split("-")[0] for f in fClass])
    # # just in case
    # metadata.createEvent(
    #     ffl1.case, config.leader, config, quiet=True, skipExisting=True
    # )
    # for fCase in fCases:
    #     metadata.createEvent(
    #         fCase, config.follower, config, quiet=True, skipExisting=True
    #     )

    lEvents = ffl1.fname.metaEvents
    lEvents = xr.open_dataset(lEvents)

    fEvents = np.unique([f.fname.metaEvents for f in fClass])
    fEvents = xr.open_mfdataset(fEvents).load()

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
            with tools.open2("%s.nodata" % fname1Match, config, "w") as f:
                f.write(f"no follower data after removal of blocked data {fname1Match}")
        log.error(f"no follower data after removal of blocked data {fname1Match}")
        errors["followerBlocked"] = True
        return fname1Match, None, None, None, None, None, None, errors

    if leader1D is None:
        if not rotationOnly:
            with tools.open2("%s.nodata" % fname1Match, config, "w") as f:
                f.write(f"no leader data after removal of blocked data {fname1Match}")
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
                with tools.open2("%s.nodata" % fname1Match, config, "w") as f:
                    f.write(
                        f"Leader ptpStatus is not Slave: {brokenDat.values} at {brokenDat.file_starttime.values}"
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
                with tools.open2("%s.nodata" % fname1Match, config, "w") as f:
                    f.write(
                        f"Follower ptpStatus is not Slave: {brokenDat.values} at {brokenDat.file_starttime.values}"
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

    timeBlocks = np.concatenate(
        (
            follower1DAll.capture_time.values[:1],
            followerRestarted,
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
    for tt, (FR1, FR2) in enumerate(zip(timeBlocks[:-1], timeBlocks[1:])):
        log.info(
            tools.concat(
                tt + 1,
                "of",
                len(timeBlocks) - 1,
                "slice for follower restart",
                FR1,
                FR2,
            )
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
            continue
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
            continue
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
            continue

        # the 2nd <= is on purpose because it is required if there is no restart. if there is a restart, there is anyway no data exactly at that time
        TIMES = (FR1 <= follower1DAll.capture_time.values) & (
            follower1DAll.capture_time.values <= FR2
        )
        if np.sum(TIMES) <= 3:
            log.warning(
                f"CONTINUE, too little follower data (#{np.sum(TIMES)}) overlapping with leader period"
            )
            continue

        errorStrs.append([])
        nSamples.append(np.sum(TIMES))

        # TIMES = REGEX nach  file_starttime
        follower1D = follower1DAll.isel(fpid=TIMES)

        if "makeCaptureTimeEven" in config.dataFixes:
            # does not make sense for leader
            # redo capture_time based on first time stamp...
            try:
                follower1D = fixes.makeCaptureTimeEven(follower1D, config, dim="fpid")
            except AssertionError as e:
                log.error("fixes.makeCaptureTimeEven FAILED")
                log.error(str(e))
                if not rotationOnly:
                    if np.sum(TIMES) <= 20:
                        log.error(
                            tools.concat(f"so little data {np.sum(TIMES)} ignore it!")
                        )
                        continue
                    else:
                        errorStrs[-1].append(
                            f"fixes.makeCaptureTimeEven FAILED {str(e)}"
                        )

                else:
                    continue

        if not np.all(np.diff(follower1D.capture_id) >= 0):
            log.error(tools.concat("follower camera reset detected"))
            if not rotationOnly:
                errorStrs[-1].append("follower camera reset detected")
            continue

        if maxDiffMs == "config":
            maxDiffMs = 1000 / config.fps / 2

        # if (minDMax4rot > 0):
        #     filt = (leader1D.Dmax>minDMax4rot).values
        #     log.info(tools.concat("DMax capture id filter leader:", minDMax4rot, np.sum(filt)/len(leader1D.fpid) * 100,"%"))
        #     leader1D = leader1D.isel(fpid=filt)

        # if (minDMax4rot > 0):
        #     filt = (follower1D.Dmax>minDMax4rot).values
        #     log.info(tools.concat("DMax capture id filter follower:", minDMax4rot, np.sum(filt)/len(follower1D.fpid) * 100,"%"))
        #     follower1D = follower1D.isel(fpid=filt)

        if (
            offsetsOnly
            or ("ptpStatus" not in lEvents.data_vars)
            or ("ptpStatus" not in fEvents.data_vars)
            or np.any(
                lEvents.ptpStatus.where(lEvents.event == "newfile", drop=True)
                == "Disabled"
            ).values
            or np.any(
                fEvents.ptpStatus.where(fEvents.event == "newfile", drop=True)
                == "Disabled"
            ).values
        ):
            ptpTime = False
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
                continue

            if (nMatched2 <= 1) and (nMatched1 <= 1):
                # if not rotationOnly:
                #     with tools.open2(f"{fname1Match}.nodata", config, "w") as f:
                #         f.write("NOT ENOUGH DATA")
                log.error(tools.concat("NOT ENOUGH DATA", fname1Match, tt, FR1, FR2))
                continue

            # In theory, capture time is much better, but there are cases were it is off. Try to identify them by chgecking whether record_time yielded more matches.
            # for mosaic, capture time is pretty much useless!
            if (nMatched2 > nMatched1) or (config.site == "mosaic"):
                if nMatched2 == -99:
                    log.error(
                        tools.concat(
                            "record_id based diff estiamtion failed",
                            fname1Match,
                            tt,
                            FR1,
                            FR2,
                        )
                    )
                    errors["offsetEstimation"] = True
                    continue

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
                return captureIdOffset, nMatched

            mu = {
                "Z": 0,
                "H": 0,
                "T": 0,
                "I": captureIdOffset,
            }
            delta = {
                "Z": 0.5,  # 0.5 because center is considered
                "Y": 0.5,  # 0.5 because center is considered
                "H": 1,
                "T": 1 / config.fps,
                "I": 1,
            }

        else:
            ptpTime = True
            mu = {
                "Z": 0,
                "H": 0,
                "T": 0,
            }
            delta = {
                "Z": 0.5,  # 0.5 because center is considered
                "Y": 0.5,  # 0.5 because center is considered
                "H": 1,
                "T": 1 / config.fps,
            }

        # figure out how cameras ae rotated, first prepare data
        dataTruncated4rot = False
        if doRot:
            rotates = []

            # for estiamting rotation, we wo not need the full data set, use subset to speed up caluculation
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
                    "not enough leader data to estimate rotation %i"
                    % len(leader1D4rot.fpid)
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

        # iterate to rotation coefficients in max. 20 steps

        if doRot:
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
                        pair_id=sorted(
                            np.argsort(matchedDat4Rot.matchScore)[-nSamples4rot:]
                        )
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
                        if np.all(
                            np.abs(rotates[ii - 1] - rotate_result) < rotate_err_result
                        ):
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

        if dataTruncated4rot or (not doRot):
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

                continue
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
        else:
            # matchDat is alread final because it was not truncated
            pass

        if (matchedDat is not None) and len(matchedDat.pair_id) > 0:
            # add position with final roation coeffs.
            matchedDat = addPosition(matchedDat, rotate_final, rotate_err_final, config)

            # fixed values would lead to confusion, so stay with original ones
            if "captureIdOverflows" in config.dataFixes:
                matchedDat = fixes.revertIdOverflowFix(matchedDat)

            matchedDats.append(matchedDat)

    # end loop camera restart FR

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
        with tools.open2(f"{fname1Match}.nodata", config, "w") as f:
            f.write("no data")
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
            raise RuntimeError(
                f"median matchScore is only {matchScoreMedian} and smaller than "
                f"minMatchScore {config.quality.minMatchScore} even though we "
                f"found {nPairs} particles"
            )

    matchedDats = tools.finishNc(matchedDats, config.site, config.visssGen)

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
        fflM = files.FilenamesFromLevel(fl.listFiles("metaEvents")[0], config)
    except IndexError:
        log.error("NO EVENT DATA %s" % case)
        return None, None

    # output file
    fnameMetaRotation = fflM.fname["metaRotation"]

    isBad, reason = tools.isBadPeriod(case, config, product="metaRotation")

    if isBad:
        raise RuntimeError(f"metaRotation data marked as broken due to: {reason}")

    # check whether output exists
    if skipExisting and tools.checkForExisting(
        fnameMetaRotation,
        events=fl.listFiles("metaEvents") + ff.listFiles(f"metaEvents"),
        parents=fl.listFilesExt(f"level1detect") + ff.listFilesExt(f"level1detect"),
    ):
        return None, None

    # figure out whether all level1detect data has been processed
    if completeDaysOnly and not fl.isCompleteL1detect:
        log.warning(
            "L1 LEADER NOT COMPLETE YET %i of %i "
            % (len(fl.listFilesExt("level1detect")), len(fl.listFiles("level0txt")))
        )
        return None, None

    # figure out whether all level1detect data has been processed
    if completeDaysOnly and not ff.isCompleteL1detect:
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
                (
                    foundLastFile,
                    lastCase,
                    lastFile,
                    lastFileTime,
                ) = files.findLastFile(config, "metaRotation", config.leader)

                log.warning(
                    f"no previous data found for {fnameMetaRotation}"
                    f"! data in config file "
                    f"{round(deltaT/np.timedelta64(1,'h'))}h old which is more "
                    f"than 48h. "
                )
                yesterdayEventFileMissing = not os.path.isfile(
                    fflM.yesterdayObject.fnamesDaily.metaEvents
                )
                dataGapSmallEnough = (
                    fflM.datetime64 - np.datetime64(lastFileTime)
                ) < np.timedelta64(8, "D")
                if yesterdayEventFileMissing and dataGapSmallEnough:
                    log.error(
                        f"I cannot find {fflM.yesterdayObject.fnamesDaily.metaEvents}"
                        " and I assume that the instrument was offline."
                        " I try to fix it with copyLastMetaRotation",
                        lastCase=lastCase,
                        yesterday=fflM.yesterday,
                    )
                    tools.copyLastMetaRotation(config, lastCase, fflM.yesterday)
                    # try again
                    prevFile = fl.yesterdayObject.listFiles("metaRotation")[0]

                else:
                    log.error(
                        f"Try running '{sys.executable} -m VISSSlib tools.copyLastMetaRotation "
                        f"{config.filename} {lastCase} {fflM.yesterday}' if instrument was offline",
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
            prevDat = xr.open_dataset(prevFile)
            prevDat = prevDat.where(prevDat.camera_Ofz.notnull(), drop=True)
            config = tools.rotXr2dict(prevDat, config)
            prevDat.close()

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
                _, _, rot, rot_err, nL, nF, nM, errors = matchParticles(
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

                # metaRotation.append(xr.DataArray([rot], ))
                # metaRotationErr.append(xr.DataArray())

            except (RuntimeError, AssertionError) as e:
                log.error(
                    "matchParticles FAILED %s, we try to fix it" % fnameMetaRotation
                )
                log.error(str(e))
                ## as a last resort, try to fix it from scratch:
                try:
                    _, _, rot, rot_err, nL, nF, nM, errors = manualRotationEstimate(
                        cases, settings, returnResultOnly=False
                    )
                except (RuntimeError, AssertionError) as e:
                    log.error("fixing attempt FAILED %s" % fnameMetaRotation)
                    log.error(str(e))
                    continue

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

    Returns
    -------
    str
        yaml Dictionary with case names as keys and rotation parameters as values
        Format: {case: {"transformation": dict, "transformation_err": dict}}

    Notes
    -----
    The function performs these steps for each case:
    1. Load level1 detection data
    2. Calculate minimum particle size (minSize) for filtering
    3. Run up to 4 iterations of particle matching:
        - 1st iteration: Full parameter set with strict filters
        - Subsequent iterations: Relaxed parameters using previous rotation estimate
    4. Validate results at each iteration
    5. Store final rotation parameters if all validations pass
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
