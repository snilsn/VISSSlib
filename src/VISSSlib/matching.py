# -*- coding: utf-8 -*-
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats
import pandas as pd
#import av
import bottleneck as bn
import pyOptimalEstimation as pyOE

from tqdm import tqdm

from copy import deepcopy

from . import __version__
from . import tools
from . import fixes
from . import files

import logging
log = logging.getLogger(__name__)


deltaY = deltaH = deltaI = 1.


def calc_Fz(phi, theta, Ofz, Lx, Lz, Fy):

    raise NotImplementedError("Do not use any more!")

    '''
    Parameters
    ----------
    phi : 
        Follower roll in deg
    theta :
        Follower pitch in deg
    Ofz :
        Offset Follower z in deg
    Lx :
        Leader x coordinate (in common xyz)
    Lz :
        Leader z coordinate (in common xyz)
    Fy :
        Follower y coordinate (in common xyz)
        
    Returns
    -------
    Fz :
        Offset and rotation corrected Follower z coordinate (in common xyz)

    
    Leader pitch, roll, and yaw as well as Follower yaw assumed to be 0.
    
    Olx (offset of leader in x) and Ofy (offset of follower in y) can be 
    ignored becuase only the difference for Fz is evaluated - 
    and Ofz can fix all shifts of the coordinate system
    '''
    Lzp = Lz #+ Olz
    Fyp = Fy #+ Ofy
    Lxp = Lx #+ Olx
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)

    Fzp = ((np.sin(theta) * Lxp) - (np.sin(phi)*Fyp) + (np.cos(theta)*Lzp))/np.cos(phi)
    Fz = Fzp - Ofz
    return Fz


def rotate_L2F(L_x, L_y, L_z, phi, theta, psi):

    ''' rotate from leader to follower coordinate system
    
    Parameters
    ----------
    L_x : float
        Leader x coordinate (in common xyz)
    L_y : float
        Leader y coordinate (in common xyz)
    L_z : float
        Leader z coordinate (in common xyz)
    phi : float
        Follower roll in deg
    theta : float
        Follower pitch in deg
    psi : float
        Follower yaw in deg
    
    Returns
    -------
    array
        Follower x, y, z coordinates
    '''

    #
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)
    
    F_xp =  np.cos(theta) * np.cos(psi) * L_x \
         +  np.cos(theta) * np.sin(psi) * L_y \
         -  np.sin(theta) * L_z
    F_yp = (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * L_x \
         + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * L_y \
         +  np.sin(phi) * np.cos(theta) * L_z 
    F_zp = (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * L_x \
         + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * L_y \
         +  np.cos(phi) * np.cos(theta) * L_z

    return F_xp, F_yp, F_zp

def shiftRotate_L2F(L_x, L_y, L_z, phi, theta, psi, Olx, Ofy, Ofz):

    ''' shift and rotate from leader to follower coordinate system
    
    Parameters
    ----------
    L_x : float
        Leader x coordinate (in common xyz)
    L_y : float
        Leader y coordinate (in common xyz)
    L_z : float
        Leader z coordinate (in common xyz)
    phi : float
        Follower roll in deg
    theta : float
        Follower pitch in deg
    psi : float
        Follower yaw in deg
    Olx : float
        leader shift in x direction
    Ofy : float
        follower shift in y direction
    Ofz : float
        follower shift in z direction

    Returns
    -------
    array
        Follower x, y, z coordinates
    '''


    L_xp = L_x + Olx

    F_x, F_yp, F_zp = rotate_L2F(L_xp, L_y, L_z, phi, theta, psi)

    F_y = F_yp - Ofy
    F_z = F_zp - Ofz

    return F_x, F_y, F_z



def rotate_F2L(F_xp, F_yp, F_zp, phi, theta, psi):

    ''' rotate from follower to leader coordinate system
    
    Parameters
    ----------
    L_x : float
        Follower x coordinate (in common xyz)
    L_y : float
        Follower y coordinate (in common xyz)
    L_z : float
        Follower z coordinate (in common xyz)
    phi : float
        Follower roll in deg
    theta : float
        Follower pitch in deg
    psi : float
        Follower yaw in deg

    Returns
    -------
    array
        Leader x, y, z coordinates
    '''    
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)

    L_x =  np.cos(theta) * np.cos(psi) * F_xp \
        + (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * F_yp \
        + (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * F_zp
    L_y =  np.cos(theta) * np.sin(psi) * F_xp \
        + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * F_yp \
        + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * F_zp
    L_z = -np.sin(theta) * F_xp \
        +  np.sin(phi) * np.cos(theta) * F_yp \
        +  np.cos(phi) * np.cos(theta) * F_zp

    return L_x, L_y, L_z

def shiftRotate_F2L(F_x, F_y, F_z, phi, theta, psi, Olx, Ofy, Ofz):

    ''' shift and rotate from follower to leader coordinate system
    
    Parameters
    ----------
    L_x : float
        Follower x coordinate (in common xyz)
    L_y : float
        Follower y coordinate (in common xyz)
    L_z : float
        Follower z coordinate (in common xyz)
    phi : float
        Follower roll in deg
    theta : float
        Follower pitch in deg
    psi : float
        Follower yaw in deg
    Olx : float
        Leader shift in x direction
    Ofy : float
        Follower shift in y direction
    Ofz : float
        Follower shift in z direction

    Returns
    -------
    array
        Leader x, y, z coordinates
    '''
    F_yp = F_y + Ofy
    F_zp = F_z + Ofz

    L_xp, L_y, L_z = rotate_F2L(F_x, F_yp, F_zp, phi, theta, psi)

    L_x = L_xp - Olx

    return L_x, L_y, L_z


def calc_L_z(L_x, F_yp, F_zp, phi, theta, psi):

    '''estimate z coordinate for the leader based on combined leade rand follower 
    measurements
    
    [description]
    
    Parameters
    ----------
    L_x : float
        x measurement of leader
    F_yp : float
        y measurement of follower without shift
    F_zp : float
        z measurement of follower without shift
    phi : float
        Follower roll in deg in deg
    theta : float
        Follower pitch in de in degg
    psi : float
        Follower yaw in deg in deg
    
    Returns
    -------
    float
        z coordinate as seen by leader ignoring offsets
    '''

    # with  wolfram simplification

    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)

    L_z = - (np.sin(theta))/(np.cos(theta) * np.cos(psi)) * L_x \
          - (np.sin(theta) * np.sin(psi) * np.cos(phi) - np.cos(psi) * np.sin(phi))/(np.cos(theta) * np.cos(psi)) * F_yp\
          + (np.sin(theta) * np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(phi))/(np.cos(theta) * np.cos(psi)) * F_zp
    return L_z


def calc_L_z_withOffsets(L_x, F_y, F_z, camera_phi=0, camera_theta=0, camera_psi=0, camera_Ofy=0, camera_Ofz=0, camera_Olx=0):
    
    '''estimate z coordinate for the leader based on combined leade rand follower 
    measurements
    
    [description]
    
    Parameters
    ----------
    L_x : float
        x measurement of leader
    F_yp : float
        y measurement of follower
    F_zp : float
        z measurement of follower
    phi : float
        Follower roll in deg in deg
    theta : float
        Follower pitch in de in degg
    psi : float
        Follower yaw in deg in deg
    Olx : float
        Leader shift in x direction
    Ofy : float
        Follower shift in y direction
    Ofz : float
        Follower shift in z direction

    Returns
    -------
    float
        z coordinate as seen by leader
    '''

    for k in camera_phi, camera_theta, camera_psi, camera_Ofy, camera_Ofz, camera_Olx:
        assert not np.any(np.isnan(k)), k

    F_yp = F_y + camera_Ofy
    F_zp = F_z  + camera_Ofz
    L_xp = L_x  + camera_Olx

    return calc_L_z(L_xp, F_yp, F_zp, camera_phi, camera_theta, camera_psi)


# def forward(x, Lx=None, Lz=None, Fy=None):
#     '''
#     forward model for pyOptimalEstimation
#     '''
#     y = calc_Fz(x.phi, x.theta, x.Ofz, Lx, Lz, Fy)
#     y = pd.Series(y, index=np.array(range(len(y))))
#     return y

def forward(x, L_x=None, F_y=None, F_z=None):
    '''forward model for pyOptimalEstimation
    
    Parameters
    ----------
    x : pandas Series
       state vector "phi", "theta", "psi", "Ofy", "Ofz", "Olx"
    L_x : array, optional
        x coordinate as seen by the leader (the default is None)
    F_y : array, optional
        y coordinate as seen by the follower (the default is None)
    F_z : array, optional
        z coordinate as seen by the follower (the default is None)
    
    Returns
    -------
    pandas Series
        z coordinate as seen by leader
    '''

    
    y = calc_L_z_withOffsets(L_x, F_y, F_z, **x.to_dict())
    y = pd.Series(y, index=np.array(range(len(y))))
    return y

def retrieveRotation(dat3, x_ap, x_cov_diag, y_cov_diag, config, verbose=False):

    '''
    apply Optimal Estimation to retrieve rotation of cameras
    '''
    
    nPart = len(dat3.pair_id)
    allVars = ["camera_phi", "camera_theta", "camera_psi", "camera_Ofy", "camera_Ofz", "camera_Olx"]
    assert dat3.camera[0].values == config.leader

    L_x, L_z, F_y, F_z = get3DPosition(dat3.sel(camera=config.leader), dat3.sel(camera=config.follower), config)

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
        x_vars, x_ap, x_cov, y_vars, y_obs, y_cov, forward, b_vars=b_vars, b_p=b_p, S_b=S_b,
        forwardKwArgs=forwardKwArgs, verbose=verbose
        )

    oe.doRetrieval()

    assert not np.any(np.isnan(oe.x_op))

    return oe.x_op, oe.x_op_err, oe.dgf_x

def probability(x, mu, sigma, delta):

    x = x.astype(float)
    mu = float(mu)
    sigma = float(sigma)
    delta = float(delta)

    x1 = x-(delta/2)
    x2 = x+(delta/2)
    return scipy.stats.norm.cdf(x2, loc=mu, scale=sigma) - scipy.stats.norm.cdf(x1, loc=mu, scale=sigma)



def removeDoubleCounts(mPart, mProp, doubleCounts):
    for doubleCount in doubleCounts:
        ii = np.where(mPart[:,0] == doubleCount)[0]
        bestProp = mProp[ii, 0].values.argmax()
#         print(doubleCount, ii, bestProp)
        for jj, i1 in enumerate(ii):
            if jj == bestProp:
                continue
            mPart[i1,:-1] = mPart[i1,1:].values
            mProp[i1,:-1] = mProp[i1,1:].values
            mPart[i1,-1] = np.nan
            mProp[i1,-1] = np.nan

    return mPart, mProp


def doMatch(leader1D, follower1D, sigma, mu, delta, config, rotate, minProp=1e-10, minNumber4Stats=10, maxMatches=100, indexOffset=0, testing=False):
    '''
    match magic function
    
    minProp: minimal required probability
    maxMatches number of best matches to consider to select best one
    minNumber4Stats: min. number of samples to estimate sigmas and mus
    '''
    
    # print("using", sigma, mu, delta)  
    #print("doMatch", len(leader1D.fpid), len(follower1D.fpid))
    prop = {}
    

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

        diffZ = L_z-L_z_estimated
        if testing:

            plt.figure()
            plt.title("diffZ")
            plt.imshow(diffZ[:20,:20], vmin=-20, vmax=20)
            plt.colorbar()
            plt.xticks(range(20))
            plt.yticks(range(20))
            plt.xlabel("follower")
            plt.ylabel("leader")

        prop["Z"] = probability(
            diffZ,
            mu["Z"],
            sigma["Z"],
            delta["Z"]
        )
    else:
        prop["Z"] = 1.
    
    # particle camera Y position difference
    if "Y" in sigma.keys():
        fyCenter = (follower1D.position_upperLeft.sel(dim2D="y") +
                    (follower1D.Droi.sel(dim2D="y")/2))
        lyCenter = (leader1D.position_upperLeft.sel(dim2D="y") +
                    (leader1D.Droi.sel(dim2D="y")/2))

        diffY = (np.array([fyCenter.values]) -
                 np.array([lyCenter.values]).T)
        prop["Y"] = probability(
            diffY,
            mu["Y"],
            sigma["Y"],
            delta["Y"]
        )
    else:
        prop["Y"] = 1.

    # particle height difference
    if "H" in sigma.keys():
        diffH = (np.array([follower1D.Droi.sel(dim2D="y").values]) -
                 np.array([leader1D.Droi.sel(dim2D="y").values]).T)

        prop["H"] = probability(
            diffH,
            mu["H"],
            sigma["H"],
            delta["H"]
        )
    else:
        prop["H"] = 1.

    # capture_time difference
    if "T" in sigma.keys():

        diffT = (np.array([follower1D.capture_time.values]) -
                 np.array([leader1D.capture_time.values]).T).astype(int)*1e-9
        prop["T"] = probability(
            diffT,
            mu["T"],
            sigma["T"],
            delta["T"]
        )
    else:
        prop["T"] = 1.
    
    # capture_id difference
    if "I" in sigma.keys():

        diffI = (np.array([follower1D.capture_id.values]) -
                 np.array([leader1D.capture_id.values]).T)
        prop["I"] = probability(
            diffI,
            mu["I"],
            sigma["I"],
            delta["I"]
        )
    else:
        prop["I"] = 1.

    # estimate joint probability
    propJoint = prop["Y"]*prop["T"]*prop["H"]*prop["I"]*prop["Z"]

    if testing:
        for k in prop.keys():
            if type(prop[k]) is not float:
                plt.figure()
                plt.title(k)
                plt.imshow(prop[k][:20,:20])
                plt.xticks(range(20))
                plt.yticks(range(20))
                plt.xlabel("follower")
                plt.ylabel("leader")
        plt.figure()
        plt.title(k)
        plt.imshow(propJoint[:20,:20])
        plt.xticks(range(20))
        plt.yticks(range(20))
        plt.xlabel("follower")
        plt.ylabel("leader")

    matchedParticles = {}
    matchedProbabilities = {}

    # try to solve this from both perspectives
    for camera, prop1, dat2 in zip(
        [config["leader"], config["follower"]], 
        [propJoint, propJoint.T], 
        [leader1D, follower1D]
    ):

        matchedParticles[camera] = np.argsort(
            prop1, axis=1)[:, -maxMatches:][:, ::-1]
        matchedProbabilities[camera] = np.sort(
            prop1, axis=1)[:, -maxMatches:][:, ::-1]

        matchedParticles[camera] = xr.DataArray(matchedParticles[camera], coords=[range(
            len(dat2.fpid)), range(matchedParticles[camera].shape[1])], dims=["fpidII", 'match'])
        matchedProbabilities[camera] = xr.DataArray(matchedProbabilities[camera], coords=[range(
            len(dat2.fpid)), range(matchedParticles[camera].shape[1])], dims=["fpidII", 'match'])

    del propJoint, prop

    for reverseFactor in [1, -1]:
        cam1, cam2 = [config["leader"], config["follower"]][::reverseFactor]

        matchedParticles[cam1] = matchedParticles[cam1].where(
            matchedProbabilities[cam1] > minProp)
        matchedProbabilities[cam1] = matchedProbabilities[cam1].where(
            matchedProbabilities[cam1] > minProp)
        
        for kk in range(maxMatches):
            u, c = np.unique(
                matchedParticles[cam1][:, 0], return_counts=True)
            doubleCounts = u[np.where(c > 1)[0]]
            doubleCounts = doubleCounts[np.isfinite(doubleCounts)]
            if len(doubleCounts) != 0:
                # print(
                    # cam1, "particles have been matched twice, fixing", kk)
                matchedParticles[cam1], matchedProbabilities[cam1] = removeDoubleCounts(
                    matchedParticles[cam1], 
                    matchedProbabilities[cam1], 
                    doubleCounts
                )
            else:
                break

        u, c = np.unique(
            matchedParticles[cam1][:, 0], return_counts=True)
        doubleCounts = u[np.where(c > 1)[0]]
        doubleCounts = doubleCounts[np.isfinite(doubleCounts)]

        assert len(
            doubleCounts) == 0, "%s particles have still been matched twice" % cam1

    for reverseFactor in [1, -1]:
        cam1, cam2 = [config["leader"],
                      config["follower"]][::reverseFactor]
        matchedParticles[cam1] = matchedParticles[cam1][:, 0]
        matchedProbabilities[cam1] = matchedProbabilities[cam1][:, 0]

        matchedParticles[cam1] = matchedParticles[cam1].dropna(
            'fpidII')
        matchedProbabilities[cam1] = matchedProbabilities[cam1].dropna(
            'fpidII')

    if np.all([len(v) == 0 for v in matchedParticles.values()]):
        noMatches = True
        print("no matched particles")
        return None

    cam1, cam2 = [config["leader"], config["follower"]]

    pairs1 = set(zip(
        matchedParticles[cam1].fpidII.values, matchedParticles[cam1].values.astype(int)))
    pairs2 = set(zip(matchedParticles[cam2].values.astype(
        int), matchedParticles[cam2].fpidII.values))

    disputedPairs = pairs1 - pairs2
    
    # sort pairs together
    dats = []
    dats.append(leader1D.isel(
        fpid=matchedParticles[config["leader"]].fpidII.values.astype(int)))
    dats.append(follower1D.isel(
        fpid=matchedParticles[config["leader"]].values.astype(int)))

    for dd, d1 in enumerate(dats):

        pid = deepcopy(d1.pid.values)
        file_starttime = deepcopy(d1.file_starttime.values)
        d1 = d1.rename(fpid='pair_id')
        d1 = d1.assign_coords(
            pair_id=np.arange(
                len(matchedParticles[config["leader"]].fpidII)
                ) + indexOffset
            )

        d1["pid"] = xr.DataArray(pid, coords=[d1.pair_id])
        d1["file_starttime"] = xr.DataArray(file_starttime, coords=[d1.pair_id])
        dats[dd] = d1

    matchedDat = xr.concat(dats, dim='camera')
    matchedDat = matchedDat.assign_coords(
        camera=[config["leader"], config["follower"]])
    # add propabilities
    matchedDat["matchScore"] = xr.DataArray(
        matchedProbabilities[config["leader"]
                             ].values.astype(np.float32),
        coords=[matchedDat.pair_id]
    )


    # estimate new offsets, potentially for the next file

    new_mu = {}
    new_sigma= {}
    
    if len(matchedDat.pair_id) >= minNumber4Stats:
        yCenter = (matchedDat.position_upperLeft.sel(dim2D='y') +
                   (matchedDat.Droi.sel(dim2D="y")/2))
        di = yCenter.diff('camera').values
        new_sigma["Y"] = bn.nanstd(di)
        new_mu["Y"] = bn.nanmedian(di)

        di = matchedDat.Droi.sel(
            dim2D='y').diff("camera").values
        new_sigma["H"] = bn.nanstd(di)
        new_mu["H"] = bn.nanmedian(di)

        di = matchedDat.capture_time.diff('camera').values
        di = di[np.isfinite(di)].astype(int)*1e-9
        new_sigma["T"] = bn.nanstd(di)
        new_mu["T"] = bn.nanmedian(di)

        di = matchedDat.capture_id.diff('camera').values
        new_sigma["I"] = bn.nanstd(di)
        new_mu["I"] = bn.nanmedian(di)

        #print(f"{len(matchedDat.pair_id)} matches found. ")
        # print(" match coefficients, ",new_mu)
    else:
        print(f"{len(matchedDat.pair_id)} matches found. Setting match coefficients to NAN")
        new_sigma["Y"] = new_mu["Y"] = new_sigma["H"] = new_mu["H"] = np.nan
        new_sigma["T"] = new_mu["T"] = new_sigma["T"] = new_mu["T"] = np.nan
    
    new_sigma = pd.Series(new_sigma)
    new_mu = pd.Series(new_mu)

    return matchedDat, disputedPairs, new_sigma, new_mu


def get3DPosition(leaderDat, followerDat, config):
    F_z = (followerDat.position_upperLeft.sel(dim2D="y") +
                (followerDat.Droi.sel(dim2D="y")/2)).values
    F_y = (followerDat.position_upperLeft.sel(dim2D="x") +
                (followerDat.Droi.sel(dim2D="x")/2)).values
    L_x = (leaderDat.position_upperLeft.sel(dim2D="x") +
                (leaderDat.Droi.sel(dim2D="x")/2)).values
    L_z = (leaderDat.position_upperLeft.sel(dim2D="y") +
                (leaderDat.Droi.sel(dim2D="y")/2)).values

    #watch out, right hand coordinate system! 
    F_y = config.frame_width - F_y

    return L_x, L_z, F_y, F_z


def addPosition(matchedDat, rotate, rotate_err, config):
    '''
    add postion variable to match dataset based on retrieved rotation parameters
    '''

    L_x, L_z, F_y, F_z = get3DPosition(matchedDat.sel(camera=config.leader), matchedDat.sel(camera=config.follower), config)

    # Fz_estimated = calc_Fz(rotate["phi"], rotate["theta"], rotate["Ofz"], Lx, Lz, Fy)

    L_z_estimated = calc_L_z_withOffsets(L_x, F_y, F_z, **rotate)

    matchedDat["dim3D"] = ["x", "y", "z", "z_rotated"]
    matchedDat["position_3D"] = xr.DataArray([L_x, F_y, L_z, L_z_estimated], coords=[matchedDat.dim3D, matchedDat.pair_id] )

    nid = len(matchedDat.pair_id)
    matchedDat["camera_rotation"] = np.array(["mean", "err"])
    for k in rotate.keys():
        matchedDat[k] = xr.DataArray(np.ones((nid,2))*np.array([rotate[k], rotate_err[k]]), coords=[matchedDat.pair_id, matchedDat["camera_rotation"] ] )

    return matchedDat



def doMatchSlicer(leader1D, follower1D, sigma, mu, delta, config, rotate, minProp=1e-10, maxMatches=100, minNumber4Stats=10, chunckSize = 700, testing=False):

    '''
    doMatch with slicing  to make sure data fits into memory
    Also, smaller chunks are computationally much more efficient, optimum appears to be around 500 for 
    a file with 50.000 particles but we use 1000 to avoid double matched particles at the gaps

    '''


    #short cut for small data sets
    if (len(leader1D.fpid) < chunckSize) or (len(follower1D.fpid) < chunckSize):
        return doMatch(leader1D, follower1D, sigma, mu, delta, config, rotate, minProp=minProp, maxMatches=maxMatches, minNumber4Stats=minNumber4Stats, testing=testing)

    # ok it is too long... 
    matchedDat = []
    new_sigma = []
    new_mu = []
    disputedPairs = []

    indexOffset = 0

    
    JJs = np.linspace(0, len(leader1D.fpid), len(leader1D.fpid)//chunckSize+1, dtype=int)

    print(f"slicing data into {len(JJs)-1} pieces")
    for ii, jj in tqdm(zip(JJs[:-1], JJs[1:]), total=len(JJs)-1):

        leader1DSlice = leader1D.isel(fpid=slice(ii,jj))
        follower1DSlice = tools.cutFollowerToLeader(leader1DSlice, follower1D)

        if len(follower1DSlice.fpid) == 0:
            res = None
            print("No follower data remains")
            continue

        res = doMatch(leader1DSlice, follower1DSlice, sigma, mu, delta, config, rotate, minProp=minProp, maxMatches=maxMatches, minNumber4Stats=minNumber4Stats, indexOffset=indexOffset)
        if res is not None:

            matchedDat1, disputedPairs1, new_sigma1, new_mu1 = res
            matchedDat.append(matchedDat1)
            indexOffset = matchedDat1.pair_id[-1].values +1
            disputedPairs += list(disputedPairs1)
            new_sigma.append(new_sigma1)
            new_mu.append(new_mu1)

    if len(matchedDat) > 0:

        new_sigma = pd.concat(new_sigma, axis=1).mean(axis=1)
        new_mu = pd.concat(new_mu, axis=1).mean(axis=1)
        matchedDat = xr.concat(matchedDat, dim="pair_id")
        return matchedDat, disputedPairs, new_sigma, new_mu
    else:
        print("doMatchSlicer: nothing matched")
        return None

def matchParticles(fnameLv1Detect, config, 
    y_cov_diag = 1.65**2, version=__version__, chunckSize=1000, 
    rotate="config", rotate_err="config", maxDiffMs = "config", 
    rotationOnly=False, nPoints=500, sigma = {
            "Z" : 1.7, # estimated from OE results
            "H" : 1.2, # estimated from OE results
            "I" : .01,
        },
        nSamples4rot = 300,
        minSamples4rot = 100,
        testing=False,
        minDMax4rot=0,
        singleParticleFramesOnly=False,
        doRot = False,
    ):

    if type(config) is str:
        config = tools.readSettings(config)

    ffl1 = files.FilenamesFromLevel(fnameLv1Detect, config)
    fname1Match = ffl1.fname["level1match"]

    ffl1.createDirs()

    matchedDat = None
    matchedDat4Rot = None
    rotate_time = None

    if not doRot:
        # get rotation estimates and add to config instead of etimating them
        fnameMetaRotation = ffl1.fname["metaRotation"]
        metaRotationDat = xr.open_dataset(fnameMetaRotation)
        config = tools.rotXr2dict(metaRotationDat, config)

    if np.any(rotate == "config"):
        rotate, rotate_time = tools.getPrevRotationEstimate(ffl1.datetime64, "transformation", config)
    assert len(rotate) != 0
    if np.any(rotate_err == "config"):
        rotate_err, rotate_time = tools.getPrevRotationEstimate(ffl1.datetime64, "transformation_err", config)
    assert len(rotate_err) != 0

    print(f"opening {fnameLv1Detect}")
    try:
        leader1D = tools.open_mflevel1detect(fnameLv1Detect, config) #with fixes
    except AssertionError as e:
        print("tools.open_mflevel1detect leader FAILED")
        error = str(e)
        print(error)

        if not rotationOnly:
            with open('%s.broken.txt' % fname1Match, 'w') as f:
                f.write("tools.open_mflevel1detect(fnameLv1Detect, config)")
                f.write("\n")
                f.write(error)
        return fname1Match, np.nan, None, None

    if leader1D is None:
        if not rotationOnly:
            with open('%s.nodata' % fname1Match, 'w') as f:
                f.write(f"no leader data in {fnameLv1Detect}")
        print(f"no leader data in {fnameLv1Detect}")
        return fname1Match, None, None, None

    print(len(leader1D.pid))

    if (leader1D is None):
        if not rotationOnly:
            with open('%s.nodata' % fname1Match, 'w') as f:
                f.write(f"no leader data in {fnameLv1Detect}")
        print(f"no leader data in {fnameLv1Detect}")
        return fname1Match, None, None, None

    if (len(leader1D.pid)<=1):
        if not rotationOnly:
            with open('%s.nodata' % fname1Match, 'w') as f:
                f.write(f"only one particle in  {fnameLv1Detect}")
        print(f"only one particle in {fnameLv1Detect}")
        return fname1Match, None, None, None


    file_starttime = leader1D.file_starttime[0].values

    #prevFile = ffl1.prevFile2("level1match")
    #if prevFile is not None:
        #prevFile = prevFile.replace("level1detect","level1match")


    # if np.any( rotate =="previous") and (prevFile is not None) and os.path.isfile(prevFile):
    #     print(f"opening prevFile for previous rotation {prevFile}")
    #     prevDat3 = xr.open_dataset(prevFile)

    #     rotate = prevDat3.isel(pair_id=0, =0)[["camera_phi", "camera_theta", "camera_Ofz"]].to_pandas()
    #     rotate_err = prevDat3.isel(pair_id=0, camera_rotation=1)[["camera_phi", "camera_theta", "camera_Ofz"]].to_pandas()
    #     prevDat3.close()
    # else:
    #     print("take default values for previous rotation:")
    #     print(rotate_default)
    # rotate = pd.Series(rotate_default)
    # rotate_err = pd.Series(rotate_err_default)

    fnames1F = ffl1.filenamesOtherCamera(graceInterval=-1, level="level1detect")
    fnames1FRAW = ffl1.filenamesOtherCamera(graceInterval=-1, level="level0txt")
    if len(fnames1FRAW) !=  len(fnames1F):
        print(f"no follower data for {fnameLv1Detect} processed YET")
        print(fnames1F)
        print(fnames1FRAW)
        return fname1Match, None, None, None
    if len(fnames1F) == 0:
        if not rotationOnly:
            with open('%s.nodata' % fname1Match, 'w') as f:
                f.write(f"no follower data for {fnameLv1Detect}")
        print(f"no follower data for {fnameLv1Detect}")
        return fname1Match, None, None, None


    lEvents = ffl1.fname.metaEvents
    fEvents= np.unique([files.FilenamesFromLevel(f, config).fname.metaEvents for f in fnames1F])

    fEvents = xr.open_mfdataset(fEvents).load()

    lEvents = xr.open_dataset(lEvents)


    start = leader1D.capture_time[0].values - np.timedelta64(2,"s")
    end = leader1D.capture_time[-1].values + np.timedelta64(2,"s")
    print(f"opening {fnames1F}")
    try:
        follower1DAll = tools.open_mflevel1detect(fnames1F, config, start=start, end=end) #with foxes
    except Exception as e:
        log.error(tools.concat("tools.open_mflevel1detect follower FAILED"))
        error = str(e)
        print(error)

        if not rotationOnly:
            with open('%s.broken.txt' % fname1Match, 'w') as f:
                f.write("tools.open_mflevel1detect(fnames1F, config)")
                f.write("\n")
                f.write(error)
        return fname1Match, np.nan, None, None

    leader1D = tools.removeBlockedData(leader1D, lEvents)
    follower1DAll = tools.removeBlockedData(follower1DAll, fEvents)

    if follower1DAll is None:
        if not rotationOnly:
            with open('%s.nodata' % fname1Match, 'w') as f:
                f.write(f"no follower data after removal of blocked data {fname1Match}")
        print(f"no follower data after removal of blocked data {fname1Match}")

        return fname1Match, None, None, None

    if leader1D is None:
        if not rotationOnly:
            with open('%s.nodata' % fname1Match, 'w') as f:
                f.write(f"no leader data after removal of blocked data {fname1Match}")
        print(f"no leader data after removal of blocked data {fname1Match}")
        return fname1Match, None, None, None

    #try to figure out when follower was restarted in leader time period
    followerRestartedII = np.where(
        [(str(e).startswith("start") or str(e).startswith("launch")) for e in fEvents.event.values]
        )[0]
    followerRestarted = fEvents.file_starttime[followerRestartedII].values


    timeBlocks = np.concatenate((
        follower1DAll.capture_time.values[:1], 
        followerRestarted, 
        follower1DAll.capture_time.values[-1:]
        ))
    timeBlocks = np.sort(timeBlocks)

    leaderMinTime = leader1D.file_starttime.min() - np.timedelta64(1,"s")
    leaderMaxTime = max(leader1D.capture_time.max(), leader1D.record_time.max()) + np.timedelta64(1,"s")

    matchedDats = []

    # lopp over all follower segments seperated by camera restarts
    for tt, (FR1, FR2) in enumerate(zip(timeBlocks[:-1], timeBlocks[1:])):
        print(tt+1, "of", len(timeBlocks) - 1, "slice for follower restart",FR1, FR2)
        if (FR1 < leaderMinTime) and (FR2 < leaderMinTime):
            log.info(tools.concat("CONTINUE, slice for follower restart",tt, FR1, FR2, "before leader time range", leaderMinTime.values))
            continue
        if (FR1 > leaderMaxTime) and (FR2 > leaderMaxTime):
            log.info(tools.concat("CONTINUE, slice for follower restart",tt, FR1, FR2, "after leader time range", leaderMaxTime.values))
            continue
        
        # the 2nd <= is on purpose because it is required if there is no restart. if there is a restart, there is anaway nod ata exactly at that time
        TIMES = (FR1 <= follower1DAll.capture_time.values) & (follower1DAll.capture_time.values  <= FR2)
        if np.sum(TIMES) == 0:
            log.warning(tools.concat("CONTINUE, no follower data"))
            continue

        # TIMES = REGEX nach  file_starttime
        follower1D = follower1DAll.isel(fpid=TIMES)

        if "makeCaptureTimeEven" in config.dataFixes:
            #does not make sense for leader
            # redo capture_time based on first time stamp...
            try: 
                follower1D = fixes.makeCaptureTimeEven(follower1D, config, dim="fpid")
            except AssertionError  as e: 
                print("fixes.makeCaptureTimeEven FAILED")
                print(str(e))
                if not rotationOnly:
                    if np.sum(TIMES) <= 20:
                        print(f"so little data {np.sum(TIMES)} ignore it!")
                        continue
                    else:
                        if not rotationOnly:
                            with open('%s.broken.txt' % fname1Match, 'w') as f:
                                f.write("fixes.makeCaptureTimeEven FAILED")
                                f.write("\n")
                                f.write(str(e))
                        print("fixes.makeCaptureTimeEven FAILED")
                        return fname1Match, None, None, None

                else:
                    continue

        assert np.all(np.diff(follower1D.capture_id) >= 0), "follower camera reset detected"

        if maxDiffMs == "config":
            maxDiffMs = 1000/config.fps/2
        try:
            captureIdOffset1, nMatched1 = tools.estimateCaptureIdDiffCore(leader1D, follower1D, "fpid", maxDiffMs=maxDiffMs, nPoints=nPoints, timeDim="capture_time")
        except Exception as e:
            captureIdOffset1 = nMatched1 = -99
            error1 = str(e)
        try:
            captureIdOffset2, nMatched2 = tools.estimateCaptureIdDiffCore(leader1D, follower1D, "fpid", maxDiffMs=maxDiffMs, nPoints=nPoints, timeDim="record_time")
        except Exception as e: 
            captureIdOffset2 = nMatched2 = -99
            error2 = str(e)

        if nMatched2 == nMatched1 == -99:
            print("tools.estimateCaptureIdDiff FAILED")
            print(error1)
            print(error2)
            if not rotationOnly:

                with open('%s.broken.txt' % fname1Match, 'w') as f:
                    f.write("tools.estimateCaptureIdDiff(ffl1, config, graceInterval=2)")
                    f.write("\n")
                    f.write(error1)
                    f.write(error2)
            continue

        if (nMatched2 <= 1) and (nMatched1 <= 1):
            # if not rotationOnly:
            #     with open(f"{fname1Match}.nodata", "w") as f:
            #         f.write("NOT ENOUGH DATA")
            print("NOT ENOUGH DATA", fname1Match, tt, FR1, FR2)
            continue


        # In theory, capture time is much better, but there are cases were it is off. Try to identify them by chgecking whether record_time yielded more matches.
        if nMatched2 > nMatched1:
            captureIdOffset = captureIdOffset2
            print(f"Taking offset from record_time {(captureIdOffset2, nMatched2)} intead of capture_time {(captureIdOffset1, nMatched1)}")
        else:
            captureIdOffset = captureIdOffset1

        mu = {
            "Z" : 0,
        #                 "Y" : 34.3,
            "H" : 0,
            "T" :  0,
            "I" : captureIdOffset,
        }
        delta = {
            "Z" : 0.5, #0.5 because center is considered
            "Y" : 0.5, #0.5 because center is considered
            "H" : 1,
            "T" : 1/config.fps,
            "I" : 1,
        }

        #figure out how cameras ae rotated, first prepare data
        dataTruncated4rot=False
        if doRot:
            rotates = []
            
            # for estiamting rotation, we wo not need the full data set, use subset to speed up caluculation

            if (minDMax4rot > 0):
                filt = (leader1D.Dmax>minDMax4rot).values
                print("DMax filter leader:", minDMax4rot, np.sum(filt)/len(leader1D.fpid) * 100,"%")
                leader1D4rot = leader1D.isel(fpid=filt)
            else:
                leader1D4rot = leader1D.copy()


            if (minDMax4rot > 0):
                filt = (follower1D.Dmax>minDMax4rot).values
                print("DMax filter follower:", minDMax4rot, np.sum(filt)/len(follower1D.fpid) * 100,"%")
                follower1D4rot = follower1D.isel(fpid=filt)
            else:
                follower1D4rot = follower1D.copy()


            # to get rotation coefficients, using frames with only a single particle is helpful!
            if singleParticleFramesOnly:
                un, ii, counts = np.unique(leader1D4rot.capture_time, return_index=True, return_counts=True)
                leader1D4rot = leader1D4rot.isel(fpid=ii[counts==1])

                un, ii, counts = np.unique(follower1D4rot.capture_time, return_index=True, return_counts=True)
                follower1D4rot = follower1D4rot.isel(fpid=ii[counts==1])


            if len(leader1D4rot.fpid) > nSamples4rot*10: #assuming we have about 10 times more particles outside the obs volume
                leader1D4rot = leader1D4rot.isel(fpid=slice(nSamples4rot*10))
                dataTruncated4rot = True
            elif len(leader1D4rot.fpid) < minSamples4rot:
                print("not enough leader data to estimate rotation")
                doRot = False


            if len(follower1D4rot.fpid) > nSamples4rot*10:
                follower1D4rot = follower1D4rot.isel(fpid=slice(nSamples4rot*10))
                dataTruncated4rot = True
            elif len(follower1D4rot.fpid) < minSamples4rot:
                print("not enough follower data to estimate rotation")
                doRot = False

        # iterate to rotation coeeficients in max. 20 steps
        if doRot:
            for ii in range(20):
                print("rotation coefficients iteration", ii, "of 20")
                # in here is all the magic
                res = doMatchSlicer(
                    leader1D4rot, follower1D4rot, sigma, mu, delta, config, rotate, chunckSize=1e6,
                testing=testing)
                if res is None:
                    print("doMatchSlicer 4 rot failed")
                    continue 
                matchedDat, disputedPairs, new_sigma, new_mu = res

                if len(matchedDat.pair_id) >= minSamples4rot:

                    matchedDat4Rot = deepcopy(matchedDat)
        #                 matchedDat4Rot = matchedDat4Rot.isel(pair_id=(matchedDat4Rot.matchScore>minMatchScore4rot))
                    matchedDat4Rot = matchedDat4Rot.isel(pair_id=sorted(np.argsort(matchedDat4Rot.matchScore)[-nSamples4rot:]))
            
                    x_ap = rotate
                    x_cov_diag = (rotate_err*10)**2
                    try: 
                        rotate, rotate_err, dgf_x = retrieveRotation(matchedDat4Rot, x_ap, x_cov_diag, y_cov_diag, config, verbose=True)
                    except AssertionError as e:
                        print(f"pyOE error, taking previous values.")
                        print(str(e))
                        break

                    print("MATCH",ii, matchedDat.matchScore.mean().values, )
                    print("ROTATE",ii,"\n", rotate ,"\n", "error","\n", rotate_err, "\n", "dgf","\n", dgf_x )
                    rotates.append(rotate)

                    if ii > 0:
                        # if the change of the coefficients is smaller than their 1std errors for all of them, stop
                        if np.all(np.abs(rotates[ii-1]-rotate) < rotate_err):
                            print("interupting loop")
                            print(rotate)
                            break
                else:
                    print(f"{len(matchedDat.pair_id)} pairs is not enough data to estimate rotation, taking previous values.")

                    break
        else:
            print(f"taking provided data for rotation from {rotate_time}")


        if rotationOnly:
            continue
            # return fname1Match, matchedDat4Rot, rotate, rotate_err

        if dataTruncated4rot or (not doRot):
            print("final doMatch")
        
            # do it again because we did not consider everything before
            res = doMatchSlicer(
                leader1D, follower1D, sigma, mu, delta, config, rotate, chunckSize=chunckSize,
            testing=testing)

            if res is None:
                print("doMatchSlicer failed")

                continue
            matchedDat, disputedPairs, new_sigma, new_mu = res
            print("doMatch ok", len(leader1D.fpid), len(follower1D.fpid))
        else:
            # matchDat is alread final because it was not truncated
            pass



        if (matchedDat is not None) and len(matchedDat.pair_id) > 0:

            #add position with final roation coeffs. 
            matchedDat = addPosition(matchedDat, rotate, rotate_err, config)

            # fixed values would lead to confusion, so stay with original ones
            if "captureIdOverflows" in config.dataFixes:
                matchedDat = fixes.revertIdOverflowFix(matchedDat)

            matchedDats.append(matchedDat)

        # end loop camera restart

    if rotationOnly:
        return fname1Match, matchedDat4Rot, rotate, rotate_err


    if len(matchedDats) == 0:
        with open(f"{fname1Match}.nodata", "w") as f:
            f.write("no data")
        print("NO DATA", fname1Match)
        return fname1Match, matchedDats, rotate, rotate_err

    elif len(matchedDats) == 1:
        # easy case
        matchedDats = matchedDat
    else:
        for ii in range(len(matchedDats)):
            del matchedDats[ii]["pair_id"]
        matchedDats = xr.concat(matchedDats, dim="pair_id")
        matchedDats["pair_id"] = range(len(matchedDats["pair_id"]))

    matchedDats = tools.finishNc(matchedDats)

    matchedDats["fitMethod"] = matchedDats.fitMethod.astype("U30")
    matchedDats["dim2D"] = matchedDats.dim2D.astype("U2")
    matchedDats["dim3D"] = matchedDats.dim3D.astype("U2")
    matchedDats["camera"] = matchedDats.camera.astype("U30")
    matchedDats["camera_rotation"] = matchedDats.camera_rotation.astype("U30")
    matchedDats.to_netcdf(fname1Match)

    print("DONE", fname1Match, "with", len(matchedDats.pair_id), "particles")

    return fname1Match, matchedDats, rotate, rotate_err


def createLevel1match(case, config, skipExisting=True, version=__version__ ,
    y_cov_diag = 1.65**2,  chunckSize=1000, 
    rotate="config", rotate_err="config", maxDiffMs = "config", 
    rotationOnly=False, nPoints=500, sigma = {
            "Z" : 1.7, # estimated from OE results
            "H" : 1.2, # estimated from OE results
            "I" : .01,
        },
        minDMax4rot = 0,
        nSamples4rot = 300,
        minSamples4rot = 100,
        testing=False,
        completeDaysOnly=False,

    ):

    # find files
    fl = files.FindFiles(case, config.leader, config, version)

    fnames1L = fl.listFilesExt("level1detect")
    if len(fnames1L) == 0:
        print("No leader files", case, config.leader , fl.fnamesPatternExt.level1detect)
        return None

    if completeDaysOnly and not fl.isCompleteL1detect:
        print("NOT COMPLETE YET %i of %i" %
              (len(fl.listFilesExt("level1detect")), len(fl.listFiles("level0txt"))))
        return None



    for fname1L in fnames1L:

        ffl1  = files.FilenamesFromLevel(fname1L, config)
        fname1Match = ffl1.fname["level1match"]

        if fname1L.endswith("broken.txt") or fname1L.endswith("nodata") or fname1L.endswith("notenoughframes"):
            ffl1.createDirs()
            with open(f"{fname1Match}.nodata", "w") as f:
                f.write("no leader data")
            print("NO leader DATA", fname1L)
            continue

        if os.path.isfile(fname1Match) and skipExisting:
            print("SKIPPING", fname1Match)
            continue
        if os.path.isfile('%s.broken.txt' % fname1Match) and skipExisting:
            print("SKIPPING BROKEN", fname1Match)
            continue
        if os.path.isfile('%s.nodata' % fname1Match) and skipExisting:
            print("SKIPPING nodata", fname1Match)
            continue

        try:
            fout, matchedDat, rot, rot_err = matchParticles(fname1L,
                                                    config,
                                                    y_cov_diag=y_cov_diag,
                                                    chunckSize=chunckSize,
                                                    rotate=rotate,
                                                    rotate_err=rotate_err,
                                                    maxDiffMs=maxDiffMs,
                                                    rotationOnly=rotationOnly,
                                                    nPoints=nPoints,
                                                    sigma=sigma,
                                                    minDMax4rot=minDMax4rot,
                                                    nSamples4rot=nSamples4rot,
                                                    minSamples4rot=minSamples4rot,
                                                    testing=testing,
                                                    )


        except (RuntimeError, AssertionError) as e:
            print("matchParticles FAILED", fname1Match)
            print(str(e))
            ffl1.createDirs()
            with open('%s.broken.txt' % fname1Match, 'w') as f:
                f.write("in scripts: matchParticles FAILED")
                f.write("\r")
                f.write(str(e))

def createMetaRotation(case, 
    config, 
    skipExisting=True, 
    version=__version__ ,
    y_cov_diag = 1.65**2,  
    chunckSize=1000, 
    rotate="config",
     rotate_err="config", 
     maxDiffMs = "config",  
     nPoints=500, 
     sigma = {
            "Z" : 1.7, # estimated from OE results
            "H" : 1.2, # estimated from OE results
            "I" : .01,
        },
        minDMax4rot = 10,
        nSamples4rot = 300,
        minSamples4rot = 50,
        testing=False,
        completeDaysOnly=True,
    ):

    # find files
    fl = files.FindFiles(case, config.leader, config, version)

    if len(fl.listFiles("metaEvents")) == 0:
        print("No event files", case, config.leader , fl.fnamesPattern.metaEvents)
        return None, None

    eventFile = fl.listFiles("metaEvents")[0]
    fflM  = files.FilenamesFromLevel(eventFile, config)
    fnameMetaRotation = fflM.fname["metaRotation"]

    if os.path.isfile(fnameMetaRotation) and skipExisting:
        print("SKIPPING", fnameMetaRotation)
        return None, None

    eventDat = xr.open_dataset(eventFile)
    nRecordedFiles = sum(eventDat.event == "newfile") + sum(eventDat.event == "brokenfile")
    #if not closed here, weired segfoults happen later when opend a 2nd times
    eventDat.close()

    fnames1L = fl.listFilesExt("level1detect")
    if (len(fnames1L) == 0) and (nRecordedFiles > 0):
        print("No leader files", case, config.leader , fl.fnamesPatternExt.level1detect)
        return None, None

    if completeDaysOnly and not fl.isCompleteL1detect:
        print("NOT COMPLETE YET %i of %i" %
              (len(fl.listFilesExt("level1detect")), len(fl.listFiles("level0txt"))))
        return None, None


    fl.createDirs()
    metaRotation = []

    # add previous rotations to config file so that most recent result can be used
    prevFile = fflM.prevFile2("metaRotation", maxOffset=np.timedelta64(24,"h"))

    if (datetime.datetime.strptime(config.start,"%Y-%m-%d") != fl.datetime.date()) and (prevFile is None):
        _, prevTime = tools.getPrevRotationEstimate(fflM.datetime64, "transformation", config)
        deltaT = (fflM.datetime64- prevTime)
        if deltaT > np.timedelta64(2,"D"):
            print(f"Skipping, no previous data found! data in config file {round(deltaT/np.timedelta64(1,'h'))}h old which is more than 48h", fnameMetaRotation)
            return None, None

    if prevFile is not None:
        prevDat = xr.open_dataset(prevFile)
        config = tools.rotXr2dict(prevDat, config)

    # add values from previous estimante to the  file so that it always contains at least one data point!
    rotate_default, prevTime = tools.getPrevRotationEstimate(fflM.datetime64, "transformation", config)
    assert len(rotate_default) != 0
    rotate_err_default, prevTime = tools.getPrevRotationEstimate(fflM.datetime64, "transformation_err", config)
    assert len(rotate_err_default) != 0
    metaRotationDf = {}
    for k in rotate_default.keys():
        metaRotationDf[k] = xr.DataArray(np.ones((1,2))*np.array([rotate_default[k], rotate_err_default[k]]), dims=["file_starttime", "camera_rotation"], coords=[[prevTime], np.array(["mean", "err"]) ] )
    metaRotation.append(xr.Dataset(metaRotationDf))

    rotate_default = pd.Series(rotate_default)
    rotate_err_default = pd.Series(rotate_err_default)

    # if os.path.isfile('%s.broken.txt' % fnameMetaRotation) and skipExisting:
    #     print("SKIPPING BROKEN", fnameMetaRotation)
    #     continue
    # if os.path.isfile('%s.nodata' % fnameMetaRotation) and skipExisting:
    #     print("SKIPPING nodata", fnameMetaRotation)
        # continue

    for fname1L in fnames1L:

        ffl1  = files.FilenamesFromLevel(fname1L, config)

        if fname1L.endswith("broken.txt") or fname1L.endswith("nodata") or fname1L.endswith("notenoughframes"):
            ffl1.createDirs()
            print("NO leader DATA", fname1L)
            continue

        try:
            _, _, rot, rot_err = matchParticles(fname1L,
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
                                                    doRot = True,
)

            # metaRotation.append(xr.DataArray([rot], ))
            # metaRotationErr.append(xr.DataArray())

        except (RuntimeError, AssertionError) as e:
            print("matchParticles FAILED", fnameMetaRotation)
            print(str(e))


        else:
            if rot is not None:

                metaRotation1 = {}
                for k in rot.keys():
                    metaRotation1[k] = xr.DataArray(np.ones((1,2))*np.array([rot[k], rot_err[k]]), dims=["file_starttime", "camera_rotation"], coords=[[ffl1.datetime64], np.array(["mean", "err"]) ] )
                metaRotation.append(xr.Dataset(metaRotation1))

                #update defautl
                rotate_default = rot
                rotate_err_default = rot_err

    if len(metaRotation) > 0:
        metaRotation = xr.concat(metaRotation, dim="file_starttime")



    metaRotation = tools.finishNc(metaRotation)
    metaRotation.to_netcdf(fnameMetaRotation)
    print("DONE", fnameMetaRotation)

    return metaRotation, fnameMetaRotation
