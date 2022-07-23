# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats
import pandas as pd
#import av
import bottleneck as bn
import pyOptimalEstimation as pyOE

import logging
log = logging.getLogger()

from copy import deepcopy

from . import __version__
from . import tools

deltaY = deltaH = deltaI = 1.


def calc_Fz(phi, theta, Ofz, Lx, Lz, Fy):
    '''
    Parameters
    ----------
    phi : 
        Follower roll
    theta :
        Follower pitch
    Ofz :
        Offset Follower z
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

def forward(x, Lx=None, Lz=None, Fy=None):
    '''
    forward model for pyOptimalEstimation
    '''
    y = calc_Fz(x.phi, x.theta, x.Ofz, Lx, Lz, Fy)
    y = pd.Series(y, index=np.array(range(len(y))))
    return y

def retrieveRotation(dat3, x_ap, x_cov_diag, y_cov_diag, verbose=False):
    '''
    apply Optimal Estimation to retrieve rotation of cameras
    '''
    
    nPart = len(dat3.pair_id)

    # Leader & Follower z coordinate
    Lz, Fz = (dat3.roi.sel(ROI_elements="y") +
                        (dat3.roi.sel(ROI_elements="h")/2)).values
    
    # LEader x and Follower y coordinate
    Lx, Fy = (dat3.roi.sel(ROI_elements="x") +
                        (dat3.roi.sel(ROI_elements="w")/2)).values

    x_vars = ["phi", "theta", "Ofz"]
    y_vars = np.array(range(nPart))

    x_cov = np.identity(len(x_vars)) * np.array(x_cov_diag)
    y_cov = np.identity(nPart) * np.array(y_cov_diag)

    y_obs = Fz

    forwardKwArgs = {"Lz": Lz, "Lx": Lx, "Fy": Fy}

    # create optimal estimation object
    oe = pyOE.optimalEstimation(
        x_vars, x_ap, x_cov, y_vars, y_obs, y_cov, forward,
        forwardKwArgs=forwardKwArgs, verbose=verbose
        )

    oe.doRetrieval()
    return oe.x_op, oe.x_op_err

def probability(x, mu, sigma, delta):

    x = x.astype(float)
    mu = np.float(mu)
    sigma = np.float(sigma)
    delta = np.float(delta)

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


def doMatch(leader1D, follower1D, sigma, mu, delta, config, minProp, maxMatches, minNumber4Stats, rotate):
    '''
    match magic function
    
    minProp: minimal required probability
    maxMatches number of best matches to consider
    minNumber4Stats: min. number of samples to estimate sigmas and mus
    '''
    
    print("using", sigma, mu, delta)  
    
    prop = {}
    
    
    # particle Z position difference in joint coordinate system
    if "Z" in sigma.keys():
        

        Fz = (follower1D.roi.sel(ROI_elements="y") +
                    (follower1D.roi.sel(ROI_elements="h")/2)).values.T
        Lz = (leader1D.roi.sel(ROI_elements="y") +
                    (leader1D.roi.sel(ROI_elements="h")/2)).values
        Fy = (follower1D.roi.sel(ROI_elements="x") +
                    (follower1D.roi.sel(ROI_elements="w")/2)).values.T
        Lx = (leader1D.roi.sel(ROI_elements="x") +
                    (leader1D.roi.sel(ROI_elements="w")/2)).values


        Fz = Fz.reshape((1, len(Fz)))
        Lz = Lz.reshape((len(Lz), 1))
        Fy = Fy.reshape((1, len(Fy)))
        Lx = Lx.reshape((len(Lx), 1))
        
        Fz_estimated = calc_Fz(rotate["phi"], rotate["theta"], rotate["Ofz"], Lx, Lz, Fy)

        diffZ = Fz-Fz_estimated
        
        prop["Z"] = probability(
            diffZ,
            mu["Z"],
            sigma["Z"],
            delta["Z"]
        )
    else:
        prop["Z"] = 1
    
    # particle camera Y position difference
    if "Y" in sigma.keys():
        fyCenter = (follower1D.roi.sel(ROI_elements="y") +
                    (follower1D.roi.sel(ROI_elements="h")/2))
        lyCenter = (leader1D.roi.sel(ROI_elements="y") +
                    (leader1D.roi.sel(ROI_elements="h")/2))

        diffY = (np.array([fyCenter.values]) -
                 np.array([lyCenter.values]).T)
        prop["Y"] = probability(
            diffY,
            mu["Y"],
            sigma["Y"],
            delta["Y"]
        )
    else:
        prop["Y"] = 1

    # particle height difference
    if "H" in sigma.keys():
        diffH = (np.array([follower1D.roi.sel(ROI_elements='h').values]) -
                 np.array([leader1D.roi.sel(ROI_elements='h').values]).T)

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
    print(propJoint.shape, propJoint.dtype)

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
                print(
                    cam1, "particles have been matched twice, fixing", kk)
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
        d1 = d1.assign_coords(pair_id=list(
            range(len(matchedParticles[config["leader"]].fpidII))))

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
        yCenter = (matchedDat.roi.sel(ROI_elements='y') +
                   (matchedDat.roi.sel(ROI_elements="h")/2))
        di = yCenter.diff('camera').values
        new_sigma["Y"] = bn.nanstd(di)
        new_mu["Y"] = bn.nanmedian(di)

        di = matchedDat.roi.sel(
            ROI_elements='h').diff("camera").values
        new_sigma["H"] = bn.nanstd(di)
        new_mu["H"] = bn.nanmedian(di)

        di = matchedDat.capture_time.diff('camera').values
        di = di[np.isfinite(di)].astype(int)*1e-9
        new_sigma["T"] = bn.nanstd(di)
        new_mu["T"] = bn.nanmedian(di)

        di = matchedDat.capture_id.diff('camera').values
        new_sigma["I"] = bn.nanstd(di)
        new_mu["I"] = bn.nanmedian(di)

        print(" match coefficients, ",new_mu)
    else:
        print("setting match coefficients to NAN")
        new_sigma["Y"] = new_mu["Y"] = new_sigma["H"] = new_mu["H"] = np.nan
        new_sigma["T"] = new_mu["T"] = new_sigma["T"] = new_mu["T"] = np.nan
    
    return matchedDat, disputedPairs, new_sigma, new_mu

def addPosition(matchedDat, rotate, rotate_err, config):
    '''
    add postion variable to match dataset based on retrieved rotation parameters
    '''

    Fz = (matchedDat.sel(camera=config.follower).roi.sel(ROI_elements="y") +
                (matchedDat.sel(camera=config.follower).roi.sel(ROI_elements="h")/2)).squeeze()
    Fy = (matchedDat.sel(camera=config.follower).roi.sel(ROI_elements="x") +
                (matchedDat.sel(camera=config.follower).roi.sel(ROI_elements="w")/2)).squeeze()
    Lx = (matchedDat.sel(camera=config.leader).roi.sel(ROI_elements="x") +
                (matchedDat.sel(camera=config.leader).roi.sel(ROI_elements="w")/2)).squeeze()
    Lz = (matchedDat.sel(camera=config.leader).roi.sel(ROI_elements="y") +
                (matchedDat.sel(camera=config.leader).roi.sel(ROI_elements="h")/2)).values

    Fz_estimated = calc_Fz(rotate["phi"], rotate["theta"], rotate["Ofz"], Lx, Lz, Fy)

    matchedDat["position_elements"] = ["x", "y", "z", "z_rotated"]
    matchedDat["position"] = xr.DataArray([Lx, Fy, Fz, Fz_estimated], coords=[matchedDat.position_elements, matchedDat.pair_id] )

    nid = len(matchedDat.pair_id)
    matchedDat["rotation"] = np.array(["mean", "err"])
    matchedDat["phi"] = xr.DataArray(np.ones((nid,2))*np.array([rotate["phi"], rotate_err["phi"]]), coords=[matchedDat.pair_id, matchedDat["rotation"] ] )
    matchedDat["theta"] = xr.DataArray(np.ones((nid,2))*np.array([rotate["theta"], rotate_err["theta"]]), coords=[matchedDat.pair_id, matchedDat["rotation"] ] )
    matchedDat["Ofz"] = xr.DataArray(np.ones((nid,2))*np.array([rotate["Ofz"], rotate_err["Ofz"]]), coords=[matchedDat.pair_id, matchedDat["rotation"] ] )

    return matchedDat



