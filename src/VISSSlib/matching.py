# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import os
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
from . import fixes
from . import files

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


def doMatch(leader1D, follower1D, sigma, mu, delta, config, rotate, minProp=1e-10, minNumber4Stats=10, maxMatches=100, indexOffset=0, testing=False):
    '''
    match magic function
    
    minProp: minimal required probability
    maxMatches number of best matches to consider to select best one
    minNumber4Stats: min. number of samples to estimate sigmas and mus
    '''
    
    # print("using", sigma, mu, delta)  
    print("doMatch", len(leader1D.fpid), len(follower1D.fpid))
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
        prop["Y"] = 1.

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

        print(f"{len(matchedDat.pair_id)} matches found. ")
        # print(" match coefficients, ",new_mu)
    else:
        print(f"{len(matchedDat.pair_id)} matches found. Setting match coefficients to NAN")
        new_sigma["Y"] = new_mu["Y"] = new_sigma["H"] = new_mu["H"] = np.nan
        new_sigma["T"] = new_mu["T"] = new_sigma["T"] = new_mu["T"] = np.nan
    
    new_sigma = pd.Series(new_sigma)
    new_mu = pd.Series(new_mu)

    return matchedDat, disputedPairs, new_sigma, new_mu





def addPosition(matchedDat, rotate, rotate_err, config):
    '''
    add postion variable to match dataset based on retrieved rotation parameters
    '''

    Fz = (matchedDat.sel(camera=config.follower).roi.sel(ROI_elements="y") +
                (matchedDat.sel(camera=config.follower).roi.sel(ROI_elements="h")/2)).values
    Fy = (matchedDat.sel(camera=config.follower).roi.sel(ROI_elements="x") +
                (matchedDat.sel(camera=config.follower).roi.sel(ROI_elements="w")/2)).values
    Lx = (matchedDat.sel(camera=config.leader).roi.sel(ROI_elements="x") +
                (matchedDat.sel(camera=config.leader).roi.sel(ROI_elements="w")/2)).values
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



def doMatchSlicer(leader1D, follower1D, sigma, mu, delta, config, rotate, minProp=1e-10, maxMatches=100, minNumber4Stats=10, chunckSize = 700, testing=False):

    '''
    doMatch with slicing  to make sure data fits into memory
    Also, smaller chunks are computationally much more efficient, optimum appears to be around 500 for 
    a file with 50.000 particles but we use 1000 to avoid double matched particles at the gaps

    '''


    #short cut for small data sets
    if (len(leader1D.fpid) < chunckSize) & (len(follower1D.fpid) < chunckSize):
        return doMatch(leader1D, follower1D, sigma, mu, delta, config, rotate, minProp=minProp, maxMatches=maxMatches, minNumber4Stats=minNumber4Stats, testing=testing)

    # ok it is too long... 
    matchedDat = []
    new_sigma = []
    new_mu = []
    disputedPairs = []

    indexOffset = 0

    
    JJs = np.linspace(0, len(leader1D.fpid), len(leader1D.fpid)//chunckSize+1, dtype=int)

    print(f"slicing data into {len(JJs)-1} pieces")
    for ii, jj in zip(JJs[:-1], JJs[1:]):

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
        new_sigma = np.mean(new_sigma)
        new_mu = np.mean(new_mu)
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
        minDMax4rot=0
    ):

    matchedDat = None
    matchedDat4Rot = None

    if np.any(rotate == "config"):
        rotate_default = config["rotate"]
        assert len(rotate_default) != 0
    else:
        rotate_default = rotate
    if np.any(rotate_err == "config"):
        rotate_err_default = config["rotate_err"]
        assert len(rotate_err_default) != 0
    else:
        rotate_err_default = rotate_err

    ffl1 = files.FilenamesFromLevel(fnameLv1Detect, config)
    fname1Match = ffl1.fname["level1match"]
    ffl1.createDirs()

    print(f"opening {fnameLv1Detect}")
    try:
        leader1D = tools.open_mflevel1detect(fnameLv1Detect, config) #with fixes
    except AssertionError as e:
        print("tools.open_mflevel1detect leader FAILED")
        error = str(e)
        print(error)

        with open('%s.broken.txt' % fname1Match, 'w') as f:
            f.write("tools.open_mflevel1detect(fnameLv1Detect, config)")
            f.write("\n")
            f.write(error)
        return fname1Match, np.nan, None, None

    if leader1D is None:
        with open('%s.nodata' % fname1Match, 'w') as f:
            f.write(f"no leader data in {fnameLv1Detect}")
        print(f"no leader data in {fnameLv1Detect}")
        return fname1Match, None, None, None

    file_starttime = leader1D.file_starttime[0].values
    
    prevFile = ffl1.prevFile("level1detect")
    if prevFile is not None:
        prevFile = prevFile.replace("level1detect","level1match")


    if np.any( rotate =="config") and (prevFile is not None) and os.path.isfile(prevFile):
        print(f"opening prevFile for previous rotation {prevFile}")
        prevDat3 = xr.open_dataset(prevFile)
        rotate = prevDat3.isel(pair_id=0, rotation=0)[["phi", "theta", "Ofz"]].to_pandas()
        rotate_err = prevDat3.isel(pair_id=0, rotation=1)[["phi", "theta", "Ofz"]].to_pandas()
        prevDat3.close()
    else:
        print("take default values for previous rotation", rotate_default)
        rotate = pd.Series(rotate_default)
        rotate_err = pd.Series(rotate_err_default)

    fnames1F = ffl1.filenamesOtherCamera(graceInterval=-1, level="level1detect")
    fnames1FRAW = ffl1.filenamesOtherCamera(graceInterval=-1, level="level0txt")
    if len(fnames1FRAW) !=  len(fnames1F):
        print(f"no follower data for {fnameLv1Detect} processed YET")
        print(fnames1F)
        print(fnames1FRAW)
        return fname1Match, None, None, None
    if len(fnames1F) == 0:
        with open('%s.nodata' % fname1Match, 'w') as f:
            f.write(f"no follower data for {fnameLv1Detect}")
        print(f"no follower data for {fnameLv1Detect}")
        return fname1Match, None, None, None


    lEvents = ffl1.fname.metaEvents
    fEvents= np.unique([files.FilenamesFromLevel(f, config).fname.metaEvents for f in fnames1F])

    lEvents = xr.open_dataset(lEvents)
    fEvents = xr.open_mfdataset(fEvents).load()


    start = leader1D.capture_time[0].values - np.timedelta64(2,"s")
    end = leader1D.capture_time[-1].values + np.timedelta64(2,"s")
    print(f"opening {fnames1F}")
    try:
        follower1DAll = tools.open_mflevel1detect(fnames1F, config, start=start, end=end) #with foxes
    except Exception as e:
        print("tools.open_mflevel1detect follower FAILED")
        error = str(e)
        print(error)

        with open('%s.broken.txt' % fname1Match, 'w') as f:
            f.write("tools.open_mflevel1detect(fnames1F, config)")
            f.write("\n")
            f.write(error)
        return fname1Match, np.nan, None, None

    leader1D = tools.removeBlockedData(leader1D, lEvents)
    follower1DAll = tools.removeBlockedData(follower1DAll, fEvents)

    if follower1DAll is None:
        with open('%s.nodata' % fname1Match, 'w') as f:
            f.write(f"no follower data after reomval of blocked data {fname1Match}")
        print(f"no follower data after reomval of blocked data {fname1Match}")
        return fname1Match, None, None, None

    if leader1D is None:
        with open('%s.nodata' % fname1Match, 'w') as f:
            f.write(f"no leader data after reomval of blocked data {fname1Match}")
        print(f"no leader data after reomval of blocked data {fname1Match}")
        return fname1Match, None, None, None


    #try to figure out when follower was restarted in leader time period
    followerRestartedII = np.where(
        [(str(e).startswith("start") or str(e).startswith("launch")) for e in fEvents.event.values]
        )[0]
    followerRestarted = fEvents.file_starttime[followerRestartedII].values


    if config.site == "mosaic":
        timeBlocks = followerRestarted
    else: #for non-mosaic data: camera  clock is reset for every file, needs to be treated seperately
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
        print(tt, "of", len(timeBlocks) - 1, "slice for follower restart",FR1, FR2)
        if (FR1 < leaderMinTime) and (FR2 < leaderMinTime):
            print("CONTINUE, slice for follower restart",tt, FR1, FR2, "before leader time range", leaderMinTime)
            continue
        if (FR1 > leaderMaxTime) and (FR2 > leaderMaxTime):
            print("CONTINUE, slice for follower restart",tt, FR1, FR2, "after leader time range", leaderMaxTime)
            continue
        
        # the 2nd <= is on purpose because it is required if there is no restart. if there is a restart, there is anaway nod ata exactly at that time
        TIMES = (FR1 <= follower1DAll.capture_time.values) & (follower1DAll.capture_time.values  <= FR2)
        if np.sum(TIMES) == 0:
            print("CONTINUE, no follower data")
            continue

        # TIMES = REGEX nach  file_starttime
        follower1D = follower1DAll.isel(fpid=TIMES)


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

        if (nMatched2 <= 1) and (nMatched1 <= 1):
            with open(f"{fname1Match}.nodata", "w") as f:
                f.write("NOT ENOUGH DATA")
                print("NOT ENOUGH DATA", fname1Match)
            return fname1Match, None, None, None

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
                return fname1Match, np.nan, None, None


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

        #figure out how cameras ae rotated

        rotates = []
        dataTruncated4rot=False
        doRot = True
        # for estiamting rotation, we wo not need the full data set, use subset to speed up caluculation

        if (minDMax4rot > 0):
            filt = (leader1D.Dmax>minDMax4rot).values
            print("DMax filter leader:", minDMax4rot, np.sum(filt)/len(leader1D.fpid) * 100,"%")
            leader1D4rot = leader1D.isel(fpid=filt)
        else:
            leader1D4rot = leader1D.copy()


        if len(leader1D4rot.fpid) > nSamples4rot*10: #assuming we have about 10 times more particles outside the obs volume
            leader1D4rot = leader1D4rot.isel(fpid=slice(nSamples4rot*20))
            dataTruncated4rot = True
        elif len(leader1D4rot.fpid) < minSamples4rot:
            print("not enough leader data to estimate rotation")
            doRot = False

        if (minDMax4rot > 0):
            filt = (follower1D.Dmax>minDMax4rot).values
            print("DMax filter follower:", minDMax4rot, np.sum(filt)/len(follower1D.fpid) * 100,"%")
            follower1D4rot = follower1D.isel(fpid=filt)
        else:
            follower1D4rot = follower1D.copy()


        if len(follower1D4rot.fpid) > nSamples4rot*10:
            follower1D4rot = follower1D4rot.isel(fpid=slice(nSamples4rot*20))
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
                    leader1D4rot, follower1D4rot, sigma, mu, delta, config, rotate, chunckSize=chunckSize,
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
                    rotate, rotate_err = retrieveRotation(matchedDat4Rot, x_ap, x_cov_diag, y_cov_diag)

                    print("MATCH",ii, matchedDat.matchScore.mean().values, )
                    print("ROTATE",ii,"\n", rotate ,"\n", "error","\n", rotate_err)
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
            print(f"not enough data to estimate rotation, taking previous values")


        if rotationOnly:
            return fname1Match, matchedDat4Rot, rotate, rotate_err

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
        testing=False

    ):

    # find files
    fl = files.FindFiles(case, config.leader, config, version)

    fnames1L = fl.listFilesExt("level1detect")
    if len(fnames1L) == 0:
        print("No leader files", case, config.leader , fl.fnamesPatternExt.level1detect)
        return None

    for fname1L in fnames1L:

        ffl1  = files.FilenamesFromLevel(fname1L, config)
        fname1Match = ffl1.fname["level1match"]

        if fname1L.endswith("broken.txt") or fname1L.endswith("nodata") or fname1L.endswith("notenoughframes"):
            ffl1.createDirs()
            with open(f"{fname1Match}.nodata", "w") as f:
                f.write("no leader data")
            print("NO leader DATA", fname1Match)
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
            fout, matchedDat, rot, rot_err = matchParticles(fname1L, config,
                    y_cov_diag = y_cov_diag,  chunckSize=chunckSize, 
    rotate=rotate, rotate_err=rotate_err, maxDiffMs = maxDiffMs, 
    rotationOnly=rotationOnly, nPoints=nPoints, sigma = sigma,
        minDMax4rot=minDMax4rot,
        nSamples4rot = nSamples4rot,
        minSamples4rot = minSamples4rot,
        testing=testing, 
)
            return fout, matchedDat, rot, rot_err 
        except RuntimeError:
            print("matchParticles FAILED", fname1Match)
            print(str(e))
            ffl1.createDirs()
            with open('%s.broken.txt' % fname1Match, 'w') as f:
                f.write("in scripts: matchParticles FAILED")
                f.write("\r")
                f.write(str(e))
            return None, None, None, None 
