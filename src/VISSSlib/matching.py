# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats
#import av
import bottleneck as bn

import logging
log = logging.getLogger()

from copy import deepcopy

from . import __version__
from . import tools

deltaY = deltaH = deltaI = 1.


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


def doMatch(leader1D, follower1D, sigmaY, sigmaH, sigmaT, sigmaI, muY, muH, muT, muI, deltaT, config, minProp, maxMatches, minNumber4Stats):
    '''
    match magic function
    
    minProp: minimal required probability
    maxMatches number of best matches to consider
    minNumber4Stats: min. number of samples to estimate sigmas and mus
    '''
    
    print("using", sigmaY, sigmaH, sigmaT, sigmaI, muY, muH, muT, muI)  
    
    if sigmaY is not None:
        fyCenter = (follower1D.roi.sel(ROI_elements="y") +
                    (follower1D.roi.sel(ROI_elements="h")/2))
        lyCenter = (leader1D.roi.sel(ROI_elements="y") +
                    (leader1D.roi.sel(ROI_elements="h")/2))

        diffY = (np.array([fyCenter.values]) -
                 np.array([lyCenter.values]).T)
        propY = probability(
            diffY,
            muY,
            sigmaY,
            deltaY
        )
    else:
        propY = 1

    if sigmaH is not None:
        diffH = (np.array([follower1D.roi.sel(ROI_elements='h').values]) -
                 np.array([leader1D.roi.sel(ROI_elements='h').values]).T)

        propH = probability(
            diffH,
            muH,
            sigmaH,
            deltaH
        )
    else:
        propH = 1.

    if sigmaT is not None:

        diffT = (np.array([follower1D.capture_time.values]) -
                 np.array([leader1D.capture_time.values]).T).astype(int)*1e-9
        propT = probability(
            diffT,
            muT,
            sigmaT,
            deltaT
        )
    else:
        propT = 1.
    
    if sigmaI is not None:

        diffI = (np.array([follower1D.capture_id.values]) -
                 np.array([leader1D.capture_id.values]).T)
        propI = probability(
            diffI,
            muI,
            sigmaI,
            deltaI
        )
    else:
        propI = 1.

    # estimate joint probability
    propP = propY*propT*propH*propI
    print(propP.shape, propP.dtype)

    matchedParticles = {}
    matchedProbabilities = {}

    # try to solve this from both perspectives
    for camera, prop1, dat2 in zip(
        [config["leader"], config["follower"]], 
        [propP, propP.T], 
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

    del propH, propP, propY, propT, propI

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

    if len(matchedDat.pair_id) >= minNumber4Stats:
        yCenter = (matchedDat.roi.sel(ROI_elements='y') +
                   (matchedDat.roi.sel(ROI_elements="h")/2))
        di = yCenter.diff('camera').values
        new_sigmaY = bn.nanstd(di)
        new_muY = bn.nanmedian(di)

        di = matchedDat.roi.sel(
            ROI_elements='h').diff("camera").values
        new_sigmaH = bn.nanstd(di)
        new_muH = bn.nanmedian(di)

        di = matchedDat.capture_time.diff('camera').values
        di = di[np.isfinite(di)].astype(int)*1e-9
        new_sigmaT = bn.nanstd(di)
        new_muT = bn.nanmedian(di)

        di = matchedDat.capture_id.diff('camera').values
        new_sigmaI = bn.nanstd(di)
        new_muI = bn.nanmedian(di)

        print(" match coefficients, ",new_muY, new_muH,new_muT, new_muI)
    else:
        print("setting match coefficients to NAN")
        new_sigmaY = new_muY = new_sigmaH = new_muH = np.nan
        new_sigmaT = new_muT = new_sigmaT = new_muT = np.nan
    
    return matchedDat, disputedPairs, new_sigmaY, new_muY, new_sigmaH, new_muH, new_sigmaT, new_muT, new_sigmaI, new_muI
