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

deltaY = deltaH = 1


def probability(x, mu, sigma, delta):
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


def doMatch(leader1D, follower1D, sigmaY, sigmaH, sigmaT, muY, muH, muT, deltaT, config, minProp, maxMatches, minNumber4Stats):
    '''
    match magic function
    
    minProp: minimal required probability
    maxMatches number of best matches to consider
    minNumber4Stats: min. number of samples to estimate sigmas and mus
    '''
    
    print("using", sigmaY, sigmaH, sigmaT, muY, muH, muT)  
    
    fyCenter = (follower1D.roi.sel(ROI_elements="y") +
                (follower1D.roi.sel(ROI_elements="h")/2))
    lyCenter = (leader1D.roi.sel(ROI_elements="y") +
                (leader1D.roi.sel(ROI_elements="h")/2))

    diffY = (np.array([fyCenter.values]) -
             np.array([lyCenter.values]).T)
    diffT = (np.array([follower1D.capture_time.values]) -
             np.array([leader1D.capture_time.values]).T).astype(int)*1e-9
    diffH = (np.array([follower1D.roi.sel(ROI_elements='h').values]) -
             np.array([leader1D.roi.sel(ROI_elements='h').values]).T)

    propH = probability(
        np.float32(diffH),
        np.float32(muH),
        np.float32(sigmaH),
        np.float32(deltaH)
    )
    propY = probability(
        np.float32(diffY),
        np.float32(muY),
        np.float32(sigmaY),
        np.float32(deltaY)
    )
    propT = probability(
        np.float32(diffT),
        np.float32(muT),
        np.float32(sigmaT),
        np.float32(deltaT)
    )
    
    # estimate joint probability
    propP = propY*propT*propH
    print(propP.shape, propP.dtype)

    matchedParticles = {}
    matchedProbabilities = {}

    # try to solve this from both perspectives
    for camera, prop1, dat2 in zip(
        [tools.nicerNames(config["leader"]), tools.nicerNames(config["follower"])], 
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

    del propH, propP, propY, propT

    for reverseFactor in [1, -1]:
        cam1, cam2 = [tools.nicerNames(config["leader"]), tools.nicerNames(config["follower"])][::reverseFactor]

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
        cam1, cam2 = [tools.nicerNames(config["leader"]),
                      tools.nicerNames(config["follower"])][::reverseFactor]
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

    cam1, cam2 = [tools.nicerNames(config["leader"]), tools.nicerNames(config["follower"])]

    pairs1 = set(zip(
        matchedParticles[cam1].fpidII.values, matchedParticles[cam1].values.astype(int)))
    pairs2 = set(zip(matchedParticles[cam2].values.astype(
        int), matchedParticles[cam2].fpidII.values))

    disputedPairs = pairs1 - pairs2
    
    # sort pairs together
    dats = []
    dats.append(leader1D.isel(
        fpid=matchedParticles[tools.nicerNames(config["leader"])].fpidII.values.astype(int)))
    dats.append(follower1D.isel(
        fpid=matchedParticles[tools.nicerNames(config["leader"])].values.astype(int)))

    for dd, d1 in enumerate(dats):

        tmpPID = deepcopy(d1.fpid)
        d1 = d1.rename(fpid='pair_id')
        d1 = d1.assign_coords(pair_id=list(
            range(len(matchedParticles[tools.nicerNames(config["leader"])].fpidII))))
        dats[dd] = d1

    matchedDat = xr.concat(dats, dim='camera')
    matchedDat = matchedDat.assign_coords(
        camera=[tools.nicerNames(config["leader"]), tools.nicerNames(config["follower"])])
    # add propabilities
    matchedDat["matchScore"] = xr.DataArray(
        matchedProbabilities[tools.nicerNames(config["leader"])
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
        print(" match coefficients, ",new_muT, new_muY)
    else:
        print("setting match coefficients to NAN")
        new_sigmaY = new_muY = new_sigmaH = new_muH = new_sigmaT = new_muT = np.nan
    
    return matchedDat, disputedPairs, new_sigmaY, new_muY, new_sigmaH, new_muH, new_sigmaT, new_muT
