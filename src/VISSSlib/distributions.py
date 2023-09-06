# -*- coding: utf-8 -*-

import sys

from .matching import *
from . import __version__
from copy import deepcopy
import numpy as np
import scipy.special
import xarray as xr
import pandas as pn
import trimesh
import dask
import dask.array
from tqdm import tqdm
from dask.diagnostics import ProgressBar

import functools
import warnings
import logging
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def _preprocess(dat):

    try:
        del dat["pair_id"]

        # we do not need all variables
        data_vars = ["capture_time","Dmax", "area","matchScore",
                                "aspectRatio", "angle", "perimeter", "position3D_center", "position3D_centroid", 
                                "camera_phi", "camera_theta", "camera_Ofz"]
        if "track_id" in dat.data_vars:
            data_vars += ["track_id","track_step"]
            #make track_ids unique, use only day-hour-minute-second, otherwise number is too large
            offset = int(dat.encoding["source"].split("_")[-1].split(".")[0].replace("-","")[6:]) * int(1e9)
            dat["track_id"].values = dat["track_id"].values + offset
        dat = dat[data_vars]
    except:
         log.error(dat.encoding["source"])
         raise KeyError
    return dat


def createLevel2match(
    case,
    config,
    freq="1T",
    minMatchScore=1e-3,
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "D"),
    blockedPixThresh=0.1, 
    blowingSnowFrameThresh=0.05,
    skipExisting = True, 
    writeNc = True,
    ):

    return createLevel2(
    case,
    config,
    freq=freq,
    minMatchScore=minMatchScore,
    DbinsPixel=DbinsPixel,
    sizeDefinitions=sizeDefinitions,
    endTime=endTime,
    blockedPixThresh=blockedPixThresh, 
    blowingSnowFrameThresh=blowingSnowFrameThresh,
    skipExisting = skipExisting, 
    writeNc = writeNc ,
    sublevel = "match"
    )

def createLevel2track(
    case,
    config,
    freq="1T",
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "D"),
    blockedPixThresh=0.1, 
    blowingSnowFrameThresh=0.05,
    skipExisting = True, 
    writeNc = True,
    ):

    return createLevel2(
    case,
    config,
    freq=freq,
    minMatchScore=None,
    DbinsPixel=DbinsPixel,
    sizeDefinitions=sizeDefinitions,
    endTime=endTime,
    blockedPixThresh=blockedPixThresh, 
    blowingSnowFrameThresh=blowingSnowFrameThresh,  
    skipExisting = skipExisting, 
    writeNc = writeNc ,
    sublevel = "track"
    )

def createLevel2(
    case,
    config,
    freq="1T",
    minMatchScore=1e-3,
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "D"),
    blockedPixThresh=0.1, 
    blowingSnowFrameThresh=0.05,
    skipExisting = True, 
    writeNc = True,
    hStep = 1,
    sublevel = "match"
    ):

    assert sublevel in ["match", "track"]
    if type(config) is str:
        config = tools.readSettings(config)

    fL = files.FindFiles(case, config.leader, config)
    lv2File = fL.fnamesDaily[f"level2{sublevel}"]


    log.info(f"Processing {lv2File}")

    if os.path.isfile(lv2File) and skipExisting:
        if os.path.getmtime(lv2File) < os.path.getmtime(fL.listFiles("metaEvents")[0]):
            print("file exists but older than event file, redoing", lv2File)
        else:
            print("SKIPPING - file exists", lv2File)
            return None, None

#    if len(fL.listFiles("metaFrames")) > len(fL.listFiles("level0")):
#        print("DATA TRANSFER INCOMPLETE ", lv2File)
#        print(len(fL.listFiles("level0")), "of", len(fL.listFiles("metaFrames")), "transmitted")
#        return None, None

    if sublevel == "match":
        if not fL.isCompleteL1match:
            print("level1match NOT COMPLETE YET %i of %i %s" %
                  (len(fL.listFilesExt("level1match")), len(fL.listFiles("level0txt")),  lv2File))
            print("look at ", fL.fnamesPatternExt["level1match"])
            return None, None
    elif sublevel == "track":
        if not fL.isCompleteL1track:
            print("level1track NOT COMPLETE YET %i of %i %s" %
                  (len(fL.listFilesExt("level1track")), len(fL.listFiles("level0txt")),  lv2File))
            print("look at ", fL.fnamesPatternExt["level1track"])
            return None, None
    else:
        raise ValueError

    lv1Files = fL.listFilesWithNeighbors(f"level1{sublevel}")

    if len(lv1Files) == 0:
        print("level1 NOT AVAILABLE %s" %
              lv2File)
        print("look at ", fL.fnamesPatternExt[f"level1{sublevel}"])
        return None, None

    timeIndex = pd.date_range(start=case, end=fL.datetime64 +
                              endTime, freq=freq, inclusive="left")
    timeIndex1 = pd.date_range(start=case, end=fL.datetime64 +
                              endTime, freq=freq, inclusive="both")

    if len(case) > 6:

            lv2Dat = createLevel2part(
            case,
            config,
            freq=freq,
            minMatchScore=None,
            DbinsPixel=DbinsPixel,
            sizeDefinitions=sizeDefinitions,
            endTime=endTime,
            skipExisting = skipExisting, 
            sublevel = sublevel
            )
            if lv2Dat is not None:
                log.info(f"load data for {case}")
                with ProgressBar():
                    lv2Dat.load()

    else:

        # due to performance reasons, split into hourly chunks and process sperately
        lv2Dat = []
        for hh in range(0,24,hStep):
            case1 = f"{case}-{hh:02d}"

            lv2Dat1 = createLevel2part(
            case1,
            config,
            freq=freq,
            minMatchScore=None,
            DbinsPixel=DbinsPixel,
            sizeDefinitions=sizeDefinitions,
            endTime=np.timedelta64(1, "h"),
            skipExisting = skipExisting, 
            sublevel = sublevel
            )

            if lv2Dat1 is not None:
                log.info(f"load data for {case1}")
                with ProgressBar():
                    lv2Dat.append(lv2Dat1.load())

        lv2Dat = xr.concat(lv2Dat, dim="time")
    #fill up missing data
    lv2Dat.reindex(time=timeIndex)

    #missing variables
    lv2Dat = addVariables(lv2Dat, case, config, timeIndex, timeIndex1, sublevel,
        blockedPixThresh=blockedPixThresh, blowingSnowFrameThresh=blowingSnowFrameThresh)


    lv2Dat = tools.finishNc(lv2Dat, config.site, config.visssGen)

    lv2Dat.D_bins.attrs.update(dict(units='m', long_name='size bins', comment="label at center of bin"))
    lv2Dat.fitMethod.attrs.update(dict(units='string', long_name='fit method to estimate aspect ratio'))
    lv2Dat.size_definition.attrs.update(dict(units='string', long_name='size definition'))
    lv2Dat.time.attrs.update(dict(long_name='time', comment='label at the end of time interval'))
    lv2Dat.camera.attrs.update(dict(units='string', long_name='camera'))

    lv2Dat.D32.attrs.update(dict(units='m', long_name='mean mass-weighted diameter'))
    lv2Dat.D43.attrs.update(dict(units='m', long_name='ratio of forth to third PSD moment'))
    lv2Dat.D_bins_left.attrs.update(dict(units='m', long_name='left edge D_bins'))
    lv2Dat.D_bins_right.attrs.update(dict(units='m', long_name='right edge D_bin'))
    lv2Dat.Dequiv_mean.attrs.update(dict(units='m', long_name='mean sphere equivalent diameter'))
    lv2Dat.Dequiv_std.attrs.update(dict(units='m', long_name='standard deviation sphere equivalent diameter'))
    lv2Dat.Dmax_mean.attrs.update(dict(units='m', long_name='mean maximum diameter'))
    lv2Dat.Dmax_std.attrs.update(dict(units='m', long_name='standard deviation maximum diameter'))
    lv2Dat.M1.attrs.update(dict(units='m', long_name='1st moment of the size distribution'))
    lv2Dat.M2.attrs.update(dict(units='m^2', long_name='2nd moment of the size distribution'))
    lv2Dat.M3.attrs.update(dict(units='m^3', long_name='3rd moment of the size distribution'))
    lv2Dat.M4.attrs.update(dict(units='m^4', long_name='4th moment of the size distribution'))
    lv2Dat.M6.attrs.update(dict(units='m^6', long_name='6th moment of the size distribution'))
    lv2Dat.N0_star_32.attrs.update(dict(units='1/m^3/m', long_name='PSD scaling parameter based on the second and third PSD moments'))
    lv2Dat.N0_star_43.attrs.update(dict(units='1/m^3/m', long_name='PSD scaling parameter based on the third and fourth PSD moments'))
    lv2Dat.Ntot.attrs.update(dict(units='1/m^3', long_name='Integral over size distribution'))
    lv2Dat.PSD.attrs.update(dict(units='1/m^3/m', long_name='Particle size distribution'))
    lv2Dat.angle_dist.attrs.update(dict(units='deg', long_name='angle dsitribution'))
    lv2Dat.angle_mean.attrs.update(dict(units='deg', long_name='mean angle'))
    lv2Dat.angle_std.attrs.update(dict(units='deg', long_name='standard deviation angle'))
    lv2Dat.area_dist.attrs.update(dict(units='m^2', long_name='area dsitribution'))
    lv2Dat.area_mean.attrs.update(dict(units='m^2', long_name='mean area'))
    lv2Dat.area_std.attrs.update(dict(units='m^2', long_name='standard deviation area'))
    lv2Dat.aspectRatio_dist.attrs.update(dict(units='-', long_name='aspectRatio dsitribution'))
    lv2Dat.aspectRatio_mean.attrs.update(dict(units='-', long_name='mean aspect ratio'))
    lv2Dat.aspectRatio_std.attrs.update(dict(units='-', long_name='standard deviation aspect ratio'))
    lv2Dat.blockedPixelRatio.attrs.update(dict(units='-', long_name='ratio of frames rejected due to blocked image filter'))
    lv2Dat.blowingSnowRatio.attrs.update(dict(units='-', long_name='ratio of frames rejected due to blowing snow filter'))
    lv2Dat.complexityBW_mean.attrs.update(dict(units='-', long_name='mean complexity (based on shape only)'))
    lv2Dat.complexityBW_std.attrs.update(dict(units='-', long_name='standard deviation complexity (based on shape only)'))
    lv2Dat.counts.attrs.update(dict(units='1/min', long_name='number of observed particles'))
    lv2Dat.matchScore_mean.attrs.update(dict(units='-', long_name='mean camera match score'))
    lv2Dat.matchScore_std.attrs.update(dict(units='-', long_name='standard deviation camera match score'))
    lv2Dat.obs_volume.attrs.update(dict(units='m^3', long_name='obs_volume'))
    lv2Dat.perimeter_dist.attrs.update(dict(units='m', long_name='perimeter dsitribution'))
    lv2Dat.perimeter_mean.attrs.update(dict(units='m', long_name='mean perimeter'))
    lv2Dat.perimeter_std.attrs.update(dict(units='m', long_name='standard deviation perimeter'))
    lv2Dat.processingFailed.attrs.update(dict(units='-', long_name='flag for faild processing'))
    lv2Dat.recordingFailed.attrs.update(dict(units='-', long_name='flag for faild recording'))

    if sublevel == "match":
        lv2Dat.nParticles.attrs.update(dict(units='-', long_name='number of particle observations'))

    elif sublevel == "track":
        lv2Dat.cameratrack.attrs.update(dict(units='string', long_name='camera and track', comment='Explains how multiple observations of the same particle by the two cameras along a track are combined'))
        lv2Dat.dim3D.attrs.update(dict(units='m', long_name='3 spatial dimensions'))

        lv2Dat.track_length_mean.attrs.update(dict(units='# frames', long_name='mean track_length'))
        lv2Dat.track_length_std.attrs.update(dict(units='# frames', long_name='standard deviation track_length'))
        lv2Dat.velocity_dist.attrs.update(dict(units='m/s', long_name='velocity dsitribution'))
        lv2Dat.velocity_mean.attrs.update(dict(units='m/s', long_name='mean velocity'))
        lv2Dat.velocity_std.attrs.update(dict(units='m/s', long_name='standard deviation velocity'))
        lv2Dat.nParticles.attrs.update(dict(units='-', long_name='number of observed unique particles'))


    if writeNc:
        tools.to_netcdf2(lv2Dat, lv2File)
    log.info(f"written {lv2File}")

    return lv2Dat, lv2File



def createLevel2part(
    case,
    config,
    freq="1T",
    minMatchScore=1e-3,
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "h"),
    skipExisting = True, 
    sublevel = "match"
    ):

    assert sublevel in ["match", "track"]

    fL = files.FindFiles(case, config.leader, config)
    lv2File = fL.fnamesDaily[f"level2{sublevel}"]

    lv1Files = fL.listFilesWithNeighbors(f"level1{sublevel}")

    if len(lv1Files) == 0:
        print("level1 NOT AVAILABLE %s" %
              lv2File)
        print("look at ", fL.fnamesPatternExt[f"level1{sublevel}"])
        return None

    timeIndex = pd.date_range(start=case, end=fL.datetime64 +
                              endTime, freq=freq, inclusive="left")
    timeIndex1 = pd.date_range(start=case, end=fL.datetime64 +
                              endTime, freq=freq, inclusive="both")

    log.info(f"open level1 files {case}")
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        level1dat = xr.open_mfdataset(lv1Files, preprocess=_preprocess, 
                combine="nested", concat_dim="pair_id")

    # limit to period of interest
    level1dat = level1dat.isel(pair_id=(level1dat.capture_time.isel(camera=0) >= fL.datetime64).values & (
        level1dat.capture_time.isel(camera=0) < (fL.datetime64 + endTime)).values)

    # make chunks more regular
    level1dat = level1dat.chunk(pair_id=10000)

    # apply matchScore threshold
    if minMatchScore is not None:
        matchCond = (level1dat.matchScore >= minMatchScore).values
        log.info(tools.concat("matchCond applies to", (matchCond.sum()/len(matchCond))*100, "% of data"))
        level1dat = level1dat.isel(pair_id=matchCond)

        if len(level1dat.matchScore) == 0:
            log.warning("no data remians after matchScore filtering %s" %
                  lv2File)
            return None   

    sizeCond = (level1dat.Dmax < max(DbinsPixel)).all("camera").values
    level1dat = level1dat.isel(pair_id=sizeCond)
    if len(level1dat.matchScore) == 0:
        log.warning("no data remians after size filtering %s" %
              lv2File)
        return None   


    # remove particles too close to the edge
    # this is possible for non symmetrical particles
    DmaxHalf = level1dat.Dmax.max("camera")/2
    farEnoughFromBorder = (
        (level1dat.position3D_center.sel(dim3D=["x", "y", "z"]) >= DmaxHalf).all("dim3D") &
        (config.frame_width - level1dat.position3D_center.sel(dim3D=["x", "y"]) >= DmaxHalf).all("dim3D") &
        (config.frame_height -
         level1dat.position3D_center.sel(dim3D="z", drop=True) >= DmaxHalf)
    )
    farEnoughFromBorder = farEnoughFromBorder.compute()

    log.info(tools.concat("farEnoughFromBorder applies to",
          (farEnoughFromBorder.sum()/len(farEnoughFromBorder)).values*100, 
          "% of data"))

    level1dat = level1dat.isel(pair_id=farEnoughFromBorder)
    if len(level1dat.pair_id) == 0:
        log.warning("no data remians after farEnoughFromBorder filtering %s" %
              lv2File)
        return None   


    if sublevel == "match":
        log.info(f"estimate camera mean values")
        data_vars = ['Dmax', 'area', 'matchScore', 'aspectRatio', 'angle', 'perimeter']

        #promote capture_time to coordimnate for later
        level1dat_time = level1dat.assign_coords(time=xr.DataArray(level1dat.capture_time.isel(camera=0).values, coords=[level1dat.pair_id]))

        # we do not need pid any more
        level1dat_time = level1dat_time.reset_coords("pid")

        # estimate max, mean and min for both cameras
        level1dat_camAve = (level1dat_time[data_vars].max("camera"), level1dat_time[data_vars].mean("camera"), level1dat_time[data_vars].min("camera"), level1dat_time[data_vars].sel(
            camera=config.leader, drop=True), level1dat_time[data_vars].sel(camera=config.follower, drop=True))
        level1dat_camAve = xr.concat(level1dat_camAve, dim="camera")
        level1dat_camAve["camera"] = ["max", "mean", "min", "leader", "follower"]
        # position_3D is the same for all
        #level1dat_camAve["position3D_center"] = level1dat_camAve["position3D_center"].sel(
        #    camera="max", drop=True)

        # fix order
        level1dat_4timeAve = level1dat_camAve.transpose(*["camera" ,"fitMethod","pair_id"])

        #clean up
        del level1dat_camAve
        # #position is not needed any more
        # del level1dat_4timeAve["position3D_center"]
        # #centroid position is not needed any more
        # del level1dat_4timeAve["position3D_centroid"]

        #save for later
        individualDataPoints = level1dat_4timeAve.pair_id
        individualDataPoints.name = "nParticles"

    elif sublevel == "track":

        log.warning("make first guess smarter by including size dependence!")

        log.info(f"reshape tracks")

        #go from pair_id to track_id and put observations along same track into new track_step dimension
        track_mi = pn.MultiIndex.from_arrays((level1dat.track_id.values,level1dat.track_step.values), names=["track_id","track_step"])
        level1dat["track_mi"] = xr.DataArray(track_mi, coords=[level1dat.pair_id])
        level1dat = level1dat.swap_dims(pair_id="track_mi")

        #this costs a lot of memeory but I do not know a better way
        level1dat_time = level1dat.unstack("track_mi")
        #promote capture_time to coordimnate for later
        level1dat_time = level1dat_time.assign_coords(time=xr.DataArray(level1dat_time.capture_time.isel(camera=0,track_step=0).values, coords=[level1dat_time.track_id]))

        #cut tracks longer than 40 elements to save memory!
        if len(level1dat_time.track_step) >40:
            log.info(f"truncating {100*level1dat_time.Dmax.isel(camera=0, track_step=40).notnull().sum().values/len(level1dat_time.track_id)} % tracks")
            level1dat_time = level1dat_time.isel(track_step=slice(40))


        log.info(f"estimate track mean values")
        #fix order
        level1dat_4trackAve = level1dat_time.transpose(*["track_id", "track_step", "camera" , "dim3D","fitMethod","camera_rotation"])
        level1dat_4trackAve = level1dat_4trackAve[['Dmax', 'area', 'matchScore', 'aspectRatio', 'angle', 'perimeter', 'position3D_centroid', 'capture_time']]

        # add velocities
        distSpace = level1dat_4trackAve.position3D_centroid.diff("track_step", label="upper")
        distTime = level1dat_4trackAve.capture_time.isel(camera=0, drop=True).diff("track_step", label="upper")
        #to fraction of seconds
        distTime = distTime/np.timedelta64(1,"s")

        #velocity in px/s
        level1dat_4trackAve["velocity"] = distSpace/distTime
        del level1dat_4trackAve["capture_time"]

        #save for later
        individualDataPoints = level1dat_4trackAve.track_id
        individualDataPoints.name = "nParticles"

        #diff output is one element shorter, so add the mean value again
        # causes problems and advantage is not clear...
        #level1dat_4trackAve["velocity"][dict(track_step=0)] = level1dat_4trackAve["velocity"].mean("track_step")
        del level1dat_4trackAve[ 'position3D_centroid']

        # estimate max, mean and min for tracks by reducing track_step
        trackOps = ["max", "mean", "min", "std"]
        level1dat_trackAve = (level1dat_4trackAve.max(["track_step", "camera"]), level1dat_4trackAve.mean(["track_step", "camera"]), level1dat_4trackAve.min(["track_step", "camera"]), level1dat_4trackAve.std(["track_step", "camera"]))
        level1dat_trackAve = xr.concat(level1dat_trackAve, dim="cameratrack")
        level1dat_trackAve["cameratrack"] = trackOps
            # position_3D is the same for all

        # use Dmax as arbitrary variable with only one dimension
        level1dat_trackAve["track_length"] = level1dat_4trackAve.Dmax.isel(camera=0, drop=True).notnull().sum("track_step")

        # becuase there are no weighted groupby operations, we have to improvise and broadcast the results
        # again to a shape including track_step - then the mean etc. values are dublicated as per track length 
        # and the result is weighted when averaging with timme
        data_vars = ['Dmax', 'area', 'matchScore', 'aspectRatio', 'angle', 'perimeter', 'velocity']
        for data_var in data_vars:
            level1dat_trackAve[data_var] = level1dat_trackAve[data_var].broadcast_like(level1dat_4trackAve.isel(camera=0, drop=True)[data_var])

        #add back the original individual values (mainly for testing)
        #level1dat_trackAve = xr.concat((level1dat_trackAve,level1dat_4trackAve.expand_dims(track=["individual"])), dim="cameratrack")

        log.info(f"reshape track data again")
        #call me crayzy but now that we have mean track properties broadcasted to every particle we can go back to pair_id!
        level1dat_4timeAve = level1dat_trackAve.stack(pair_id=("track_id","track_step"))
        # make sure only data is used within original track length
        notNull = level1dat_4trackAve.Dmax.isel(camera=0, drop=True).notnull().stack(pair_id=("track_id","track_step")).compute()
        level1dat_4timeAve = level1dat_4timeAve.isel(pair_id=notNull)
        #multiindex causes trouble below, so just swap with time

        # switch to time coordinate
        level1dat_4timeAve = level1dat_4timeAve.swap_dims(pair_id="time")
        

        #clean up
        del level1dat_trackAve


    #clean up 
    level1dat.close()

    log.info(f"load data")
    #turned out, it runs about 3 to 4 times faster when NOT using dask beyond this point. 
    level1dat_4timeAve = level1dat_4timeAve.load()

    log.info(f"add additonal variables")
    level1dat_4timeAve = addPerParticleVariables(level1dat_4timeAve)

    # split data in 1 min chunks
    level1datG = level1dat_4timeAve.groupby_bins("time", timeIndex1, right=False, squeeze=False)
    individualDataPointsG = individualDataPoints.groupby_bins("time", timeIndex1, right=False, squeeze=False)

    del level1dat_4timeAve

    sizeDefinitions = ["Dmax", "Dequiv"]
    data_vars = ["area", "angle", "aspectRatio", "perimeter"]
    if sublevel == "track":
        data_vars.append("velocity")

    log.info(f"get time resolved distributions")
    # process each 1 min chunks
    res = {}
    nParticles = {}

    if sublevel == "track":
        coordVar = "cameratrack"
    else:
        coordVar = "camera"

    # iterate through every 1 min piece
    for interv, level1datG1 in tqdm(level1datG, file=sys.stdout):
        #print(interv)
        tmp = []
        # for each track&camera/min/max/mean seperately
        for coord in level1datG1[coordVar]:
            # estimate counts
            tmpXr = []
            for sizeDefinition in sizeDefinitions:
                tmpXr1 = level1datG1[[sizeDefinition]].sel(**{coordVar:coord}).groupby_bins(
                    sizeDefinition, DbinsPixel, right=False).count().fillna(0)

                tmpXr1 = tmpXr1.rename({sizeDefinition: "N"})
                tmpXr1 = tmpXr1.rename({f"{sizeDefinition}_bins": "D_bins"})

                #import pdb; pdb.set_trace()
                # estimate mean values for "area", "angle", "aspectRatio", "perimeter"
                # Dmax is only for technical resaons and is removed afterwards
                data_vars1 = data_vars + [sizeDefinition]
                otherVars1 = level1datG1[data_vars1].sel(
                    **{coordVar:coord}).groupby_bins(sizeDefinition, DbinsPixel, right=False).mean()
                del otherVars1[sizeDefinition]
                otherVars1 = otherVars1.rename({k: f"{k}_dist" for k in otherVars1.data_vars})
                otherVars1 = otherVars1.rename({f"{sizeDefinition}_bins": "D_bins"})
                tmpXr1.update(otherVars1)
                tmpXr.append(tmpXr1)


            tmpXr = xr.concat(tmpXr, dim="size_definition")
            tmpXr["size_definition"] = sizeDefinitions

            tmp.append(xr.Dataset(tmpXr))
        # merge camera/min/max/mean reults
        res[interv.left] = xr.concat(tmp, dim=coordVar)
        # add camera/min/max/mean information
        res[interv.left][coordVar] = level1datG1[coordVar]

    #clean up
    del tmpXr, tmp, tmpXr1

    dist = xr.concat(res.values(), dim="time")
    dist["time"] = list(res.keys())

    # fill data gaps with zeros
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=dask.array.core.PerformanceWarning)
        dist = dist.reindex(time=timeIndex)
    dist["N"] = dist["N"].fillna(0)

    log.info("do temporal mean values")
    # estimate mean values
    # to do: data is weighted with number of obs not considering the smalle robservation volume for larger particles


    meanValues = level1datG.mean()
    meanValues = meanValues.rename({k: f"{k}_mean" for k in meanValues.data_vars})
    meanValues = meanValues.rename(time_bins="time")
    # we want tiem stamps not intervals
    meanValues["time"] = [a.left for a in meanValues["time"].values]

    log.info("do temporal std values")
    # estimate mean values
    # to do: data is weighted with number of obs not considering the smalle robservation volume for larger particles
    stdValues = level1datG.std()
    stdValues = stdValues.rename({k: f"{k}_std" for k in stdValues.data_vars})
    stdValues = stdValues.rename(time_bins="time")
    # we want tiem stamps not intervals
    stdValues["time"] = meanValues["time"]

    nParticles = individualDataPointsG.count()
    nParticles = nParticles.rename(time_bins="time")
    nParticles["time"] = meanValues["time"]

    log.info("merge data")
    level2dat = xr.merge((dist, meanValues, stdValues, nParticles))

    log.info("calibrate data")
    calibDat = calibrateData(level2dat, level1dat_time, config, DbinsPixel, timeIndex1)

    #clean up!
    del level1datG, level1dat_time
    return calibDat

def addPerParticleVariables(level1dat_camAve):
    # add area equivalent radius
    level1dat_camAve["Dequiv"] = np.sqrt(4*level1dat_camAve["area"]/np.pi)

    #based on Garrett, T. J., and S. E. Yuter, 2014: Observed influence of riming, temperature, and turbulence on the fallspeed of solid precipitation. Geophys. Res. Lett., 41, 6515–6522, doi:10.1002/2014GL061016.
    level1dat_camAve["complexityBW"] = level1dat_camAve["perimeter"]/(np.pi * level1dat_camAve["Dequiv"])
    # level1dat_camAve["complexity"] = level1dat_camAve["complexityBW"] * 

    return level1dat_camAve

def addVariables(calibDat, case, config, timeIndex, timeIndex1, sublevel, blockedPixThresh=0.1, blowingSnowFrameThresh=0.05):

    # 1 min data
    deltaT = int(timeIndex.freq.nanos * 1e-9) * config.fps
    # 1 pixel size bins
    deltaD = config.resolution * 1e-6

    calibDat["PSD"] = calibDat["counts"] / deltaT / \
        calibDat["obs_volume"] / deltaD  # now in 1/m4
    calibDat["Ntot"] = (calibDat["counts"] / deltaT /
                        calibDat["obs_volume"]).sum("D_bins")

    M = {}
    for mm in [1,2,3,4,6]:
        M[mm] = (calibDat.PSD.fillna(0)*deltaD*calibDat.D_bins**mm).sum("D_bins")
        calibDat[f"M{mm}"] = M[mm]

    for b in [2,3]:
        calibDat[f"D{b+1}{b}"] = M[b+1]/M[b]
        calibDat[f"N0_star_{b+1}{b}"] = (M[b]**(b+2)/M[b+1]**(b+1)) * ((b+1)**(b+1))/scipy.special.gamma(b+1)

    # quality variables
    recordingFailed, processingFailed, blockedPixels, blowingSnowRatio = getDataQuality(
        case, config, timeIndex, timeIndex1, sublevel)
    assert np.all(blockedPixels.time == calibDat.time)

    cameraBlocked = blockedPixels.max("camera") > blockedPixThresh
    blowingSnow = blowingSnowRatio.max("camera") > blowingSnowFrameThresh

    #apply quality
    log.info("apply quality filters...")
    log.info(tools.concat("recordingFailed filter removed", recordingFailed.values.sum()/len(recordingFailed)*100, "% of data"))
    log.info(tools.concat("processingFailed filter removed", processingFailed.values.sum()/len(processingFailed)*100, "% of data"))
    log.info(tools.concat("cameraBlocked filter removed", cameraBlocked.values.sum()/len(cameraBlocked)*100, "% of data"))
    log.info(tools.concat("blowingSnow filter removed", blowingSnow.values.sum()/len(blowingSnow)*100, "% of data"))

    allFilter = recordingFailed | processingFailed | cameraBlocked | blowingSnow
    log.info(tools.concat("all filter together removed", allFilter.values.sum()/len(allFilter)*100, "% of data"))

    assert (allFilter.time == calibDat.time).all()

    calibDatFilt = calibDat.where(~allFilter)
    #reverse for D_bins_left and D_bins_right
    calibDatFilt["D_bins_left"] = calibDat["D_bins_left"]
    calibDatFilt["D_bins_right"] = calibDat["D_bins_right"]

    calibDatFilt["recordingFailed"] = recordingFailed
    calibDatFilt["processingFailed"] = processingFailed
    calibDatFilt["blowingSnowRatio"] = blowingSnowRatio
    if sublevel == "match":
        blockedVars = (blockedPixels.max("camera"), 
            blockedPixels.mean("camera"), 
            blockedPixels.min("camera"), 
            blockedPixels.sel(camera="leader", drop=True), 
            blockedPixels.sel(camera="follower", drop=True)
            )
        calibDatFilt["blockedPixelRatio"] = xr.concat(blockedVars, dim="camera").T
        blowingVars = (blowingSnowRatio.max("camera"), 
            blowingSnowRatio.mean("camera"), 
            blowingSnowRatio.min("camera"), 
            blowingSnowRatio.sel(camera="leader", drop=True), 
            blowingSnowRatio.sel(camera="follower", drop=True)
            )
        calibDatFilt["blowingSnowRatio"] = xr.concat(blowingVars, dim="camera").T
    else:
        calibDatFilt["blockedPixelRatio"] = blockedPixels.T
        calibDatFilt["blowingSnowRatio"] = blowingSnowRatio.T

    return calibDatFilt


def estimateObservationVolume(level1dat_time, config, DbinsPixel, timeIndex1):

    '''
    in pixel

    '''

    rotDat = level1dat_time[["camera_phi", "camera_theta", "camera_Ofz"]].sel(
        camera_rotation="mean").groupby_bins("time", timeIndex1, right=False, squeeze=False).mean()
    rotDat = rotDat.rename(time_bins="time")
    # we want time stamps not intervals
    rotDat["time"] = [a.left for a in rotDat["time"].values]

    rotDat.load()
    rotDat = rotDat.round(2)

    volumes = []
    for ii in range(len(rotDat.time)):

        rotDat1 = rotDat.isel(time=ii, drop=True)
        try:
            rotDat1 = rotDat1.isel(track_step=0, drop=True)
        except ValueError:
            pass

        # print(config.frame_width,
        #                              config.frame_height,
        #                              rotDat1.camera_phi.values,
        #                              rotDat1.camera_theta.values,
        #                              rotDat1.camera_Ofz.values, DbinsPixel)
        Ds, volume = estimateVolumes(config.frame_width,
                                     config.frame_height,
                                     float(rotDat1.camera_phi.values),
                                     float(rotDat1.camera_theta.values),
                                     float(rotDat1.camera_Ofz.values),
                                     sizeBins=DbinsPixel)

        volumes.append(volume[1:])
    return volumes


def calibrateData(level2dat, level1dat_time, config, DbinsPixel, timeIndex1):
    '''go from pixel to SI units'''

    assert "intercept" in config.calibration.keys()
    assert "slope" in config.calibration.keys()

    slope = config.calibration.slope
    intercept = config.calibration.intercept

    calibDat = level2dat.rename(N="counts").copy()

    volumes = estimateObservationVolume(level1dat_time, config, DbinsPixel, timeIndex1)
    calibDat["obs_volume"] = xr.DataArray(
        volumes, coords=[level2dat.time, level2dat.D_bins])

    # apply resolution
    calibDat["D_bins"] = (calibDat["D_bins"] - intercept) / slope / 1e6
    # assume that intercept is an artifact of Dmax estimation
    calibDat["obs_volume"] = calibDat["obs_volume"]  / slope**3 / 1e6**3

    # go from intervals to center values
    calibDat["D_bins_left"] = xr.DataArray(
        [b.left for b in calibDat.D_bins.values], dims=["D_bins"])
    calibDat["D_bins_right"] = xr.DataArray(
        [b.right for b in calibDat.D_bins.values], dims=["D_bins"])
    calibDat = calibDat.assign_coords(
        D_bins=[b.mid for b in calibDat.D_bins.values])

    #remaining variables
    # assume that intercept is an artifact of Dmax estimation
    calibDat["area_dist"] = calibDat["area_dist"]  / slope**2 / 1e6**2
    calibDat["perimeter_dist"] = calibDat["perimeter_dist"] / slope / 1e6

    calibDat["Dmax_mean"] = (calibDat["Dmax_mean"] -intercept )/ slope / 1e6
    calibDat["Dmax_std"] = (calibDat["Dmax_std"] -intercept )/ slope / 1e6
    calibDat["area_mean"] = calibDat["area_mean"]  / slope**2 / 1e6**2
    calibDat["area_std"] = calibDat["area_std"]  / slope**2 / 1e6**2
    calibDat["perimeter_mean"] = calibDat["perimeter_mean"] / slope / 1e6
    calibDat["perimeter_std"] = calibDat["perimeter_std"] / slope / 1e6
    # is the intercept correction needed for Dequiv? good question...
    calibDat["Dequiv_mean"] = (calibDat["Dequiv_mean"]) / slope / 1e6

    if "velocity_dist" in calibDat.data_vars:
        calibDat["velocity_dist"] = (calibDat["velocity_dist"]) / slope / 1e6
        calibDat["velocity_mean"] = (calibDat["velocity_mean"]) / slope / 1e6
        calibDat["velocity_std"] = (calibDat["velocity_std"]) / slope / 1e6

    return calibDat


def applyCalib(pixel, slope, intercept):
    #pix = slope*um + intercept
    um = (pixel - intercept)/slope
    m = um / 1e6
    return m

def getDataQuality(case, config, timeIndex, timeIndex1, sublevel):
    """Estimate data quality for level2 
    """

    fL = files.FindFiles(case, config.leader, config)
    fF = files.FindFiles(case, config.follower, config)

    fnameL = fL.listFilesWithNeighbors("metaEvents")
    fnameF = fF.listFilesWithNeighbors("metaEvents")

    eventL = xr.open_mfdataset(fnameL).load()
    eventF = xr.open_mfdataset(fnameF).load()

    matchFilesAll = fL.listFilesExt(f"level1{sublevel}")
    matchFilesBroken = [f for f in matchFilesAll if f.endswith("broken.txt")]
    brokenTimes = [files.FilenamesFromLevel(
        f, config).datetime64 for f in matchFilesBroken]
    matchFilesBroken = xr.DataArray(
        matchFilesBroken, dims=["file_starttime"], coords=[brokenTimes])

    graceTime = 2  # s
    newfilesF = eventF.isel(file_starttime=(eventF.event == "newfile"))
    newfilesL = eventL.isel(file_starttime=(eventL.event == "newfile"))

    dataRecorded = []
    processingFailed = []
    for tt, tI1min in enumerate(timeIndex):
        tDiffF = (np.datetime64(tI1min) -
                  newfilesF.file_starttime).values/np.timedelta64(1, "s")
        tDiffL = (np.datetime64(tI1min) -
                  newfilesL.file_starttime).values/np.timedelta64(1, "s")
        dataRecordedF = np.any(
            tDiffF[tDiffF >= -graceTime] < (config.newFileInt - graceTime))
        dataRecordedL = np.any(
            tDiffL[tDiffL >= -graceTime] < (config.newFileInt - graceTime))
    #     print(tI1min, dataRecordedF, dataRecordedL)
        dataRecorded.append(dataRecordedF and dataRecordedL)

        if len(matchFilesBroken) > 0:
            tDiffBroken = (np.datetime64(
                tI1min)-matchFilesBroken.file_starttime).values/np.timedelta64(1, "s")
            processingFailed1 = np.any(
                tDiffBroken[tDiffBroken >= -graceTime] < (config.newFileInt - graceTime))
            processingFailed.append(processingFailed1)
        else:
            processingFailed.append(False)

    recordingFailed = ~xr.DataArray(
        dataRecorded, dims=["time"], coords=[timeIndex])
    processingFailed = xr.DataArray(
        processingFailed, dims=["time"], coords=[timeIndex])

    blockedPixelsF = (eventF.blocking.sel(blockingThreshold=50, drop=True))
    blockedPixelsL = (eventL.blocking.sel(blockingThreshold=50, drop=True))

    blockedPixelsF = blockedPixelsF.reindex(
        file_starttime=timeIndex, method="nearest", tolerance=np.timedelta64(int(config.newFileInt/1.9), "s"))
    blockedPixelsL = blockedPixelsL.reindex(
        file_starttime=timeIndex, method="nearest", tolerance=np.timedelta64(int(config.newFileInt/1.9), "s"))

    blockedPixelsL = blockedPixelsL.rename(file_starttime="time")
    blockedPixelsF = blockedPixelsF.rename(file_starttime="time")
    blockedPixels = xr.concat((blockedPixelsL, blockedPixelsF), dim="camera")
    blockedPixels["camera"] = ["leader", "follower"]

    fnames = {}    
    fnames["leader"] = fL.listFilesWithNeighbors("metaDetection")
    fnames["follower"] = fF.listFilesWithNeighbors("metaDetection")
    blowingSnowRatio = tools.identifyBlowingSnowData(fnames, config, timeIndex1)
    blowingSnowRatio = blowingSnowRatio

    return recordingFailed, processingFailed, blockedPixels, blowingSnowRatio


def _createBox(p1, p2, p3, p4, p5, p6, p7, p8):

    vertices = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
    faces = np.array([[1, 3, 0], [4, 1, 0], [0, 3, 2], [2, 4, 0], [1, 7, 3],
                      [5, 1, 4], [5, 7, 1], [3, 7, 2], [6, 4, 2], [2, 7, 6],
                      [6, 5, 4], [7, 5, 6]])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def createLeaderBox(width, height, delta=0):
    '''get trimesh representing the leader observation volume

    Parameters
    ----------
    width : int
        image width
    height : int
        image height
    delta : number, optional
        distance to the edges (the default is 0)

    Returns
    -------
    trimesh
        trimesh object
    '''

    X0 = -width
    X1 = 2*width
    Y0 = 0 + delta
    Y1 = width - delta
    Z0 = 0 + delta
    Z1 = height - delta

    p1 = (X0, Y0, Z0)
    p2 = (X0, Y0, Z1)
    p3 = (X0, Y1, Z0)
    p4 = (X0, Y1, Z1)
    p5 = (X1, Y0, Z0)
    p6 = (X1, Y0, Z1)
    p7 = (X1, Y1, Z0)
    p8 = (X1, Y1, Z1)

    return _createBox(p1, p2, p3, p4, p5, p6, p7, p8)


def createFollowerBox(width, height, camera_phi, camera_theta, camera_Ofz, delta=0):
    '''get trimesh representing the follower observation volume

    Parameters
    ----------
    width : int
        image width
    height : int
        image height
    camera_phi : float
        roll of follower camera
    camera_theta : float
        pitch of follower camera
    camera_Ofz : float
        offset in z direction
    delta : number, optional
        distance to the edges (the default is 0)

    Returns
    -------
    trimesh
        trimesh object
    '''
    X0 = 0 + delta
    X1 = width - delta
    Y0 = -width
    Y1 = 2*width
    Z0 = 0 + delta
    Z1 = height - delta

    psi = Olx = Ofy = 0.

    p1 = shiftRotate_F2L(X0, Y0, Z0, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz)
    p2 = shiftRotate_F2L(X0, Y0, Z1, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz)
    p3 = shiftRotate_F2L(X0, Y1, Z0, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz)
    p4 = shiftRotate_F2L(X0, Y1, Z1, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz)
    p5 = shiftRotate_F2L(X1, Y0, Z0, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz)
    p6 = shiftRotate_F2L(X1, Y0, Z1, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz)
    p7 = shiftRotate_F2L(X1, Y1, Z0, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz)
    p8 = shiftRotate_F2L(X1, Y1, Z1, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz)

    return _createBox(p1, p2, p3, p4, p5, p6, p7, p8)


def estimateVolume(width, height, camera_phi, camera_theta, camera_Ofz, delta=0):
    '''estimate intersecting volume of leader and follower


    Parameters
    ----------
    width : int
        image width
    height : int
        image height
    camera_phi : float
        roll of follower camera
    camera_theta : float
        pitch of follower camera
    camera_Ofz : float
        offset in z direction
    delta : number, optional
        distance to the edges (the default is 0)

    Returns
    -------
    float
        intersection volume
    '''

    follower = createFollowerBox(width, height, camera_phi, camera_theta, camera_Ofz, delta=delta)
    leader = createLeaderBox(width, height, delta=delta)


    volume = leader.intersection(follower).volume

    return volume

@functools.cache
def estimateVolumes(width, height, camera_phi, camera_theta, camera_Ofz, sizeBins, nSteps=5, interpolate=True):
    '''estimate intersecting volume of leader and follower for different distances to 
    the edge of the volume

    Parameters
    ----------
    width : int
        image width
    height : int
        image height
    camera_phi : float
        roll of follower camera
    camera_theta : float
        pitch of follower camera
    camera_Ofz : float
        offset in z direction
    minSize : int
        minimum size to consider for distamce to the edge
    maxSize : int
        maximum size to consider for distamce to the edge
    nSteps : int, optional
        number of points were the colume is estimated (the default is 5)

    Returns
    -------
    array
        distances to the edge
    array
        volumes of corresponding distances
    '''


    # distance to the edge is only half of the particle size!
    Ds = np.ceil(np.array(sizeBins)/2).astype(int)


    if np.any(np.isnan((camera_phi, camera_theta, camera_Ofz))):
        volumes = np.full(len(sizeBins), np.nan)
    else:
        volumes = []
        # select only nStep distances and interpolate rest
        iiInter = np.linspace(0,len(sizeBins)-1,nSteps, endpoint=True, dtype=int)
        for ii in iiInter:
            volumes.append(estimateVolume(
                width, height, camera_phi, camera_theta, camera_Ofz, delta=Ds[ii]))

        if interpolate:
            Ds_inter, volumes = interpolateVolumes(Ds, Ds[iiInter], volumes)
            assert np.all(Ds_inter==Ds)
    return Ds, np.array(volumes)

def interpolateVolumes(Dfull, Dstep, volumes1):
    '''interpolate volumes considering the cube dependency

    Parameters
    ----------
    Dfull : array
        list of distances wanted
    Dstep : array
        list of calculates distances
    volumes1 : array
        list of volumes

    Returns
    -------
    array
        distances to the edge used for interpolation
    array
        volumes of corresponding interpolated distances
    '''

    volumes = np.interp(Dfull, Dstep, np.array(volumes1)**(1/3.))**3
    return Dfull, volumes



# def velocity(dat):
#     coords = ["max", "mean", "min", "std"]#, "median"]
#     diffs = dat.position3D_centroid.diff("pair_id")
#     if diffs.shape[1] == 0:
#         datJoint = xr.DataArray(np.zeros((len(coords), len(dat.dim3D.values)))*np.nan, coords=[coords, dat.dim3D.values], dims=["track", "dim3D"])
#         return datJoint
#     maxs = diffs.max("pair_id")
#     mins = diffs.min("pair_id")
#     means = diffs.mean("pair_id")
# #     medians = diffs.median("pair_id")
#     stds = diffs.std("pair_id")
#     datJoint = xr.concat([maxs,means,mins,stds],#,medians], 
#                          dim="track")
#     datJoint["track"] = coords
#     return datJoint



# def trackProperties(dat):
#     coords = ["max", "mean", "min", "std"]#, "median"]
#     maxs = dat.max("pair_id")
#     mins = dat.min("pair_id")
#     means = dat.mean("pair_id")
# #     medians = dat.median("pair_id")
#     stds = dat.std("pair_id")
#     datJoint = xr.concat([maxs,means,mins,stds],#,medians], 
#                          dim="track")
#     datJoint["track"] = coords
#     return datJoint

# def averageTracks(lv1track):
    
#     lv1track["Dequiv"] = np.sqrt(4*lv1track["area"]/np.pi)
#     #based on Garrett, T. J., and S. E. Yuter, 2014: Observed influence of riming, temperature, and turbulence on the fallspeed of solid precipitation. Geophys. Res. Lett., 41, 6515–6522, doi:10.1002/2014GL061016.
#     lv1track["complexityBW"] = lv1track["perimeter"]/(np.pi * lv1track["Dequiv"])

#     gp = lv1track.groupby("track_id")
#     lv1trackJoint = []
#     log.info(f"calculating max")
#     lv1trackJoint.append(gp.max())
#     log.info(f"calculating mean")
#     lv1trackJoint.append(gp.mean())
#     log.info(f"calculating min")
#     lv1trackJoint.append(gp.min())
#     log.info(f"calculating std")
#     lv1trackJoint.append(gp.std())
#     log.info(f"calculating median")
#     lv1trackJoint.append(gp.median())
    
#     log.info(f"joining data")
#     lv1trackJoint = xr.concat(lv1trackJoint, dim="track")
#     lv1trackJoint["track"] = ["max", "mean", "min", "std", "median"]
    
#     log.info(f"calculate counts")
#     # use matchscore as arbitrary variable with only one dimension
#     counts = lv1track[["matchScore","track_id"]].groupby("track_id").count()["matchScore"]

#     log.info(f"calculate velocity")
#     lv1trackJoint["track_length"] = counts
#     lv1trackJoint["velocity"] = lv1track[["position3D_centroid", "track_id"]].groupby("track_id").map(velocity)
#     #lv1trackJoint["absVelocity"] = lv1track[["position_3D", "track_id"]].groupby("track_id").map(absVelocity)


    # return lv1trackJoint