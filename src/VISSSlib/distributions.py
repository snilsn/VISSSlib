# -*- coding: utf-8 -*-

from .matching import *
from . import __version__
from copy import deepcopy
import numpy as np
import xarray as xr
import trimesh
import dask
from tqdm import tqdm

import functools
import logging
log = logging.getLogger(__name__)

def _preprocess(dat):
    del dat["pair_id"]
    # we do not need all variables
    dat = dat[["capture_time","Dmax", "area","matchScore",
                            "aspectRatio", "angle", "perimeter", "position_3D", 
                            "camera_phi", "camera_theta", "camera_Ofz"]]
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
    ):

    fL = files.FindFiles(case, config.leader, config)
    lv2File = fL.fnamesDaily.level2match

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

    if not fL.isCompleteL1match:
        print("level1match NOT COMPLETE YET %i of %i %s" %
              (len(fL.listFilesExt("level1match")), len(fL.listFiles("level0txt")),  lv2File))
        print("look at ", fL.fnamesPatternExt["level1match"])
        return None, None



    matchFiles = fL.listFilesWithNeighbors("level1match")

    if len(matchFiles) == 0:
        print("level1match NOT AVAILABLE %s" %
              lv2File)
        print("look at ", fL.fnamesPatternExt["level1match"])
        return None, None

    timeIndex = pd.date_range(start=case, end=fL.datetime64 +
                              endTime, freq=freq, inclusive="left")
    timeIndex1 = pd.date_range(start=case, end=fL.datetime64 +
                              endTime, freq=freq, inclusive="both")


    log.info(f"open level1match files {case}")
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        matchDat = xr.open_mfdataset(
            matchFiles, preprocess=_preprocess, combine="nested", concat_dim="pair_id")

    # limit to period of interest
    matchDat = matchDat.isel(pair_id=(matchDat.capture_time.isel(camera=0) >= fL.datetime64).values & (
        matchDat.capture_time.isel(camera=0) < (fL.datetime64 + endTime)).values)

    # make chunks more regular
    matchDat = matchDat.chunk(pair_id=10000)

    # apply matchScore threshold
    matchCond = (matchDat.matchScore >= minMatchScore).values
    log.info(tools.concat("matchCond applies to", (matchCond.sum()/len(matchCond))*100, "% of data"))
    matchDat = matchDat.isel(pair_id=matchCond)

    if len(matchDat.matchScore) == 0:
        log.warning("no data remians after matchScore filtering %s" %
              lv2File)
        return None, None   

    sizeCond = (matchDat.Dmax < max(DbinsPixel)).all("camera").values
    matchDat = matchDat.isel(pair_id=sizeCond)
    if len(matchDat.matchScore) == 0:
        log.warning("no data remians after size filtering %s" %
              lv2File)
        return None, None   


    # switch to time coordinate
    matchDatT = matchDat.assign_coords(time=xr.DataArray(
        matchDat.capture_time.isel(camera=0).values, coords=[matchDat.pair_id]))
    del matchDatT["pid"]
    del matchDatT["capture_time"]

    # estimate max, mean and min for both cameras
    matchDatTJ = (matchDatT.max("camera"), matchDatT.mean("camera"), matchDatT.min("camera"), matchDatT.sel(
        camera=config.leader, drop=True), matchDatT.sel(camera=config.follower, drop=True))
    matchDatTJ = xr.concat(matchDatTJ, dim="camera")
    matchDatTJ["camera"] = ["max", "mean", "min", "leader", "follower"]
    # position_3D is the same for all
    matchDatTJ["position_3D"] = matchDatTJ["position_3D"].sel(
        camera="max", drop=True)

    del matchDatTJ["camera_phi"]
    del matchDatTJ["camera_theta"]
    del matchDatTJ["camera_Ofz"]


    # remove particles too close to the edge
    # this is possible for non symmetrical particles
    DmaxHalf = matchDatTJ.Dmax.sel(camera="max", drop=True)/2
    farEnoughFromBorder = (
        (matchDatTJ.position_3D.sel(dim3D=["x", "y", "z"]) >= DmaxHalf).all("dim3D") &
        (config.frame_width - matchDatTJ.position_3D.sel(dim3D=["x", "y"]) >= DmaxHalf).all("dim3D") &
        (config.frame_height -
         matchDatTJ.position_3D.sel(dim3D="z", drop=True) >= DmaxHalf)
    )
    log.info(tools.concat("farEnoughFromBorder applies to",
          (farEnoughFromBorder.sum()/len(farEnoughFromBorder)).values*100, 
          "% of data"))

    log.info(tools.concat("farEnoughFromBorder applies to",
          (farEnoughFromBorder.sum()/len(farEnoughFromBorder)).values*100, 
          "% of data"))
    matchDatTJ = matchDatTJ.isel(pair_id=farEnoughFromBorder)
    if len(matchDatTJ.pair_id) == 0:
        log.warning("no data remians after farEnoughFromBorder filtering %s" %
              lv2File)
        return None, None   

    matchDatTJ = addPerParticleVariables(matchDatTJ)

    #position is not needed nay more
    del matchDatTJ["position_3D"]

    # now load data
    matchDatTJ.load()

    # split data in 1 min chunks
    matchDatG = matchDatTJ.groupby_bins("time", timeIndex1, squeeze=False)

    sizeDefinitions = ["Dmax", "Dequiv"]
    # process each 1 min chunks

    # iterate through every 1 min piece
    res = {}
    for interv, matchDatG1 in tqdm(matchDatG):
        #print(interv)
        tmp = []
        # for each camera/min/max/mean seperately
        for cam in matchDatG1.camera:
            # estimate counts
            tmpXr = []
            for sizeDefinition in sizeDefinitions:
                tmpXr1 = matchDatG1[[sizeDefinition]].sel(camera=cam).groupby_bins(
                    sizeDefinition, DbinsPixel).count().fillna(0)
                tmpXr1 = tmpXr1.rename({sizeDefinition: "N"})
                tmpXr1 = tmpXr1.rename({f"{sizeDefinition}_bins": "D_bins"})
                # estimate mean values for "area", "angle", "aspectRatio", "perimeter"
                # Dmax is only for technical resaons and is removed afterwards
                otherVars1 = matchDatG1[["area", "angle", "aspectRatio", "perimeter", sizeDefinition]].sel(
                    camera=cam).groupby_bins(sizeDefinition, DbinsPixel).mean()
                del otherVars1[sizeDefinition]
                otherVars1 = otherVars1.rename({k: f"{k}_dist" for k in otherVars1.data_vars})
                otherVars1 = otherVars1.rename({f"{sizeDefinition}_bins": "D_bins"})
                tmpXr1.update(otherVars1)
                tmpXr.append(tmpXr1)

            tmpXr = xr.concat(tmpXr, dim="size_definition")
            tmpXr["size_definition"] = sizeDefinitions

            tmp.append(xr.Dataset(tmpXr))
        # merge camera/min/max/mean reults
        res[interv.left] = xr.concat(tmp, dim="camera")
        # add camera/min/max/mean information
        res[interv.left]["camera"] = matchDatG1.camera


    dist = xr.concat(res.values(), dim="time")
    dist["time"] = list(res.keys())

    # fill data gaps with zeros
    dist = dist.reindex(time=timeIndex)
    dist["N"] = dist["N"].fillna(0)

    log.info("do mean values")
    # estimate mean values
    meanValues = matchDatG.mean()
    meanValues = meanValues.rename({k: f"{k}_mean" for k in meanValues.data_vars})
    meanValues = meanValues.rename(time_bins="time")
    # we want tiem stamps not intervals
    meanValues["time"] = [a.left for a in meanValues["time"].values]

    log.info("merge and load data")
    level2dat = xr.merge((dist, meanValues))
    level2dat.load()

    calibDat = calibrateData(level2dat, matchDatT, config, DbinsPixel, timeIndex1)
    calibDat = addVariables(calibDat, case, config, timeIndex, timeIndex1, 
        blockedPixThresh=blockedPixThresh, blowingSnowFrameThresh=blowingSnowFrameThresh)

    #clean up!
    matchDat.close()
    del matchDat, matchDatG, matchDatTJ, matchDatT, tmpXr, tmp, tmpXr1

    fL.createDirs()
    calibDat = tools.finishNc(calibDat)
    calibDat.to_netcdf(lv2File)
    print("DONE", lv2File)

    return calibDat, lv2File

def addPerParticleVariables(matchDatTJ):
    # add area equivalent radius
    matchDatTJ["Dequiv"] = np.sqrt(4*matchDatTJ["area"]/np.pi)

    #based on Garrett, T. J., and S. E. Yuter, 2014: Observed influence of riming, temperature, and turbulence on the fallspeed of solid precipitation. Geophys. Res. Lett., 41, 6515–6522, doi:10.1002/2014GL061016.
    matchDatTJ["complexityBW"] = matchDatTJ["perimeter"]/(np.pi * matchDatTJ["Dequiv"])
    # matchDatTJ["complexity"] = matchDatTJ["complexityBW"] * 

    return matchDatTJ

def addVariables(calibDat, case, config, timeIndex, timeIndex1, blockedPixThresh=0.1, blowingSnowFrameThresh=0.05):


    # 1 min data
    deltaT = int(timeIndex.freq.nanos * 1e-9) * config.fps
    # 1 pixel size bins
    deltaD = config.resolution * 1e-6

    calibDat["PSD"] = calibDat["counts"] / deltaT / \
        calibDat["obs_volume"] / deltaD  # now in 1/m4
    calibDat["Ntot"] = (calibDat["counts"] / deltaT /
                        calibDat["obs_volume"]).sum("D_bins")

    M2 = (calibDat.PSD.fillna(0)*deltaD*calibDat.D_bins**2).sum("D_bins")
    M3 = (calibDat.PSD.fillna(0)*deltaD*calibDat.D_bins**3).sum("D_bins")
    M4 = (calibDat.PSD.fillna(0)*deltaD*calibDat.D_bins**4).sum("D_bins")

    calibDat["D23"] = M3/M2
    calibDat["D34"] = M4/M3

    # quality variables
    recordingFailed, processingFailed, blockedPixels, blowingSnowRatio = getDataQuality(
        case, config, timeIndex, timeIndex1)
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

    calibDatFilt["recordingFailed"] = recordingFailed
    calibDatFilt["processingFailed"] = processingFailed
    calibDatFilt["blowingSnowRatio"] = blowingSnowRatio
    calibDatFilt["blockedPixelRatio"] = xr.concat((blockedPixels.max("camera"), 
        blockedPixels.mean(
        "camera"), 
        blockedPixels.min("camera"), 
        blockedPixels.sel(camera="leader", drop=True), 
        blockedPixels.sel(camera="follower", drop=True)
        ), dim="camera").T
    calibDatFilt["blowingSnowRatio"] = xr.concat((blowingSnowRatio.max("camera"), 
        blowingSnowRatio.mean(
        "camera"), 
        blowingSnowRatio.min("camera"), 
        blowingSnowRatio.sel(camera="leader", drop=True), 
        blowingSnowRatio.sel(camera="follower", drop=True)
        ), dim="camera").T


    return calibDatFilt


def estimateObservationVolume(matchDatT, config, DbinsPixel, timeIndex1):

    '''
    in pixel

    '''

    rotDat = matchDatT[["camera_phi", "camera_theta", "camera_Ofz"]].sel(
        camera_rotation="mean").groupby_bins("time", timeIndex1, squeeze=False).mean()
    rotDat = rotDat.rename(time_bins="time")
    # we want time stamps not intervals
    rotDat["time"] = [a.left for a in rotDat["time"].values]

    rotDat.load()
    rotDat = rotDat.round(2)

    volumes = []
    for ii in range(len(rotDat.time)):
        Ds, volume = estimateVolumes(config.frame_width,
                                     config.frame_height,
                                     rotDat.camera_phi.values[ii],
                                     rotDat.camera_theta.values[ii],
                                     rotDat.camera_Ofz.values[ii],
                                     sizeBins=DbinsPixel)

        volumes.append(volume[1:])
    return volumes


def calibrateData(level2dat, matchDatT, config, DbinsPixel, timeIndex1):
    '''go from pixel to SI units'''

    assert "intercept" in config.calibration.keys()
    assert "slope" in config.calibration.keys()

    slope = config.calibration.slope
    intercept = config.calibration.intercept

    calibDat = level2dat.rename(N="counts").copy()

    volumes = estimateObservationVolume(matchDatT, config, DbinsPixel, timeIndex1)
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
    calibDat["area_mean"] = calibDat["area_mean"]  / slope**2 / 1e6**2
    calibDat["perimeter_mean"] = calibDat["perimeter_mean"] / slope / 1e6
    # is the intercept correction needed for Dequiv? good question...
    calibDat["Dequiv_mean"] = (calibDat["Dequiv_mean"]) / slope / 1e6

    return calibDat


def applyCalib(pixel, slope, intercept):
    #pix = slope*um + intercept
    um = (pixel - intercept)/slope
    m = um / 1e6
    return m

def getDataQuality(case, config, timeIndex, timeIndex1):
    """Estimate data quality for level2 
    """

    fL = files.FindFiles(case, config.leader, config)
    fF = files.FindFiles(case, config.follower, config)

    fnameL = fL.listFilesWithNeighbors("metaEvents")
    fnameF = fF.listFilesWithNeighbors("metaEvents")

    eventL = xr.open_mfdataset(fnameL).load()
    eventF = xr.open_mfdataset(fnameF).load()

    matchFilesAll = fL.listFilesExt("level1match")
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
