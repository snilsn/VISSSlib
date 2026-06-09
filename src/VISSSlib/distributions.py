# -*- coding: utf-8 -*-

import functools
import operator
import sys
import warnings
from copy import deepcopy

import numpy as np
import xarray as xr
from loguru import logger as log

from . import __version__, quicklooks
from .matching import *

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _preprocess(dat):
    """
    Preprocess the input data for further processing in distributions.

    This function handles the initial data preparation steps, such as filtering
    variables and renaming coordinates, to ensure consistency in subsequent
    processing steps.

    Parameters
    ----------
    dat : xarray.Dataset
        Input dataset to be preprocessed.

    Returns
    -------
    xarray.Dataset
        Preprocessed dataset ready for further analysis.
    """
    try:
        # we do not need all variables
        data_vars = [
            "capture_time",
            "Dmax",
            "area",
            "aspectRatio",
            "angle",
            "perimeter"
        ]
        if "pair_id" in dat.coords:
            del dat["pair_id"]
            data_vars += [
                "matchScore",
                "position3D_center",
                "position3D_centroid",
                "position_centroid",
                "camera_phi",
                "camera_theta",
                "camera_Ofz",
            ]
        else:
            dat = dat.rename(
                pid="pair_id"
            )  # saves lots of trouble later of the main dimension has the same name
            data_vars += ["blur", "Droi", "position_upperLeft"]

        if "track_id" in dat.data_vars:
            data_vars += ["track_id", "track_step"]
            # make track_ids unique, use only day-hour-minute-second, otherwise number is too large
            offset = int(
                dat.encoding["source"].split("_")[-1].split(".")[0].replace("-", "")[6:]
            ) * int(1e9)
            dat["track_id"].values = dat["track_id"].values + offset
        dat = dat[data_vars]
    except:
        log.error(dat.encoding["source"])
        raise KeyError
    return dat


_operators = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}

_select = {
    "max": np.max,
    "mean": np.mean,
    "min": np.min,
}


@tools.loopify_with_camera
def createLevel2detect(
    case,
    camera,
    config,
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "D"),
    skipExisting=True,
    writeNc=True,
    applyFilters=[],
    doPlot=True,
    doParticlePlot=True,
):
    """
    Create level2detect data for a given case and camera.

    Processes raw particle detection data to generate statistical distributions
    and quality metrics for precipitation analysis.

    Parameters
    ----------
    case : str
        Case identifier.
        This can be a date string (YYYYMMDD) or date range (YYYYMMDD-YYYYMMDD)
        or comma-separated dates (YYYYMMDD,YYYYMMDD,YYYYMMDD).
    camera : str
        Camera identifier.
        this can be also "all", "leader", or "follower".
    config : dict
        Configuration settings.
    DbinsPixel : range, optional
        Pixel bins for size distribution, defaults to range(301).
    sizeDefinitions : list, optional
        Size definitions, defaults to ["Dmax", "Dequiv"].
    endTime : numpy.timedelta64, optional
        End time for processing, defaults to np.timedelta64(1, "D").
    skipExisting : bool, optional
        Skip existing outputs, defaults to True.
    writeNc : bool, optional
        Write NetCDF outputs, defaults to True.
    applyFilters : list, optional
        Filters to apply, defaults to [].
        Each filter is a tuple with:
        1) variable name (e.g. aspectRatio, all lv1 variables work)
        2) Operator, one of '>','<','>=','<=','=='
        3) Value for comparison or a list of two values, used as intercept and slope (in this order) for comparison with a linear function of Dmax
        4) if variable contains extra dimensions, which one to select, {} otherwise

        Example to get all particles > 10 pixels (using max of both cameras) with
        aspectRatio >= 0.7 (using min of both cameras)
        applyFilters = [
            ("Dmax",">",10,"max",{}),
            ("aspectRatio",">",0.7,"min",{"fitMethod":'cv2.fitEllipseDirect'}),
        ]
    doPlot : bool, optional
        Create plots, defaults to True.
    doParticlePlot : bool, optional
        Create particle plots, defaults to True.

    Returns
    -------
    tuple
        Processed dataset and file path.
    """
    out = _createLevel2(
        case,
        config,
        DbinsPixel=DbinsPixel,
        sizeDefinitions=sizeDefinitions,
        endTime=endTime,
        skipExisting=skipExisting,
        writeNc=writeNc,
        applyFilters=applyFilters,
        sublevel="detect",
        camera=camera,
    )
    if doPlot:
        print(f"Running createLevel2detectQuicklook for {case}")
        quicklooks.createLevel2detectQuicklook(
            case, camera, config, skipExisting=skipExisting
        )
    if doParticlePlot:
        print(f"Running createLevel1detectQuicklook for {case}")
        quicklooks.createLevel1detectQuicklook(
            case, camera, config, skipExisting=skipExisting
        )

    return out


@tools.loopify
def createLevel2match(
    case,
    config,
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "D"),
    skipExisting=True,
    writeNc=True,
    applyFilters=[],
    doPlot=True,
    doParticlePlot=True,
):
    """
    Create level2match data for a given case and camera.

    Processes raw particle detection data to generate statistical distributions
    and quality metrics for precipitation analysis.

    Parameters
    ----------
    case : str
        Case identifier.
        This can be a date string (YYYYMMDD) or date range (YYYYMMDD-YYYYMMDD)
        or comma-separated dates (YYYYMMDD,YYYYMMDD,YYYYMMDD).
    config : dict
        Configuration settings.
    DbinsPixel : range, optional
        Pixel bins for size distribution, defaults to range(301).
    sizeDefinitions : list, optional
        Size definitions, defaults to ["Dmax", "Dequiv"].
    endTime : numpy.timedelta64, optional
        End time for processing, defaults to np.timedelta64(1, "D").
    skipExisting : bool, optional
        Skip existing outputs, defaults to True.
    writeNc : bool, optional
        Write NetCDF outputs, defaults to True.
    applyFilters : list, optional
        Filters to apply, defaults to [].
        Each filter is a tuple with:
        1) variable name (e.g. aspectRatio, all lv1 variables work)
        2) Operator, one of '>','<','>=','<=','=='
        3) Value for comparison or a list of two values, used as intercept and slope (in this order) for comparison with a linear function of Dmax
        4) if variable contains extra dimensions, which one to select, {} otherwise

        Example to get all particles > 10 pixels (using max of both cameras) with
        aspectRatio >= 0.7 (using min of both cameras)
        applyFilters = [
            ("Dmax",">",10,"max",{}),
            ("aspectRatio",">",0.7,"min",{"fitMethod":'cv2.fitEllipseDirect'}),
        ]
    doPlot : bool, optional
        Create plots, defaults to True.
    doParticlePlot : bool, optional
        Create particle plots, defaults to True.

    Returns
    -------
    tuple
        Processed dataset and file path.
    """
    out = _createLevel2(
        case,
        config,
        DbinsPixel=DbinsPixel,
        sizeDefinitions=sizeDefinitions,
        endTime=endTime,
        skipExisting=skipExisting,
        writeNc=writeNc,
        applyFilters=applyFilters,
        sublevel="match",
    )
    if doPlot:
        print(f"Running createLevel2matchQuicklook for {case}")
        quicklooks.createLevel2matchQuicklook(case, config, skipExisting=skipExisting)
    if doParticlePlot:
        print(f"Running createLevel1matchParticlesQuicklook for {case}")
        quicklooks.createLevel1matchParticlesQuicklook(
            case, config, skipExisting=skipExisting
        )

    return out


@tools.loopify
def createLevel2track(
    case,
    config,
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "D"),
    skipExisting=True,
    writeNc=True,
    applyFilters=[],
    doPlot=True,
):
    """
    Create level2track data for a given case and camera.

    Processes raw particle detection data to generate statistical distributions
    and quality metrics for precipitation analysis.

    Parameters
    ----------
    case : str
        Case identifier.
        This can be a date string (YYYYMMDD) or date range (YYYYMMDD-YYYYMMDD)
        or comma-separated dates (YYYYMMDD,YYYYMMDD,YYYYMMDD).
    config : dict
        Configuration settings.
    DbinsPixel : range, optional
        Pixel bins for size distribution, defaults to range(301).
    sizeDefinitions : list, optional
        Size definitions, defaults to ["Dmax", "Dequiv"].
    endTime : numpy.timedelta64, optional
        End time for processing, defaults to np.timedelta64(1, "D").
    skipExisting : bool, optional
        Skip existing outputs, defaults to True.
    writeNc : bool, optional
        Write NetCDF outputs, defaults to True.
    applyFilters : list, optional
        Filters to apply, defaults to [].
        Each filter is a tuple with:
        1) variable name (e.g. aspectRatio, all lv1 variables work)
        2) Operator, one of '>','<','>=','<=','=='
        3) Value for comparison or a list of two values, used as intercept and slope (in this order) for comparison with a linear function of Dmax
        4) if variable contains extra dimensions, which one to select, {} otherwise

        Example to get all particles > 10 pixels (using max of both cameras) with
        aspectRatio >= 0.7 (using min of both cameras)
        applyFilters = [
            ("Dmax",">",10,"max",{}),
            ("aspectRatio",">",0.7,"min",{"fitMethod":'cv2.fitEllipseDirect'}),
        ]
    doPlot : bool, optional
        Create plots, defaults to True.
    doParticlePlot : bool, optional
        Create particle plots, defaults to True.

    Returns
    -------
    tuple
        Processed dataset and file path.
    """
    out = _createLevel2(
        case,
        config,
        DbinsPixel=DbinsPixel,
        sizeDefinitions=sizeDefinitions,
        endTime=endTime,
        skipExisting=skipExisting,
        writeNc=writeNc,
        applyFilters=applyFilters,
        sublevel="track",
    )
    if doPlot:
        print(f"Running createLevel2trackQuicklook for {case}")
        quicklooks.createLevel2trackQuicklook(case, config, skipExisting=skipExisting)

    return out


def _createLevel2(
    case,
    config,
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "D"),
    skipExisting=True,
    writeNc=True,
    hStep=1,
    sublevel="match",
    camera="leader",
    applyFilters=[],
):
    """
    Internal function to create level 2 data for different sublevels.

    Core processing function that orchestrates the creation of level 2 data
    for match, track, or detect sublevels.

    """
    import dask.array
    import pandas as pd
    from dask.diagnostics import ProgressBar

    assert sublevel in ["match", "track", "detect"]
    if type(config) is str:
        config = tools.readSettings(config)

    fL = files.FindFiles(case, camera, config)
    lv2File = fL.fnamesDaily[f"level2{sublevel}"]

    log.info(f"Processing {lv2File}")

    if skipExisting and tools.checkForExisting(
        lv2File,
        events=fL.listFiles("metaEvents"),
        parents=fL.listFilesExt(f"level1{sublevel}"),
    ):
        return None, None
    if skipExisting and tools.checkForExisting(
        "%s.nodata" % lv2File,
        events=fL.listFiles("metaEvents"),
        parents=fL.listFilesExt(f"level1{sublevel}"),
    ):
        return None, None

    #    if len(fL.listFiles("metaFrames")) > len(fL.listFiles("level0")):
    #        print("DATA TRANSFER INCOMPLETE ", lv2File)
    #        print(len(fL.listFiles("level0")), "of", len(fL.listFiles("metaFrames")), "transmitted")
    #        return None, None

    # is that smart??
    noLevel0 = len(fL.listFilesExt(f"level0txt")) == 0
    if noLevel0:
        with tools.open2("%s.nodata" % lv2File, "w") as f:
            f.write("no level 0 data for %s" % case)

    if sublevel == "match":
        if not fL.isCompleteL1match:
            log.error(
                "level1match NOT COMPLETE YET %i of %i %s"
                % (
                    len(fL.listFilesExt("level1match")),
                    len(fL.listFiles("level0txt")),
                    lv2File,
                )
            )
            log.error("look at %s" % fL.fnamesPatternExt["level1match"])
            return None, None
    elif sublevel == "track":
        if not fL.isCompleteL1track:
            log.error(
                "level1track NOT COMPLETE YET %i of %i %s"
                % (
                    len(fL.listFilesExt("level1track")),
                    len(fL.listFiles("level0txt")),
                    lv2File,
                )
            )
            log.error("look at %s" % fL.fnamesPatternExt["level1track"])
            return None, None
    elif sublevel == "detect":
        if not fL.isCompleteL1detect:
            log.error(
                "level1detect NOT COMPLETE YET %i of %i %s"
                % (
                    len(fL.listFilesExt("level1detect")),
                    len(fL.listFiles("level0txt")),
                    lv2File,
                )
            )
            log.error("look at %s" % fL.fnamesPatternExt["level1detect"])
            return None, None
    else:
        raise ValueError

    allBroken = np.all(
        [f.endswith("broken.txt") for f in fL.listFilesExt(f"level1{sublevel}")]
    )

    if allBroken:
        raise RuntimeError(
            f"All level1{sublevel} in {fL.fnamesPatternExt[f'level1{sublevel}']} are broken."
        )

    lv1Files = fL.listFilesWithNeighbors(f"level1{sublevel}")

    # if len(lv1Files) == 0:
    #     log.error("level1 NOT AVAILABLE %s" % lv2File)
    #     log.error("look at %s" % fL.fnamesPatternExt[f"level1{sublevel}"])
    #     return None, None

    timeIndex = pd.date_range(
        start=case,
        end=fL.datetime64 + endTime,
        freq=config.level2.freq,
        inclusive="left",
    )
    timeIndex1 = pd.date_range(
        start=case,
        end=fL.datetime64 + endTime,
        freq=config.level2.freq,
        inclusive="both",
    )

    allEmpty = False
    if len(case) > 8:
        lv2Dat = _createLevel2part(
            case,
            config,
            freq=config.level2.freq,
            DbinsPixel=DbinsPixel,
            sizeDefinitions=sizeDefinitions,
            endTime=endTime,
            skipExisting=skipExisting,
            sublevel=sublevel,
            applyFilters=applyFilters,
            camera=camera,
        )
        if lv2Dat is not None:
            log.info(f"load data for {case}")
            with ProgressBar():
                lv2Dat.load()
        else:
            allEmpty = True

    else:
        # due to performance reasons, split into hourly chunks and process sperately
        lv2Dat = []
        for hh in range(0, 24, hStep):
            case1 = f"{case}-{hh:02d}"

            lv2Dat1 = _createLevel2part(
                case1,
                config,
                freq=config.level2.freq,
                DbinsPixel=DbinsPixel,
                sizeDefinitions=sizeDefinitions,
                endTime=np.timedelta64(1, "h"),
                skipExisting=skipExisting,
                sublevel=sublevel,
                applyFilters=applyFilters,
                camera=camera,
            )

            if lv2Dat1 is not None:
                log.info(f"load data for {case1}")
                with ProgressBar():
                    lv2Dat.append(lv2Dat1.load())

        try:
            lv2Dat = xr.concat(lv2Dat, dim="time")
        except:
            allEmpty = True
    if allEmpty:
        with tools.open2("%s.nodata" % lv2File, config, "w") as f:
            f.write("no data for %s" % case)
        log.warning("no data for %s" % case)
        log.warning("written: %s.nodata" % lv2File)
        return lv2Dat, lv2File

    # fill up missing data
    lv2Dat = lv2Dat.reindex(time=timeIndex)

    # missing variables
    lv2Dat = addVariables(
        lv2Dat,
        case,
        config,
        timeIndex,
        timeIndex1,
        sublevel,
        camera=camera,
    )

    lv2Dat = tools.finishNc(lv2Dat, config.site, config.visssGen)

    lv2Dat.D_bins.attrs.update(
        dict(units="m", long_name="size bins", comment="label at center of bin")
    )
    lv2Dat.fitMethod.attrs.update(
        dict(units="string", long_name="fit method to estimate aspect ratio")
    )
    lv2Dat.size_definition.attrs.update(
        dict(units="string", long_name="size definition")
    )
    lv2Dat.time.attrs.update(
        dict(long_name="time", comment="label at the end of time interval")
    )

    if sublevel != "detect":
        lv2Dat.camera.attrs.update(dict(units="string", long_name="camera"))

    lv2Dat.D32.attrs.update(dict(units="m", long_name="mean mass-weighted diameter"))
    lv2Dat.D43.attrs.update(
        dict(units="m", long_name="ratio of forth to third PSD moment")
    )
    lv2Dat.D_bins_left.attrs.update(dict(units="m", long_name="left edge D_bins"))
    lv2Dat.D_bins_right.attrs.update(dict(units="m", long_name="right edge D_bin"))
    lv2Dat.Dequiv_mean.attrs.update(
        dict(units="m", long_name="mean sphere equivalent diameter")
    )
    lv2Dat.Dequiv_std.attrs.update(
        dict(units="m", long_name="standard deviation sphere equivalent diameter")
    )
    lv2Dat.Dmax_mean.attrs.update(dict(units="m", long_name="mean maximum diameter"))
    lv2Dat.Dmax_std.attrs.update(
        dict(units="m", long_name="standard deviation maximum diameter")
    )
    lv2Dat.Dmax_dist.attrs.update(
        dict(units="m", long_name="maximum diameter distribution")
    )
    lv2Dat.M1.attrs.update(
        dict(units="m", long_name="1st moment of the size distribution")
    )
    lv2Dat.M2.attrs.update(
        dict(units="m^2", long_name="2nd moment of the size distribution")
    )
    lv2Dat.M3.attrs.update(
        dict(units="m^3", long_name="3rd moment of the size distribution")
    )
    lv2Dat.M4.attrs.update(
        dict(units="m^4", long_name="4th moment of the size distribution")
    )
    lv2Dat.M6.attrs.update(
        dict(units="m^6", long_name="6th moment of the size distribution")
    )
    lv2Dat.N0_star_32.attrs.update(
        dict(
            units="1/m^3/m",
            long_name="PSD scaling parameter based on the second and third PSD moments",
        )
    )
    lv2Dat.N0_star_43.attrs.update(
        dict(
            units="1/m^3/m",
            long_name="PSD scaling parameter based on the third and fourth PSD moments",
        )
    )
    lv2Dat.Ntot.attrs.update(
        dict(units="1/m^3", long_name="Integral over size distribution")
    )
    lv2Dat.PSD.attrs.update(
        dict(units="1/m^3/m", long_name="Particle size distribution")
    )
    lv2Dat.angle_dist.attrs.update(dict(units="deg", long_name="angle distribution"))
    lv2Dat.angle_mean.attrs.update(dict(units="deg", long_name="mean angle"))
    lv2Dat.angle_std.attrs.update(
        dict(units="deg", long_name="standard deviation angle")
    )
    lv2Dat.area_dist.attrs.update(dict(units="m^2", long_name="area distribution"))
    lv2Dat.area_mean.attrs.update(dict(units="m^2", long_name="mean area"))
    lv2Dat.area_std.attrs.update(dict(units="m^2", long_name="standard deviation area"))

    lv2Dat.aspectRatio_dist.attrs.update(
        dict(units="-", long_name="aspectRatio distribution")
    )
    lv2Dat.aspectRatio_mean.attrs.update(dict(units="-", long_name="mean aspect ratio"))
    lv2Dat.aspectRatio_std.attrs.update(
        dict(units="-", long_name="standard deviation aspect ratio")
    )
    lv2Dat.blockedPixelRatio.attrs.update(
        dict(
            units="-", long_name="ratio of frames rejected due to blocked image filter"
        )
    )
    lv2Dat.blowingSnowRatio.attrs.update(
        dict(units="-", long_name="ratio of frames rejected due to blowing snow filter")
    )

    lv2Dat.complexityBW_mean.attrs.update(
        dict(units="-", long_name="complexity distribution (based on shape only)")
    )
    lv2Dat.complexityBW_mean.attrs.update(
        dict(units="-", long_name="mean complexity (based on shape only)")
    )
    lv2Dat.complexityBW_std.attrs.update(
        dict(units="-", long_name="standard deviation complexity (based on shape only)")
    )

    lv2Dat.normalizedRimeMass_mean.attrs.update(
        dict(
            units="-",
            long_name="normalized rime mass distribution (based on shape only)",
        )
    )
    lv2Dat.normalizedRimeMass_mean.attrs.update(
        dict(units="-", long_name="mean normalized rime mass (based on shape only)")
    )
    lv2Dat.normalizedRimeMass_std.attrs.update(
        dict(
            units="-",
            long_name="standard deviation normalized rime mass (based on shape only)",
        )
    )

    lv2Dat.counts.attrs.update(
        dict(
            units="1/min",
            long_name="number of observed particles (not calibrated to observation volume)",
        )
    )
    if sublevel != "detect":
        lv2Dat.matchScore_mean.attrs.update(
            dict(units="-", long_name="mean camera match score")
        )
        lv2Dat.matchScore_std.attrs.update(
            dict(units="-", long_name="standard deviation camera match score")
        )
        lv2Dat.observationsRatio.attrs.update(
            dict(
                units="-",
                long_name="ratio of detected particles by leader and follower",
            )
        )
    lv2Dat.obs_volume.attrs.update(dict(units="m^3", long_name="obs_volume"))
    lv2Dat.perimeter_dist.attrs.update(
        dict(units="m", long_name="perimeter distribution")
    )
    lv2Dat.perimeter_mean.attrs.update(dict(units="m", long_name="mean perimeter"))
    lv2Dat.perimeter_std.attrs.update(
        dict(units="m", long_name="standard deviation perimeter")
    )

    lv2Dat.qualityFlags.attrs.update(
        dict(
            units="m",
            long_name="binary quality Flags",
            comment="For recordingFailed, "
            "processingFailed, cameraBlocked, blowingSnow, obervationsDiffer, "
            "and tracksTooShort. "
            "Use VISSSlib.tools.unpackQualityFlags to unpack",
        )
    )

    if sublevel == "match":
        lv2Dat.nParticles.attrs.update(
            dict(units="-", long_name="number of particle observations")
        )

    elif sublevel == "track":
        lv2Dat.cameratrack.attrs.update(
            dict(
                units="string",
                long_name="camera and track",
                comment="Explains how multiple observations of the same particle by the two cameras along a track are combined",
            )
        )
        lv2Dat.dim3D.attrs.update(dict(units="m", long_name="3 spatial dimensions"))

        lv2Dat.track_length_mean.attrs.update(
            dict(units="# frames", long_name="mean track_length")
        )
        lv2Dat.track_length_std.attrs.update(
            dict(units="# frames", long_name="standard deviation track_length")
        )
        lv2Dat.velocity_dist.attrs.update(
            dict(units="m/s", long_name="velocity distribution")
        )
        lv2Dat.velocity_mean.attrs.update(dict(units="m/s", long_name="mean velocity"))
        lv2Dat.velocity_std.attrs.update(
            dict(units="m/s", long_name="standard deviation velocity")
        )
        lv2Dat.track_angle_dist.attrs.update(
            dict(units="deg", long_name="track_angle distribution to 0,0,1 vector")
        )
        lv2Dat.track_angle_mean.attrs.update(
            dict(units="deg", long_name="mean track_angle to 0,0,1 vector")
        )
        lv2Dat.track_angle_std.attrs.update(
            dict(
                units="deg", long_name="standard deviation track_angle to 0,0,1 vector"
            )
        )
        lv2Dat.nParticles.attrs.update(
            dict(units="-", long_name="number of observed unique particles")
        )

    if writeNc:
        tools.to_netcdf2(lv2Dat, config, lv2File)

    return lv2Dat, lv2File

def createLevel2_multiple_classes(
    level1dat,
    config,
    freq="1min",
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "h"),
    skipExisting=True,
    sublevel="track",
    camera="leader",
    applyFilters=[],
):
    
    lvl2_list = []

    for particle_class in np.unique(level1dat.category):

        mask = level1dat.category == particle_class
        class_dataset = level1dat.where(mask, drop=True)

        lvl2class = createLevel2_single_class(
        class_dataset,
        config)

        lvl2_list.append((particle_class, lvl2class))

    names = [name for name, _ in lvl2_list]
    datasets = [ds for _, ds in lvl2_list]

    combined = xr.concat(datasets, dim="particle_class")
    combined = combined.assign_coords(particle_class=names)

    return combined

def createLevel2_single_class(
    level1dat,
    config,
    freq="1min",
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "h"),
    skipExisting=True,
    sublevel="track",
    camera="leader",
    applyFilters=[],
):
    """
    Create level 2 data for chunk of time
    """
    import dask
    import pandas as pd
    import scipy.stats
    from tqdm import tqdm

    assert sublevel in ["match", "track"]

    start = level1dat.capture_time.min().values
    end = level1dat.capture_time.max().values
    case = pd.Timestamp(level1dat.capture_time.min().values).strftime("%Y%m%d")

    rounded_start = pd.Timestamp(start).floor('min')
    rounded_end = pd.Timestamp(end).ceil('min')
    
    timeIndex = pd.date_range(
        start=rounded_start, end=rounded_end, freq=freq, inclusive="left"
    )
    timeIndex1 = pd.date_range(
        start=rounded_start, end=rounded_end, freq=freq, inclusive="both"
    )

    if sublevel == "detect":
        # apply blur threshold
        if config.visssGen == "visss":
            """
            # coefficients developed from winer 2021/22 in Hyytiälä
            The idea is to estimate the cummulated total PSD for detect and match data
            and find a blur threshold so that both distributions agree. See
            match_analyze_blur_vs_size.ipynb Filters are applied:

                processing failed by selecting only detect files where match is present
                blocked data by VISSSlib quality control
                smaller sampling volume of matched data is considered by estimatign correction factor

            """
            blurThresh = np.array(
                [
                    np.nan,
                    800.0,
                    773.0,
                    360.0,
                    363.0,
                    436.0,
                    523.0,
                    568.0,
                    633.0,
                    621.0,
                    607.0,
                    574.0,
                    556.0,
                    534.0,
                    506.0,
                    477.0,
                    465.0,
                    435.0,
                    416.0,
                    394.0,
                    374.0,
                    356.0,
                    327.0,
                    314.0,
                    300.0,
                    292.0,
                    275.0,
                    263.0,
                    256.0,
                    248.0,
                    237.0,
                    232.0,
                    226.0,
                    220.0,
                    207.0,
                    202.0,
                    196.0,
                    191.0,
                    183.0,
                    176.0,
                    172.0,
                    170.0,
                    161.0,
                    158.0,
                    153.0,
                    151.0,
                    148.0,
                    144.0,
                    141.0,
                    138.0,
                    140.0,
                    132.0,
                    132.0,
                    131.0,
                    130.0,
                    129.0,
                    125.0,
                    126.0,
                    124.0,
                    123.0,
                    119.0,
                    119.0,
                    120.0,
                    120.0,
                    119.0,
                    117.0,
                    117.0,
                    117.0,
                    118.0,
                    117.0,
                    115.0,
                    114.0,
                    115.0,
                    115.0,
                    115.0,
                    118.0,
                    113.0,
                    114.0,
                    116.0,
                    113.0,
                    112.0,
                    114.0,
                    112.0,
                    112.0,
                    116.0,
                    115.0,
                    112.0,
                    113.0,
                    114.0,
                    111.0,
                    111.0,
                    113.0,
                    112.0,
                ]
                + 400 * [110.0]
            )  # by using 2000 we make sure even huge particles are treated and do not raise an error
        elif config.visssGen == "visss2":
            # coefficients developed from early 2023 in NYA cases with low wind speed
            # "20230213", "20230223", "20230409", "20230429",
            blurThresh = np.array(
                [
                    np.nan,
                    323.0,
                    450.0,
                    282.0,
                    330.0,
                    351.0,
                    378.0,
                    363.0,
                    370.0,
                    352.0,
                    339.0,
                    322.0,
                    305.0,
                    291.0,
                    275.0,
                    259.0,
                    252.0,
                    234.0,
                    225.0,
                    214.0,
                    206.0,
                    196.0,
                    188.0,
                    181.0,
                    173.0,
                    167.0,
                    159.0,
                    154.0,
                    149.0,
                    142.0,
                    137.0,
                    133.0,
                    130.0,
                    124.0,
                    120.0,
                    117.0,
                    114.0,
                    111.0,
                    107.0,
                    105.0,
                    102.0,
                    99.0,
                    97.0,
                    95.0,
                    93.0,
                    91.0,
                    89.0,
                    88.0,
                    85.0,
                    83.0,
                    82.0,
                    81.0,
                    79.0,
                    78.0,
                    77.0,
                    76.0,
                    75.0,
                    74.0,
                    73.0,
                    72.0,
                    72.0,
                    70.0,
                    69.0,
                    69.0,
                    68.0,
                    67.0,
                    66.0,
                    66.0,
                    66.0,
                    64.0,
                    64.0,
                    63.0,
                    63.0,
                    62.0,
                    62.0,
                    61.0,
                    60.0,
                ]
                + 400 * [60.0]
            )

        elif config.visssGen == "visss3":
            # coefficients developed from winer 2023/24 in Hyytiälä
            blurThresh = np.array(
                [
                    np.nan,
                    584.0,
                    1005.0,
                    1181.0,
                    1537.0,
                    1587.0,
                    1397.0,
                    1312.0,
                    1261.0,
                    1181.0,
                    1107.0,
                    1054.0,
                    986.0,
                    931.0,
                    875.0,
                    818.0,
                    788.0,
                    735.0,
                    704.0,
                    668.0,
                    641.0,
                    611.0,
                    585.0,
                    567.0,
                    545.0,
                    525.0,
                    501.0,
                    487.0,
                    472.0,
                    455.0,
                    441.0,
                    427.0,
                    419.0,
                    404.0,
                    391.0,
                    384.0,
                    373.0,
                    366.0,
                    354.0,
                    348.0,
                    342.0,
                    335.0,
                    327.0,
                    320.0,
                    315.0,
                    309.0,
                    303.0,
                    297.0,
                    292.0,
                    289.0,
                    283.0,
                    278.0,
                    275.0,
                    271.0,
                    266.0,
                    263.0,
                    259.0,
                    257.0,
                    253.0,
                    251.0,
                    247.0,
                    244.0,
                    242.0,
                    238.0,
                    235.0,
                    232.0,
                    229.0,
                    228.0,
                    226.0,
                    224.0,
                    221.0,
                    220.0,
                    219.0,
                    216.0,
                    214.0,
                    211.0,
                    209.0,
                    207.0,
                    205.0,
                    203.0,
                    201.0,
                    203.0,
                    199.0,
                    200.0,
                    198.0,
                    198.0,
                    195.0,
                    194.0,
                    194.0,
                    192.0,
                    192.0,
                    190.0,
                    192.0,
                    189.0,
                    187.0,
                    188.0,
                    187.0,
                    184.0,
                    184.0,
                    187.0,
                    179.0,
                    181.0,
                    183.0,
                    183.0,
                    183.0,
                    184.0,
                    181.0,
                    177.0,
                    181.0,
                    180.0,
                    176.0,
                    175.0,
                    175.0,
                    177.0,
                    172.0,
                    173.0,
                    174.0,
                    175.0,
                    172.0,
                    174.0,
                    174.0,
                    176.0,
                    176.0,
                    174.0,
                    171.0,
                ]
                + 400 * [170.0]
            )

        else:
            raise ValueError(f"VISSS Generation {config.visssGen} not supported")

        # discard huge ones so that the trick with the look up table works
        sizeCond = (level1dat.Dmax <= 350).values
        level1dat = level1dat.isel(pair_id=sizeCond)

        # this works liek a lookup table. We use the Dmax rounded to next
        # integer as an index for blurThresh
        appliedblurThresh = blurThresh[np.around(level1dat.Dmax.values).astype(int)]

        blurCond = (level1dat.blur >= appliedblurThresh).values
        log.info(
            tools.concat(
                "blurCond applies to",
                (blurCond.sum() / len(blurCond)) * 100,
                "% of data",
            )
        )
        level1dat = level1dat.isel(pair_id=blurCond)

        del level1dat["blur"]

        if len(level1dat.pair_id) == 0:
            log.warning("no data remains after blurCond filtering %s" % lv2File)
            return None

    else:  # match or track
        # apply matchScore threshold
        if config.quality.minMatchScore is not None:
            matchCond = (level1dat.matchScore >= config.quality.minMatchScore).values
            log.info(
                tools.concat(
                    "matchCond applies to",
                    (matchCond.sum() / len(matchCond)) * 100,
                    "% of data",
                )
            )
            level1dat = level1dat.isel(pair_id=matchCond)

            if len(level1dat.matchScore) == 0:
                log.warning("no data remains after matchScore filtering %s" % lv2File)
                return None

    # only for debuging

    for aa, applyFilter in enumerate(applyFilters):
        assert (
            len(applyFilter) == 5
        ), "applyFilters elements must contain filterVar, operator, filterValue, extraDims"
        filterVar, opStr, filterValue, selectCameraStr, extraDims = applyFilter

        # add complexity to data if filtered by complexity
        if filterVar in ["complexityBW"]:
            level1dat = addPerParticleVariables(level1dat, config)

        if selectCameraStr == "max":
            thisDat = level1dat[filterVar].sel(**extraDims).max("camera")
        elif selectCameraStr == "min":
            thisDat = level1dat[filterVar].sel(**extraDims).min("camera")
        elif selectCameraStr == "mean":
            thisDat = level1dat[filterVar].sel(**extraDims).mean("camera")
        else:
            raise ValueError(
                "selectCameraStr must be max, min or mean, received %s", selectCameraStr
            )

        if isinstance(filterValue, list):
            assert (
                len(filterValue) == 2
            ), "filterValue is list, but contains not 2 elements; at the moment only linear functions of Dmax are supported"
            matchCond = _operators[opStr](
                thisDat,
                (filterValue[0] + filterValue[1] * level1dat["Dmax"]).max(dim="camera"),
            ).values

        else:
            matchCond = _operators[opStr](thisDat, filterValue).values

        level1dat = level1dat.isel(pair_id=matchCond)

        if len(level1dat.matchScore) == 0:
            log.warning("no data remains after additional filtering")
            return None

    try:
        sizeCond = (level1dat.Dmax < max(DbinsPixel)).all("camera").values
    except ValueError:
        sizeCond = (level1dat.Dmax < max(DbinsPixel)).values
    level1dat = level1dat.isel(pair_id=sizeCond)
    if len(level1dat.pair_id) == 0:
        log.warning("no data remains after size filtering %s" % lv2File)
        return None

    # remove particles too close to the edge
    # this is possible for non symmetrical particles
    if sublevel == "detect":
        level1dat["position_center"] = level1dat.position_upperLeft + (
            level1dat.Droi / 2
        )
        DmaxHalf = level1dat.Dmax / 2

        farEnoughFromBorder = (
            (level1dat.position_center >= DmaxHalf).all("dim2D")
            & (
                (
                    config.frame_width
                    - level1dat.position_center.sel(dim2D="x", drop=True)
                )
                >= DmaxHalf
            )
            & (
                (
                    config.frame_height
                    - level1dat.position_center.sel(dim2D="y", drop=True)
                )
                >= DmaxHalf
            )
        )

    else:
        DmaxHalf = level1dat.Dmax.max("camera") / 2

        farEnoughFromBorder = (
            (level1dat.position3D_center.sel(dim3D=["x", "y", "z"]) >= DmaxHalf).all(
                "dim3D"
            )
            & (
                (config.frame_width - level1dat.position3D_center.sel(dim3D=["x", "y"]))
                >= DmaxHalf
            ).all("dim3D")
            & (
                (
                    config.frame_height
                    - level1dat.position3D_center.sel(dim3D="z", drop=True)
                )
                >= DmaxHalf
            )
        )

    farEnoughFromBorder = farEnoughFromBorder.compute()

    log.info(
        tools.concat(
            "farEnoughFromBorder applies to",
            (farEnoughFromBorder.sum() / len(farEnoughFromBorder)).values * 100,
            "% of data",
        )
    )

    level1dat = level1dat.isel(pair_id=farEnoughFromBorder)
    if len(level1dat.pair_id) == 0:
        log.warning("no data remains after farEnoughFromBorder filtering %s" % lv2File)
        return None

    if config.level2.correctForSmallOnes and "maxSharpness" in config.keys():
        # remove small particles that are not in the very sharpest region
        # IN DEVELOPMENT
        assert sublevel != "detect"
        log.info("removing small particles outside of most sharp volume")
        isOutsideSharp = False
        l1datL = level1dat.position_centroid.sel(
            dim2D="x", camera=config.leader, drop=True
        ).drop_vars("pid")
        l1datF = level1dat.position_centroid.sel(
            dim2D="x", camera=config.follower, drop=True
        ).drop_vars("pid")
        for size in config.maxSharpness.leader.keys():
            isSize = (level1dat.Dmax.max("camera") >= size) & (
                level1dat.Dmax.max("camera") < size + 1
            )
            outsideLeader = isSize & (
                (l1datL < config.maxSharpness.leader[size][0])
                | (l1datL > config.maxSharpness.leader[size][1])
            )
            outsideFollower = isSize & (
                (l1datF < config.maxSharpness.follower[size][0])
                | (l1datF > config.maxSharpness.follower[size][1])
            )
            isOutsideSharp = isOutsideSharp | outsideLeader | outsideFollower

        isOutsideSharp = isOutsideSharp.compute()
        del l1datL, l1datF

        log.info(
            tools.concat(
                "isOutsideSharp applies NOT to",
                100 - (isOutsideSharp.sum() / len(isOutsideSharp)).values * 100,
                "% of data",
            )
        )

        level1dat = level1dat.isel(pair_id=~isOutsideSharp)
        if len(level1dat.pair_id) == 0:
            log.warning("no data remains after isOutsideSharp filtering %s" % lv2File)
            return None
    else:
        log.info(f"do not remove particles outside of most sharp area")

    if sublevel == "detect":
        level1dat = level1dat.drop_vars(
            ["position_center", "Droi", "position_upperLeft", "dim2D"]
        )
    else:
        level1dat = level1dat.drop_vars("position_centroid")

    # done with filtering the data.
    # the following code is much faster with a loaded netcdf object
    log.info("loading data")
    level1dat.load()

    if sublevel == "detect":
        log.info(f"estimate mean values")

        # promote capture_time to coordimnate for later
        level1dat_time = level1dat.assign_coords(
            time=xr.DataArray(level1dat.capture_time.values, coords=[level1dat.pair_id])
        )

        # fix order
        level1dat_4timeAve = level1dat_time.transpose(*["fitMethod", "pair_id"])

        # save for later
        individualDataPoints = level1dat_4timeAve.pair_id
        individualDataPoints.name = "nParticles"

    elif sublevel == "match":
        log.info(f"estimate camera mean values")
        data_vars = [
            "Dmax",
            "area",
            "matchScore",
            "aspectRatio",
            "angle",
            "perimeter"
        ]

        # promote capture_time to coordimnate for later
        level1dat_time = level1dat.assign_coords(
            time=xr.DataArray(
                level1dat.capture_time.isel(camera=0).values, coords=[level1dat.pair_id]
            )
        )

        # we do not need pid any more
        level1dat_time = level1dat_time.reset_coords("pid")

        # estimate max, mean and min for both cameras
        level1dat_camAve = (
            level1dat_time[data_vars].max("camera"),
            level1dat_time[data_vars].mean("camera"),
            level1dat_time[data_vars].min("camera"),
            level1dat_time[data_vars].sel(camera=config.leader, drop=True),
            level1dat_time[data_vars].sel(camera=config.follower, drop=True),
        )
        level1dat_camAve = xr.concat(level1dat_camAve, dim="camera")
        level1dat_camAve["camera"] = ["max", "mean", "min", "leader", "follower"]

        # fix angle becuase we want the circular mean
        level1dat_camAve["angle"].loc["mean"] = level1dat_time["angle"].reduce(
            scipy.stats.circmean, "camera", high=360, nan_policy="omit"
        )

        # position_3D is the same for all
        # level1dat_camAve["position3D_center"] = level1dat_camAve["position3D_center"].sel(
        #    camera="max", drop=True)

        # fix order
        level1dat_4timeAve = level1dat_camAve.transpose(
            *["camera", "fitMethod", "pair_id"]
        )

        # clean up
        del level1dat_camAve
        # #position is not needed any more
        # del level1dat_4timeAve["position3D_center"]
        # #centroid position is not needed any more
        # del level1dat_4timeAve["position3D_centroid"]

        # save for later
        individualDataPoints = level1dat_4timeAve.pair_id
        individualDataPoints.name = "nParticles"

    elif sublevel == "track":
        (
            level1dat_trackAve,
            level1dat_track2D,
            level1dat_time,
            individualDataPoints,
            _,
        ) = getPerTrackStatistics(level1dat)

        # because there are no weighted groupby operations
        # https://github.com/pydata/xarray/issues/3937, we have to improvise
        # and broadcast the results again to a shape including track_step
        # - then the mean etc. values are dublicated as per track length
        # and the result is weighted when averaging with time
        data_vars = [
            "Dmax",
            "area",
            "matchScore",
            "aspectRatio",
            "angle",
            "perimeter",
            "velocity"
        ]
        for data_var in data_vars:
            level1dat_trackAve[data_var] = level1dat_trackAve[data_var].broadcast_like(
                level1dat_track2D.isel(camera=0, drop=True)[data_var]
            )

        # add back the original individual values (mainly for testing)
        # level1dat_trackAve = xr.concat((level1dat_trackAve,level1dat_track2D.expand_dims(track=["individual"])), dim="cameratrack")

        log.info(f"reshape track data again")
        # call me crazy but now that we have mean track properties broadcasted to every particle we can go back to pair_id!
        level1dat_4timeAve = level1dat_trackAve.stack(
            pair_id=("track_id", "track_step")
        )

        # make sure only data is used within original track length
        notNull = (
            level1dat_track2D.Dmax.isel(camera=0, drop=True)
            .notnull()
            .stack(pair_id=("track_id", "track_step"))
            .compute()
        )

        level1dat_4timeAve = level1dat_4timeAve.isel(pair_id=notNull)
        # multiindex causes trouble below, so just swap with time
        level1dat_4timeAve = level1dat_4timeAve.swap_dims(pair_id="time")
        # for reasons I do not understand there are sometimes a few values with NaT timestamops
        level1dat_4timeAve = level1dat_4timeAve.isel(
            time=~np.isnan(level1dat_4timeAve.time)
        )
        individualDataPoints = individualDataPoints.isel(
            track_id=~np.isnan(individualDataPoints.time)
        )

        # clean up
        del level1dat_trackAve

    else:
        raise ValueError("do not know sublevel")

    # log.info(f"load data")
    ##turned out, it runs about 3 to 4 times faster when NOT using dask beyond this point.
    # level1dat_4timeAve = level1dat_4timeAve.load()

    log.info(f"add additonal variables")
    level1dat_4timeAve = addPerParticleVariables(level1dat_4timeAve, config)

    # split data in 1 min chunks
    level1datG = level1dat_4timeAve.groupby_bins(
        "time", timeIndex1, right=False, squeeze=False
    )
    level1datG_angle = level1dat_4timeAve["angle"].groupby_bins(
        "time", timeIndex1, right=False, squeeze=False
    )
    individualDataPointsG = individualDataPoints.groupby_bins(
        "time", timeIndex1, right=False, squeeze=False
    )

    del level1dat_4timeAve

    sizeDefinitions = ["Dmax", "Dequiv"]
    data_vars = [
        "area",
        "angle",
        "aspectRatio",
        "perimeter",
        "complexityBW",
        "normalizedRimeMass"
    ]
    if sublevel == "track":
        data_vars += ["velocity", "track_angle"]

    log.info(f"get time resolved distributions")
    # process each 1 min chunks
    res = {}
    nParticles = {}

    if sublevel == "detect":
        # iterate through every 1 min piece
        for interv, level1datG1 in tqdm(level1datG, file=sys.stdout):
            # estimate counts
            tmpXr = []
            for sizeDefinition in sizeDefinitions:
                tmpXr1 = (
                    level1datG1[[sizeDefinition]]
                    .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                    .count()
                    .fillna(0)
                )

                tmpXr1 = tmpXr1.rename({sizeDefinition: "N"})
                tmpXr1 = tmpXr1.rename({f"{sizeDefinition}_bins": "D_bins"})

                # import pdb; pdb.set_trace()
                # estimate mean values for "area", "angle", "aspectRatio", "perimeter"
                # Dmax is only for technical resaons and is removed afterwards
                data_vars1 = data_vars + [sizeDefinition]
                data_vars1.remove("angle")  # treated seperately

                otherVars1 = (
                    level1datG1[data_vars1]
                    .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                    .mean()
                )
                angleVars = (
                    level1datG1[["angle", sizeDefinition]]
                    .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                    .reduce(scipy.stats.circmean, high=360, nan_policy="omit")
                )
                otherVars1["angle"] = angleVars["angle"]
                del otherVars1[sizeDefinition]

                otherVars1 = otherVars1.rename(
                    {k: f"{k}_dist" for k in otherVars1.data_vars}
                )
                otherVars1 = otherVars1.rename({f"{sizeDefinition}_bins": "D_bins"})
                tmpXr1.update(otherVars1)
                tmpXr.append(tmpXr1)

            tmpXr = xr.concat(tmpXr, dim="size_definition")
            tmpXr["size_definition"] = sizeDefinitions

            res[interv.left] = xr.Dataset(tmpXr)

        # clean up
        del tmpXr, tmpXr1

        dist = xr.concat(res.values(), dim="time")
        dist["time"] = list(res.keys())

    else:
        if sublevel == "track":
            coordVar = "cameratrack"
        elif sublevel == "match":
            coordVar = "camera"

        # iterate through every 1 min piece
        for interv, level1datG1 in tqdm(level1datG, file=sys.stdout):
            # print(interv)
            tmp = []
            # for each track&camera/min/max/mean seperately
            for coord in level1datG1[coordVar].values:
                # estimate counts
                tmpXr = []
                for sizeDefinition in sizeDefinitions:
                    tmpXr1 = (
                        level1datG1[[sizeDefinition]]
                        .sel(**{coordVar: coord})
                        .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                        .count()
                        .fillna(0)
                    )

                    tmpXr1 = tmpXr1.rename({sizeDefinition: "N"})
                    tmpXr1 = tmpXr1.rename({f"{sizeDefinition}_bins": "D_bins"})

                    # import pdb; pdb.set_trace()
                    # estimate mean values for "area", "angle", "aspectRatio", "perimeter"
                    # Dmax is only for technical resaons and is removed afterwards
                    data_vars1 = data_vars + [sizeDefinition]
                    data_vars1.remove("angle")  # treated seperately
                    otherVars1 = (
                        level1datG1[data_vars1]
                        .sel(**{coordVar: coord})
                        .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                        .mean()
                    )
                    angleVars = (
                        level1datG1[["angle", sizeDefinition]]
                        .sel(**{coordVar: coord})
                        .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                        .reduce(scipy.stats.circmean, high=360, nan_policy="omit")
                    )
                    otherVars1["angle"] = angleVars["angle"]
                    del otherVars1[sizeDefinition]

                    otherVars1 = otherVars1.rename(
                        {k: f"{k}_dist" for k in otherVars1.data_vars}
                    )
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

        # clean up
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
    meanValues["angle"] = level1datG_angle.reduce(
        scipy.stats.circmean, high=360, nan_policy="omit"
    )
    meanValues = meanValues.rename({k: f"{k}_mean" for k in meanValues.data_vars})
    meanValues = meanValues.rename(time_bins="time")
    # we want time stamps not intervals
    meanValues["time"] = [a.left for a in meanValues["time"].values]

    log.info("do temporal std values")
    # estimate mean values
    # to do: data is weighted with number of obs not considering the smalle robservation volume for larger particles
    stdValues = level1datG.std()
    stdValues["angle"] = level1datG_angle.reduce(
        scipy.stats.circstd, high=360, nan_policy="omit"
    )
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

    # clean up!
    del level1datG, level1dat_time

    calibDat = calibDat.reindex(time=timeIndex)

    calibDat = addVariables(
        calibDat,
        case,
        config,
        timeIndex,
        timeIndex1,
        sublevel,
        camera=camera, single_class=True
    )

    calibDat = tools.finishNc(calibDat, config.site, config.visssGen)

    calibDat.D_bins.attrs.update(
        dict(units="m", long_name="size bins", comment="label at center of bin")
    )
    calibDat.fitMethod.attrs.update(
        dict(units="string", long_name="fit method to estimate aspect ratio")
    )
    calibDat.size_definition.attrs.update(
        dict(units="string", long_name="size definition")
    )
    calibDat.time.attrs.update(
        dict(long_name="time", comment="label at the end of time interval")
    )

    #if sublevel != "detect":
        #calibDat.camera.attrs.update(dict(units="string", long_name="camera"))

    calibDat.D32.attrs.update(dict(units="m", long_name="mean mass-weighted diameter"))
    calibDat.D43.attrs.update(
        dict(units="m", long_name="ratio of forth to third PSD moment")
    )
    calibDat.D_bins_left.attrs.update(dict(units="m", long_name="left edge D_bins"))
    calibDat.D_bins_right.attrs.update(dict(units="m", long_name="right edge D_bin"))
    calibDat.Dequiv_mean.attrs.update(
        dict(units="m", long_name="mean sphere equivalent diameter")
    )
    calibDat.Dequiv_std.attrs.update(
        dict(units="m", long_name="standard deviation sphere equivalent diameter")
    )
    calibDat.Dmax_mean.attrs.update(dict(units="m", long_name="mean maximum diameter"))
    calibDat.Dmax_std.attrs.update(
        dict(units="m", long_name="standard deviation maximum diameter")
    )
    calibDat.M1.attrs.update(
        dict(units="m", long_name="1st moment of the size distribution")
    )
    calibDat.M2.attrs.update(
        dict(units="m^2", long_name="2nd moment of the size distribution")
    )
    calibDat.M3.attrs.update(
        dict(units="m^3", long_name="3rd moment of the size distribution")
    )
    calibDat.M4.attrs.update(
        dict(units="m^4", long_name="4th moment of the size distribution")
    )
    calibDat.M6.attrs.update(
        dict(units="m^6", long_name="6th moment of the size distribution")
    )
    calibDat.N0_star_32.attrs.update(
        dict(
            units="1/m^3/m",
            long_name="PSD scaling parameter based on the second and third PSD moments",
        )
    )
    calibDat.N0_star_43.attrs.update(
        dict(
            units="1/m^3/m",
            long_name="PSD scaling parameter based on the third and fourth PSD moments",
        )
    )
    calibDat.Ntot.attrs.update(
        dict(units="1/m^3", long_name="Integral over size distribution")
    )
    calibDat.PSD.attrs.update(
        dict(units="1/m^3/m", long_name="Particle size distribution")
    )
    calibDat.angle_dist.attrs.update(dict(units="deg", long_name="angle distribution"))
    calibDat.angle_mean.attrs.update(dict(units="deg", long_name="mean angle"))
    calibDat.angle_std.attrs.update(
        dict(units="deg", long_name="standard deviation angle")
    )
    calibDat.area_dist.attrs.update(dict(units="m^2", long_name="area distribution"))
    calibDat.area_mean.attrs.update(dict(units="m^2", long_name="mean area"))
    calibDat.area_std.attrs.update(dict(units="m^2", long_name="standard deviation area"))

    calibDat.aspectRatio_dist.attrs.update(
        dict(units="-", long_name="aspectRatio distribution")
    )
    calibDat.aspectRatio_mean.attrs.update(dict(units="-", long_name="mean aspect ratio"))
    calibDat.aspectRatio_std.attrs.update(
        dict(units="-", long_name="standard deviation aspect ratio")
    )
    """
    calibDat.blockedPixelRatio.attrs.update(
        dict(
            units="-", long_name="ratio of frames rejected due to blocked image filter"
        )
    )
    calibDat.blowingSnowRatio.attrs.update(
        dict(units="-", long_name="ratio of frames rejected due to blowing snow filter")
    )
    """
    calibDat.complexityBW_mean.attrs.update(
        dict(units="-", long_name="complexity distribution (based on shape only)")
    )
    calibDat.complexityBW_mean.attrs.update(
        dict(units="-", long_name="mean complexity (based on shape only)")
    )
    calibDat.complexityBW_std.attrs.update(
        dict(units="-", long_name="standard deviation complexity (based on shape only)")
    )

    calibDat.normalizedRimeMass_mean.attrs.update(
        dict(
            units="-",
            long_name="normalized rime mass distribution (based on shape only)",
        )
    )
    calibDat.normalizedRimeMass_mean.attrs.update(
        dict(units="-", long_name="mean normalized rime mass (based on shape only)")
    )
    calibDat.normalizedRimeMass_std.attrs.update(
        dict(
            units="-",
            long_name="standard deviation normalized rime mass (based on shape only)",
        )
    )

    calibDat.counts.attrs.update(
        dict(
            units="1/min",
            long_name="number of observed particles (not calibrated to observation volume)",
        )
    )
    if sublevel != "detect":
        calibDat.matchScore_mean.attrs.update(
            dict(units="-", long_name="mean camera match score")
        )
        calibDat.matchScore_std.attrs.update(
            dict(units="-", long_name="standard deviation camera match score")
        )
        """
        calibDat.observationsRatio.attrs.update(
            dict(
                units="-",
                long_name="ratio of detected particles by leader and follower",
            )
        )
        """
    calibDat.obs_volume.attrs.update(dict(units="m^3", long_name="obs_volume"))
    calibDat.perimeter_dist.attrs.update(
        dict(units="m", long_name="perimeter distribution")
    )
    calibDat.perimeter_mean.attrs.update(dict(units="m", long_name="mean perimeter"))
    calibDat.perimeter_std.attrs.update(
        dict(units="m", long_name="standard deviation perimeter")
    )
    """
    calibDat.qualityFlags.attrs.update(
        dict(
            units="m",
            long_name="binary quality Flags",
            comment="For recordingFailed, "
            "processingFailed, cameraBlocked, blowingSnow, obervationsDiffer, "
            "and tracksTooShort. "
            "Use VISSSlib.tools.unpackQualityFlags to unpack",
        )
    )
    """
    if sublevel == "match":
        calibDat.nParticles.attrs.update(
            dict(units="-", long_name="number of particle observations")
        )

    elif sublevel == "track":
        calibDat.cameratrack.attrs.update(
            dict(
                units="string",
                long_name="camera and track",
                comment="Explains how multiple observations of the same particle by the two cameras along a track are combined",
            )
        )
        calibDat.dim3D.attrs.update(dict(units="m", long_name="3 spatial dimensions"))

        calibDat.track_length_mean.attrs.update(
            dict(units="# frames", long_name="mean track_length")
        )
        calibDat.track_length_std.attrs.update(
            dict(units="# frames", long_name="standard deviation track_length")
        )
        calibDat.velocity_dist.attrs.update(
            dict(units="m/s", long_name="velocity distribution")
        )
        calibDat.velocity_mean.attrs.update(dict(units="m/s", long_name="mean velocity"))
        calibDat.velocity_std.attrs.update(
            dict(units="m/s", long_name="standard deviation velocity")
        )
        calibDat.track_angle_dist.attrs.update(
            dict(units="deg", long_name="track_angle distribution to 0,0,1 vector")
        )
        calibDat.track_angle_mean.attrs.update(
            dict(units="deg", long_name="mean track_angle to 0,0,1 vector")
        )
        calibDat.track_angle_std.attrs.update(
            dict(
                units="deg", long_name="standard deviation track_angle to 0,0,1 vector"
            )
        )
        calibDat.nParticles.attrs.update(
            dict(units="-", long_name="number of observed unique particles")
        )

    return calibDat

def _createLevel2part(
    case,
    config,
    freq="1min",
    DbinsPixel=range(301),
    sizeDefinitions=["Dmax", "Dequiv"],
    endTime=np.timedelta64(1, "h"),
    skipExisting=True,
    sublevel="match",
    camera="leader",
    applyFilters=[],
):
    """
    Create level 2 data for chunk of time
    """
    import dask
    import pandas as pd
    import scipy.stats
    from tqdm import tqdm

    assert sublevel in ["match", "track", "detect"]

    fL = files.FindFiles(case, camera, config)
    lv2File = fL.fnamesDaily[f"level2{sublevel}"]

    lv1Files = fL.listFilesWithNeighbors(f"level1{sublevel}")

    if len(lv1Files) == 0:
        log.warning("level1 NOT AVAILABLE %s (might be nodata)" % lv2File)
        log.warning("look at %s" % fL.fnamesPatternExt[f"level1{sublevel}"])
        return None

    timeIndex = pd.date_range(
        start=case, end=fL.datetime64 + endTime, freq=freq, inclusive="left"
    )
    timeIndex1 = pd.date_range(
        start=case, end=fL.datetime64 + endTime, freq=freq, inclusive="both"
    )

    log.info(f"open level1 files {case}")

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        # with statement makes sure file references are really closed avoiding segfaults
        with xr.open_mfdataset(
            lv1Files, preprocess=_preprocess, combine="nested", concat_dim="pair_id"
        ) as level1dat:
            # camera variable exists only for track and match
            try:
                # limit to period of interest
                level1dat = level1dat.isel(
                    pair_id=(
                        level1dat.capture_time.isel(camera=0) >= fL.datetime64
                    ).values
                    & (
                        level1dat.capture_time.isel(camera=0)
                        < (fL.datetime64 + endTime)
                    ).values
                )
            except ValueError:
                level1dat = level1dat.isel(
                    pair_id=(level1dat.capture_time >= fL.datetime64).values
                    & (level1dat.capture_time < (fL.datetime64 + endTime)).values
                )

            level1dat.load()
    # make chunks more regular
    # level1dat = level1dat.chunk(pair_id=10000)

    if sublevel == "detect":
        # apply blur threshold
        if config.visssGen == "visss":
            """
            # coefficients developed from winer 2021/22 in Hyytiälä
            The idea is to estimate the cummulated total PSD for detect and match data
            and find a blur threshold so that both distributions agree. See
            match_analyze_blur_vs_size.ipynb Filters are applied:

                processing failed by selecting only detect files where match is present
                blocked data by VISSSlib quality control
                smaller sampling volume of matched data is considered by estimatign correction factor

            """
            blurThresh = np.array(
                [
                    np.nan,
                    800.0,
                    773.0,
                    360.0,
                    363.0,
                    436.0,
                    523.0,
                    568.0,
                    633.0,
                    621.0,
                    607.0,
                    574.0,
                    556.0,
                    534.0,
                    506.0,
                    477.0,
                    465.0,
                    435.0,
                    416.0,
                    394.0,
                    374.0,
                    356.0,
                    327.0,
                    314.0,
                    300.0,
                    292.0,
                    275.0,
                    263.0,
                    256.0,
                    248.0,
                    237.0,
                    232.0,
                    226.0,
                    220.0,
                    207.0,
                    202.0,
                    196.0,
                    191.0,
                    183.0,
                    176.0,
                    172.0,
                    170.0,
                    161.0,
                    158.0,
                    153.0,
                    151.0,
                    148.0,
                    144.0,
                    141.0,
                    138.0,
                    140.0,
                    132.0,
                    132.0,
                    131.0,
                    130.0,
                    129.0,
                    125.0,
                    126.0,
                    124.0,
                    123.0,
                    119.0,
                    119.0,
                    120.0,
                    120.0,
                    119.0,
                    117.0,
                    117.0,
                    117.0,
                    118.0,
                    117.0,
                    115.0,
                    114.0,
                    115.0,
                    115.0,
                    115.0,
                    118.0,
                    113.0,
                    114.0,
                    116.0,
                    113.0,
                    112.0,
                    114.0,
                    112.0,
                    112.0,
                    116.0,
                    115.0,
                    112.0,
                    113.0,
                    114.0,
                    111.0,
                    111.0,
                    113.0,
                    112.0,
                ]
                + 400 * [110.0]
            )  # by using 2000 we make sure even huge particles are treated and do not raise an error
        elif config.visssGen == "visss2":
            # coefficients developed from early 2023 in NYA cases with low wind speed
            # "20230213", "20230223", "20230409", "20230429",
            blurThresh = np.array(
                [
                    np.nan,
                    323.0,
                    450.0,
                    282.0,
                    330.0,
                    351.0,
                    378.0,
                    363.0,
                    370.0,
                    352.0,
                    339.0,
                    322.0,
                    305.0,
                    291.0,
                    275.0,
                    259.0,
                    252.0,
                    234.0,
                    225.0,
                    214.0,
                    206.0,
                    196.0,
                    188.0,
                    181.0,
                    173.0,
                    167.0,
                    159.0,
                    154.0,
                    149.0,
                    142.0,
                    137.0,
                    133.0,
                    130.0,
                    124.0,
                    120.0,
                    117.0,
                    114.0,
                    111.0,
                    107.0,
                    105.0,
                    102.0,
                    99.0,
                    97.0,
                    95.0,
                    93.0,
                    91.0,
                    89.0,
                    88.0,
                    85.0,
                    83.0,
                    82.0,
                    81.0,
                    79.0,
                    78.0,
                    77.0,
                    76.0,
                    75.0,
                    74.0,
                    73.0,
                    72.0,
                    72.0,
                    70.0,
                    69.0,
                    69.0,
                    68.0,
                    67.0,
                    66.0,
                    66.0,
                    66.0,
                    64.0,
                    64.0,
                    63.0,
                    63.0,
                    62.0,
                    62.0,
                    61.0,
                    60.0,
                ]
                + 400 * [60.0]
            )

        elif config.visssGen == "visss3":
            # coefficients developed from winer 2023/24 in Hyytiälä
            blurThresh = np.array(
                [
                    np.nan,
                    584.0,
                    1005.0,
                    1181.0,
                    1537.0,
                    1587.0,
                    1397.0,
                    1312.0,
                    1261.0,
                    1181.0,
                    1107.0,
                    1054.0,
                    986.0,
                    931.0,
                    875.0,
                    818.0,
                    788.0,
                    735.0,
                    704.0,
                    668.0,
                    641.0,
                    611.0,
                    585.0,
                    567.0,
                    545.0,
                    525.0,
                    501.0,
                    487.0,
                    472.0,
                    455.0,
                    441.0,
                    427.0,
                    419.0,
                    404.0,
                    391.0,
                    384.0,
                    373.0,
                    366.0,
                    354.0,
                    348.0,
                    342.0,
                    335.0,
                    327.0,
                    320.0,
                    315.0,
                    309.0,
                    303.0,
                    297.0,
                    292.0,
                    289.0,
                    283.0,
                    278.0,
                    275.0,
                    271.0,
                    266.0,
                    263.0,
                    259.0,
                    257.0,
                    253.0,
                    251.0,
                    247.0,
                    244.0,
                    242.0,
                    238.0,
                    235.0,
                    232.0,
                    229.0,
                    228.0,
                    226.0,
                    224.0,
                    221.0,
                    220.0,
                    219.0,
                    216.0,
                    214.0,
                    211.0,
                    209.0,
                    207.0,
                    205.0,
                    203.0,
                    201.0,
                    203.0,
                    199.0,
                    200.0,
                    198.0,
                    198.0,
                    195.0,
                    194.0,
                    194.0,
                    192.0,
                    192.0,
                    190.0,
                    192.0,
                    189.0,
                    187.0,
                    188.0,
                    187.0,
                    184.0,
                    184.0,
                    187.0,
                    179.0,
                    181.0,
                    183.0,
                    183.0,
                    183.0,
                    184.0,
                    181.0,
                    177.0,
                    181.0,
                    180.0,
                    176.0,
                    175.0,
                    175.0,
                    177.0,
                    172.0,
                    173.0,
                    174.0,
                    175.0,
                    172.0,
                    174.0,
                    174.0,
                    176.0,
                    176.0,
                    174.0,
                    171.0,
                ]
                + 400 * [170.0]
            )

        else:
            raise ValueError(f"VISSS Generation {config.visssGen} not supported")

        # discard huge ones so that the trick with the look up table works
        sizeCond = (level1dat.Dmax <= 350).values
        level1dat = level1dat.isel(pair_id=sizeCond)

        # this works liek a lookup table. We use the Dmax rounded to next
        # integer as an index for blurThresh
        appliedblurThresh = blurThresh[np.around(level1dat.Dmax.values).astype(int)]

        blurCond = (level1dat.blur >= appliedblurThresh).values
        log.info(
            tools.concat(
                "blurCond applies to",
                (blurCond.sum() / len(blurCond)) * 100,
                "% of data",
            )
        )
        level1dat = level1dat.isel(pair_id=blurCond)

        del level1dat["blur"]

        if len(level1dat.pair_id) == 0:
            log.warning("no data remains after blurCond filtering %s" % lv2File)
            return None

    else:  # match or track
        # apply matchScore threshold
        if config.quality.minMatchScore is not None:
            matchCond = (level1dat.matchScore >= config.quality.minMatchScore).values
            log.info(
                tools.concat(
                    "matchCond applies to",
                    (matchCond.sum() / len(matchCond)) * 100,
                    "% of data",
                )
            )
            level1dat = level1dat.isel(pair_id=matchCond)

            if len(level1dat.matchScore) == 0:
                log.warning("no data remains after matchScore filtering %s" % lv2File)
                return None

    # only for debuging

    for aa, applyFilter in enumerate(applyFilters):
        assert (
            len(applyFilter) == 5
        ), "applyFilters elements must contain filterVar, operator, filterValue, extraDims"
        filterVar, opStr, filterValue, selectCameraStr, extraDims = applyFilter

        # add complexity to data if filtered by complexity
        if filterVar in ["complexityBW"]:
            level1dat = addPerParticleVariables(level1dat, config)

        if selectCameraStr == "max":
            thisDat = level1dat[filterVar].sel(**extraDims).max("camera")
        elif selectCameraStr == "min":
            thisDat = level1dat[filterVar].sel(**extraDims).min("camera")
        elif selectCameraStr == "mean":
            thisDat = level1dat[filterVar].sel(**extraDims).mean("camera")
        else:
            raise ValueError(
                "selectCameraStr must be max, min or mean, received %s", selectCameraStr
            )

        if isinstance(filterValue, list):
            assert (
                len(filterValue) == 2
            ), "filterValue is list, but contains not 2 elements; at the moment only linear functions of Dmax are supported"
            matchCond = _operators[opStr](
                thisDat,
                (filterValue[0] + filterValue[1] * level1dat["Dmax"]).max(dim="camera"),
            ).values

        else:
            matchCond = _operators[opStr](thisDat, filterValue).values

        level1dat = level1dat.isel(pair_id=matchCond)

        if len(level1dat.matchScore) == 0:
            log.warning("no data remains after additional filtering")
            return None

    try:
        sizeCond = (level1dat.Dmax < max(DbinsPixel)).all("camera").values
    except ValueError:
        sizeCond = (level1dat.Dmax < max(DbinsPixel)).values
    level1dat = level1dat.isel(pair_id=sizeCond)
    if len(level1dat.pair_id) == 0:
        log.warning("no data remains after size filtering %s" % lv2File)
        return None

    # remove particles too close to the edge
    # this is possible for non symmetrical particles
    if sublevel == "detect":
        level1dat["position_center"] = level1dat.position_upperLeft + (
            level1dat.Droi / 2
        )
        DmaxHalf = level1dat.Dmax / 2

        farEnoughFromBorder = (
            (level1dat.position_center >= DmaxHalf).all("dim2D")
            & (
                (
                    config.frame_width
                    - level1dat.position_center.sel(dim2D="x", drop=True)
                )
                >= DmaxHalf
            )
            & (
                (
                    config.frame_height
                    - level1dat.position_center.sel(dim2D="y", drop=True)
                )
                >= DmaxHalf
            )
        )

    else:
        DmaxHalf = level1dat.Dmax.max("camera") / 2

        farEnoughFromBorder = (
            (level1dat.position3D_center.sel(dim3D=["x", "y", "z"]) >= DmaxHalf).all(
                "dim3D"
            )
            & (
                (config.frame_width - level1dat.position3D_center.sel(dim3D=["x", "y"]))
                >= DmaxHalf
            ).all("dim3D")
            & (
                (
                    config.frame_height
                    - level1dat.position3D_center.sel(dim3D="z", drop=True)
                )
                >= DmaxHalf
            )
        )

    farEnoughFromBorder = farEnoughFromBorder.compute()

    log.info(
        tools.concat(
            "farEnoughFromBorder applies to",
            (farEnoughFromBorder.sum() / len(farEnoughFromBorder)).values * 100,
            "% of data",
        )
    )

    level1dat = level1dat.isel(pair_id=farEnoughFromBorder)
    if len(level1dat.pair_id) == 0:
        log.warning("no data remains after farEnoughFromBorder filtering %s" % lv2File)
        return None

    if config.level2.correctForSmallOnes and "maxSharpness" in config.keys():
        # remove small particles that are not in the very sharpest region
        # IN DEVELOPMENT
        assert sublevel != "detect"
        log.info("removing small particles outside of most sharp volume")
        isOutsideSharp = False
        l1datL = level1dat.position_centroid.sel(
            dim2D="x", camera=config.leader, drop=True
        ).drop_vars("pid")
        l1datF = level1dat.position_centroid.sel(
            dim2D="x", camera=config.follower, drop=True
        ).drop_vars("pid")
        for size in config.maxSharpness.leader.keys():
            isSize = (level1dat.Dmax.max("camera") >= size) & (
                level1dat.Dmax.max("camera") < size + 1
            )
            outsideLeader = isSize & (
                (l1datL < config.maxSharpness.leader[size][0])
                | (l1datL > config.maxSharpness.leader[size][1])
            )
            outsideFollower = isSize & (
                (l1datF < config.maxSharpness.follower[size][0])
                | (l1datF > config.maxSharpness.follower[size][1])
            )
            isOutsideSharp = isOutsideSharp | outsideLeader | outsideFollower

        isOutsideSharp = isOutsideSharp.compute()
        del l1datL, l1datF

        log.info(
            tools.concat(
                "isOutsideSharp applies NOT to",
                100 - (isOutsideSharp.sum() / len(isOutsideSharp)).values * 100,
                "% of data",
            )
        )

        level1dat = level1dat.isel(pair_id=~isOutsideSharp)
        if len(level1dat.pair_id) == 0:
            log.warning("no data remains after isOutsideSharp filtering %s" % lv2File)
            return None
    else:
        log.info(f"do not remove particles outside of most sharp area")

    if sublevel == "detect":
        level1dat = level1dat.drop_vars(
            ["position_center", "Droi", "position_upperLeft", "dim2D"]
        )
    else:
        level1dat = level1dat.drop_vars("position_centroid")

    # done with filtering the data.
    # the following code is much faster with a loaded netcdf object
    log.info("loading data")
    level1dat.load()

    if sublevel == "detect":
        log.info(f"estimate mean values")

        # promote capture_time to coordimnate for later
        level1dat_time = level1dat.assign_coords(
            time=xr.DataArray(level1dat.capture_time.values, coords=[level1dat.pair_id])
        )

        # fix order
        level1dat_4timeAve = level1dat_time.transpose(*["fitMethod", "pair_id"])

        # save for later
        individualDataPoints = level1dat_4timeAve.pair_id
        individualDataPoints.name = "nParticles"

    elif sublevel == "match":
        log.info(f"estimate camera mean values")
        data_vars = [
            "Dmax",
            "area",
            "matchScore",
            "aspectRatio",
            "angle",
            "perimeter",
        ]

        # promote capture_time to coordimnate for later
        level1dat_time = level1dat.assign_coords(
            time=xr.DataArray(
                level1dat.capture_time.isel(camera=0).values, coords=[level1dat.pair_id]
            )
        )

        # we do not need pid any more
        level1dat_time = level1dat_time.reset_coords("pid")

        # estimate max, mean and min for both cameras
        level1dat_camAve = (
            level1dat_time[data_vars].max("camera"),
            level1dat_time[data_vars].mean("camera"),
            level1dat_time[data_vars].min("camera"),
            level1dat_time[data_vars].sel(camera=config.leader, drop=True),
            level1dat_time[data_vars].sel(camera=config.follower, drop=True),
            level1dat_time[data_vars].std("camera"),
        )
        level1dat_camAve = xr.concat(level1dat_camAve, dim="camera")

        level1dat_camAve["camera"] = [
            "max",
            "mean",
            "min",
            "leader",
            "follower",
            "std",
        ]

        # fix angle because we want the circular mean
        level1dat_camAve["angle"].loc["mean"] = level1dat_time["angle"].reduce(
            scipy.stats.circmean, "camera", high=360, nan_policy="omit"
        )

        # position_3D is the same for all
        # level1dat_camAve["position3D_center"] = level1dat_camAve["position3D_center"].sel(
        #    camera="max", drop=True)

        # fix order
        level1dat_4timeAve = level1dat_camAve.transpose(
            *["camera", "fitMethod", "pair_id"]
        )

        # clean up
        del level1dat_camAve
        # #position is not needed any more
        # del level1dat_4timeAve["position3D_center"]
        # #centroid position is not needed any more
        # del level1dat_4timeAve["position3D_centroid"]

        # save for later
        individualDataPoints = level1dat_4timeAve.pair_id
        individualDataPoints.name = "nParticles"

    elif sublevel == "track":
        (
            level1dat_trackAve,
            level1dat_track2D,
            level1dat_time,
            individualDataPoints,
            _,
        ) = getPerTrackStatistics(level1dat)

        # because there are no weighted groupby operations
        # https://github.com/pydata/xarray/issues/3937, we have to improvise
        # and broadcast the results again to a shape including track_step
        # - then the mean etc. values are dublicated as per track length
        # and the result is weighted when averaging with time
        data_vars = [
            "Dmax",
            "area",
            "matchScore",
            "aspectRatio",
            "angle",
            "perimeter",
            "velocity",
        ]
        for data_var in data_vars:
            level1dat_trackAve[data_var] = level1dat_trackAve[data_var].broadcast_like(
                level1dat_track2D.isel(camera=0, drop=True)[data_var]
            )

        # add back the original individual values (mainly for testing)
        # level1dat_trackAve = xr.concat((level1dat_trackAve,level1dat_track2D.expand_dims(track=["individual"])), dim="cameratrack")

        log.info(f"reshape track data again")
        # call me crazy but now that we have mean track properties broadcasted to every particle we can go back to pair_id!
        level1dat_4timeAve = level1dat_trackAve.stack(
            pair_id=("track_id", "track_step")
        )

        # make sure only data is used within original track length
        notNull = (
            level1dat_track2D.Dmax.isel(camera=0, drop=True)
            .notnull()
            .stack(pair_id=("track_id", "track_step"))
            .compute()
        )

        level1dat_4timeAve = level1dat_4timeAve.isel(pair_id=notNull)
        # multiindex causes trouble below, so just swap with time
        level1dat_4timeAve = level1dat_4timeAve.swap_dims(pair_id="time")
        # for reasons I do not understand there are sometimes a few values with NaT timestamops
        level1dat_4timeAve = level1dat_4timeAve.isel(
            time=~np.isnan(level1dat_4timeAve.time)
        )
        individualDataPoints = individualDataPoints.isel(
            track_id=~np.isnan(individualDataPoints.time)
        )

        # clean up
        del level1dat_trackAve

    else:
        raise ValueError("do not know sublevel")

    # log.info(f"load data")
    ##turned out, it runs about 3 to 4 times faster when NOT using dask beyond this point.
    # level1dat_4timeAve = level1dat_4timeAve.load()

    log.info(f"add additonal variables")
    level1dat_4timeAve = addPerParticleVariables(level1dat_4timeAve, config)

    # split data in 1 min chunks
    level1datG = level1dat_4timeAve.groupby_bins(
        "time", timeIndex1, right=False, squeeze=False
    )
    level1datG_angle = level1dat_4timeAve["angle"].groupby_bins(
        "time", timeIndex1, right=False, squeeze=False
    )
    individualDataPointsG = individualDataPoints.groupby_bins(
        "time", timeIndex1, right=False, squeeze=False
    )

    del level1dat_4timeAve

    sizeDefinitions = ["Dmax", "Dequiv"]
    data_vars = [
        "area",
        "angle",
        "aspectRatio",
        "perimeter",
        "complexityBW",
        "normalizedRimeMass",
    ]
    if sublevel == "track":
        data_vars += ["velocity", "track_angle"]

    log.info(f"get time resolved distributions")
    # process each 1 min chunks
    res = {}
    nParticles = {}

    if sublevel == "detect":
        # iterate through every 1 min piece
        for interv, level1datG1 in tqdm(level1datG, file=sys.stdout, desc=case):
            # estimate counts
            tmpXr = []
            for sizeDefinition in sizeDefinitions:
                tmpXr1 = (
                    level1datG1[[sizeDefinition]]
                    .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                    .count()
                    .fillna(0)
                )

                tmpXr1 = tmpXr1.rename({sizeDefinition: "N"})
                tmpXr1 = tmpXr1.rename({f"{sizeDefinition}_bins": "D_bins"})

                # import pdb; pdb.set_trace()
                # estimate mean values for "area", "angle", "aspectRatio", "perimeter"
                # Dmax is only for technical resaons and is removed afterwards
                data_vars1 = data_vars + [sizeDefinition]
                data_vars1.remove("angle")  # treated seperately

                otherVars1 = (
                    level1datG1[data_vars1]
                    .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                    .mean()
                )
                angleVars = (
                    level1datG1[["angle", sizeDefinition]]
                    .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                    .reduce(scipy.stats.circmean, high=360, nan_policy="omit")
                )
                otherVars1["angle"] = angleVars["angle"]
                del otherVars1[sizeDefinition]

                otherVars1 = otherVars1.rename(
                    {k: f"{k}_dist" for k in otherVars1.data_vars}
                )
                otherVars1 = otherVars1.rename({f"{sizeDefinition}_bins": "D_bins"})
                tmpXr1.update(otherVars1)
                tmpXr.append(tmpXr1)

            tmpXr = xr.concat(tmpXr, dim="size_definition")
            tmpXr["size_definition"] = sizeDefinitions

            res[interv.left] = xr.Dataset(tmpXr)

        # clean up
        del tmpXr, tmpXr1

        dist = xr.concat(res.values(), dim="time")
        dist["time"] = list(res.keys())

    else:
        if sublevel == "track":
            coordVar = "cameratrack"
        elif sublevel == "match":
            coordVar = "camera"

        # iterate through every 1 min piece
        for interv, level1datG1 in tqdm(level1datG, file=sys.stdout, desc=case):
            # print(interv)
            tmp = []
            # for each track&camera/min/max/mean seperately
            for coord in level1datG1[coordVar].values:
                # estimate counts
                tmpXr = []
                for sizeDefinition in sizeDefinitions:
                    # for the PSD, caemra min/max etc applied to the binning
                    tmpXr1 = (
                        level1datG1[[sizeDefinition]]
                        .sel(**{coordVar: coord})
                        .groupby_bins(sizeDefinition, DbinsPixel, right=False)
                        .count()
                        .fillna(0)
                    )

                    tmpXr1 = tmpXr1.rename({sizeDefinition: "N"})
                    tmpXr1 = tmpXr1.rename({f"{sizeDefinition}_bins": "D_bins"})

                    # import pdb; pdb.set_trace()
                    # estimate mean values for "area", "angle", "aspectRatio", "perimeter"
                    # Dmax is only for technical resaons and is removed afterwards
                    data_vars1 = data_vars + [sizeDefinition]
                    data_vars1.remove("angle")  # treated seperately

                    # for all variabvles except the PSD, min/max etc is applied to the variable
                    # the binning uses max observed Dmax or Deq
                    binningVar = level1datG1[sizeDefinition].sel(**{coordVar: "max"})
                    otherVars1 = (
                        level1datG1[data_vars1]
                        .sel(drop=True, **{coordVar: coord})
                        .groupby_bins(binningVar, DbinsPixel, right=False)
                        .mean()
                    )
                    angleVars = (
                        level1datG1[["angle", sizeDefinition]]
                        .sel(drop=True, **{coordVar: coord})
                        .groupby_bins(binningVar, DbinsPixel, right=False)
                        .reduce(scipy.stats.circmean, high=360, nan_policy="omit")
                    )
                    otherVars1["angle"] = angleVars["angle"]

                    otherVars1 = otherVars1.rename(
                        {k: f"{k}_dist" for k in otherVars1.data_vars}
                    )
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

        # clean up
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
    meanValues["angle"] = level1datG_angle.reduce(
        scipy.stats.circmean, high=360, nan_policy="omit"
    )
    meanValues = meanValues.rename({k: f"{k}_mean" for k in meanValues.data_vars})
    meanValues = meanValues.rename(time_bins="time")
    # we want time stamps not intervals
    meanValues["time"] = [a.left for a in meanValues["time"].values]

    log.info("do temporal std values")
    # estimate mean values
    # to do: data is weighted with number of obs not considering the smalle robservation volume for larger particles
    stdValues = level1datG.std()
    stdValues["angle"] = level1datG_angle.reduce(
        scipy.stats.circstd, high=360, nan_policy="omit"
    )
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

    # clean up!
    del level1datG, level1dat_time
    return calibDat


def addPerParticleVariables(level1dat_camAve, config):
    """
    Add per-particle variables to the dataset.

    This function computes additional particle characteristics like complexity
    and normalized rime mass based on existing particle data.

    Parameters
    ----------
    level1dat_camAve : xarray.Dataset
        Dataset containing particle data.
    config : dict
        Configuration settings for computation.

    Returns
    -------
    xarray.Dataset
        Dataset with additional particle variables.
    """

    # add area equivalent radius
    level1dat_camAve["Dequiv"] = np.sqrt(
        4 * level1dat_camAve["area"] / np.pi
    )

    # based on Garrett, T. J., and S. E. Yuter, 2014: Observed influence of riming,
    # temperature, and turbulence on the fallspeed of solid precipitation.
    # Geophys. Res. Lett., 41, 6515–6522, doi:10.1002/2014GL061016.

    # N. Maherndl did not find a signficant difference between using variables
    # considering holes and not considering holes

    level1dat_camAve["complexityBW"] = level1dat_camAve["perimeter"] / (
        np.pi * level1dat_camAve["Dequiv"]
    )

    # for fit Complexity = a0 + a1Mlogd + a2d + a3Mlog
    # Maherndl, N., M. Moser, J. Lucke, M. Mech, N. Risse, I. Schirmacher, and
    # M. Maahn, 2024: Quantifying riming from airborne data during the HALO-(AC)3
    # campaign. Atmos. Meas. Tech., 17, 1475–1495, doi:10.5194/amt-17-1475-2024.
    # coefficients adapted to VISSS resolution

    if config.visssGen == "visss":
        a0, a1, a2, a3 = (
            1.20692082,
            -0.0051769,
            -0.00143907,
            -0.31902574,
        )
    elif config.visssGen == "visss2":
        a0, a1, a2, a3 = (
            1.22641410e00,
            -4.12985039e-03,
            -8.40670401e-04,
            -3.04855738e-01,
        )
    elif config.visssGen == "visss3":
        a0, a1, a2, a3 = (
            1.22888785e00,
            -4.40436778e-03,
            -9.32105780e-04,
            -3.04682514e-01,
        )
    else:
        log.error(f"VISSS generation {config.visssGen} not implemented")
        a0, a1, a2, a3 = (np.nan, np.nan, np.nan, np.nan)

    M = 10 ** (
        (a0 - level1dat_camAve["complexityBW"] + (a2 * level1dat_camAve["Dmax"]))
        / -((a1 * level1dat_camAve["Dmax"]) + a3)
    )

    level1dat_camAve["normalizedRimeMass"] = M.where(
        level1dat_camAve["Dmax"] >= config.quality.minSize4insituM
    )

    return level1dat_camAve


def addVariables(
    calibDat,
    case,
    config,
    timeIndex,
    timeIndex1,
    sublevel,
    camera="leader",
    single_class=False,
):
    """
    Add additional variables to the calibrated dataset.

    Computes derived quantities such as particle size distribution (PSD),
    moments, and quality flags based on the calibrated particle data.

    Parameters
    ----------
    calibDat : xarray.Dataset
        Calibrated dataset with basic particle properties.
    case : str
        Case identifier for data context. For loopify decorator, this is
        a date string (YYYYMMDD) or date range (YYYYMMDD-YYYYMMDD) or comma-separated dates.
    config : dict
        Configuration settings for computations.
    timeIndex : numpy.ndarray
        Time index for data aggregation.
    timeIndex1 : numpy.ndarray
        Detailed time index for computations.
    sublevel : str
        Processing sublevel. Must be one of ["match", "track", "detect"].
    camera : str, optional
        Camera identifier, defaults to "leader". Used for match and track sublevels.

    Returns
    -------
    xarray.Dataset
        Dataset with added variables.
    """
    import scipy.special

    # 1 min data
    deltaT = int(timeIndex.freq.nanos * 1e-9) * config.fps
    # 1 pixel size bins
    deltaD = (1 / config.calibration.slope) * 1e-6

    calibDat["PSD"] = (
        calibDat["counts"] / deltaT / calibDat["obs_volume"] / deltaD
    )  # now in 1/m4
    calibDat["Ntot"] = (calibDat["counts"] / deltaT / calibDat["obs_volume"]).sum(
        "D_bins"
    )

    M = {}
    for mm in [1, 2, 3, 4, 6]:
        M[mm] = (calibDat.PSD.fillna(0) * deltaD * calibDat.D_bins**mm).sum("D_bins")
        calibDat[f"M{mm}"] = M[mm]

    for b in [2, 3]:
        calibDat[f"D{b+1}{b}"] = M[b + 1] / M[b]
        calibDat[f"N0_star_{b+1}{b}"] = (
            (M[b] ** (b + 2) / M[b + 1] ** (b + 1))
            * ((b + 1) ** (b + 1))
            / scipy.special.gamma(b + 1)
        )
    if single_class==True:
        return calibDat
        
    # quality variables
    (
        recordingFailed,
        processingFailed,
        blockedPixels,
        blowingSnowRatio,
        observationsRatio,
    ) = getDataQuality(case, config, timeIndex, timeIndex1, sublevel, camera=camera)
    assert np.all(blockedPixels.time == calibDat.time)

    if sublevel == "detect":
        processingFailed.values[:] = False  # not relevant becuase it is about matching
        cameraBlocked = blockedPixels > config.quality.blockedPixThresh
        blowingSnow = blowingSnowRatio > config.quality.blowingSnowFrameThresh
        obervationsDiffer = np.array([False] * len(blockedPixels))
        tracksTooShort = np.array([False] * len(blockedPixels))

    else:
        recordingFailed = recordingFailed.any("camera")
        cameraBlocked = blockedPixels.max("camera") > config.quality.blockedPixThresh
        blowingSnow = (
            blowingSnowRatio.max("camera") > config.quality.blowingSnowFrameThresh
        )
        obervationsDiffer = (observationsRatio < config.quality.obsRatioThreshold) | (
            observationsRatio > (1 / config.quality.obsRatioThreshold)
        )
        if sublevel == "match":
            tracksTooShort = np.array([False] * len(cameraBlocked))
        else:
            tracksTooShort = (
                calibDat.track_length_mean <= config.quality.trackLengthThreshold
            )

    # apply quality
    log.info("apply quality filters...")
    log.info(
        tools.concat(
            "recordingFailed filter removed",
            recordingFailed.values.sum() / len(recordingFailed) * 100,
            "% of data",
        )
    )
    log.info(
        tools.concat(
            "processingFailed filter removed",
            processingFailed.values.sum() / len(processingFailed) * 100,
            "% of data",
        )
    )
    log.info(
        tools.concat(
            "cameraBlocked filter removed",
            cameraBlocked.values.sum() / len(cameraBlocked) * 100,
            "% of data",
        )
    )
    log.info(
        tools.concat(
            "blowingSnow filter removed",
            blowingSnow.values.sum() / len(blowingSnow) * 100,
            "% of data",
        )
    )
    if sublevel != "detect":
        log.info(
            tools.concat(
                "observationsRatio filter removed",
                obervationsDiffer.values.sum() / len(cameraBlocked) * 100,
                "% of data",
            )
        )
    if sublevel == "track":
        log.info(
            tools.concat(
                "tracksTooShort filter removed",
                tracksTooShort.values.sum() / len(cameraBlocked) * 100,
                "% of data",
            )
        )
    allFilter = (
        recordingFailed
        | processingFailed
        | cameraBlocked
        | blowingSnow
        | obervationsDiffer
        | tracksTooShort
    )
    log.info(
        tools.concat(
            "all filter together applied to",
            allFilter.values.sum() / len(allFilter) * 100,
            "% of data",
        )
    )

    qualityFlag = np.stack(
        [
            recordingFailed,
            processingFailed,
            cameraBlocked,
            blowingSnow,
            obervationsDiffer,
            tracksTooShort,
        ],
        axis=-1,
    )
    qualityFlag = np.packbits(qualityFlag, axis=-1).squeeze()
    calibDat["qualityFlags"] = xr.DataArray(qualityFlag, coords=[calibDat.time])

    assert (allFilter.time == calibDat.time).all()

    # calibDat["recordingFailed"] = recordingFailed
    # calibDat["processingFailed"] = processingFailed
    calibDat["blowingSnowRatio"] = blowingSnowRatio
    if sublevel == "match":
        blockedVars = (
            blockedPixels.max("camera"),
            blockedPixels.mean("camera"),
            blockedPixels.min("camera"),
            blockedPixels.sel(camera="leader", drop=True),
            blockedPixels.sel(camera="follower", drop=True),
            blockedPixels.std("camera"),
        )
        calibDat["blockedPixelRatio"] = xr.concat(blockedVars, dim="camera").T
        blowingVars = (
            blowingSnowRatio.max("camera"),
            blowingSnowRatio.mean("camera"),
            blowingSnowRatio.min("camera"),
            blowingSnowRatio.sel(camera="leader", drop=True),
            blowingSnowRatio.sel(camera="follower", drop=True),
            blowingSnowRatio.std("camera"),
        )
        calibDat["blowingSnowRatio"] = xr.concat(blowingVars, dim="camera").T
    else:
        calibDat["blockedPixelRatio"] = blockedPixels.T
        calibDat["blowingSnowRatio"] = blowingSnowRatio.T

    if sublevel != "detect":
        calibDat["observationsRatio"] = observationsRatio

    return calibDat


def getPerTrackStatistics(level1dat, maxAngleDiff=20, extraVars=[]):
    """
    Extract statistical information from particle tracks.

    Computes track-level statistics by aggregating particle data along
    trajectory dimensions and calculating descriptive metrics.

    Parameters
    ----------
    level1dat : xarray.Dataset
        Input dataset containing particle tracking data.
    maxAngleDiff : float, optional
        Maximum angle difference for track continuation, defaults to 20.
    extraVars : list, optional
        Extra variables to include in computation, defaults to [].

    Returns
    -------
    tuple
        Tuple containing track statistics dataset, 2D track data,
        time-resolved track data, individual particle data, and number of cuts.
    """
    import pandas as pd
    import scipy.stats

    log.info(f"reshape tracks")

    # go from pair_id to track_id and put observations along same track into new track_step dimension
    track_mi = pd.MultiIndex.from_arrays(
        (level1dat.track_id.values, level1dat.track_step.values),
        names=["track_id", "track_step"],
    )
    level1dat["track_mi"] = xr.DataArray(track_mi, coords=[level1dat.pair_id])
    level1dat = level1dat.swap_dims(pair_id="track_mi")

    # very very rarely, there are duplicate values in the index, reason is unclear
    _, uniqueII = np.unique(level1dat.track_mi, return_index=True)
    level1dat = level1dat.isel(track_mi=uniqueII)

    # this costs a lot of memeory but I do not know a better way
    level1dat_time = level1dat.unstack("track_mi")
    # promote capture_time to coordinate for later
    level1dat_time = level1dat_time.assign_coords(
        time=xr.DataArray(
            level1dat_time.capture_time.isel(camera=0, track_step=0).values,
            coords=[level1dat_time.track_id],
        )
    )

    # cut tracks longer than 40 elements to save memory!
    if len(level1dat_time.track_step) > 40:
        log.info(
            f"truncating {100*level1dat_time.Dmax.isel(camera=0, track_step=40).notnull().sum().values/len(level1dat_time.track_id)} % tracks"
        )
        level1dat_time = level1dat_time.isel(track_step=slice(40))

    log.info(f"estimate track mean values")

    # fix order
    orderedCoords = [
        "track_id",
        "track_step",
        "camera",
        "dim3D",
        "fitMethod",
        "camera_rotation",
    ]
    datCoords = level1dat_time.reset_coords(drop=True).coords
    remainingCoords = list(set(datCoords) - set(orderedCoords))
    orderedCoords = orderedCoords + remainingCoords
    level1dat_track2D = level1dat_time.transpose(*orderedCoords)

    # which variables do we need?
    level1dat_track2D = level1dat_track2D[
        set(
            [
                "Dmax",
                "area",
                "matchScore",
                "aspectRatio",
                "angle",
                "perimeter",
                "position3D_centroid",
                "capture_time"
            ]
            + extraVars
        )
    ]

    # try to find and remove sudden turns in the tracks,
    tracksCut = 0
    for kk in range(2):
        log.info(f"try to find and remove sudden turns in the tracks {kk}")
        level1dat_track2D, nCuts = removeTrackEdges(level1dat_track2D, int(maxAngleDiff))
        tracksCut += nCuts

    # add velocities
    distSpace = level1dat_track2D.position3D_centroid.diff("track_step", label="upper")
    distTime = level1dat_track2D.capture_time.isel(camera=0, drop=True).diff(
        "track_step", label="upper"
    )
    # to fraction of seconds
    distTime = distTime / np.timedelta64(1, "s")

    # velocity in px/s
    level1dat_track2D["velocity"] = distSpace / distTime

    # save for later
    individualDataPoints = level1dat_track2D.track_id
    individualDataPoints.name = "nParticles"

    # diff output is one element shorter, so add the mean value again
    # causes problems and advantage is not clear...
    # level1dat_track2D["velocity"][dict(track_step=0)] = level1dat_track2D["velocity"].mean("track_step")
    del level1dat_track2D["position3D_centroid"]

    # estimate max, mean and min for tracks by reducing track_step
    level1dat_track2D_4ave = (
        level1dat_track2D.copy()
    )  # copy required becuase std of time does not work
    del level1dat_track2D_4ave["capture_time"]
    trackOps = ["max", "mean", "min", "std"]
    level1dat_trackAve = (
        level1dat_track2D_4ave.max(["track_step", "camera"]),
        level1dat_track2D_4ave.mean(["track_step", "camera"]),
        level1dat_track2D_4ave.min(["track_step", "camera"]),
        level1dat_track2D_4ave.std(["track_step", "camera"]),
    )
    level1dat_trackAve = xr.concat(level1dat_trackAve, dim="cameratrack")
    level1dat_trackAve["cameratrack"] = trackOps

    # fix  circular mean & std for angle
    level1dat_trackAve["angle"].loc["mean"] = level1dat_track2D_4ave["angle"].reduce(
        scipy.stats.circmean, ["track_step", "camera"], high=360, nan_policy="omit"
    )
    level1dat_trackAve["angle"].loc["std"] = level1dat_track2D_4ave["angle"].reduce(
        scipy.stats.circstd, ["track_step", "camera"], high=360, nan_policy="omit"
    )

    # use Dmax as arbitrary variable with only one dimension
    level1dat_trackAve["track_length"] = (
        level1dat_track2D_4ave.Dmax.isel(camera=0, drop=True)
        .notnull()
        .sum("track_step")
    )

    del level1dat_track2D_4ave

    return (
        level1dat_trackAve,
        level1dat_track2D,
        level1dat_time,
        individualDataPoints,
        tracksCut,
    )


def removeTrackEdges(level1dat_track2D, maxAngleDiff):
    """
    Remove track edges identified by large angular differences.

    Identifies potentially erroneous track terminations by detecting large
    angular deviations and removes those sections from the data.

    Parameters
    ----------
    level1dat_track2D : xarray.Dataset
        Track dataset with particle trajectory data.
    maxAngleDiff : float
        Maximum allowable angular difference for track continuity.

    Returns
    -------
    tuple
        Updated track dataset and number of removed edges.
    """
    import vg
    import xarray_extras.sort

    distSpace = level1dat_track2D.position3D_centroid.diff("track_step", label="upper")

    # compute 3d angle to 0,0,1 vector, vg library doesnt like ND arrays or xr:
    di = distSpace.sel(dim3D=["x", "y", "z"]).values[:]
    an = vg.angle(
        np.array([0, 0, 1]), di.reshape(di.shape[0] * di.shape[1], di.shape[2])
    ).reshape(di.shape[:2])
    level1dat_track2D["track_angle"] = xr.DataArray(
        an, coords=[distSpace.track_id, distSpace.track_step]
    )

    # try to find and remove edges in the tracks,
    tts = []
    if maxAngleDiff != 0:
        ang = np.abs(level1dat_track2D.track_angle.diff("track_step"))
        # angles can be large, but for natural tracks the 2nd is also large
        # sort cannot handle nans
        twoLargest = xarray_extras.sort.topk(ang.fillna(-999), 2, "track_step")
        twoLargest = twoLargest.where(twoLargest != -999)
        # identify tracks
        tts = np.where(twoLargest.diff("track_step") < -maxAngleDiff)[0]
        tts = level1dat_track2D.track_id[tts].values
        extraTracks = []
        for tt in tts:
            # where is the edge?
            maxII = ang.sel(track_id=tt).argmax().values + 2
            # log.info(tools.concat("cutting track", tt, "max angle diff.", ang.sel(track_id=tt).max().values, "at index",  maxII))
            # move to extra array
            extraTracks.append(
                deepcopy(
                    level1dat_track2D.loc[
                        {"track_id": tt, "track_step": slice(maxII, None)}
                    ]
                )
            )
            # delete information
            level1dat_track2D.loc[
                {"track_id": tt, "track_step": slice(maxII, None)}
            ] = np.nan

        # create index and append extraTracks to end
        if len(extraTracks) > 0:
            oldMax = level1dat_track2D.track_id.max().values
            extraIds = range(int(oldMax + 1), int(oldMax + 1 + len(extraTracks)))
            extraTracks = xr.concat(extraTracks, dim="track_id")
            extraTracks["track_id"] = extraIds
            level1dat_track2D = xr.concat(
                (level1dat_track2D, extraTracks), dim="track_id"
            )
    log.info(f"track cut at {len(tts)} positions")
    return level1dat_track2D, len(tts)


def estimateObservationVolume(level1dat_time, config, DbinsPixel, timeIndex1):
    """
    Estimate observation volumes for different particle sizes.

    Calculates the effective observation volume for each particle size bin
    based on camera configuration and particle characteristics.

    Parameters
    ----------
    level1dat_time : xarray.Dataset
        Time-resolved dataset with camera rotation parameters.
    config : dict
        Configuration settings for volume estimation.
    DbinsPixel : range
        Pixel bins for size distribution.
    timeIndex1 : numpy.ndarray
        Detailed time index for volume calculation.

    Returns
    -------
    list
        List of estimated observation volumes for each bin.
    """
    if config.level2.correctForSmallOnes:
        log.info("adjust observation volujme for smallest particles")

        assert np.all(
            np.array(config.maxSharpness.leader.keys())
            == np.array(config.maxSharpness.follower.keys())
        )
        maxSharpnessSizes = (tuple(config.maxSharpness.leader.keys()),)
        maxSharpnessLeader = (
            tuple(tuple(c) for c in config.maxSharpness.leader.values()),
        )
        maxSharpnessFollower = (
            tuple(tuple(c) for c in config.maxSharpness.follower.values()),
        )

    else:
        log.info("do NOT adjust observation volujme for smallest particles")
        maxSharpnessSizes = tuple()
        maxSharpnessLeader = tuple()
        maxSharpnessFollower = tuple()

    if "camera_phi" in level1dat_time.data_vars:
        rotDat = level1dat_time[["camera_phi", "camera_theta", "camera_Ofz"]].sel(
            camera_rotation="mean"
        )
        try:
            rotDat = rotDat.isel(track_id=~np.isnan(rotDat.time).values)
        except ValueError:
            rotDat = rotDat.isel(pair_id=~np.isnan(rotDat.time).values)

        rotDat = rotDat.groupby_bins(
            "time", timeIndex1, right=False, squeeze=False
        ).mean()
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
            Ds, volume = _estimateVolumes(
                config.frame_width,
                config.frame_height,
                config.level2.correctForSmallOnes,
                float(rotDat1.camera_phi.values),
                float(rotDat1.camera_theta.values),
                float(rotDat1.camera_Ofz.values),
                DbinsPixel,
                maxSharpnessSizes,
                maxSharpnessLeader,
                maxSharpnessFollower,
            )

            volumes.append(volume[1:])
    else:  # for detect only:
        Ds, volume = _estimateVolumes(
            config.frame_width,
            config.frame_height,
            config.level2.correctForSmallOnes,
            0.0,
            0.0,
            0.0,
            DbinsPixel,
            maxSharpnessSizes,
            maxSharpnessLeader,
            maxSharpnessFollower,
        )

        # add time dimension
        volumes = []
        for ii in range(len(timeIndex1) - 1):
            volumes.append(volume[1:])
    return volumes


def calibrateData(level2dat, level1dat_time, config, DbinsPixel, timeIndex1):
    """
    Apply calibration to level 2 data.

    Transforms pixel-based measurements into calibrated physical units
    and adjusts for instrument-specific characteristics.

    Parameters
    ----------
    level2dat : xarray.Dataset
        Level 2 dataset with raw pixel measurements.
    level1dat_time : xarray.Dataset
        Time-resolved dataset with camera parameters.
    config : dict
        Configuration settings for calibration.
    DbinsPixel : range
        Pixel bins for size distribution.
    timeIndex1 : numpy.ndarray
        Detailed time index for calibration.

    Returns
    -------
    xarray.Dataset
        Calibrated dataset with physical units.
    """

    assert "slope" in config.calibration.keys()

    slope = config.calibration.slope

    calibDat = level2dat.rename(N="counts").copy()

    volumes = estimateObservationVolume(level1dat_time, config, DbinsPixel, timeIndex1)
    calibDat["obs_volume"] = xr.DataArray(
        volumes, coords=[level2dat.time, level2dat.D_bins]
    )

    # apply resolution
    calibDat = calibDat.assign_coords(D_bins=calibDat["D_bins"].values / slope / 1e6)
    calibDat["obs_volume"] = calibDat["obs_volume"] / slope**3 / 1e6**3

    # go from intervals to center values
    calibDat["D_bins_left"] = xr.DataArray(
        [b.left for b in calibDat.D_bins.values], dims=["D_bins"]
    )
    calibDat["D_bins_right"] = xr.DataArray(
        [b.right for b in calibDat.D_bins.values], dims=["D_bins"]
    )
    calibDat = calibDat.assign_coords(D_bins=[b.mid for b in calibDat.D_bins.values])

    # remaining variables
    calibDat["area_dist"] = calibDat["area_dist"] / slope**2 / 1e6**2
    calibDat["perimeter_dist"] = calibDat["perimeter_dist"] / slope / 1e6

    calibDat["Dmax_mean"] = (calibDat["Dmax_mean"]) / slope / 1e6
    calibDat["Dmax_std"] = (calibDat["Dmax_std"]) / slope / 1e6
    calibDat["area_mean"] = calibDat["area_mean"] / slope**2 / 1e6**2
    calibDat["area_std"] = calibDat["area_std"] / slope**2 / 1e6**2
    calibDat["perimeter_mean"] = calibDat["perimeter_mean"] / slope / 1e6
    calibDat["perimeter_std"] = calibDat["perimeter_std"] / slope / 1e6

    calibDat["Dequiv_mean"] = (calibDat["Dequiv_mean"]) / slope / 1e6

    if "velocity_dist" in calibDat.data_vars:
        calibDat["velocity_dist"] = (calibDat["velocity_dist"]) / slope / 1e6
        calibDat["velocity_mean"] = (calibDat["velocity_mean"]) / slope / 1e6
        calibDat["velocity_std"] = (calibDat["velocity_std"]) / slope / 1e6

    return calibDat


def _getDataQuality1(case, config, timeIndex, timeIndex1, sublevel, camera):
    """Estimate data quality for level2"""

    f1 = files.FindFiles(case, camera, config, version='1.2')

    fname1 = f1.listFilesWithNeighbors("metaEvents")

    event1 = xr.open_mfdataset(fname1).load()

    if camera == "leader":
        matchFilesAll = f1.listFilesExt(f"level1{sublevel}")
        matchFilesBroken = [f for f in matchFilesAll if f.endswith("broken.txt")]
        brokenTimes = [
            files.FilenamesFromLevel(f, config).datetime64 for f in matchFilesBroken
        ]
        matchFilesBroken = xr.DataArray(
            matchFilesBroken, dims=["file_starttime"], coords=[brokenTimes]
        )
    else:
        matchFilesBroken = xr.DataArray(
            [], dims=["file_starttime"], coords={"file_starttime": []}
        )

    graceTime = 2  # s
    newfiles1 = event1.isel(file_starttime=(event1.event == "newfile"))

    dataRecorded = []
    processingFailed = []
    for tt, tI1min in enumerate(timeIndex):
        tDiff1 = (
            np.datetime64(tI1min) - newfiles1.file_starttime
        ).values / np.timedelta64(1, "s")
        dataRecorded1 = np.any(
            tDiff1[tDiff1 >= -graceTime] < (config.newFileInt - graceTime)
        )
        #     print(tI1min, dataRecordedF, dataRecorded)
        dataRecorded.append(dataRecorded1)

        if len(matchFilesBroken) > 0:
            tDiffBroken = (
                np.datetime64(tI1min) - matchFilesBroken.file_starttime
            ).values / np.timedelta64(1, "s")
            processingFailed1 = np.any(
                tDiffBroken[tDiffBroken >= -graceTime] < (config.newFileInt - graceTime)
            )
            processingFailed.append(processingFailed1)
        else:
            processingFailed.append(False)

    recordingFailed1 = xr.DataArray(
        ~np.array(dataRecorded), dims=["time"], coords=[timeIndex]
    )
    processingFailed = xr.DataArray(processingFailed, dims=["time"], coords=[timeIndex])

    blowingSnowRatio1, nDetected = tools.identifyBlockedBlowingSnowData(
        f1.listFilesWithNeighbors("metaDetection"), config, timeIndex1, sublevel
    )

    blockedPixels1 = event1.blocking.sel(blockingThreshold=50, drop=True)

    blockedPixels1 = blockedPixels1.reindex(
        file_starttime=timeIndex,
        method="nearest",
        tolerance=np.timedelta64(int(config.newFileInt / 1.9), "s"),
    )
    blockedPixels1 = blockedPixels1.rename(file_starttime="time")

    return (
        recordingFailed1,
        processingFailed,
        blockedPixels1,
        blowingSnowRatio1,
        nDetected,
    )


def getDataQuality(case, config, timeIndex, timeIndex1, sublevel, camera=None):
    """
    Estimate comprehensive data quality for level 2 processing.

    This function aggregates data quality metrics across both cameras for
    match and track processing levels.

    Parameters
    ----------
    case : str
        Case identifier for reference data. For loopify decorator, this is
        a date string (YYYYMMDD) or date range (YYYYMMDD-YYYYMMDD) or comma-separated dates.
    config : dict
        Configuration settings for quality assessment.
    timeIndex : numpy.ndarray
        Time index for quality evaluation.
    timeIndex1 : numpy.ndarray
        Detailed time index for quality calculations.
    sublevel : str
        Processing sublevel. Must be one of ["match", "track", "detect"].
    camera : str, optional
        Camera identifier for quality metrics, defaults to None.
        For loopify_with_camera decorator, can be "all", "leader", or "follower".

    Returns
    -------
    tuple
        Quality metrics including recording failures, processing failures,
        blocked pixels ratios, blowing snow ratios, and observations ratio.
    """
    if sublevel in ["match", "track"]:
        (
            recordingFailedL,
            processingFailed,
            blockedPixelsL,
            blowingSnowRatioL,
            nDetectedL,
        ) = _getDataQuality1(case, config, timeIndex, timeIndex1, sublevel, "leader")

        (
            recordingFailedF,
            _________,
            blockedPixelsF,
            blowingSnowRatioF,
            nDetectedF,
        ) = _getDataQuality1(case, config, timeIndex, timeIndex1, sublevel, "follower")

        recordingFailed = xr.concat((recordingFailedL, recordingFailedF), dim="camera")
        recordingFailed["camera"] = ["leader", "follower"]

        blockedPixels = xr.concat((blockedPixelsL, blockedPixelsF), dim="camera")
        blockedPixels["camera"] = ["leader", "follower"]

        blowingSnowRatio = xr.concat(
            (blowingSnowRatioL, blowingSnowRatioF), dim="camera"
        )
        blowingSnowRatio["camera"] = ["leader", "follower"]

        observationsRatio = tools.compareNDetected(nDetectedL, nDetectedF)

        return (
            recordingFailed,
            processingFailed,
            blockedPixels,
            blowingSnowRatio,
            observationsRatio,
        )

    else:
        (
            recordingFailed,
            processingFailed,
            blockedPixels,
            blowingSnowRatio,
            nDetected,
        ) = _getDataQuality1(case, config, timeIndex, timeIndex1, sublevel, camera)
        return recordingFailed, processingFailed, blockedPixels, blowingSnowRatio, 1


def _createBox(p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Create a 3D box geometry for observation volume calculation.

    Constructs a trimesh representation of a box defined by eight corner points.

    Parameters
    ----------
    p1 : array-like
        First corner point.
    p2 : array-like
        Second corner point.
    p3 : array-like
        Third corner point.
    p4 : array-like
        Fourth corner point.
    p5 : array-like
        Fifth corner point.
    p6 : array-like
        Sixth corner point.
    p7 : array-like
        Seventh corner point.
    p8 : array-like
        Eighth corner point.

    Returns
    -------
    trimesh.Trimesh
        Trimesh object representing the box geometry.
    """
    import trimesh

    vertices = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
    faces = np.array(
        [
            [1, 3, 0],
            [4, 1, 0],
            [0, 3, 2],
            [2, 4, 0],
            [1, 7, 3],
            [5, 1, 4],
            [5, 7, 1],
            [3, 7, 2],
            [6, 4, 2],
            [2, 7, 6],
            [6, 5, 4],
            [7, 5, 6],
        ]
    )

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def createLeaderBox(width, height, delta=0, deltaExtra1=0, deltaExtra2=0):
    """
    Create a 3D box representing the leader camera observation volume.

    Defines the physical volume observable by the leader camera based on
    image dimensions and boundary constraints.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    delta : number, optional
        Distance to all edges, defaults to 0.
    deltaExtra1 : number, optional
        Distance to left vertical edge, defaults to 0.
    deltaExtra2 : number, optional
        Distance to right vertical edge, defaults to 0.

    Returns
    -------
    trimesh.Trimesh
        Trimesh object representing the leader observation volume.
    """

    X0 = -width
    X1 = 2 * width
    Y0 = 0 + max(delta, deltaExtra1)
    Y1 = width - max(delta, deltaExtra2)
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


def createFollowerBox(
    width,
    height,
    camera_phi,
    camera_theta,
    camera_Ofz,
    delta=0,
    deltaExtra1=0,
    deltaExtra2=0,
):
    """
    Create a 3D box representing the follower camera observation volume.

    Defines the physical volume observable by the follower camera based on
    image dimensions, camera orientation, and boundary constraints.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    camera_phi : float
        Roll of follower camera.
    camera_theta : float
        Pitch of follower camera.
    camera_Ofz : float
        Offset in z direction.
    delta : number, optional
        Distance to all edges, defaults to 0.
    deltaExtra1 : number, optional
        Distance to left vertical edge, defaults to 0.
    deltaExtra2 : number, optional
        Distance to right vertical edge, defaults to 0.

    Returns
    -------
    trimesh.Trimesh
        Trimesh object representing the follower observation volume.
    """
    X0 = 0 + max(delta, deltaExtra1)
    X1 = width - max(delta, deltaExtra2)
    Y0 = -width
    Y1 = 2 * width
    Z0 = 0 + delta
    Z1 = height - delta

    psi = Olx = Ofy = 0.0

    p1 = shiftRotate_F2L(
        X0, Y0, Z0, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz
    )
    p2 = shiftRotate_F2L(
        X0, Y0, Z1, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz
    )
    p3 = shiftRotate_F2L(
        X0, Y1, Z0, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz
    )
    p4 = shiftRotate_F2L(
        X0, Y1, Z1, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz
    )
    p5 = shiftRotate_F2L(
        X1, Y0, Z0, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz
    )
    p6 = shiftRotate_F2L(
        X1, Y0, Z1, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz
    )
    p7 = shiftRotate_F2L(
        X1, Y1, Z0, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz
    )
    p8 = shiftRotate_F2L(
        X1, Y1, Z1, camera_phi, camera_theta, psi, Olx, Ofy, camera_Ofz
    )

    return _createBox(p1, p2, p3, p4, p5, p6, p7, p8)


def _estimateVolume(
    width,
    height,
    camera_phi,
    camera_theta,
    camera_Ofz,
    delta=0,
    deltaLeaderExtra1=0,
    deltaLeaderExtra2=0,
    deltaFollowerExtra1=0,
    deltaFollowerExtra2=0,
):
    """
    Estimate intersecting volume of leader and follower observation volumes.

    Calculates the three-dimensional intersection volume shared by both
    cameras' observation regions.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    camera_phi : float
        Roll of follower camera.
    camera_theta : float
        Pitch of follower camera.
    camera_Ofz : float
        Offset in z direction.
    delta : number, optional
        Distance to all edges, defaults to 0.
    deltaLeaderExtra1 : number, optional
        Distance to left vertical edge for leader, defaults to 0.
    deltaLeaderExtra2 : number, optional
        Distance to right vertical edge for leader, defaults to 0.
    deltaFollowerExtra1 : number, optional
        Distance to left vertical edge for follower, defaults to 0.
    deltaFollowerExtra2 : number, optional
        Distance to right vertical edge for follower, defaults to 0.

    Returns
    -------
    float
        Intersection volume in cubic units.
    """

    follower = createFollowerBox(
        width,
        height,
        camera_phi,
        camera_theta,
        camera_Ofz,
        delta=delta,
        deltaExtra1=deltaFollowerExtra1,
        deltaExtra2=deltaFollowerExtra2,
    )
    leader = createLeaderBox(
        width,
        height,
        delta=delta,
        deltaExtra1=deltaLeaderExtra1,
        deltaExtra2=deltaLeaderExtra2,
    )

    volume = leader.intersection(follower).volume

    return volume


@functools.cache
def _estimateVolumes(
    width,
    height,
    correctForSmallOnes,
    camera_phi,
    camera_theta,
    camera_Ofz,
    sizeBins,
    maxSharpnessSizes,
    maxSharpnessLeader,
    maxSharpnessFollower,
    nSteps=5,
    interpolate=True,
):
    """
    Estimate intersecting observation volumes for different particle sizes.

    Calculates the observation volume for multiple particle size bins based on
    camera configuration and geometric constraints.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    correctForSmallOnes : bool
        Flag for small particle correction.
    camera_phi : float
        Roll of follower camera.
    camera_theta : float
        Pitch of follower camera.
    camera_Ofz : float
        Offset in z direction.
    sizeBins : array-like
        Particle size bins.
    maxSharpnessSizes : array-like
        Sizes requiring special treatment.
    maxSharpnessLeader : array-like
        Leader camera adjustment parameters.
    maxSharpnessFollower : array-like
        Follower camera adjustment parameters.
    nSteps : int, optional
        Number of interpolation points, defaults to 5.
    interpolate : bool, optional
        Apply interpolation, defaults to True.

    Returns
    -------
    tuple
        Size bins and corresponding volumes.
    """
    if np.any(np.isnan((camera_phi, camera_theta, camera_Ofz))):
        volumes = np.full(len(sizeBins), np.nan)
    else:
        volumes = []
        # select only nStep distances and interpolate rest
        ddInter = np.linspace(0, len(sizeBins) - 1, nSteps, endpoint=True, dtype=int)
        # for the smallest one we use an extra large distance to the edge, so interpolation won't work

        # add one  point which won't be limited for interpolation
        limtedDs = list(maxSharpnessSizes)
        try:
            limtedDs = limtedDs + [max(maxSharpnessSizes) + 1]
        except ValueError:  # ie maxSharpnessSizes empty
            pass
        ddInter = sorted(set(list(ddInter) + limtedDs))

        for ii, dd in enumerate(ddInter):
            if correctForSmallOnes:
                try:
                    ss = np.where(np.array(maxSharpnessSizes) == dd)[0][0]
                except IndexError:  # i.e. dd not in maxSharpnessSizes
                    leaderExtra = [0, 0]
                    followerExtra = [0, 0]
                else:
                    leaderExtra = np.array(maxSharpnessLeader[ss])
                    followerExtra = np.array(maxSharpnessFollower[ss])
                    leaderExtra[1] = width - leaderExtra[1]
                    followerExtra[1] = width - followerExtra[1]
            else:
                leaderExtra = [0, 0]
                followerExtra = [0, 0]

                # distance to the edge is only half of the particle size!
            delta = np.ceil(dd / 2).astype(int)

            volumes.append(
                _estimateVolume(
                    width,
                    height,
                    camera_phi,
                    camera_theta,
                    camera_Ofz,
                    delta=delta,
                    deltaLeaderExtra1=leaderExtra[0],
                    deltaLeaderExtra2=leaderExtra[1],
                    deltaFollowerExtra1=followerExtra[0],
                    deltaFollowerExtra2=followerExtra[1],
                )
            )
            log.debug(
                tools.concat(ii, dd, delta, leaderExtra, followerExtra, volumes[-1])
            )

        if interpolate:
            Ds_inter, volumes = interpolateVolumes(np.array(sizeBins), ddInter, volumes)
            assert np.all(Ds_inter == np.array(sizeBins))
    return np.array(sizeBins), np.array(volumes)


def interpolateVolumes(Dfull, Dstep, volumes1):
    """
    Interpolate volumes considering the cubic dependency.

    Applies cubic interpolation to volume data between specified intervals
    to obtain smooth volume estimates across all particle sizes.

    Parameters
    ----------
    Dfull : array-like
        Full range of particle sizes.
    Dstep : array-like
        Step sizes for interpolation.
    volumes1 : array-like
        Volume values at step sizes.

    Returns
    -------
    tuple
        Interpolated size bins and volumes.
    """

    volumes = np.interp(Dfull, Dstep, np.array(volumes1) ** (1 / 3.0)) ** 3
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
