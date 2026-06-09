# -*- coding: utf-8 -*-

import datetime
import glob
import os
import shutil
import sys
import uuid
import warnings
from copy import deepcopy

import numpy as np
import xarray as xr
from image_packer import packer
from loguru import logger as log

from . import *
from . import __version__, av, files, tools


@tools.loopify
def generate(
    case,
    config,
    level,
    camera="all",
    version=__version__,
    skipExisting=True,
    endYesterday=False,
):
    """
    Generate quicklooks for a specified level over a range of days.

    Parameters
    ----------
    case : str
        case identifier. e.g. string in format YYYYMMDD
    config : dict
        Configuration dictionary or settings file
    camera : str, optional
        Camera identifier. Default is "all"".
    level : str
        The data level for which to generate quicklooks. Options include:
        'level0', 'level1detect', 'metaFrames', 'level2detect', 'level1match',
        'level2match', 'level2track', 'level3combinedRiming'
    version : str, optional
        Version identifier for the processing, by default __version__
    skipExisting : bool, optional
        Whether to skip generation if output already exists, by default True
    endYesterday : bool, optional
        Whether to end the case range at yesterday. Default is True.
        This parameter is passed to getCaseRange.

    Returns
    -------
    None
        Function processes files but doesn't return anything directly

    Raises
    ------
    ValueError
        If the specified level is not recognized
    """

    if camera == "all":
        if level in ["level0", "level1detect", "metaFrames", "level2detect"]:
            cameras = config["instruments"]
        else:
            cameras = [config.leader]
    else:
        cameras = [camera]
    for camera in cameras:
        if level == "level0":
            res = level0Quicklook(
                case, camera, config, version=version, skipExisting=skipExisting
            )
        elif level == "metaFrames":
            res = metaFramesQuicklook(
                case, camera, config, version=version, skipExisting=skipExisting
            )
        elif level == "metaRotation":
            res = metaRotationQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        elif level == "level1detect":
            res = createLevel1detectQuicklook(
                case, camera, config, version=version, skipExisting=skipExisting
            )
        elif level == "level1match":
            res = createLevel1matchParticlesQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        elif level == "level2detect":
            res = createLevel2detectQuicklook(
                case, camera, config, version=version, skipExisting=skipExisting
            )
        elif level == "level2match":
            res = createLevel2matchQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        elif level == "level2track":
            res = createLevel2matchQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        elif level == "level3combinedRiming":
            res = createLevel3RimingQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        else:
            raise ValueError(f"Do not know level {level}")
    return res


def _plotVar(
    pVar,
    capture_time,
    ax,
    ylabel=None,
    axhline=None,
    xlabel=None,
    resample="5min",
    func="mean",
    color="C1",
    label=None,
    ratiovar=None,
):
    """
    Plot variable data with error bars or statistics.

    Parameters
    ----------
    pVar : array-like
        Variable data to plot
    capture_time : xarray.DataArray
        Time coordinates for the data
    ax : matplotlib.axes.Axes
        Axes object to plot on
    ylabel : str, optional
        Label for y-axis, by default None
    axhline : float, optional
        Horizontal line to draw, by default None
    xlabel : str, optional
        Label for x-axis, by default None
    resample : str, optional
        Resampling frequency, by default "5min"
    func : str, optional
        Function to apply for aggregation ('mean', 'count', 'first', 'ratio'),
        by default "mean"
    color : str, optional
        Color for the plot, by default "C1"
    label : str, optional
        Label for legend, by default None
    ratiovar : xarray.DataArray, optional
        Variable for ratio calculation, by default None

    Returns
    -------
    tuple
        (axes, resampled_data)
    """
    if axhline is not None:
        ax.axhline(axhline, color="magenta", lw=0.5)

    try:
        capture_time = capture_time.isel(camera=0)
    except ValueError:
        pass

    pVar = xr.DataArray(pVar, coords=[capture_time.values], dims=["time"])
    pVar = pVar.sortby("time")  # this is to avoid making a fuzz about jumping indices
    pVar = pVar.resample(time=resample)

    if func == "mean":
        pMean = pVar.mean()
        pStd = pVar.std()

        try:
            pMean.plot(ax=ax, marker=".", color=color, label=label)
        except TypeError:
            log.warning("no data to plot pMean")
        ax.fill_between(pMean.time, pMean - pStd, pMean + pStd, alpha=0.3, color=color)
    elif func == "count":
        pCount = pVar.count()
        try:
            pCount.plot(ax=ax, marker=".", color=color, label=label)
        except TypeError:
            log.warning("no data to plot pCount")
    elif func == "first":
        pFirst = pVar.first()
        try:
            pFirst.plot(ax=ax, marker=".", color=color, label=label)
        except TypeError:
            log.warning("no data to plot pFirst")
    elif func == "ratio":
        try:
            (ratiovar.count() / pVar.count()).plot(
                ax=ax, marker=".", color=color, label=label
            )
        except TypeError:
            log.warning("no data to plot ratiovar/pvar")
    else:
        raise ValueError(f"Do not know {func}")

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return ax, pVar


def _plot2dhist(
    pVar,
    capture_time,
    ax,
    cax,
    bins,
    ylabel=None,
    logScale=True,
    resample="5min",
    cbarlabel=None,
):
    """
    Plot 2D histogram data.

    Parameters
    ----------
    pVar : array-like
        Variable data to plot
    capture_time : xarray.DataArray
        Time coordinates for the data
    ax : matplotlib.axes.Axes
        Axes object to plot on
    cax : matplotlib.axes.Axes
        Axes object for colorbar
    bins : array-like
        Bin edges for histogram
    ylabel : str, optional
        Label for y-axis, by default None
    logScale : bool, optional
        Whether to use logarithmic scale, by default True
    resample : str, optional
        Resampling frequency, by default "5min"
    cbarlabel : str, optional
        Label for colorbar, by default None

    Returns
    -------
    tuple
        (axes, resampled_data)
    """
    import matplotlib.pyplot as plt

    pVar = xr.DataArray(
        pVar, coords=[capture_time.isel(camera=0).values], dims=["time"]
    )
    pVar = pVar.sortby("time")  # this is to avoid making a fuzz about jumping indices
    pVar = pVar.resample(time=resample)

    binMeans = (bins[1:] + bins[:-1]) / 2.0
    hists = []
    labels = []
    for l, p in pVar:
        labels.append(l)
        hists.append(
            100 * np.histogram(p, bins=bins)[0] / np.sum(np.isfinite(p.values))
        )

    hists = xr.DataArray(hists, coords=[labels, binMeans], dims=["time", "bins"])
    hists = hists.where(hists != 0)

    hists = hists.resample(
        time=resample
    ).first()  # little trick to fill up missing values
    # import pdb; pdb.set_trace()

    if np.any(np.array(hists.shape) <= 1):
        log.warning("no data to plot hists")
    else:
        pc = hists.T.plot.pcolormesh(
            ax=ax, cbar_ax=cax, cbar_kwargs={"label": cbarlabel}
        )

    if logScale:
        ax.set_yscale("log")

    ax.set_ylabel(ylabel)
    ax.set_xlabel(None)
    ax.set_ylim(bins[0], bins[-1])

    return ax, pVar


def _crop(image):
    """
    Crop black image parts from an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array

    Returns
    -------
    numpy.ndarray
        Cropped image array
    """
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[
        np.min(y_nonzero) : np.max(y_nonzero) + 1,
        np.min(x_nonzero) : np.max(x_nonzero) + 1,
    ]


@tools.loopify_with_camera(endYesterday=False)
def createLevel1detectQuicklookHourly(
    case,
    camera,
    config,
    hours=range(1, 24),
    version=__version__,
    container_width=200,
    container_height_max=600,
    nTiles=120,
    nRows=4,
    extra=1,
    readParticlesFromFiles=True,
    skipExisting=True,
    ffOut="default",
    timeStep="fixed",
    minBlur="config",
    minSize=8,
    omitLabel4small="config",
    timedelta=np.timedelta64(1, "h"),
):
    """
    Create hourly version of particle quicklook.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    hours : range, optional
        Hours to process, by default range(1, 24)
    version : str, optional
        Version identifier, by default __version__
    container_width : int, optional
        Width of each tile container, by default 200
    container_height_max : int, optional
        Maximum height of each tile container, by default 600
    nTiles : int, optional
        Number of tiles per row, by default 120
    nRows : int, optional
        Number of rows in output, by default 4
    extra : int, optional
        Extra spacing between tiles, by default 1
    readParticlesFromFiles : bool, optional
        Whether to read particles from files, by default True
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    ffOut : str, optional
        Output file path, by default "default"
    timeStep : str, optional
        Time step method ('fixed' or 'variable'), by default "fixed"
    minBlur : float or str, optional
        Minimum blur threshold, by default "config"
    minSize : int or str, optional
        Minimum particle size, by default 8
    omitLabel4small : bool or str, optional
        Whether to omit labels for small particles, by default "config"
    timedelta : numpy.timedelta64, optional
        Time window for data selection, by default np.timedelta64(1, "h")

    Returns
    -------
    None
        Function processes files but doesn't return anything directly
    """
    import matplotlib.pyplot as plt

    for hh in hours:
        timestamp = f"{case}-{hh:02d}"
        f = createLevel1detectQuicklook(
            timestamp,
            camera,
            config,
            version=version,
            container_width=container_width,
            container_height_max=container_height_max,
            nTiles=nTiles,
            nRows=nRows,
            extra=extra,
            readParticlesFromFiles=readParticlesFromFiles,
            skipExisting=skipExisting,
            ffOut=ffOut,
            timeStep=timeStep,  # attempt to fill plot equally
            minBlur=minBlur,
            minSize=minSize,
            omitLabel4small=omitLabel4small,
            timedelta=timedelta,
            returnFig=False,
        )
        print(f)

    return


@tools.loopify_with_camera(endYesterday=False)
def createLevel1detectQuicklook(
    timestamp,
    camera,
    config,
    version=__version__,
    container_width=200,
    container_height_max=300,
    nTiles=60,
    nRows=4,
    extra=1,
    readParticlesFromFiles=True,
    skipExisting=True,
    ffOut="default",
    timeStep="variable",  # attempt to fill plot equally
    minBlur="config",
    minSize="config",
    omitLabel4small="config",
    timedelta=np.timedelta64(1, "D"),
    returnFig=True,
):
    """
    Create level1detect quicklook for a given timestamp.

    Parameters
    ----------
    timestamp : str
        Timestamp string in format YYYYMMDDHH or YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    container_width : int, optional
        Width of each tile container, by default 200
    container_height_max : int, optional
        Maximum height of each tile container, by default 300
    nTiles : int, optional
        Number of tiles per row, by default 60
    nRows : int, optional
        Number of rows in output, by default 4
    extra : int, optional
        Extra spacing between tiles, by default 1
    readParticlesFromFiles : bool, optional
        Whether to read particles from files, by default True
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    ffOut : str, optional
        Output file path, by default "default"
    timeStep : str, optional
        Time step method ('fixed' or 'variable'), by default "variable"
    minBlur : float or str, optional
        Minimum blur threshold, by default "config"
    minSize : int or str, optional
        Minimum particle size, by default "config"
    omitLabel4small : bool or str, optional
        Whether to omit labels for small particles, by default "config"
    timedelta : numpy.timedelta64, optional
        Time window for data selection, by default np.timedelta64(1, "D")
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, figure) if returnFig=True, otherwise just output_file_path
    """
    import cv2
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
    from tqdm import tqdm

    if minBlur == "config":
        minBlur = config["level1detectQuicklook"]["minBlur"]
    if minSize == "config":
        minSize = config["level1detectQuicklook"]["minSize"]
    if omitLabel4small == "config":
        omitLabel4small = config["level1detectQuicklook"]["omitLabel4small"]

    ff = files.FindFiles(timestamp, camera, config, version)
    if ffOut == "default":
        ffOut = ff.quicklook.level1detect

    site = config["site"]

    particlesPloted = 0

    if skipExisting and tools.checkForExisting(
        ffOut,
        level0=ff.listFiles("level0"),
        events=ff.listFiles("metaEvents"),
        parents=ff.listFilesExt("level1detect"),
    ):
        return None, None

    if site != "mosaic":
        if (len(ff.listFiles("level0")) == 0) and (
            len(ff.listFiles("level0status")) == 0
        ):
            print("NO DATA YET (TRANSFERRED?)", ffOut)
            return None, None

    if len(ff.listFiles("metaFrames")) > len(ff.listFiles("level0")):
        print("DATA TRANSFER INCOMPLETE ", ffOut)
        print(
            len(ff.listFiles("level0")),
            "of",
            len(ff.listFiles("metaFrames")),
            "transmitted",
        )
        return None, None

    if (len(ff.listFilesExt("level1detect")) == 0) and (
        len(ff.listFiles("level0")) > 0
    ):
        print("NO DATA YET ", ffOut)
        return None, None

    if not ff.isCompleteL1detect:
        print(
            "NOT COMPLETE YET %i of %i L1detect %s "
            % (
                len(ff.listFilesExt("level1detect")),
                len(ff.listFiles("level0txt")),
                ffOut,
            )
        )

        #         if (len(ff.listFilesExt("level1detect")) == len(ff.listFiles("level0"))):
        #             afshgsa
        return None, None

    #     else:

    total_width = (container_width + extra) * nTiles // nRows
    max_height = (20 + extra + container_height_max) * nRows + 60

    # let use a matplotlib font becuase we can be sure it is there
    mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
    mpl_ttf_dir = os.path.join(mpl_data_dir, "fonts", "ttf")
    font = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 35)
    fontL = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 16)

    print("RUNNING open files ", ffOut, len(ff.listFiles("metaFrames")))

    dats2 = []
    l1Files = ff.listFilesWithNeighbors("level1detect")

    nParticles = 0

    for fname2 in tqdm(l1Files):
        fname1 = fname2.replace("level1detect", "metaFrames")

        try:
            dat2 = xr.open_dataset(fname2)
        except FileNotFoundError:
            if os.path.isfile(f"{fname2}.nodata"):
                continue
            elif os.path.isfile(f"{fname2}.broken.txt"):
                continue
            elif os.path.isfile(f"{fname2}.notenoughframes"):
                continue
            else:
                raise FileNotFoundError(fname2)

        nParticles += len(dat2.pid)

        dat2 = dat2[
            ["Dmax", "blur", "record_time", "record_id", "position_upperLeft", "Droi"]
        ]
        # it is more efficient to load the data now in comparison to after isel
        dat2 = dat2.load()

        dat2 = dat2.isel(pid=((dat2.blur > minBlur) & (dat2.Dmax > minSize)))

        if len(dat2.pid) == 0:
            continue
        dat2 = dat2[["record_time", "record_id", "Droi", "position_upperLeft"]]

        dat2 = dat2.expand_dims(dict(file=[fname1]))
        #     dat2 = dat2.set_coords(dict(file = fname2))
        dat2 = dat2.stack(fpid=("file", "pid"))

        dats2.append(dat2)

    print("opened")
    new_im = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    if len(dats2) == 0:
        draw = ImageDraw.Draw(new_im)
        draw.text(
            (total_width // 3, max_height // 3), "no raw data", (0, 0, 0), font=font
        )

        nParticles = 0

    else:
        limDat = xr.concat(dats2, dim="fpid")
        limDat = limDat.isel(
            fpid=(limDat.record_time >= ff.datetime64)
            & (limDat.record_time < (ff.datetime64 + timedelta))
        )
        #         limDat = dats2
        print("merged")

        if len(limDat.fpid) == 0:
            print("TOO FEW DATA ", ffOut)

            draw = ImageDraw.Draw(new_im)
            draw.text(
                (total_width // 3, max_height // 3),
                "no data recorded",
                (0, 0, 0),
                font=font,
            )

            nParticles = 0

        else:
            print("Total number of particles for plotting %i" % len(limDat.fpid))

            if timeStep == "variable":
                timeSteps = np.percentile(
                    limDat.record_time, np.linspace(0, 100, nTiles + 1)
                )
            elif timeStep == "fixed":
                timeSteps = np.array(
                    pd.date_range(
                        start=ff.datetime64,
                        periods=nTiles + 1,
                        end=ff.datetime64 + timedelta,
                    )
                )
            else:
                raise ValueError("do not understand timeStep")
            mosaics = []

            videos = {}
            for tt, (t1, t2) in enumerate(zip(timeSteps[:-1], timeSteps[1:])):
                if tt == len(timeSteps) - 2:
                    whereCond = limDat.record_time >= t1
                else:
                    whereCond = (limDat.record_time >= t1) & (limDat.record_time < t2)

                thisDat = limDat.where(whereCond).dropna("fpid")
                totalArea = 0

                # select pids randomly, figure out how much we need, and sort them again
                pids = deepcopy(thisDat.fpid.values)
                fnames = thisDat.fpid.file.values
                particleImages = {}
                for fname in np.unique(fnames):
                    fn = files.FilenamesFromLevel(fname, config)
                    # tarRoot = fn.fname.imagesL1detect.split("/")[-1].replace(".tar.bz2","")
                    # particleImages[fname] = (tools.imageTarFile.open(fn.fname.imagesL1detect, "r:bz2"), tarRoot)
                    # print(fn.fname.imagesL1detect)

                    for imageFile in [
                        fn.fname.imagesL1detect,
                        fn.fname.imagesL1detect.replace(".bin", ".zip"),
                    ]:
                        try:
                            particleImages[fname] = tools.imageZipFile(
                                imageFile, mode="r"
                            )
                        except FileNotFoundError:
                            log.warning(f"did not find {imageFile}")
                        except tools.zipfile.BadZipFile:
                            log.warning(f"is broken {imageFile}")
                        else:
                            break
                nPids = len(pids)
                rng = np.random.default_rng(tt)
                rng.shuffle(pids)

                containerSize = container_width * container_height_max

                if nPids < 5:  # fo very few particles I want to see them all!
                    nParticlesNeeded = len(pids)
                else:
                    try:
                        nParticlesNeeded = (
                            np.where(
                                (thisDat.sel(fpid=pids).Droi + 1)
                                .prod("dim2D")
                                .cumsum("fpid")
                                / containerSize
                                > 0.95
                            )[0][0]
                            + 1
                        )
                    except IndexError:
                        nParticlesNeeded = nPids

                pids = np.sort(pids[:nParticlesNeeded])
                print(tt, "/", nTiles, t1, t2, nParticlesNeeded, "of", nPids)
                particlesPloted += nParticlesNeeded
                ims = []

                videos = {}

                for fname, pid in pids:
                    # basenameImg = fname.split('/')[-1]

                    if not readParticlesFromFiles:
                        f1 = files.FilenamesFromLevel(fname, config)
                        thisfname_lv0 = f1.fname.level0

                        if thisfname_lv0 not in videos.keys():
                            for k in videos.keys():
                                videos[k].release()

                            videos[thisfname_lv0] = av.VideoReaderMeta(
                                thisfname_lv0, fname, safeMode=True
                            )
                    #             print('opened %s'%thisfname_lv0)

                    particle = thisDat.sel(fpid=(fname, pid))
                    kk = int(particle.record_id.values)
                    background = 0
                    if not readParticlesFromFiles:
                        _, frame1, _ = videos[thisfname_lv0].getFrameByIndex(kk)

                        if frame1 is None:
                            continue

                        w, h = particle.Droi.values.astype(int)
                        x, y = particle.position_upperLeft.values.astype(int)
                        if len(frame1.shape) == 3:
                            frame1 = frame1[:, :, 0]
                        im = frame1[
                            y
                            + config.level1detect.height_offset : y
                            + config.level1detect.height_offset
                            + h,
                            x : x + w,
                        ]
                    else:
                        pidStr = "%07i" % pid
                        # imName = '%s.png' % (pidStr)
                        # imfname = '%s/%s/%s' % (particleImages[fname][1],pidStr[:4], imName)
                        try:
                            # im = particleImages[fname][0].extractimage(imfname)
                            im = particleImages[fname].extractnpy(pidStr)
                        except KeyError:
                            print("NOT FOUND ", pidStr)
                            raise ValueError
                            continue
                        # apply alpha channel
                        # im[...,0][im[...,1] == 0] = background
                        # drop alpha channel
                        im = im[..., 0]

                    # im = av.doubleDynamicRange(im, offset=2)

                    im = np.pad(im, [(0, 1), (0, 1)])
                    try:
                        fid = np.where(fname == np.array(ff.listFiles("metaFrames")))[
                            0
                        ][0]
                    except:
                        fid = -1
                    text = np.full((100, 100), background, dtype=np.uint8)

                    textStr = "%i.%i" % (fid, pid)
                    # print(textStr)
                    text = cv2.putText(
                        text,
                        textStr,
                        (0, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.75,
                        255,
                        1,
                    )
                    text = _crop(text)

                    y1, x1 = im.shape
                    y2, x2 = text.shape

                    # only add label if large enough
                    if (omitLabel4small == "all") or (
                        (omitLabel4small == True) and (x1 < x2)
                    ):
                        imT = im
                    else:
                        y3 = y1 + y2
                        x3 = max(x1, x2)
                        imT = np.full((y3, x3), background, dtype=np.uint8)
                        imT[:y1, :x1] = im
                        imT[y1:, :x2] = text
                    ims.append(imT)
                    totalArea += np.prod(imT.shape)

                for fname in fnames:
                    try:
                        particleImages[fname].close()
                    except KeyError:
                        pass

                # make tile
                images = [Image.fromarray(im) for im in ims]
                if len(images) == 0:
                    mosaic = (
                        np.ones(
                            (container_height_max, container_width, 3), dtype=np.uint8
                        )
                        * 0
                    )
                else:
                    mosaic = Packer_patched(images).pack(
                        container_width=container_width,
                        container_height_max=container_height_max,
                    )
                    mosaic = np.array(mosaic)

                    if container_width > mosaic.shape[1]:
                        mosaic = np.pad(
                            mosaic,
                            [(0, 0), (0, container_width - mosaic.shape[1]), (0, 0)],
                        )

                    # sometimes container is too large...
                    mosaic = mosaic[:container_height_max, :container_width]

                label = Image.fromarray(
                    np.ones((20, mosaic.shape[1], 3), dtype=np.uint8) * 255
                )
                drawL = ImageDraw.Draw(label)
                textStr = "%s-%s" % (
                    str(t1).split(".")[0].split("T")[1],
                    str(t2).split(".")[0].split("T")[1],
                )
                if nParticlesNeeded != nPids:
                    textStr += " (R)"
                drawL.text((0, 0), textStr, (0, 0, 0), font=fontL)

                mosaic = Image.fromarray(np.vstack((label, mosaic)))
                #             display(mosaic)
                mosaics.append(mosaic)

            nCols = nTiles // nRows

            widths, heights = zip(*(i.size for i in mosaics))

            for nRow in range(nRows):
                x_offset = 0

                for im in mosaics[nCols * (nRow) : nCols * (nRow + 1)]:
                    new_im.paste(im, (x_offset, max(heights) * nRow + 50))
                    x_offset += im.size[0] + extra

                # x_offset = 0
            # for im in mosaics[len(mosaics)//nRows:]:
            #   new_im.paste(im, (x_offset,max(heights) +50))
            #   x_offset += im.size[0] + extra

    tenmm = 1e6 / (1 / config.calibration.slope) / 100

    if ff.hour == "":
        title = (
            "%s-%s-%s %s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger detections plotted, 10 mm = %.1f px ="
            % (
                ff.year,
                ff.month,
                ff.day,
                tools.nicerNames(camera),
                config["name"],
                minSize,
                minSize * (1 / config.calibration.slope) * 1e-6 * 1000,
                particlesPloted,
                nParticles,
                tenmm,
            )
        )
    else:
        title = (
            "%s-%s-%sT%s %s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger detections plotted, 10 mm = %.1f px ="
            % (
                ff.year,
                ff.month,
                ff.day,
                ff.hour,
                tools.nicerNames(camera),
                config["name"],
                minSize,
                minSize * (1 / config.calibration.slope) * 1e-6 * 1000,
                particlesPloted,
                nParticles,
                tenmm,
            )
        )

    # new_im = cv2.putText(np.array(new_im), title,
    #                      (0, 45), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS,)
    # (label_width, label_height), baseline = cv2.getTextSize(
    #     title, FONT, FONT_SCALE, FONT_THICKNESS)

    draw = ImageDraw.Draw(new_im)
    draw.text((0, 0), title, (0, 0, 0), font=font)
    width = draw.textlength(title, font=font)
    # Call textbbox to get the bounding box
    bbox = draw.textbbox((0, 0), title, font=font)

    # Extract top and bottom values from the returned tuple
    left, top, right, bottom = bbox
    height = top - bottom
    # width, height = draw.textsize(title, font=font)##Depricated
    draw.line((width + 15, 30, width + 15 + round(tenmm), 30), fill=0, width=5)

    tools.createParentDir(ffOut, mode=config.dirMode)
    new_im.save(ffOut)
    print("SAVED ", ffOut)

    if returnFig:
        return ffOut, new_im
    else:
        return ffOut


class Packer_patched(packer.Packer):
    """
    Patched image_packer routine that works without files.

    This class patches the image_packer library to work with in-memory images
    rather than requiring temporary files.

    Attributes
    ----------
    _uid_to_filepath : dict
        Mapping of unique identifiers to image objects
    _pieces : list
        List of image pieces to pack
    _has_alpha : bool
        Flag indicating whether images have alpha channels
    """

    def __init__(self, images):
        # Ensure plugins are fully loaded so that Image.EXTENSION is populated.
        from PIL import Image, ImageDraw, ImageFont

        Image.init()

        self._uid_to_filepath = dict()
        self._pieces = list()
        self._has_alpha = False

        for im in images:
            width = im.width
            height = im.height
            uid = uuid.uuid4()
            self._uid_to_filepath[uid] = deepcopy(im)
            self._pieces.append(
                packer.blf.Piece(uid=uid, size=packer.blf.Size(width, height))
            )
            if im.mode in ("RGBA", "LA") or (
                im.mode == "P" and "transparency" in im.info
            ):
                self._has_alpha = True

    def pack(self, container_width, options=None, container_height_max=100):
        """Packs multiple images of different sizes or formats into one image.

        Parameters
        ----------
        container_width : int
            Width of the container image
        options : dict, optional
            Packing options, by default None
        container_height_max : int, optional
            Maximum height of the container, by default 100

        Returns
        -------
        PIL.Image.Image
            Packed image
        """
        if options is None:
            options = self._DEFAULT_OPTIONS
        else:
            options = {
                key: options[key] if key in options else self._DEFAULT_OPTIONS[key]
                for key in self._DEFAULT_OPTIONS.keys()
            }

        margin_ = options["margin"]
        assert isinstance(margin_, tuple) and len(margin_) == 4

        if options["enable_vertical_flip"]:
            margin = packer.blf.Thickness(
                top=margin_[2], right=margin_[1], bottom=margin_[0], left=margin_[3]
            )
        else:
            margin = packer.blf.Thickness(
                top=margin_[0], right=margin_[1], bottom=margin_[2], left=margin_[3]
            )

        blf_options = {
            "margin": margin,
            "collapse_margin": options["collapse_margin"],
            "enable_auto_size": options["enable_auto_size"],
            "force_pow2": options["force_pow2"],
        }

        container_width, container_height, regions = packer.blf_solver.solve(
            pieces=self._pieces, container_width=container_width, options=blf_options
        )

        compImage = self._save_image(
            container_width=container_width,
            container_height=container_height_max,
            regions=regions,
            options=options,
        )
        return compImage

    def _save_image(self, container_width, container_height, regions, options):
        from PIL import Image, ImageDraw, ImageFont

        bg_color_ = options["bg_color"]
        assert isinstance(bg_color_, tuple) and (3 <= len(bg_color_) <= 4)
        bg_color = tuple(int(channel * 255.0) for channel in bg_color_)
        if len(bg_color) == 3:
            bg_color += (255,)

        if self._has_alpha:
            blank_image = Image.new(
                mode="RGBA", size=(container_width, container_height), color=bg_color
            )
        else:
            blank_image = Image.new(
                mode="RGB",
                size=(container_width, container_height),
                color=bg_color[0:3],
            )

        enable_vertical_flip = options["enable_vertical_flip"]

        for region in regions:
            x = region.left
            if enable_vertical_flip:
                y = region.bottom
            else:
                y = container_height - region.top

            im = self._uid_to_filepath.get(region.uid)

            blank_image.paste(im=im, box=(x, y))

        return blank_image


@tools.loopify_with_camera(endYesterday=False)
def level0Quicklook(case, camera, config, version=__version__, skipExisting=True):
    """
    Create level0 quicklook for a given case and camera.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True

    Returns
    -------
    tuple
        (output_file_path, None) if successful, (None, None) otherwise
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    global metaDats
    metaDats = []

    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level0
    tools.createParentDir(fOut, mode=config.dirMode)

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
    ):
        return None, None

    print(case, camera, fOut)

    if len(ff.listFiles("level0txt")) == 0 and len(ff.listFiles("level0status")) == 0:
        print(case, "no data")

        return None, None

    fnames = []
    cases = []
    for fname in ff.listFiles("level0jpg"):
        fn = files.Filenames(fname, config, version)
        if fn.case.endswith("0000") or fn.case.endswith("5959"):
            fnames.append(fname)
            cases.append(fn.case)
    fnames = fnames[:24]
    cases = cases[:24]
    execStr = ""
    for f, c in zip(fnames, cases):
        execStr += f"-label {c} {f} "
    if len(fnames) > 0:
        tools.execute_stdout(
            f"montage {execStr} -geometry 225x180+1+1 -tile 6x4 -quality 70% -title '{config.name} {camera} {case}' {fOut}"
        )
    else:
        print("no input files")
        return None
    tools.copyCurrentQuicklook("level0", ff)
    return fOut


@tools.loopify_with_camera(endYesterday=False)
def metaFramesQuicklook(
    case, camera, config, version=__version__, skipExisting=True, plotCompleteOnly=True
):
    """
    Create metaFrames quicklook for a given case and camera.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    plotCompleteOnly : bool, optional
        Whether to only plot if data is complete, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if successful, (None, None) otherwise
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    global metaDats
    metaDats = []

    camera_new = camera.split("_")[0] + "-" + camera.split("_")[1]

    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.metaFrames

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("metaFrames"),
    ):
        return None, None

    if plotCompleteOnly and not ff.isCompleteMetaFrames:
        print(
            "NOT COMPLETE YET %i of %i MetaFrames %s"
            % (
                len(ff.listFilesExt("metaFrames")),
                len(ff.listFiles("level0txt")),
                ff.fnamesPattern.metaFrames,
            )
        )
        return None, None

    print(case, camera, fOut)

    if len(ff.listFiles("level0txt")) == 0 and len(ff.listFiles("level0status")) == 0:
        print(case, "no data")

        return None, None

    try:
        level0status = ff.listFiles("level0status")[0]
    except IndexError:
        level0status = None

    print("reading events")
    if len(ff.listFiles("metaEvents")) == 0:
        log.error(f"event data not found")
        return None, None
    try:
        events = xr.open_dataset(ff.listFiles("metaEvents")[0])
    except:
        log.error(f'{ff.listFiles("metaEvents")[0]} broken')
        return None, None

    if len(events.data_vars) == 0:
        log.error(f'{ff.listFiles("metaEvents")[0]} empty')
        return None, None

    if not ff.isCompleteMetaFrames:
        log.warning(f"meta frames not complete")
        return None, None

    # iterate to get meta data
    # if len(ff.listFiles("metaFrames")) > 0:
    #     print("reading metaFrames")
    # try: #faster?
    #     metaDats = xr.open_mfdataset(ff.listFiles("metaFrames"), combine="by_coords")
    # except ValueError:
    #     metaDats = xr.open_mfdataset(ff.listFiles("metaFrames"), combine="nested")

    # else:
    #     print(f'metaFrames not found')
    #     return None, None
    print("reading metaFrames")
    metaDats = []
    for fname1 in ff.listFiles("metaFrames"):
        metaDat = xr.open_dataset(fname1)
        keys = ["capture_time"]
        if "queue_size" in metaDat.variables:
            keys.append("queue_size")
        else:
            keys.append("record_time")
        metaDat = metaDat[keys]
        metaDats.append(deepcopy(metaDat))
        metaDat.close()
    if len(metaDats) == 0:
        metaDats = None
    else:
        metaDats = xr.concat(metaDats, dim="capture_time")
        metaDats = metaDats.sortby("capture_time")

    ts = events.file_starttime.where(
        (events.event == "sleep-trigger") | (events.event == "stop-trigger")
    )

    print("plotting")
    # plotting
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, figsize=(20, 15), gridspec_kw={"hspace": 0.3}
    )
    # plt.rcParams['text.usetex'] = False
    # plt.rcParams['lines.linewidth'] = 1.5
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    plt.suptitle(
        "VISSS Status-Plot \n"
        + f"{ff.year}-{ff.month}-{ff.day}"
        + ", "
        + config["name"]
        + "",
        fontsize=25,
        y=0.995,
        fontweight="bold",
        x=mid,
    )
    fig.text(mid, 0.07, "time", fontsize=20, ha="center")

    if metaDats is not None:
        if "queue_size" in keys:
            queue = metaDats.queue_size
            ax1.plot(metaDats.capture_time, queue)
            ax1.set_ylabel("frames in queue", fontsize=20)
            ax1.set_title(
                r"$\bf{" + str(camera_new) + "}$" "\n" "queue size", fontsize=20
            )
        else:
            delay = (metaDats.record_time - metaDats.capture_time).astype(int) / 1e9
            ax1.plot(metaDats.capture_time, delay)
            ax1.set_ylabel("record delay [s]", fontsize=20)
            ax1.set_title(
                r"$\bf{" + str(camera_new) + "}$" "\n" "record delay", fontsize=20
            )

    isBlocked = events.blocking.dropna("file_starttime").sel(blockingThreshold=50) > 0.1
    isBlocked = isBlocked.file_starttime.where(isBlocked).values

    ylim = ax1.get_ylim()
    if isBlocked.any():
        ax1.fill_between(
            isBlocked,
            [ylim[0]] * len(isBlocked),
            [ylim[1]] * len(isBlocked),
            color="red",
            alpha=0.25,
            label="blocked",
        )
    if ts.notnull().sum() > 0:
        ax1.fill_between(
            ts,
            [ylim[0]] * len(ts),
            [ylim[1]] * len(ts),
            color="orange",
            alpha=0.5,
            label="idle",
        )
    ax1.set_ylim(ylim)
    ax1.set_xlim(
        np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
        np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
        + np.timedelta64(1, "D"),
    )
    ax1.tick_params(axis="both", labelsize=15)
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    ax1.grid()

    if metaDats is not None:
        frames = metaDats.capture_time.notnull().resample(capture_time="5min").sum()
        ax2.plot(frames.capture_time, frames)
    ylim = [0, config.fps * 5 * 60 * 1.05]
    if isBlocked.any():
        ax2.fill_between(
            isBlocked,
            [ylim[0]] * len(isBlocked),
            [ylim[1]] * len(isBlocked),
            color="red",
            alpha=0.25,
            label="blocked",
        )
    if ts.notnull().sum() > 0:
        ax2.fill_between(
            ts,
            [ylim[0]] * len(ts),
            [ylim[1]] * len(ts),
            color="orange",
            alpha=0.5,
            label="idle",
        )
    ax2.set_ylim(ylim)
    ax2.set_xlim(
        np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
        np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
        + np.timedelta64(1, "D"),
    )
    ax2.set_title("total frames / 5 min", fontsize=20)
    ax2.set_ylabel("frames", fontsize=20)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    ax2.grid()

    if (
        len(
            events.blocking.isel(
                blockingThreshold=range(0, len(events.blockingThreshold), 2)
            ).dropna("file_starttime")
        )
        != 0
    ):
        events.blocking.isel(
            blockingThreshold=range(0, len(events.blockingThreshold), 2)
        ).dropna("file_starttime").plot.line(x="file_starttime")
        ylim = [0, 1]
        if isBlocked.any():
            ax3.fill_between(
                isBlocked,
                [ylim[0]] * len(isBlocked),
                [ylim[1]] * len(isBlocked),
                color="red",
                alpha=0.25,
                label="blocked",
            )
        if ts.notnull().sum() > 0:
            ax3.fill_between(
                ts, [ylim[0]] * len(ts), [ylim[1]] * len(ts), color="orange", alpha=0.5
            )
        ax3.set_ylim(ylim)
        ax3.set_xlim(
            np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
            np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
            + np.timedelta64(1, "D"),
        )
        ax3.set_title("blocking", fontsize=20)
        ax3.set_ylabel("pixels below \n blocking threshold", fontsize=20)
        ax3.set_xlabel("")
        plt.setp(
            ax3.get_xticklabels(), rotation=False, ha="center", rotation_mode="anchor"
        )
        ax3.tick_params(axis="both", labelsize=15)
        ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        ax3.grid()

    else:
        if isBlocked.any():
            ax3.fill_between(
                isBlocked,
                [ylim[0]] * len(isBlocked),
                [ylim[1]] * len(isBlocked),
                color="red",
                alpha=0.25,
                label="blocked",
            )
        if ts.notnull().sum() > 0:
            ax3.fill_between(
                ts, [ylim[0]] * len(ts), [ylim[1]] * len(ts), color="orange", alpha=0.5
            )
        ax3.set_ylim(0, 1)
        ax3.set_xlim(
            np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
            np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
            + np.timedelta64(1, "D"),
        )
        ax3.set_title("blocking", fontsize=20)
        ax3.set_ylabel("pixels below \n blocking threshold", fontsize=20)
        ax3.set_xlabel("")
        ax3.tick_params(axis="both", labelsize=15)
        ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        ax3.grid()

    firstEvent = True
    for event in events.event:
        if str(event.values).startswith("start") or str(event.values).startswith(
            "launch"
        ):
            for bx in [ax1, ax2, ax3]:
                if firstEvent:
                    label = "restarted"
                    firstEvent = False
                else:
                    label = None
                bx.axvline(
                    event.file_starttime.values, color="red", ls=":", label=label
                )

    ax1.legend(fontsize=15, bbox_to_anchor=(1, 1.4))

    # _statusText(fig, ff.listFiles("metaFrames"), config)
    tools.savefig(fig, config, fOut, fnames=ff.listFiles("metaFrames"))
    tools.copyCurrentQuicklook("metaFrames", ff)

    if metaDats is not None:
        metaDats.close()
    events.close()

    return fOut, fig


@tools.loopify
def createLevel1matchQuicklook(
    case,
    config,
    skipExisting=True,
    version=__version__,
    plotCompleteOnly=True,
    returnFig=True,
):
    """
    Create level1match quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    version : str, optional
        Version identifier, by default __version__
    plotCompleteOnly : bool, optional
        Whether to only plot if data is complete, by default True
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    resample = "5min"  # 5 mins

    # find files
    fl = files.FindFiles(case, config["leader"], config, version)
    ff = files.FindFiles(case, config["follower"], config, version)
    # get level 0 file names
    fOut = fl.quicklook.level1match

    if skipExisting and tools.checkForExisting(
        fOut,
        level0=fl.listFiles("level0") + ff.listFiles("level0"),
        events=fl.listFiles("metaEvents") + ff.listFiles("metaEvents"),
        parents=fl.listFilesExt("level1match"),
    ):
        return None, None

    if (len(fl.listFiles("level0")) == 0) and (len(fl.listFiles("level0status")) == 0):
        print(
            "NO DATA YET (TRANSFERRED?)",
            fl.fnamesPattern["level0"],
            fl.quicklook.level1match,
        )
        return None, None

    if (len(fl.listFilesExt("level1match")) == 0) and (len(fl.listFiles("level0")) > 0):
        print(
            "NO DATA YET ", fl.fnamesPatternExt["level1match"], fl.quicklook.level1match
        )
        return None, None

    if plotCompleteOnly and not fl.isCompleteL1match:
        print(
            "NOT COMPLETE YET %i of %i L1match %s"
            % (
                len(fl.listFilesExt("level1match")),
                len(fl.listFiles("level0txt")),
                fl.fnamesPattern.level1match,
            )
        )
        return None, None

    fnames1M = fl.listFiles("level1match")
    if len(fnames1M) == 0:
        print("No precipitation", case, fl.fnamesPattern.level1match)
        fig, axcax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
        axcax.axis("off")
        axcax.set_title(f"VISSS level1match {config.name} {case} \n No precipitation")
        # _statusText(fig, [], config)
        tools.savefig(fig, config, fOut)
        return fOut, fig

    print("Running", fOut)

    fnames1DL = fl.listFiles("level1detect")
    ff = files.FindFiles(case, config.follower, config, version)
    fnames1DF = ff.listFiles("level1detect")

    assert len(fnames1DL) > 0
    assert len(fnames1DF) > 0

    datMfull = tools.open_mflevel1match(
        fnames1M,
        config,
        datVars=[
            "Dmax",
            "capture_time",
            "matchScore",
            "position3D_centroid",
            "position_upperLeft",
            "Droi",
            "camera_theta",
            "camera_phi",
            "camera_Ofz",
        ],
    )
    datM = datMfull.isel(fpair_id=(datMfull.matchScore >= config.quality.minMatchScore))

    if len(datM.fpair_id) <= 1:
        print("No precipitation (2)", case, fl.fnamesPattern.level1match)
        fig, axcax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
        axcax.axis("off")
        axcax.set_title(
            f"VISSS level1match {config.name} {case} \n No precipitation (2)"
        )
        # _statusText(fig, [], config)
        tools.savefig(fig, config, fOut)
        return fOut, fig

    datDL = tools.open_mflevel1detect(
        fnames1DL, config, skipFixes="all", datVars=["Dmax", "capture_time"]
    )
    datDF = tools.open_mflevel1detect(
        fnames1DF, config, skipFixes="all", datVars=["Dmax", "capture_time"]
    )

    fig, axcax = plt.subplots(
        nrows=9,
        ncols=2,
        figsize=(10, 15),
        gridspec_kw={
            "width_ratios": [1, 0.01],
            "height_ratios": [2, 2, 2, 1, 1, 1, 1, 1, 1],
        },
    )

    fig.suptitle(f"VISSS level1match {config.name} {case}")

    ax = axcax[:, 0]
    cax = axcax[:, 1]

    Dmax = datM.Dmax.mean("camera").values
    _, rs = _plotVar(
        Dmax,
        datM.capture_time,
        ax[0],
        "Counts [-]",
        func="count",
        label="matched",
        color="C1",
        resample=resample,
    )
    _, rs1 = _plotVar(
        datDL.Dmax,
        datDL.capture_time,
        ax[0],
        "Counts [-]",
        func="ratio",
        label="ratio leader",
        color="C2",
        ratiovar=rs,
        resample=resample,
    )
    _, rs1 = _plotVar(
        datDF.Dmax,
        datDF.capture_time,
        ax[0],
        "Counts [-]",
        func="ratio",
        label="ratio follower",
        color="C3",
        ratiovar=rs,
        resample=resample,
    )
    ax[0].set_yscale("log")
    cax[0].axis("off")
    ax[0].legend()

    bins = np.logspace(0, 2.5, 21)
    _, _ = _plot2dhist(
        Dmax, datM.capture_time, ax[1], cax[1], bins, ylabel="Dmax [px]", cbarlabel="%"
    )

    matchScore = datMfull.matchScore.values
    bins = np.logspace(-10, 0, 41)
    _, _ = _plot2dhist(
        matchScore,
        datMfull.capture_time,
        ax[2],
        cax[2],
        bins,
        ylabel="match score [-]",
        cbarlabel="%",
    )
    ax[2].axhline(config.quality.minMatchScore)

    zDiff = datM.position_upperLeft.sel(dim2D="y").diff("camera").values.squeeze()
    _, rs = _plotVar(
        zDiff,
        datM.capture_time,
        ax[3],
        "z difference [px]",
        axhline=0,
        resample=resample,
    )
    cax[3].axis("off")

    hDiff = datM.Droi.sel(dim2D="y").diff("camera").values.squeeze()
    _, _ = _plotVar(
        hDiff,
        datM.capture_time,
        ax[4],
        "h difference [px]",
        axhline=0,
        resample=resample,
    )
    cax[4].axis("off")

    tDiff = datM.capture_time.diff("camera").values.squeeze().astype(int) * 1e-9
    _, _ = _plotVar(
        tDiff,
        datM.capture_time,
        ax[5],
        "t difference [s]",
        axhline=0,
        resample=resample,
    )
    cax[5].axis("off")

    defaultRotation, prevTime = tools.getPrevRotationEstimate(
        ff.datetime64, "transformation", config
    )

    theta = datM.camera_theta.sel(camera_rotation="mean").values.squeeze()
    _, _ = _plotVar(
        theta,
        datM.capture_time.isel(camera=0),
        ax[6],
        "theta",
        axhline=defaultRotation["camera_theta"],
        resample=resample,
    )
    phi = datM.camera_phi.sel(camera_rotation="mean").values.squeeze()
    _, _ = _plotVar(
        phi,
        datM.capture_time.isel(camera=0),
        ax[7],
        "phi",
        axhline=defaultRotation["camera_phi"],
        resample=resample,
    )
    Ofz = datM.camera_Ofz.sel(camera_rotation="mean").values.squeeze()
    _, _ = _plotVar(
        Ofz,
        datM.capture_time.isel(camera=0),
        ax[8],
        "Ofz",
        axhline=defaultRotation["camera_Ofz"],
        resample=resample,
    )

    cax[6].axis("off")
    cax[7].axis("off")
    cax[8].axis("off")

    timeIndex1 = pd.date_range(
        start=case,
        end=fl.datetime64 + np.timedelta64(1, "D"),
        freq=resample,
        inclusive="both",
    )

    blowingSnowL, nDetectedL = tools.identifyBlockedBlowingSnowData(
        fl.listFilesWithNeighbors("metaDetection"), config, timeIndex1, "match"
    )
    blowingSnowF, nDetectedF = tools.identifyBlockedBlowingSnowData(
        ff.listFilesWithNeighbors("metaDetection"), config, timeIndex1, "match"
    )
    blowingSnowL = blowingSnowL.time.where(
        blowingSnowL > config.quality.blowingSnowFrameThresh
    ).values
    blowingSnowF = blowingSnowF.time.where(
        blowingSnowF > config.quality.blowingSnowFrameThresh
    ).values

    observationsRatio = tools.compareNDetected(nDetectedL, nDetectedF)
    obervationsDiffer = observationsRatio < config.quality.obsRatioThreshold
    # required to show individual outliers
    obervationsDiffer = obervationsDiffer.resample(time="1min").pad()
    obervationsDiffer = obervationsDiffer.time.where(obervationsDiffer).values

    for bx in ax:
        ylim = bx.get_ylim()
        try:
            bx.fill_between(
                obervationsDiffer,
                [ylim[0]] * len(obervationsDiffer),
                [ylim[1]] * len(obervationsDiffer),
                color="magenta",
                alpha=0.25,
                label="observed # of particles differ",
            )
        except TypeError:
            log.warning("no data to plot obervationsDiffer")

    eventDatL = xr.open_dataset(fl.listFiles("metaEvents")[0])
    for event in eventDatL.event:
        if str(event.values).startswith("start") or str(event.values).startswith(
            "launch"
        ):
            for bx in ax:
                bx.axvline(event.file_starttime.values, color="r")
    lBlocked = (
        eventDatL.blocking.sel(blockingThreshold=50) > config.quality.blockedPixThresh
    )

    lBlocked = lBlocked.file_starttime.where(lBlocked).values
    for bx in ax:
        ylim = bx.get_ylim()
        if not np.all(np.isnan(lBlocked)):
            bx.fill_between(
                lBlocked,
                [ylim[0]] * len(lBlocked),
                [ylim[1]] * len(lBlocked),
                color="red",
                alpha=0.25,
                label="blocked leader",
            )

        if not np.all(np.isnan(blowingSnowL)):
            bx.fill_between(
                blowingSnowL,
                [ylim[0]] * len(blowingSnowL),
                [ylim[1]] * len(blowingSnowL),
                color="orange",
                alpha=0.25,
                label="blow. snow leader",
                hatch="///",
            )

        bx.set_ylim(ylim)

    eventDatF = xr.open_dataset(ff.listFiles("metaEvents")[0])
    for event in eventDatF.event:
        if str(event.values).startswith("start") or str(event.values).startswith(
            "launch"
        ):
            for bx in ax:
                bx.axvline(event.file_starttime.values, color="blue")
    fBlocked = (
        eventDatF.blocking.sel(blockingThreshold=50) > config.quality.blockedPixThresh
    )
    fBlocked = fBlocked.file_starttime.where(fBlocked).values
    for bx in ax:
        ylim = bx.get_ylim()
        if not np.all(np.isnan(fBlocked)):
            bx.fill_between(
                fBlocked,
                [ylim[0]] * len(fBlocked),
                [ylim[1]] * len(fBlocked),
                color="blue",
                alpha=0.25,
                label="blocked follower",
            )
        if not np.all(np.isnan(blowingSnowF)):
            bx.fill_between(
                blowingSnowF,
                [ylim[0]] * len(blowingSnowF),
                [ylim[1]] * len(blowingSnowF),
                color="purple",
                alpha=0.25,
                label="blow. snow follower",
                hatch="///",
            )

        bx.set_ylim(ylim)

    ax[1].legend()

    for ii in range(8):
        ax[ii].set_xticklabels([])
    for ii in range(9):
        if ii > 0:
            ax[ii - 1].sharex(ax[ii])
        ax[ii].grid(True)

    ax[8].set_xlim(
        np.datetime64(f"{fl.year}-{fl.month}-{fl.day}T00:00"),
        np.datetime64(f"{fl.year}-{fl.month}-{fl.day}T00:00") + np.timedelta64(1, "D"),
    )

    print("DONE", fOut)
    tools.savefig(fig, config, fOut, fnames=fnames1M, w_pad=0.05, h_pad=0.005)
    if returnFig:
        return fOut, fig
    else:
        return fOut


def metaRotationYearlyQuicklook(year, config, version=__version__, skipExisting=True):
    """
    Create yearly meta rotation quicklook for a given year.

    Parameters
    ----------
    year : str
        Year string in format YYYY
    config : dict or str
        Configuration dictionary or path to config file
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    config = tools.readSettings(config)

    ff = files.FindFiles(f"{year}0101", config.leader, config, version)
    ff1 = files.FindFiles(f"{int(year)-1}0101", config.leader, config, version)
    fOut = ff.quicklook.metaRotation
    fOut = fOut.replace("0101.png", ".png")

    fPattern = ff.fnamesPattern.metaRotation.replace("0101.nc", "*.nc")
    rotFiles = sorted(glob.glob(fPattern))

    if (
        skipExisting
        and tools.checkForExisting(fOut, parents=rotFiles)
        # and (
        #     int(year)
        #     < int((datetime.datetime.utcnow() - datetime.timedelta(days=60)).year)
        # )
    ):
        print(f"{year} skip exisiting {fOut}")
        return None, None

    rotDat = []
    # open_mfdataset does not work due to duplicate file_starttimes...
    for rotFile in rotFiles:
        rotDat.append(xr.open_dataset(rotFile))
    if len(rotDat) == 0:
        log.error(f"{fPattern} not found")
        return None, None
    rotDat = xr.concat(rotDat, "file_starttime")
    rotDat = rotDat.isel(
        file_starttime=np.unique(rotDat.file_starttime, return_index=True)[1]
    ).sortby("file_starttime")

    # handle current year a bit differently
    if int(year) == int(datetime.datetime.utcnow().year):
        rotFiles1 = ff1.fnamesPattern.metaRotation.replace("0101.nc", "*.nc")
        rotFiles1 = sorted(glob.glob(rotFiles1))
        rotDat1 = []
        # open_mfdataset does not work due to duplicate file_starttimes...
        for rotFile in rotFiles:
            rotDat1.append(xr.open_dataset(rotFile))
        if len(rotDat1) == 0:
            log.error(f"{fPattern} not found")
            return None, None
        rotDat1 = xr.concat(rotDat1, "file_starttime")
        rotDat1 = rotDat1.isel(
            file_starttime=np.unique(rotDat1.file_starttime, return_index=True)[1]
        ).sortby("file_starttime")
        rotDat = xr.concat((rotDat1, rotDat), dim="file_starttime")
        lastYear = np.datetime64(
            datetime.datetime.utcnow() - datetime.timedelta(days=365)
        )
        rotDat = rotDat.isel(file_starttime=(rotDat.file_starttime > lastYear))
        rotDat = rotDat.sortby("file_starttime")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, figsize=(20, 15), gridspec_kw={"hspace": 0.0}, sharex=True
    )

    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    plt.suptitle(
        "VISSS rotation \n" + f"{year}" + ", " + config["name"] + "",
        fontsize=25,
        y=0.995,
        fontweight="bold",
        x=mid,
    )

    rotDat.camera_phi.sel(camera_rotation="mean").plot(ax=ax1)
    rotDat.camera_theta.sel(camera_rotation="mean").plot(ax=ax2)
    rotDat.camera_Ofz.sel(camera_rotation="mean").plot(ax=ax3)

    ax1.fill_between(
        rotDat.file_starttime,
        rotDat.camera_phi.sel(camera_rotation="mean")
        - rotDat.camera_phi.sel(camera_rotation="err"),
        rotDat.camera_phi.sel(camera_rotation="mean")
        + rotDat.camera_phi.sel(camera_rotation="err"),
        alpha=0.5,
    )

    ax2.fill_between(
        rotDat.file_starttime,
        rotDat.camera_theta.sel(camera_rotation="mean")
        - rotDat.camera_theta.sel(camera_rotation="err"),
        rotDat.camera_theta.sel(camera_rotation="mean")
        + rotDat.camera_theta.sel(camera_rotation="err"),
        alpha=0.5,
    )
    ax3.fill_between(
        rotDat.file_starttime,
        rotDat.camera_Ofz.sel(camera_rotation="mean")
        - rotDat.camera_Ofz.sel(camera_rotation="err"),
        rotDat.camera_Ofz.sel(camera_rotation="mean")
        + rotDat.camera_Ofz.sel(camera_rotation="err"),
        alpha=0.5,
    )

    resets = [
        np.datetime64(
            datetime.datetime.strptime(d.ljust(15, "0"), "%Y%m%d-%H%M%S")
        ).astype(rotDat.file_starttime.dtype)
        for d in config.rotate.keys()
    ]

    ofz = rotDat.camera_Ofz.sel(camera_rotation="mean")
    cond = ofz.where(np.isnan(ofz))
    for ax in [ax1, ax2, ax3]:
        ylim = ax.get_ylim()
        if cond.notnull().any():
            ax.fill_between(
                cond,
                [ylim[0]] * len(rotDat.file_starttime),
                [ylim[1]] * len(rotDat.file_starttime),
                color="pink",
                alpha=0.25,
                label="rotation failed",
            )

        firstReset = True
        for reset in resets:
            print(reset)
            if firstReset:
                ax.axvline(reset, color="k", label="rotation from config", alpha=0.5)
            else:
                ax.axvline(reset, color="k", alpha=0.5)
            firstReset = False

    ax1.set_xlim(rotDat.file_starttime.min(), rotDat.file_starttime.max())

    ax1.legend(fontsize=15, bbox_to_anchor=(1, 1.4))

    ax1.grid()
    ax1.set_title(None)
    ax1.set_ylabel("phi rotation [°]", fontsize=20)
    ax1.set_xlabel(None)

    ax2.set_title(None)
    ax2.set_ylabel("theta rotation [°]", fontsize=20)
    ax2.grid()
    ax2.set_xlabel(None)

    ax3.set_title(None)
    ax3.set_ylabel("z offset [px]", fontsize=20)
    ax3.tick_params(axis="both", labelsize=15)
    # ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    ax3.grid()
    ax3.set_xlabel("time")

    # _statusText(fig, rotFiles, config)
    tools.savefig(fig, config, fOut, fnames=rotFiles)

    rotDat.close()

    if year == str(datetime.datetime.today().year):
        try:
            shutil.copy(fOut, ff.quicklookCurrent.metaRotation)
        except PermissionError:
            log.error(f"No permission to write {ff.quicklookCurrent.metaRotation}")

    return fOut, fig


@tools.loopify
def metaRotationQuicklook(case, config, version=__version__, skipExisting=True):
    """
    Create meta rotation quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    camera = config.leader

    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.metaRotation

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("metaRotation"),
    ):
        return None, None

    fnames1D = {}
    fnames1D["leader"] = ff.listFiles("level1detect")
    ff2 = files.FindFiles(case, config.follower, config, version)
    fnames1D["follower"] = ff2.listFiles("level1detect")

    nParticles = {}
    for camera in fnames1D.keys():
        nParticles[camera] = {}
        for fname in fnames1D[camera]:
            d1 = xr.open_dataset(fname)
            nP = len(d1.pid)
            d1.close()
            tt = files.FilenamesFromLevel(fname, config).datetime64
            nParticles[camera][tt] = nP
        nParticles[camera] = pd.Series(nParticles[camera])

    print(case, camera, fOut)

    if len(ff.listFiles("level0txt")) == 0 and len(ff.listFiles("level0status")) == 0:
        print(case, "no data")

        return None, None

    print("reading events")
    if len(ff.listFiles("metaEvents")) == 0:
        log.error(f"event data not found")
        return None, None

    try:
        eventFile = ff.listFiles("metaEvents")[0]
    except IndexError:
        print(f"no leader event file")
        return None, None

    try:
        events = xr.open_dataset(eventFile)
    except:
        print(f"{eventFile} broken")
        return None, None

    try:
        eventFile = ff2.listFiles("metaEvents")[0]
    except IndexError:
        print(f"no follower event file")
        return None, None

    try:
        events2 = xr.open_dataset(eventFile)
    except:
        print(f"{eventFile} broken")
        return None, None

    if len(events.data_vars) == 0:
        print(f'{ff.listFiles("metaEvents")[0]} empty')
        return None, None

    if len(events2.data_vars) == 0:
        print(f'{ff.listFiles("metaEvents")[0]} empty')
        return None, None

    if len(ff.listFiles("metaRotation")) == 0:
        print(f"no metaRotation data yet")
        return None, None

    try:
        rotDat = xr.open_dataset(ff.listFiles("metaRotation")[0])
    except:
        print(f'{ff.listFiles("metaRotation")[0]} broken')
        return None, None

    if len(rotDat.file_starttime) <= 1:
        print(f'{ff.listFiles("metaRotation")[0]} empty')
        return None, None

    ts = events.file_starttime.where(
        (events.event == "sleep-trigger") | (events.event == "stop-trigger")
    )
    ts2 = events2.file_starttime.where(
        (events2.event == "sleep-trigger") | (events2.event == "stop-trigger")
    )

    print("plotting")
    # plotting
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, figsize=(20, 15), gridspec_kw={"hspace": 0.0}, sharex=True
    )
    ax3.set_yscale("log")

    # plt.rcParams['text.usetex'] = False
    # plt.rcParams['lines.linewidth'] = 1.5
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    plt.suptitle(
        "VISSS rotation \n"
        + f"{ff.year}-{ff.month}-{ff.day}"
        + ", "
        + config["name"]
        + "",
        fontsize=25,
        y=0.995,
        fontweight="bold",
        x=mid,
    )

    rotDat.camera_phi.sel(camera_rotation="mean").plot(ax=ax1, marker="x", label="phi")
    rotDat.camera_theta.sel(camera_rotation="mean").plot(
        ax=ax1, marker="x", label="theta"
    )
    rotDat.camera_Ofz.sel(camera_rotation="mean").plot(ax=ax2, marker="x")

    for camera in nParticles.keys():
        try:
            xr.DataArray(nParticles[camera]).plot(ax=ax3, marker="x", label=camera)
        except TypeError:
            pass

    ofz = rotDat.camera_Ofz.sel(camera_rotation="mean")
    cond = ofz.file_starttime.where(np.isnan(ofz))
    for ax in [ax1, ax2, ax3]:
        ylim = ax.get_ylim()
        if cond.notnull().any():
            ax.fill_between(
                cond,
                [ylim[0]] * len(rotDat.file_starttime),
                [ylim[1]] * len(rotDat.file_starttime),
                color="red",
                label="rotation failed",
            )

    ax1.fill_between(
        rotDat.file_starttime,
        rotDat.camera_phi.sel(camera_rotation="mean")
        - rotDat.camera_phi.sel(camera_rotation="err"),
        rotDat.camera_phi.sel(camera_rotation="mean")
        + rotDat.camera_phi.sel(camera_rotation="err"),
        alpha=0.5,
    )

    ax2.fill_between(
        rotDat.file_starttime,
        rotDat.camera_theta.sel(camera_rotation="mean")
        - rotDat.camera_theta.sel(camera_rotation="err"),
        rotDat.camera_theta.sel(camera_rotation="mean")
        + rotDat.camera_theta.sel(camera_rotation="err"),
        alpha=0.5,
    )
    ax2.fill_between(
        rotDat.file_starttime,
        rotDat.camera_Ofz.sel(camera_rotation="mean")
        - rotDat.camera_Ofz.sel(camera_rotation="err"),
        rotDat.camera_Ofz.sel(camera_rotation="mean")
        + rotDat.camera_Ofz.sel(camera_rotation="err"),
        alpha=0.5,
    )
    isBlocked = events.blocking.dropna("file_starttime").sel(blockingThreshold=50) > 0.1
    isBlocked = isBlocked.file_starttime.where(isBlocked).values
    isBlocked2 = (
        events2.blocking.dropna("file_starttime").sel(blockingThreshold=50) > 0.1
    )
    isBlocked2 = isBlocked2.file_starttime.where(isBlocked2).values

    resets = [
        np.datetime64(
            datetime.datetime.strptime(d.ljust(15, "0"), "%Y%m%d-%H%M%S")
        ).astype(rotDat.file_starttime.dtype)
        for d in config.rotate.keys()
    ]

    for ax in [ax1, ax2, ax3]:
        ylim = ax.get_ylim()
        if isBlocked.any():
            ax.fill_between(
                isBlocked,
                [ylim[0]] * len(isBlocked),
                [ylim[1]] * len(isBlocked),
                color="red",
                alpha=0.25,
                label="leader blocked",
            )
        if isBlocked2.any():
            ax.fill_between(
                isBlocked,
                [ylim[0]] * len(isBlocked),
                [ylim[1]] * len(isBlocked),
                color="blue",
                alpha=0.25,
                label="follower blocked",
            )
        if ts.notnull().sum() > 0:
            ax.fill_between(
                ts,
                [ylim[0]] * len(ts),
                [ylim[1]] * len(ts),
                color="orange",
                alpha=0.5,
                label="idle",
            )
        if ts2.notnull().sum() > 0:
            ax.fill_between(
                ts2,
                [ylim[0]] * len(ts2),
                [ylim[1]] * len(ts2),
                color="orange",
                alpha=0.5,
                label="idle",
            )
        firstReset = True
        for reset in resets:
            if firstReset:
                ax.axvline(reset, color="k", label="rotation from config")
            else:
                ax.axvline(reset, color="k")
            firstReset = False
        ax.set_ylim(ylim)

        ax.grid()

    ax1.set_title(None)
    ax1.set_ylabel("angular rotation [°]", fontsize=20)
    ax1.set_xlabel(None)

    ax2.set_title(None)
    ax2.set_ylabel("z offset [px]", fontsize=20)
    ax2.set_xlabel(None)

    ax3.set_xlim(
        np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
        np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
        + np.timedelta64(1, "D"),
    )
    ax3.set_title(None)
    ax3.set_ylabel("# particles in lv1detect", fontsize=20)
    ax3.tick_params(axis="both", labelsize=15)
    ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    ax3.set_xlabel("time")

    firstEvent = True
    for event in events.event:
        if str(event.values).startswith("start") or str(event.values).startswith(
            "launch"
        ):
            for bx in [ax1, ax2, ax3]:
                if firstEvent:
                    label = "leader restarted"
                    firstEvent = False
                else:
                    label = None
                bx.axvline(
                    event.file_starttime.values, color="red", ls=":", label=label
                )
    firstEvent = True
    for event in events2.event:
        if str(event.values).startswith("start") or str(event.values).startswith(
            "launch"
        ):
            for bx in [ax1, ax2, ax3]:
                if firstEvent:
                    label = "follower restarted"
                    firstEvent = False
                else:
                    label = None
                bx.axvline(
                    event.file_starttime.values, color="red", ls="--", label=label
                )

    ax1.legend(fontsize=15)
    ax3.legend(fontsize=15)

    # _statusText(fig, ff.listFiles("metaRotation"), config)
    tools.savefig(fig, config, fOut, fnames=ff.listFiles("metaRotation"))
    rotDat.close()
    events.close()

    return fOut, fig


@tools.loopify_with_camera(endYesterday=False)
def createLevel2detectQuicklook(
    case, camera, config, version=__version__, skipExisting=True, returnFig=True
):
    """
    Create level2detect quicklook for a given case and camera.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    version = __version__

    nodata = False
    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level2detect

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("level2detect"),
    ):
        return None, None

    lv2 = ff.listFiles("level2detect")
    if len(lv2) == 0:
        if len(ff.listFilesExt("level2detect")) == 0:
            log.error(f"{case} level2detect data not found")
            return None, None
        else:
            nodata = True
    else:
        lv2 = lv2[0]

    if len(ff.listFiles("metaEvents")) == 0:
        log.error(f"{case} event data not found")
        return None, None

    log.info(f"running {case} {fOut}")

    fig, axs = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(20, 20),
        gridspec_kw={
            "width_ratios": [1, 0.01],
            "height_ratios": [2.5, 1, 1, 1],
            "hspace": 0.02,
            "wspace": 0.1,
        },
    )
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    fig.suptitle(
        f"VISSS level2detect {camera.split('_')[0]}\n"
        + f"{ff.year}-{ff.month}-{ff.day}"
        + ", "
        + config["name"]
        + "",
        fontsize=25,
        y=0.995,
        fontweight="bold",
        x=mid,
    )
    if nodata:
        axs[0, 0].set_title("no data")
    else:
        dat2 = xr.open_dataset(lv2)
        # dat2 = dat2.where(dat2.qualityFlags == 0)
        quality = tools.unpackQualityFlags(dat2.qualityFlags, doubleTimestamps=True)
        fEvents1 = ff.listFiles("metaEvents")[0]

        try:
            events = xr.open_dataset(fEvents1)
        except:
            print(f"{fEvents1} broken")
            return None, None

        (ax1, ax2, ax3, ax4) = axs[:, 0]
        (bx1, bx2, bx3, bx4) = axs[:, 1]

        ax1a = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        (dat2.PSD.sel(size_definition="Dmax")).T.plot(
            ax=ax1,
            norm=mpl.colors.LogNorm(vmin=1, vmax=dat2.PSD.max()),
            cbar_kwargs={"label": "Particle Size Distribution [1/m^4]"},
            cbar_ax=bx1,
        )
        ax1.set_ylabel("Dmax [m]")

        lns3 = dat2.Ntot.sel(size_definition="Dmax").plot(
            ax=ax1a, label="N_tot", color="k", alpha=0.7
        )
        ax1a.set_ylabel("N_tot [1/m3]")
        ax1a.set_yscale("log")
        ax1a.set_ylim(1e0, 1e11)
        ax1a.set_title("")

        # (dat2.aspectRatio_dist.sel(size_definition="Dmax", fitMethod="cv2.fitEllipseDirect")).T.plot(ax=ax2, vmin=0,vmax=1, cbar_kwargs={"label":"aspect ratio [-]"})
        dat2.Dmax_mean.plot(ax=ax2, label="mean Dmax")
        dat2.D32.sel(size_definition="Dmax").plot(ax=ax2, label="D32")
        dat2.D43.sel(size_definition="Dmax").plot(ax=ax2, label="D43")
        ax2.set_ylabel("Size [m]")
        ax2.legend()

        lns1 = (
            dat2.normalizedRimeMass_dist.sel(size_definition="Dmax")
            .mean("D_bins")
            .plot(ax=ax3, label="normalized rime mass [-]")
        )
        lns2 = dat2.aspectRatio_mean.sel(fitMethod="cv2.fitEllipseDirect").plot(
            ax=ax3, label="aspect ratio [-]"
        )
        ax3.set_ylabel("Shape [-]")
        ax3.legend()

        dat2.counts.sel(size_definition="Dmax").sum("D_bins").plot(
            label="observed particles [1/min]", ax=ax4
        )
        ax4.set_yscale("log")
        ax4.set_ylabel("Performance")
        ax4.legend()

        obervationsDiffer = quality.sel(flag="obervationsDiffer", drop=True)
        recordingFailed = quality.sel(flag="recordingFailed", drop=True)
        processingFailed = quality.sel(flag="processingFailed", drop=True)
        blowingSnow = quality.sel(flag="blowingSnow", drop=True)
        cameraBlocked = quality.sel(flag="cameraBlocked", drop=True)

        for ax in axs[:, 0]:
            ax.set_title(None)

            ylim = ax.get_ylim()
            cond = quality.time.where(recordingFailed)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="pink",
                    alpha=0.25,
                    label="cameras off",
                )
            cond = quality.time.where(obervationsDiffer)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="magenta",
                    alpha=0.25,
                    label="observed # of particles differ",
                )

            cond = quality.time.where(processingFailed)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="purple",
                    alpha=0.25,
                    label="processing failed",
                )
            cond = quality.time.where(cameraBlocked)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="red",
                    alpha=0.25,
                    label=f"camera blocked > {config.quality.blockedPixThresh*100}%",
                )
            cond = quality.time.where(blowingSnow)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="orange",
                    alpha=0.25,
                    label=f"blowing snow > {config.quality.blowingSnowFrameThresh*100}%",
                )  # , hatch='///')
            ax.set_ylim(ylim)

            ax.tick_params(axis="both", labelsize=15)
            ax.grid()
            ax.set_xlim(
                np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
                np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
                + np.timedelta64(1, "D"),
            )

        for ax in axs[:-1, 0]:
            ax.set_xticklabels([])

        # mark relaunch
        firstEvent = True
        for event in events.event:
            if str(event.values).startswith("start") or str(event.values).startswith(
                "launch"
            ):
                for ax in axs[:, 0]:
                    if firstEvent:
                        label = "restarted"
                        firstEvent = False
                    else:
                        label = None
                    ax.axvline(
                        event.file_starttime.values, color="red", ls=":", label=label
                    )

        ax1.legend()
        ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))

        for bx in axs[1:, 1]:
            bx.axis("off")

    tools.savefig(fig, config, fOut, fnames=ff.listFiles("level2detect"))
    if returnFig:
        return fOut, fig
    else:
        return fOut


@tools.loopify
def createLevel2matchQuicklook(
    case, config, version=__version__, skipExisting=True, returnFig=True
):
    """
    Create level2match quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    camera = config.leader
    nodata = False
    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level2match

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("level2match"),
    ):
        return None, None

    lv2match = ff.listFiles("level2match")
    if len(lv2match) == 0:
        if len(ff.listFilesExt("level2match")) == 0:
            log.error(
                f"{case} level2match data not found {ff.fnamesPattern.level2match}"
            )
            return None, None
        else:
            nodata = True
    else:
        lv2match = lv2match[0]

    if len(ff.listFiles("metaEvents")) == 0:
        log.error(f"{case} event data not found")
        return None, None

    log.info(f"running {case} {fOut}")

    fig, axs = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(20, 20),
        gridspec_kw={
            "width_ratios": [1, 0.01],
            "height_ratios": [2.5, 1, 1, 1],
            "hspace": 0.02,
            "wspace": 0.1,
        },
    )
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    fig.suptitle(
        "VISSS level2match \n"
        + f"{ff.year}-{ff.month}-{ff.day}"
        + ", "
        + config["name"]
        + "",
        fontsize=25,
        y=0.995,
        fontweight="bold",
        x=mid,
    )
    if nodata:
        axs[0, 0].set_title("no data")
    else:
        dat2 = xr.open_dataset(lv2match)
        # dat2 = dat2.where(dat2.qualityFlags == 0)
        quality = tools.unpackQualityFlags(dat2.qualityFlags, doubleTimestamps=True)
        fEvents1 = ff.listFiles("metaEvents")[0]
        fEvents2 = files.FilenamesFromLevel(fEvents1, config).filenamesOtherCamera(
            level="metaEvents"
        )[0]

        try:
            events1 = xr.open_dataset(fEvents1)
        except:
            print(f"{fEvents1} broken")
            return None, None
        try:
            events2 = xr.open_dataset(fEvents2)
        except:
            print(f"{fEvents2} broken")
            return None, None

        events = xr.concat((events1, events2), dim="file_starttime")

        (ax1, ax2, ax3, ax4) = axs[:, 0]
        (bx1, bx2, bx3, bx4) = axs[:, 1]
        ax1a = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        (dat2.PSD.sel(camera="max", size_definition="Dmax")).T.plot(
            ax=ax1,
            norm=mpl.colors.LogNorm(vmin=1, vmax=dat2.PSD.max()),
            cbar_kwargs={"label": "Particle Size Distribution [1/m^4]"},
            cbar_ax=bx1,
        )
        ax1.set_ylabel("Dmax [m]")

        lns3 = dat2.Ntot.sel(camera="max", size_definition="Dmax").plot(
            ax=ax1a, label="N_tot", color="k", alpha=0.7
        )
        ax1a.set_ylabel("N_tot [1/m3]")
        ax1a.set_yscale("log")
        ax1a.set_ylim(1e0, 1e11)
        ax1a.set_title("")

        # (dat2.aspectRatio_dist.sel(camera="max", size_definition="Dmax", fitMethod="cv2.fitEllipseDirect")).T.plot(ax=ax2, vmin=0,vmax=1, cbar_kwargs={"label":"aspect ratio [-]"})
        dat2.Dmax_mean.sel(camera="max").plot(ax=ax2, label="mean Dmax")
        dat2.D32.sel(camera="max", size_definition="Dmax").plot(ax=ax2, label="D32")
        dat2.D43.sel(camera="max", size_definition="Dmax").plot(ax=ax2, label="D43")
        ax2.set_ylabel("Size [m]")
        ax2.legend()

        dat2.normalizedRimeMass_dist.sel(camera="max", size_definition="Dmax").mean(
            "D_bins"
        ).plot(ax=ax3, label="normalized rime mass [-]")
        dat2.aspectRatio_mean.sel(camera="max", fitMethod="cv2.fitEllipseDirect").plot(
            ax=ax3, label="aspect ratio [-]"
        )
        ax3.set_ylabel("Shape [-]")
        ax3.legend()

        dat2.counts.sel(camera="max", size_definition="Dmax").sum("D_bins").plot(
            label="observed particles [1/min]", ax=ax4
        )
        dat2.matchScore_mean.sel(camera="max").plot(label="match score [-]", ax=ax4)
        ax4.set_yscale("log")
        ax4.set_ylabel("Performance")
        ax4.legend()

        obervationsDiffer = quality.sel(flag="obervationsDiffer", drop=True)
        recordingFailed = quality.sel(flag="recordingFailed", drop=True)
        processingFailed = quality.sel(flag="processingFailed", drop=True)
        blowingSnow = quality.sel(flag="blowingSnow", drop=True)
        cameraBlocked = quality.sel(flag="cameraBlocked", drop=True)

        for ax in axs[:, 0]:
            ax.set_title(None)

            ylim = ax.get_ylim()
            cond = quality.time.where(obervationsDiffer)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="magenta",
                    alpha=0.25,
                    label="observed # of particles differ",
                )
            cond = quality.time.where(recordingFailed)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="pink",
                    alpha=0.25,
                    label="cameras off",
                )
            cond = quality.time.where(processingFailed)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="purple",
                    alpha=0.25,
                    label="processing failed",
                )
            cond = quality.time.where(cameraBlocked)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="red",
                    alpha=0.25,
                    label=f"camera blocked > {config.quality.blockedPixThresh*100}%",
                )
            cond = quality.time.where(blowingSnow)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="orange",
                    alpha=0.25,
                    label=f"blowing snow > {config.quality.blowingSnowFrameThresh*100}%",
                )  # , hatch='///')
            ax.set_ylim(ylim)

            ax.tick_params(axis="both", labelsize=15)
            ax.grid()
            ax.set_xlim(
                np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
                np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
                + np.timedelta64(1, "D"),
            )

        for ax in axs[:-1, 0]:
            ax.set_xticklabels([])

        # mark relaunch
        firstEvent = True
        for event in events.event:
            if str(event.values).startswith("start") or str(event.values).startswith(
                "launch"
            ):
                for ax in axs[:, 0]:
                    if firstEvent:
                        label = "restarted"
                        firstEvent = False
                    else:
                        label = None
                    ax.axvline(
                        event.file_starttime.values, color="red", ls=":", label=label
                    )

        ax1.legend()
        ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))

        for bx in axs[1:, 1]:
            bx.axis("off")

    tools.savefig(fig, config, fOut, fnames=ff.listFiles("level2match"))
    if returnFig:
        return fOut, fig
    else:
        return fOut


@tools.loopify
def createLevel2trackQuicklook(
    case, config, version=__version__, skipExisting=True, returnFig=True
):
    """
    Create level2track quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    camera = config.leader
    nodata = False
    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level2track
    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("level2track"),
    ):
        return None, None

    lv2track = ff.listFiles("level2track")
    if len(lv2track) == 0:
        if len(ff.listFilesExt("level2track")) == 0:
            log.error(f"{case} lv2track data not found")
            return None, None
        else:
            nodata = True
    else:
        lv2track = lv2track[0]

    if len(ff.listFiles("metaEvents")) == 0:
        log.error(f"{case} event data not found")
        return None, None

    log.info(f"running {case} {fOut}")

    fig, axs = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(20, 20),
        gridspec_kw={
            "width_ratios": [1, 0.01],
            "height_ratios": [1.25, 1.25, 1, 1, 1],
            "hspace": 0.02,
            "wspace": 0.1,
        },
    )
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    fig.suptitle(
        "VISSS level2track \n"
        + f"{ff.year}-{ff.month}-{ff.day}"
        + ", "
        + config["name"]
        + "",
        fontsize=25,
        y=0.995,
        fontweight="bold",
        x=mid,
    )
    if nodata:
        axs[0, 0].set_title("no data")
    else:
        dat2 = xr.open_dataset(lv2track)
        # dat2 = dat2.where(dat2.qualityFlags == 0)
        quality = tools.unpackQualityFlags(dat2.qualityFlags, doubleTimestamps=True)

        print(lv2track)

        fEvents1 = ff.listFiles("metaEvents")[0]
        fEvents2 = files.FilenamesFromLevel(fEvents1, config).filenamesOtherCamera(
            level="metaEvents"
        )[0]

        try:
            events1 = xr.open_dataset(fEvents1)
        except:
            print(f"{fEvents1} broken")
            return None, None
        try:
            events2 = xr.open_dataset(fEvents2)
        except:
            print(f"{fEvents2} broken")
            return None, None

        events = xr.concat((events1, events2), dim="file_starttime")

        (ax1, ax1a, ax2, ax3, ax4) = axs[:, 0]
        (bx1, bx1a, bx2, bx3, bx4) = axs[:, 1]
        ax1b = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        lns3 = dat2.Ntot.sel(cameratrack="max", size_definition="Dmax").plot(
            ax=ax1b, label="N_tot", color="k", alpha=0.7
        )
        ax1b.set_ylabel("N_tot [1/m3]")
        ax1b.set_yscale("log")
        ax1b.set_ylim(1e0, 1e11)
        ax1b.set_title("")

        plotDat = (dat2.PSD.sel(cameratrack="max", size_definition="Dmax")).T
        if np.any(plotDat.notnull()):
            plotDat.plot(
                ax=ax1,
                norm=mpl.colors.LogNorm(vmin=1, vmax=dat2.PSD.max()),
                cbar_kwargs={"label": "Particle Size Distribution [1/m^4]"},
                cbar_ax=bx1,
            )

        ax1.set_ylabel("Dmax [m]")

        plotDat = (
            dat2.velocity_dist.sel(
                cameratrack="mean", size_definition="Dmax", dim3D="z"
            )
        ).T
        if np.any(plotDat.notnull()):
            plotDat.plot(
                ax=ax1a,
                cbar_kwargs={"label": "Particle Sedimentation velocity [m/s]"},
                cbar_ax=bx1a,
                vmin=0,
                vmax=3,
            )

        ax1a.set_ylabel("Dmax [m]")

        # (dat2.aspectRatio_dist.sel(cameratrack="max", size_definition="Dmax", fitMethod="cv2.fitEllipseDirect")).T.plot(ax=ax2, vmin=0,vmax=1, cbar_kwargs={"label":"aspect ratio [-]"})
        dat2.Dmax_mean.sel(cameratrack="max").plot(ax=ax2, label="mean Dmax")
        dat2.D32.sel(cameratrack="max", size_definition="Dmax").plot(
            ax=ax2, label="D32"
        )
        dat2.D43.sel(cameratrack="max", size_definition="Dmax").plot(
            ax=ax2, label="D34"
        )
        ax2.set_ylabel("Size [m]")
        ax2.legend()

        dat2.normalizedRimeMass_dist.sel(
            cameratrack="max", size_definition="Dmax"
        ).mean("D_bins").plot(ax=ax3, label="normalized rime mass [-]")
        dat2.aspectRatio_mean.sel(
            cameratrack="min", fitMethod="cv2.fitEllipseDirect"
        ).plot(ax=ax3, label="aspect ratio [-]")
        ax3.set_ylabel("Shape [-]")
        ax3.legend()

        dat2.counts.sel(cameratrack="max", size_definition="Dmax").sum("D_bins").plot(
            label="observed particles [1/min]", ax=ax4
        )
        dat2.matchScore_mean.sel(cameratrack="max").plot(
            label="match score [-]", ax=ax4
        )
        ax4.set_yscale("log")
        ax4.set_ylabel("Performance")
        ax4.legend()

        obervationsDiffer = quality.sel(flag="obervationsDiffer", drop=True)
        recordingFailed = quality.sel(flag="recordingFailed", drop=True)
        processingFailed = quality.sel(flag="processingFailed", drop=True)
        blowingSnow = quality.sel(flag="blowingSnow", drop=True)
        cameraBlocked = quality.sel(flag="cameraBlocked", drop=True)
        tracksTooShort = quality.sel(flag="tracksTooShort", drop=True)

        for ax in axs[:, 0]:
            ax.set_title(None)

            ylim = ax.get_ylim()
            cond = quality.time.where(recordingFailed)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="pink",
                    alpha=0.25,
                    label="cameras off",
                )
            cond = quality.time.where(obervationsDiffer)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="magenta",
                    alpha=0.25,
                    label="observed # of particles differ",
                )

            cond = quality.time.where(processingFailed)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="purple",
                    alpha=0.25,
                    label="processing failed",
                )
            cond = quality.time.where(cameraBlocked)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="red",
                    alpha=0.25,
                    label=f"camera blocked > {config.quality.blockedPixThresh*100}%",
                )
            cond = quality.time.where(blowingSnow)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="orange",
                    alpha=0.25,
                    label=f"blowing snow > {config.quality.blowingSnowFrameThresh*100}%",
                )  # , hatch='///')
            cond = quality.time.where(tracksTooShort)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="blue",
                    alpha=0.2,
                    label=f"mean track length < {config.quality.trackLengthThreshold}",
                    # hatch="X",
                )

            ax.set_ylim(ylim)

            ax.tick_params(axis="both", labelsize=15)
            ax.grid()
            ax.set_xlim(
                np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
                np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
                + np.timedelta64(1, "D"),
            )

        for ax in axs[:-1, 0]:
            ax.set_xticklabels([])

        # mark relaunch
        firstEvent = True
        for event in events.event:
            if str(event.values).startswith("start") or str(event.values).startswith(
                "launch"
            ):
                for ax in axs[:, 0]:
                    if firstEvent:
                        label = "restarted"
                        firstEvent = False
                    else:
                        label = None
                    ax.axvline(
                        event.file_starttime.values, color="red", ls=":", label=label
                    )

        ax1.legend()
        ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))

        for bx in axs[2:, 1]:
            bx.axis("off")

    tools.savefig(fig, config, fOut, fnames=ff.listFiles("level2track"))
    if returnFig:
        return fOut, fig
    else:
        return fOut


@tools.loopify
def createLevel1matchParticlesQuicklook(
    timestamp,
    config,
    version=__version__,
    container_width=200,
    container_height_max=300,
    nTiles=60,
    nRows=4,
    extra=1,
    readParticlesFromFiles=True,
    skipExisting=True,
    ffOut="default",
    timeStep="variable",  # attempt to fill plot equally
    minBlur="config",
    minSize="config",
    omitLabel4small="config",
    timedelta=np.timedelta64(1, "D"),
    returnFig=True,
    doLevel1matchQuicklook=True,
):
    """
    Create level1match particles quicklook for a given timestamp.

    Parameters
    ----------
    timestamp : str
        Timestamp string in format YYYYMMDDHH or YYYYMMDD
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    container_width : int, optional
        Width of each tile container, by default 200
    container_height_max : int, optional
        Maximum height of each tile container, by default 300
    nTiles : int, optional
        Number of tiles per row, by default 60
    nRows : int, optional
        Number of rows in output, by default 4
    extra : int, optional
        Extra spacing between tiles, by default 1
    readParticlesFromFiles : bool, optional
        Whether to read particles from files, by default True
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    ffOut : str, optional
        Output file path, by default "default"
    timeStep : str, optional
        Time step method ('fixed' or 'variable'), by default "variable"
    minBlur : float or str, optional
        Minimum blur threshold, by default "config"
    minSize : int or str, optional
        Minimum particle size, by default "config"
    omitLabel4small : bool or str, optional
        Whether to omit labels for small particles, by default "config"
    timedelta : numpy.timedelta64, optional
        Time window for data selection, by default np.timedelta64(1, "D")
    returnFig : bool, optional
        Whether to return the figure, by default True
    doLevel1matchQuicklook : bool, optional
        Whether to also run level1match quicklook, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    import cv2
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
    from tqdm import tqdm

    camera = config["leader"]

    # for convinience, do the other L1match quicklook as well
    if doLevel1matchQuicklook:
        createLevel1matchQuicklook(timestamp, config, skipExisting=skipExisting)

    if minBlur == "config":
        minBlur = config["level1detectQuicklook"]["minBlur"]
    if minSize == "config":
        minSize = config["level1detectQuicklook"]["minSize"]
    if omitLabel4small == "config":
        omitLabel4small = config["level1detectQuicklook"]["omitLabel4small"]

    ff = files.FindFiles(timestamp, camera, config, version)
    if ffOut == "default":
        ffOut = ff.quicklook.level1matchParticles

    site = config["site"]

    particlesPloted = 0

    if skipExisting and tools.checkForExisting(
        ffOut,
        level0=ff.listFiles("level0"),
        events=ff.listFiles("metaEvents"),
        parents=ff.listFilesExt("level1match"),
    ):
        return None, None

    if site != "mosaic":
        if (len(ff.listFiles("level0")) == 0) and (
            len(ff.listFiles("level0status")) == 0
        ):
            print("NO DATA YET (TRANSFERRED?)", ffOut)
            return None, None

    if len(ff.listFiles("metaFrames")) > len(ff.listFiles("level0")):
        print("DATA TRANSFER INCOMPLETE ", ffOut)
        print(
            len(ff.listFiles("level0")),
            "of",
            len(ff.listFiles("metaFrames")),
            "transmitted",
        )
        return None, None

    if (len(ff.listFilesExt("level1match")) == 0) and (len(ff.listFiles("level0")) > 0):
        print("NO DATA YET ", ffOut)
        return None, None

    if not ff.isCompleteL1match:
        print(
            "NOT COMPLETE YET %i of %i L1match %s "
            % (
                len(ff.listFilesExt("level1match")),
                len(ff.listFiles("level0txt")),
                ffOut,
            )
        )

        #         if (len(ff.listFilesExt("level1match")) == len(ff.listFiles("level0"))):
        #             afshgsa
        return None, None

    #     else:

    total_width = (container_width + extra) * nTiles // nRows
    max_height = (20 + extra + container_height_max) * nRows + 60

    # let use a matplotlib font becuase we can be sure it is there
    mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
    mpl_ttf_dir = os.path.join(mpl_data_dir, "fonts", "ttf")
    font = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 35)
    fontL = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 16)

    print("RUNNING open files ", ffOut, len(ff.listFiles("metaFrames")))

    dats2 = []
    l1Files = ff.listFilesWithNeighbors("level1match")

    nParticles = 0
    for fname2 in tqdm(l1Files):
        fname1 = fname2.replace("level1match", "metaFrames")
        try:
            dat2 = xr.open_dataset(fname2)

        except FileNotFoundError:
            if os.path.isfile(f"{fname2}.nodata"):
                continue
            elif os.path.isfile(f"{fname2}.broken.txt"):
                continue
            elif os.path.isfile(f"{fname2}.notenoughframes"):
                continue
            else:
                raise FileNotFoundError(fname2)

        nParticles += len(dat2.pair_id)

        # to speed things up, lets look only at the first 30000 particles. this avoids crazily large files
        dat2 = dat2.isel(pair_id=slice(30000))

        dat2 = dat2[
            [
                "Dmax",
                "blur",
                "record_time",
                "record_id",
                "Droi",
                "file_starttime",
                "matchScore",
            ]
        ]

        # it is more efficient to load the data now in comparison to after isel
        dat2 = dat2.load()

        # print(fname2, len(dat2.pair_id))
        dat2 = dat2.isel(
            pair_id=(
                (dat2.blur > minBlur).all("camera")
                & (dat2.Dmax > minSize).all("camera")
                & (dat2.matchScore > config.quality.minMatchScore)
            )
        )
        # print(fname2, len(dat2.pair_id))

        if len(dat2.pair_id) == 0:
            continue
        dat2 = dat2[["record_time", "record_id", "Droi", "file_starttime"]]

        dat2 = dat2.expand_dims(dict(file=[fname1]))
        #     dat2 = dat2.set_coords(dict(file = fname2))
        dat2 = dat2.stack(fpair_id=("file", "pair_id"))
        dats2.append(dat2)

    print("opened")
    new_im = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    if len(dats2) == 0:
        draw = ImageDraw.Draw(new_im)
        draw.text(
            (total_width // 3, max_height // 3), "no raw data", (0, 0, 0), font=font
        )

    else:
        limDat = xr.concat(dats2, dim="fpair_id")
        limDat = limDat.isel(
            fpair_id=(limDat.record_time.isel(camera=0).values >= ff.datetime64)
            & (limDat.record_time.isel(camera=0).values < (ff.datetime64 + timedelta))
        )
        #         limDat = dats2
        print("merged")

        if len(limDat.fpair_id) == 0:
            print("TOO FEW DATA ", ffOut)

            draw = ImageDraw.Draw(new_im)
            draw.text(
                (total_width // 3, max_height // 3),
                "no data recorded",
                (0, 0, 0),
                font=font,
            )

        else:
            print("Total number of particles for plotting %i" % len(limDat.fpair_id))

            if timeStep == "variable":
                timeSteps = np.percentile(
                    limDat.record_time, np.linspace(0, 100, nTiles + 1)
                )
            elif timeStep == "fixed":
                timeSteps = np.array(
                    pd.date_range(
                        start=ff.datetime64,
                        periods=nTiles + 1,
                        end=ff.datetime64 + timedelta,
                    )
                )
            else:
                raise ValueError("do not understand timeStep")
            mosaics = []

            videos = {}
            for tt, (t1, t2) in enumerate(zip(timeSteps[:-1], timeSteps[1:])):
                if tt == len(timeSteps) - 2:
                    whereCond = limDat.record_time >= t1
                else:
                    whereCond = (limDat.record_time >= t1) & (limDat.record_time < t2)

                thisDat = limDat.isel(fpair_id=whereCond.any("camera").values)
                totalArea = 0

                # select pids randomly, figure out how much we need, and sort them again
                starttimes = deepcopy(thisDat.file_starttime.values).T
                fpair_ids = deepcopy(thisDat.fpair_id.values)

                fnames = []
                time2Fname = {}
                for cc, camera in enumerate(config.instruments):
                    time2Fname[camera] = {}
                    starttimes1 = np.unique(starttimes[:, cc])
                    for starttime1 in starttimes1:
                        case = pd.to_datetime(str(starttime1)).strftime("%Y%m%d-%H%M%S")
                        ff1 = files.FindFiles(case, camera, config, version=version)
                        zipF = ff1.listFiles("imagesL1detect")
                        if len(zipF) > 0:
                            fnames.append(zipF[0])
                            time2Fname[camera][starttime1] = zipF[0]
                        else:
                            log.error(f"no zip file for {case} {camera}")
                particleImages = {}
                for fname in np.unique(fnames):
                    fn = files.FilenamesFromLevel(fname, config)
                    # tarRoot = fn.fname.imagesL1detect.split("/")[-1].replace(".tar.bz2","")
                    # particleImages[fname] = (tools.imageTarFile.open(fn.fname.imagesL1detect, "r:bz2"), tarRoot)
                    # print(fn.fname.imagesL1detect)
                    particleImages[fname] = tools.imageZipFile(
                        fn.fname.imagesL1detect, mode="r"
                    )

                nPids = fpair_ids.shape[0]
                # import pdb

                # pdb.set_trace()
                rng = np.random.default_rng(tt)
                rng.shuffle(fpair_ids)
                # apply shuffling
                thisDat = thisDat.sel(fpair_id=fpair_ids)

                containerSize = container_width * container_height_max

                if nPids < 5:  # fo very few particles I want to see them all!
                    nParticlesNeeded = len(fpair_ids)
                else:
                    try:
                        nParticlesNeeded = (
                            np.where(
                                (thisDat.Droi + 1)
                                .prod("dim2D")
                                .sum("camera")
                                .cumsum("fpair_id")
                                / containerSize
                                > 0.95
                            )[0][0]
                            + 1
                        )
                    except IndexError:
                        nParticlesNeeded = nPids

                fpair_ids = fpair_ids[:nParticlesNeeded]
                print(tt, "/", nTiles, t1, t2, nParticlesNeeded, "of", nPids)
                particlesPloted += nParticlesNeeded
                ims = []

                for fp, fpair_id in enumerate(fpair_ids):
                    im = [None, None]
                    fname2, pair_id = fpair_id
                    particle_pair = thisDat.sel(fpair_id=fpair_id)
                    for cc, camera in enumerate(config.instruments):
                        particle = particle_pair.sel(camera=camera)
                        pid = particle.pid.values
                        background = 0

                        pidStr = "%07i" % pid
                        # imName = '%s.png' % (pidStr)
                        # imfname = '%s/%s/%s' % (particleImages[fname][1],pidStr[:4], imName)

                        starttime1 = particle.file_starttime
                        fname1 = time2Fname[camera][starttime1.values]
                        try:
                            # im = particleImages[fname][0].extractimage(imfname)
                            im[cc] = particleImages[fname1].extractnpy(pidStr)
                        except KeyError:
                            print("NOT FOUND ", pidStr)
                            im[cc] = np.array([[0]])
                            continue
                        # apply alpha channel
                        # im[...,0][im[...,1] == 0] = background
                        # drop alpha channel
                        im[cc] = im[cc][..., 0]

                    im = tools.concatImgX(*im, background=50)

                    # im = av.doubleDynamicRange(im, offset=2)

                    im = np.pad(im, [(0, 1), (0, 1)], constant_values=0)
                    try:
                        fid = np.where(fname2 == np.array(ff.listFiles("metaFrames")))[
                            0
                        ][0]
                    except:
                        fid = -1
                    text = np.full((100, 100), background, dtype=np.uint8)

                    textStr = "%i.%i" % (fid, pid)

                    text = cv2.putText(
                        text,
                        textStr,
                        (0, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.75,
                        255,
                        1,
                    )
                    text = _crop(text)

                    y1, x1 = im.shape
                    y2, x2 = text.shape

                    # only add label if large enough
                    if (omitLabel4small == "all") or (
                        (omitLabel4small == True) and (x1 < x2)
                    ):
                        imT = im
                    else:
                        imT = tools.concatImgY(im, text)
                        # import pdb;pdb.set_trace()
                    ims.append(imT)
                    totalArea += np.prod(imT.shape)

                for fname in fnames:
                    particleImages[fname].close()

                # make tile
                images = [Image.fromarray(im) for im in ims]
                if len(images) == 0:
                    mosaic = (
                        np.ones(
                            (container_height_max, container_width, 3), dtype=np.uint8
                        )
                        * 0
                    )
                else:
                    mosaic = Packer_patched(images).pack(
                        container_width=container_width,
                        container_height_max=container_height_max,
                    )
                    mosaic = np.array(mosaic)

                    if container_width > mosaic.shape[1]:
                        mosaic = np.pad(
                            mosaic,
                            [(0, 0), (0, container_width - mosaic.shape[1]), (0, 0)],
                        )

                    # sometimes container is too large...
                    mosaic = mosaic[:container_height_max, :container_width]

                label = Image.fromarray(
                    np.ones((20, mosaic.shape[1], 3), dtype=np.uint8) * 255
                )
                drawL = ImageDraw.Draw(label)
                textStr = "%s-%s" % (
                    str(t1).split(".")[0].split("T")[1],
                    str(t2).split(".")[0].split("T")[1],
                )
                if nParticlesNeeded != nPids:
                    textStr += " (R)"
                drawL.text((0, 0), textStr, (0, 0, 0), font=fontL)

                mosaic = Image.fromarray(np.vstack((label, mosaic)))
                #             display(mosaic)
                mosaics.append(mosaic)

            nCols = nTiles // nRows

            widths, heights = zip(*(i.size for i in mosaics))

            for nRow in range(nRows):
                x_offset = 0

                for im in mosaics[nCols * (nRow) : nCols * (nRow + 1)]:
                    new_im.paste(im, (x_offset, max(heights) * nRow + 50))
                    x_offset += im.size[0] + extra

                # x_offset = 0
            # for im in mosaics[len(mosaics)//nRows:]:
            #   new_im.paste(im, (x_offset,max(heights) +50))
            #   x_offset += im.size[0] + extra

    tenmm = 1e6 / (1 / config.calibration.slope) / 100

    if ff.hour == "":
        title = (
            "%s-%s-%s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger matched particles plotted, 10 mm = %.1f px ="
            % (
                ff.year,
                ff.month,
                ff.day,
                config["name"],
                minSize,
                minSize * (1 / config.calibration.slope) * 1e-6 * 1000,
                particlesPloted,
                nParticles,
                tenmm,
            )
        )
    else:
        title = (
            "%s-%s-%sT%s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger matched particles plotted, 10 mm = %.1f px ="
            % (
                ff.year,
                ff.month,
                ff.day,
                ff.hour,
                config["name"],
                minSize,
                minSize * (1 / config.calibration.slope) * 1e-6 * 1000,
                particlesPloted,
                nParticles,
                tenmm,
            )
        )

    # new_im = cv2.putText(np.array(new_im), title,
    #                      (0, 45), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS,)
    # (label_width, label_height), baseline = cv2.getTextSize(
    #     title, FONT, FONT_SCALE, FONT_THICKNESS)

    draw = ImageDraw.Draw(new_im)
    draw.text((0, 0), title, (0, 0, 0), font=font)
    width = draw.textlength(title, font=font)

    draw.line((width + 15, 30, width + 15 + round(tenmm), 30), fill=0, width=5)

    tools.createParentDir(ffOut, mode=config.dirMode)
    new_im.save(ffOut)
    print("SAVED ", ffOut)

    if returnFig:
        return ffOut, new_im
    else:
        return ffOut


@tools.loopify
def createLevel3RimingQuicklook(
    case,
    config,
    skipExisting=True,
    version=__version__,
    returnFig=True,
):
    """
    Create level3 combined rime quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    version : str, optional
        Version identifier, by default __version__
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    camera = config.leader
    nodata = False
    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level3combinedRiming

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("level3combinedRiming"),
    ):
        return None, None

    lv3 = ff.listFiles("level3combinedRiming")
    if len(lv3) == 0:
        if len(ff.listFilesExt("level3combinedRiming")) == 0:
            log.error(
                f"{case} level3combinedRiming data not found {ff.fnamesPattern.level3combinedRiming}"
            )
            return None, None
        else:
            nodata = True
    else:
        lv3 = lv3[0]

    if len(ff.listFiles("metaEvents")) == 0:
        log.error(f"{case} event data not found")
        return None, None

    log.info(f"running {case} {fOut}")

    fig, axs = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(20, 20),
        gridspec_kw={
            # "width_ratios": [1, 0.01],
            "height_ratios": [1, 1, 1, 1],
            "hspace": 0.02,
            "wspace": 0.1,
        },
    )
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    fig.suptitle(
        f"VISSS level3combinedRiming {config.level3.combinedRiming.extraFileStr}\n"
        + f"{ff.year}-{ff.month}-{ff.day}"
        + ", "
        + config["name"]
        + "",
        fontsize=25,
        y=0.995,
        fontweight="bold",
        x=mid,
    )
    if nodata:
        axs[0].set_title("no data")
    else:
        dat3 = xr.open_dataset(lv3)  # .sel(size_definition="Dmax", drop=True)

        dat2 = xr.open_dataset(ff.listFiles("level2track")[0]).sel(
            size_definition="Dmax", cameratrack="max", drop=True
        )

        quality = tools.unpackQualityFlags(dat2.qualityFlags, doubleTimestamps=True)

        fEvents1 = ff.listFiles("metaEvents")[0]
        fEvents2 = files.FilenamesFromLevel(fEvents1, config).filenamesOtherCamera(
            level="metaEvents"
        )[0]

        try:
            events1 = xr.open_dataset(fEvents1)
        except:
            print(f"{fEvents1} broken")
            return None, None
        try:
            events2 = xr.open_dataset(fEvents2)
        except:
            print(f"{fEvents2} broken")
            return None, None

        events = xr.concat((events1, events2), dim="file_starttime")

        (ax1, ax2, ax3, ax4) = axs

        dat3.Ze_0.plot(ax=ax1, label="Ze_0")
        dat3.Ze_ground.plot(ax=ax1, label="Ze_ground")
        dat3.Ze_combinedRetrieval.plot(ax=ax1, label="Ze_combinedRetrieval")
        dat3.Ze_ground_fitResidual.where(dat3.Ze_0 > -10).plot(
            ax=ax1, label="Ze_ground_fitResidual"
        )
        dat3.Ze_std.where(dat3.Ze_0 > -10).plot(ax=ax1, label="Ze_std")

        ax1.set_ylabel("Ze [dBz]")
        ax1.legend()
        ax1.set_ylim(-20, 40)

        dat3.combinedNormalizedRimeMass.plot(ax=ax2, label="M (combined)")
        dat2.normalizedRimeMass_mean.plot(ax=ax2, label="M (in situ weighted mean)")
        dat2.normalizedRimeMass_dist.mean("D_bins").plot(
            ax=ax2, label="M (in situ mean)"
        )

        ax2.set_ylabel("M [-]")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.set_ylim(1e-3, 10)

        dat3.IWC.plot(ax=ax3, label="IWC (combined)")
        ax3.set_ylabel("IWC [kg/m$^3$]")
        ax3.set_yscale("log")
        ax3.legend()

        dat3.SR_M.plot(ax=ax4, label="SR (combined with meas. fall vel.)")
        dat3.SR_M_heymsfield10.plot(ax=ax4, label="SR (combined with param. fall vel.)")
        ax4.set_ylabel("SR [mm/h w.e.]")
        ax4.set_yscale("log")
        ax4.legend()

        obervationsDiffer = quality.sel(flag="obervationsDiffer", drop=True)
        recordingFailed = quality.sel(flag="recordingFailed", drop=True)
        processingFailed = quality.sel(flag="processingFailed", drop=True)
        blowingSnow = quality.sel(flag="blowingSnow", drop=True)
        cameraBlocked = quality.sel(flag="cameraBlocked", drop=True)
        tracksTooShort = quality.sel(flag="tracksTooShort", drop=True)

        for ax in axs:
            ax.set_title(None)

            ylim = ax.get_ylim()
            cond = quality.time.where(recordingFailed)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="pink",
                    alpha=0.25,
                    label="cameras off",
                )
            cond = quality.time.where(obervationsDiffer)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="magenta",
                    alpha=0.25,
                    label="observed # of particles differ",
                )

            cond = quality.time.where(processingFailed)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="purple",
                    alpha=0.25,
                    label="processing failed",
                )
            cond = quality.time.where(cameraBlocked)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="red",
                    alpha=0.25,
                    label=f"camera blocked > {config.quality.blockedPixThresh*100}%",
                )
            cond = quality.time.where(blowingSnow)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="orange",
                    alpha=0.25,
                    label=f"blowing snow > {config.quality.blowingSnowFrameThresh*100}%",
                )  # , hatch='///')
            cond = quality.time.where(tracksTooShort)
            if cond.notnull().any():
                ax.fill_between(
                    cond,
                    [ylim[0]] * len(quality.time),
                    [ylim[1]] * len(quality.time),
                    color="blue",
                    alpha=0.2,
                    label=f"mean track length < {config.quality.trackLengthThreshold}",
                    # hatch="X",
                )

            ax.set_ylim(ylim)

            ax.tick_params(axis="both", labelsize=15)
            ax.grid()
            ax.set_xlim(
                np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00"),
                np.datetime64(f"{ff.year}-{ff.month}-{ff.day}" + "T00:00")
                + np.timedelta64(1, "D"),
            )

        for ax in axs[:-1]:
            ax.set_xticklabels([])

        # mark relaunch
        firstEvent = True
        for event in events.event:
            if str(event.values).startswith("start") or str(event.values).startswith(
                "launch"
            ):
                for ax in axs[:]:
                    if firstEvent:
                        label = "restarted"
                        firstEvent = False
                    else:
                        label = None
                    ax.axvline(
                        event.file_starttime.values, color="red", ls=":", label=label
                    )

        ax1.legend()
        ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        dat2.close()
        dat3.close()
    tools.savefig(fig, config, fOut, fnames=ff.listFiles("level3combinedRiming"))
    return fig
