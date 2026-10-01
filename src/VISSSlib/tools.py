# -*- coding: utf-8 -*-

import argparse
import datetime
import glob
import io
import json
import multiprocessing
import os
import re
import shutil
import socket
import struct
import subprocess
import time
import uuid
import warnings
import zipfile
import zlib
from copy import deepcopy
from functools import wraps

import numba
import numpy as np
import taskqueue
import xarray as xr
from addict import Dict
from loguru import logger as log

from . import __version__, files, fixes


def _allDoneParents(camera, config):
    parents = [
        "leader_metaEvents",
        "follower_metaEvents",
    ]
    if config.level1match.processL1match:
        parents += ["leader_level2track", "leader_level2match"]
    if config.level2.processL2detect:
        parents += ["leader_level2detect", "follower_level2detect"]
    # combinedRiming depends on level2track, so it can only be pulled in
    # when the matching/tracking branch is actually enabled -- otherwise
    # a deployment with processL1match: false but a stray
    # combinedRiming.processRetrieval: true would make allDone try (and
    # fail) to build level2track anyway.
    if config.level1match.processL1match and config.level3.combinedRiming.processRetrieval:
        parents += ["leader_level3combinedRiming"]
    return parents


# Single source of truth for "what depends on what" and "how is it built"
# for every processing level. `parents` is a callable(camera, config) ->
# list of f"{camera}_{level}" parent names (a callable because a few
# levels' parents depend on the camera or on config flags, e.g. allDone).
# `leaderOnly=True` means the level only ever exists for camera="leader".
# `command` describes how products.DataProduct.generateCommands() builds
# the shell command:
#   ("none",)                                  no command (raw input levels)
#   ("daily", call)                             one command for the whole day
#   ("l1", originLevel, call, extraOrigin)      one command per level0/L1 file
#   ("touch",)                                  the allDone sentinel file
#
# Every processing function's own skip-check (tools.checkForExisting) should
# resolve its parents/events list from this via resolveLevelParents rather
# than hand-writing one -- a hand-written list can silently drift out of
# sync with what's declared here (e.g. distributions._createLevel2's
# level2track check not knowing about its level2match dependency, fixed
# alongside this comment), leaving the DAG orchestrator (products.py)
# correctly regenerating a command that the processing function itself
# then immediately skips again, forever.
LEVEL_REGISTRY = {
    "level0": {
        "parents": lambda camera, config: [],
        "command": ("none",),
    },
    "level0txt": {
        "parents": lambda camera, config: [],
        "command": ("none",),
    },
    "metaEvents": {
        "parents": lambda camera, config: [f"{camera}_level0txt"],
        "command": ("daily", "metadata.createEvent"),
    },
    "metaFrames": {
        "parents": lambda camera, config: [f"{camera}_level0txt"],
        "command": ("daily", "metadata.createMetaFrames"),
    },
    "level1detect": {
        "parents": lambda camera, config: [],
        "command": ("l1", "level0txt", "detection.detectParticles", None),
    },
    "metaRotation": {
        "parents": lambda camera, config: [
            "leader_level1detect",
            "follower_level1detect",
            # metaEvents are added to all the L2 products to force
            # regeneration when event file is updated (ie more data is
            # transferred)
            "leader_metaEvents",
            "follower_metaEvents",
        ],
        "command": ("daily", "matching.createMetaRotation"),
        "leaderOnly": True,
    },
    "level1match": {
        "parents": lambda camera, config: [f"{camera}_metaRotation"],
        "command": (
            "l1",
            "level1detect",
            "matching.matchParticles",
            "metaRotation",
        ),
        "leaderOnly": True,
    },
    "level1track": {
        "parents": lambda camera, config: [f"{camera}_level1match"],
        "command": ("l1", "level1match", "tracking.trackParticles", None),
        "leaderOnly": True,
    },
    "level2detect": {
        "parents": lambda camera, config: [
            f"{camera}_level1detect",
            f"{camera}_metaEvents",
        ],
        "command": ("daily", "distributions.createLevel2detect"),
    },
    "level2match": {
        "parents": lambda camera, config: [
            f"{camera}_level1match",
            # metaEvents are added to all the L2 products to force
            # regeneration when events file is updated (ie more data is
            # transferred)
            "leader_metaEvents",
            "follower_metaEvents",
        ],
        "command": ("daily", "distributions.createLevel2match"),
        "leaderOnly": True,
    },
    "level2track": {
        "parents": lambda camera, config: [
            f"{camera}_level1track",
            "leader_level2match",
            "leader_metaEvents",
            "follower_metaEvents",
        ],
        "command": ("daily", "distributions.createLevel2track"),
        "leaderOnly": True,
    },
    "level3combinedRiming": {
        "parents": lambda camera, config: [
            f"{camera}_level2track",
            "leader_metaEvents",
            "follower_metaEvents",
        ],
        "command": ("daily", "level3.retrieveCombinedRiming"),
        "leaderOnly": True,
    },
    "allDone": {
        "parents": _allDoneParents,
        "command": ("touch",),
        "leaderOnly": True,
    },
}


def resolveLevelParents(level, camera, case, config):
    """
    Resolve LEVEL_REGISTRY[level]'s declared parents into (FindFiles,
    parentLevel) pairs for this camera+case, ready to pass straight into
    checkForExisting's `parents=` (or `events=`) kwarg.

    This exists so a processing function's own skip-check can't silently
    drift out of sync with the DAG orchestrator's dependency declaration
    the way distributions._createLevel2's did for level2track/level2match
    (level2track reuses level2match's own zResidualTooWide flag, so it's
    genuinely stale whenever level2match changes -- LEVEL_REGISTRY already
    declared that dependency, the skip-check just never asked about it,
    so products.py kept correctly regenerating a command that immediately
    skipped itself again, forever). Every checkForExisting call site
    should build its parents/events list from this instead of
    hand-writing one, so LEVEL_REGISTRY stays the actual enforced source
    of truth.

    Parameters
    ----------
    level : str
        The level that is checking for existing/up-to-date output --
        i.e. whose declared parents (not its own identity) get resolved.
    camera : str
        "leader" or "follower" -- this level's own camera role.
    case : str
        Case identifier (day, "YYYYMMDD").
    config : dict
        Settings (as returned by readSettings).

    Returns
    -------
    list of (files.FindFiles, str)
        One (FindFiles, parentLevel) pair per declared parent, each
        FindFiles instance built for that parent's own camera role
        (which may differ from `camera`, e.g. a "follower_metaEvents"
        parent of a "leader"-camera level).
    """
    # LEVEL_REGISTRY's "parents" lambdas format their {camera}_{level}
    # parent names against the camera *role* ("leader"/"follower"), but
    # callers of this function are inconsistent about which form of
    # `camera` they hold at that point -- e.g. distributions._createLevel2
    # receives the role for "match"/"track" (via @tools.loopify, which
    # never touches camera at all) but the fully-resolved camera id for
    # "detect" (via @tools.loopify_with_camera, which resolves "leader"/
    # "follower" to config.leader/config.follower before calling in).
    # files.FindFiles already tolerates both forms transparently; do the
    # same normalization here instead of assuming the role.
    if camera not in ("leader", "follower"):
        camera = "leader" if camera == config.leader else "follower"
    parentNames = LEVEL_REGISTRY[level]["parents"](camera, config)
    resolved = []
    for parentName in parentNames:
        parentCamera, parentLevel = parentName.split("_")
        cameraFull = config.leader if parentCamera == "leader" else config.follower
        resolved.append((files.FindFiles(case, cameraFull, config), parentLevel))
    return resolved


DEFAULT_SETTINGS = {
    # settings that must be provided in YAML file
    "computers": None,
    "fps": None,
    "frame_height": None,
    "frame_width": None,
    "leader": None,
    "follower": None,
    "nThreads": None,
    "path": None,
    "pathOut": None,
    "pathQuicklooks": None,
    "visssGen": None,
    "site": None,
    "start": None,
    "end": None,
    "name": None,
    "model": None,
    # mostly default settings
    "aux": {
        "arm": {},
        "cloudnet": {
            "site": None,
        },
        "meteo": {
            "downloadData": True,
            "source": None,
            "path": None,
            "doi": None,
        },
        "pangaea": {},
        "radar": {
            # "source": "cloudnetCategorize",
            "calibrationOffset": {
                "2000-01-01": 0,  # values that need to be added to radar Z. always previos value is used
            },
            "downloadData": True,
            "elevation": 90,
            "heightRange": (120, 360),
            "minHeightBins": 4,
            "path": None,
            "source": None,
            "timeOffset": 120,
        },
    },
    "calibration": {
        "slope": None,
        "slope_err": None,
    },
    # applied to every deployment regardless of what a yaml's own dataFixes
    # lists -- see the merge in readSettings(), which unions this with the
    # yaml's value instead of letting it be overridden like other settings
    "dataFixes": ["removeFlippedCaptureTimeFrames", "dropUnrecordedTrailingFrames"],
    "dirMode": 0o775,  # 509
    "fileMode": 0o664,  # 436
    "goodFiles": ["None", "None"],
    "badData": [],
    "level1detect": {
        "applyCanny2Particle": True,  # canny filter gets edges better
        "backSub": "cv2.createBackgroundSubtractorKNN",
        "backSubKW": {
            "dist2Threshold": 400,
            "detectShadows": False,
            "history": 100,
        },  #       dist2Threshold of 100 was extensively tested, but this makes small particles larger even though it helps with wings etc.
        #  this is compensated by the canny filter, but not for holes in the particles (the default is {"dist2Threshold": 400,
        #                               "detectShadows": False, "history": 100})
        "blurSigma": 1,
        "check4childCntLength": True,  # discard short child contours instead of dilate/erose
        "cropImage": None,  # (offsetX, offsetY)
        "cv2NumThreads": None,  # cv2's own thread pool is not covered by the OMP_NUM_THREADS/
        # OPENBLAS_NUM_THREADS/etc. env vars products.py already exports for detect jobs, so
        # by default it happily grabs os.cpu_count() threads per process; with
        # tools.workers() launching os.cpu_count() worker processes, that oversubscribes
        # cores badly (measured ~2-3x slower aggregate throughput). Left at the default
        # (None), detectParticles follows the OMP_NUM_THREADS env var products.py already
        # sets per job (falling back to cv2's own default if that isn't set either, e.g.
        # interactive use outside the task queue). Set an explicit int here to override
        # both and force a specific value regardless of environment -- including -1, which
        # cv2 itself defines as "reset to system default", i.e. the escape hatch to force
        # cv2 back to using all cores even when OMP_NUM_THREADS is set for everything else.
        "dilateErodeFgMask": False,  # turns out to be not so smart because it makes holes insides particles smaller
        "dilateFgMask4Contours": True,
        "dilateIterations": 1,  # to close gaps in canny edges (the default is 1 whic is sufficient)
        "doubleDynamicRange": True,
        "erosionTestThreshold": 0.06,
        "height_offset": 64,
        "maskCorners": None,
        "maxMovingObjects": 1000,  # 60 until 18.9.24
        "minArea": 0,
        "minAspectRatio": None,  # testing only
        "minBlur": 10,
        "minBlur4picturewrite": 250,
        "minContrast": 20,
        "minDmax": 0,
        "minMovingPixels": [20, 10, 5, 2, 2, 2, 2],
        "minSize4picturewrite": 8,
        "testMovieFile": True,
        "threshs": [20, 30, 40, 60, 80, 100, 120],
        "trainingSize": 100,
        "writeImg": True,
    },
    "level1detectQuicklook": {
        "minBlur": 500,
        "minSize": 10,
        "omitLabel4small": True,
    },
    "level1match": {
        "maxMovingObjects": 300,  # 60 until 18.9.24
        "processL1match": True,
    },
    "level1shape": {},
    "level1track": {
        "maxMovingObjects": 300,  # 60 until 18.9.24
    },
    "level2": {
        "correctForSmallOnes": False,
        "freq": "1min",
        "processL2detect": True,
    },
    "level3": {
        "combinedRiming": {
            "extraFileStr": "",
            "maxTemp": 275.15,  # +2°C
            "minNParticles": 100,
            "minZe": -10,
            "processRetrieval": False,
            "Zvar": "Ze_ground",  # extrapolated to surface using aux.radar.heightIndices
        }
    },
    "logo": None,
    "movieExtension": "mkv",
    "newFileInt": 600,
    "quality": {
        "blowingSnowFrameThresh": 0.05,
        "blockedPixThresh": 0.1,
        # pixels (level1's native unit -- this residual is computed
        # directly on level1-stage pixel data, before level2's SI/meter
        # calibration is ever applied); max tolerable std of the per-pair
        # Z-consistency residual (matching.zResidualSigma) before a
        # rotation refit is considered
        # futile -- a refit can only correct a *biased* residual (shift
        # the mean), not shrink a wide one. Used both to skip pointless
        # refit attempts during matchParticles' self-heal escalation and,
        # independent of matchScore, to flag already-passing level1match
        # files with suspiciously inconsistent pairs (scripts/qc_report.py
        # --matchscore-check). Empirically derived on hyytiala2_v3
        # (2026-08-30): a fixable file had pre-refit sigma ~1.4, a
        # partially-fixable extreme-density file ~3.9, a confirmed
        # unfixable file (15 refit attempts, ~zero bias) ~5.5 -- a
        # starting heuristic, expect it to need tuning per deployment.
        "maxZSigma": 5.0,
        "minMatchScore": 1e-3,
        "minSize4insituM": 10,
        "obsRatioThreshold": 0.7,
        # level2track flag trackingIncomplete: fraction of the expected
        # observations (given velocity and observation volume) that were
        # actually tracked. Derived from 41 VISSS1/2/3 test cases: <= 0.17 for
        # dense data with poor matching or many tiny particles, >= 0.56 for
        # windy, fast, rainy and low-concentration cases.
        "minTrackCompleteness": 0.25,
    },
    "rotate": {},
}

niceNames = (
    ("master", "leader"),
    ("trigger", "leader"),
    ("slave", "follower"),
)


def loopify_with_camera(func=None, *, endYesterday=True):
    """
    Decorator to make function loop over cases and cameras.

    Parameters
    ----------
    endYesterday : bool, optional
        Whether to end the case range at yesterday. Default is True.
        This parameter is passed to getCaseRange.

    Returns
    -------
    callable
        Decorator function or wrapped function.

    Examples
    --------
    ::

        @loopify_with_camera
        def my_func(case, camera, config):
            pass

        @loopify_with_camera(endYesterday=False)
        def my_func(case, camera, config):
            pass
    """

    def decorator(f):
        @wraps(f)
        @log.catch(
            reraise=True
        )  # catches exceptions from the wrapped function
        def loopify_with_camera_(case, camera, settings, *args, **kwargs):
            config = readSettings(settings)
            if camera == "all":
                cameras = [config.leader, config.follower]
            elif camera == "leader":
                cameras = [config.leader]
            elif camera == "follower":
                cameras = [config.follower]
            else:
                cameras = [camera]
            cases = getCaseRange(case, config, endYesterday=endYesterday)
            if len(cases) > 1:
                log.info(f"Converted case string '{case}' to case range: {cases}")
            returns = list()
            for case1 in cases:
                for camera1 in cameras:
                    log.info(
                        f"Processing {case1} with {f.__name__} for {camera1} at {config.basename}"
                    )
                    returns.append(f(case1, camera1, config, *args, **kwargs))
            if len(returns) == 1:
                return returns[0]
            else:
                return returns

        return loopify_with_camera_

    if func is None:
        # Called with parentheses: @loopify_with_camera() or @loopify_with_camera(endYesterday=False)
        return decorator
    else:
        # Called without parentheses: @loopify_with_camera
        return decorator(func)


def loopify(func=None, *, endYesterday=True):
    """
    Decorator to make function loop over cases.

    Parameters
    ----------
    endYesterday : bool, optional
        Whether to end the case range at yesterday. Default is True.
        This parameter is passed to getCaseRange.

    Returns
    -------
    callable
        Decorator function or wrapped function.

    Examples
    --------
    ::

        @loopify
        def my_func(case, config):
            pass

        @loopify(endYesterday=False)
        def my_func(case, config):
            pass
    """

    def decorator(f):
        @wraps(f)
        @log.catch(
            reraise=True
        )  # catches exceptions from the wrapped function
        # catches exceptions from the wrapped function
        def loopify_(case, settings, *args, **kwargs):
            config = readSettings(settings)
            cases = getCaseRange(case, config, endYesterday=endYesterday)
            if len(cases) > 1:
                log.info(f"Converted case string '{case}' to case range: {cases}")
            returns = list()
            for case1 in cases:
                log.info(f"Processing {case1} with {f.__name__} at {config.basename}")
                returns.append(f(case1, config, *args, **kwargs))
            if len(returns) == 1:
                return returns[0]
            else:
                return returns

        return loopify_

    if func is None:
        # Called with parentheses: @loopify() or @loopify(endYesterday=False)
        return decorator
    else:
        # Called without parentheses: @loopify
        return decorator(func)


class DictNoDefault(Dict):
    """
    Dictionary class that raises KeyError when accessing missing keys.

    Inherits from addict.Dict.
    """

    def __missing__(self, key):
        raise KeyError(key)


def nicerNames(string):
    """
    Replace VISSS names with up-to-date equivalents.

    Parameters
    ----------
    string : str
        Input string to process.

    Returns
    -------
    str
        String with replaced names.
    """
    for i in range(len(niceNames)):
        string = string.replace(*niceNames[i])
    return string


def readSettings(fname):
    """
    Read configuration settings from a YAML file. Default is taken from DEFAULT_SETTINGS

    Parameters
    ----------
    fname : str or dict
        Path to YAML file or already parsed config dict.

    Returns
    -------
    addict.Dict
        Configuration settings.
    """
    import flatten_dict
    import yaml

    if type(fname) is str:
        # we have to flatten the dictionary so that update works
        config = flatten_dict.flatten(DEFAULT_SETTINGS)
        with open(fname, "r") as stream:
            loadedSettings = flatten_dict.flatten(yaml.load(stream, Loader=yaml.Loader))
            # Check for keys in loadedSettings that are not in DEFAULT_SETTINGS
            default_keys = set(flatten_dict.flatten(DEFAULT_SETTINGS).keys())
            exception_top = {}
            for key in loadedSettings.keys():
                if key not in default_keys:
                    # Skip keys in exception_top (top-level) and any key under 'rotate'
                    if (len(key) == 1 and key[0] in exception_top) or (
                        len(key) > 0 and key[0] in ["rotate"]
                    ):
                        continue
                    log.warning(
                        f"Key {key} in settings file is not in the default settings and might be unused."
                    )
            # dataFixes is additive, not overridden like every other setting:
            # DEFAULT_SETTINGS["dataFixes"] holds fixes meant to apply to every
            # deployment regardless of what a specific yaml lists, so union it
            # with the yaml's own value instead of letting the plain update()
            # below silently drop the defaults whenever a yaml declares its
            # own (even empty) dataFixes list.
            dataFixesKey = ("dataFixes",)
            if dataFixesKey in loadedSettings:
                merged = list(DEFAULT_SETTINGS["dataFixes"])
                for fix in loadedSettings[dataFixesKey] or []:
                    if fix not in merged:
                        merged.append(fix)
                loadedSettings[dataFixesKey] = merged
            config.update(loadedSettings)
        # unflatten again and convert to addict.Dict
        config = DictNoDefault(flatten_dict.unflatten(config))

        config["filename"] = fname
        config["basename"] = os.path.basename(fname)
        config["dirname"] = os.path.dirname(fname)

        config["instruments"] = [config.leader, config.follower]
        # check for relative paths (with respect to the yaml file and make them absolute
        for key in ["path", "pathOut", "pathQuicklooks"]:
            if not config[key].startswith("/"):
                config[key] = f"{config["dirname"]}/{config[key]}"
            config[key] = config[key].replace("$HOSTNAME", socket.gethostname())
        return config
    else:  # is already config
        return fname
        
def isBadPeriod(case, config, product=None):
    """
    Check if a case falls within a bad data period.
    
    Parameters
    ----------
    case : str
        Case identifier (YYYYMMDD or YYYYMMDD-HHMMSS)
    config : dict
        Configuration dictionary
    product : str, optional
        Product level to check. If None, returns True if case is bad for any product.
    
    Returns
    -------
    tuple(bool, str)
        (is_bad, reason) where reason is empty string if not bad
    """

    config = readSettings(config)

    if not hasattr(config, 'badData') or config.badData is None:
        return False, ""
    
    #files.FindFiles(case, camera.leader, config).datetime
    case_dt =  datetime.datetime.strptime(
        case.ljust(15, "0"), "%Y%m%d-%H%M%S"
    ) if '-' in case else datetime.datetime.strptime(case, "%Y%m%d")
    
    for period in config.badData:

        if product in files.dailyLevels:
            start = datetime.datetime.strptime(
                str(period.start.split("-")[0]), "%Y%m%d"
            )
            # end-of-day exclusive upper bound (the day after period.end);
            # must be compared with `<`, not `<=` -- the case string for
            # that following day itself parses to this same midnight, so
            # `<=` would incorrectly also flag the day right after the
            # configured end date as bad.
            end = datetime.datetime.strptime(
                str(period.end.split("-")[0]), "%Y%m%d"
            ) + datetime.timedelta(days=1)
            inRange = start <= case_dt < end
        else:
            start = datetime.datetime.strptime(
                str(period.start).ljust(15, "0"), "%Y%m%d-%H%M%S"
            )
            end = datetime.datetime.strptime(
                str(period.end).ljust(15, "0"), "%Y%m%d-%H%M%S"
            )
            inRange = start <= case_dt <= end

        if not inRange:
            continue
        # Period matches time range - check product
        if (period.products is None) or (product is None) or (product in period.products):
            return True, period.reason
    
    return False, ""


def getCaseRange(nDays, config, endYesterday=True):
    """
    Get list of case identifiers from a date range.

    Parameters
    ----------
    nDays : int, str
        Number of days going back or date string "YYYYMMDD" or "YYYYMMDD-YYYYMMDD"
        or "YYYYMMDD,YYYYMMDD,YYYYMMDD".
    config : dict
        Configuration settings.
    endYesterday : bool, optional
        Whether to end yesterday, by default True.

    Returns
    -------
    list of str
        List of case identifiers.
    """
    # shortcut to detect timestamps of YYYYMMDD-HH or YYYYMMDD-HHMMSS
    if type(nDays) is str:
        if len(nDays) < 6:
            nDays = int(nDays)
        elif (nDays[-3] == "-") or (nDays[-7] == "-"):
            return [nDays]
    days = getDateRange(nDays, config, endYesterday=endYesterday)
    cases = []
    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"
        cases.append(case)
    return cases


def getDateRange(nDays, config, endYesterday=True):
    """
    Generate date range based on input parameters.

    Parameters
    ----------
    nDays : int, str
        Number of days going back or date string "YYYYMMDD" or "YYYYMMDD-YYYYMMDD"
        or "YYYYMMDD,YYYYMMDD,YYYYMMDD".
    config : dict
        Configuration settings.
    endYesterday : bool, optional
        Whether to end yesterday, by default True.

    Returns
    -------
    pandas.DatetimeIndex
        Date range.
    """
    import pandas as pd

    if type(nDays) is np.str_:
        nDays = str(nDays)

    config = readSettings(config)
    if config["end"] == "today":
        end = datetime.datetime.now(datetime.UTC)
        if endYesterday:
            end2 = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)
        else:
            end2 = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)
    else:
        end = end2 = pd.Timestamp(config["end"], tz="UTC")

    if (type(nDays) is int) or type(nDays) is float:
        if nDays > 1000:
            nDays = str(nDays)
    elif (type(nDays) is str) and (len(nDays) < 6):
        nDays = int(nDays)

    if nDays == 0:
        days = pd.date_range(
            start=pd.Timestamp(config["start"], tz="UTC"),
            end=end2,
            freq="1D",
            normalize=True,
            name=None,
            inclusive="both",
        )

    elif type(nDays) is str:
        days = nDays.split(",")
        if len(days) == 1:
            days = nDays.split("-")
            if len(days) == 1:
                days = pd.date_range(
                    start=nDays,
                    periods=1,
                    freq="1D",
                    tz="UTC",
                    normalize=True,
                    name=None,
                    inclusive="both",
                )
            else:
                days = pd.date_range(
                    start=days[0],
                    end=days[1],
                    freq="1D",
                    tz="UTC",
                    normalize=True,
                    name=None,
                    inclusive="both",
                )
        else:
            days = pd.DatetimeIndex(days).tz_localize("UTC")

    else:
        days = pd.date_range(
            end=end2,
            periods=nDays,
            freq="1D",
            normalize=True,
            name=None,
            inclusive="both",
        )

    # double check to make sure we did not add too much
    if np.any(days < pd.Timestamp(config.start, tz="UTC")):
        log.warning(
            f"Date range {nDays} includes cases that are before the specified start {config.start}"
        )
        days = days[days >= pd.Timestamp(config.start, tz="UTC")]

    if config.end == "today":
        end = datetime.datetime.now(datetime.UTC)
    else:
        end = pd.Timestamp(config.end, tz="UTC")

    if np.any(days > end):
        log.warning(
            f"Date range {nDays} includes cases that are after the specified end {end}"
        )
        days = days[days <= pd.Timestamp(config.end, tz="UTC")]

    return days


def getPreviousKey(thisDict, case):
    """
    Get the previous key-value pair in a dictionary based on timestamp.

    Parameters
    ----------
    thisDict : dict
        Dictionary with datetime keys.
    case : str
        Case identifier.

    Returns
    -------
    tuple
        Previous value and timestamp, or None if not found.
    """
    thisDict1 = {}
    for k, v in thisDict.items():
        thisDict1[np.datetime64(k)] = v

    dates = np.array(list(thisDict1.keys()))

    # 1. Filter for dates before the case
    previous_dates = dates[dates < case2timestamp(case)]

    # 2. Get the maximum of the remaining dates
    result = np.max(previous_dates) if previous_dates.size > 0 else None

    if result is None:
        return None
    else:
        return thisDict1[result], result


def getPreviousCalibrationOffset(case, config):
    """
    Get the previous calibration offset for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : dict
        Configuration settings.

    Returns
    -------
    tuple
        Calibration offset and timestamp, or None if not found.
    """
    return getPreviousKey(config.aux.radar.calibrationOffset, case)


def getMode(a):
    """
    Compute the mode of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        Mode of the array.
    """
    (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    mode = a[index]
    return a


def open_mfmetaFrames(fnames, config, start=None, end=None, skipFixes=[]):
    """
    Open multiple metaFrame files.

    Parameters
    ----------
    fnames : list of str
        File names.
    config : dict
        Configuration settings.
    start : datetime, optional
        Start time, by default None.
    end : datetime, optional
        End time, by default None.
    skipFixes : list of str, optional
        Fixes to skip, by default [].

    Returns
    -------
    xarray.Dataset
        Combined dataset.
    """

    def preprocess(dat):
        # keep track of file start time
        fname = dat.encoding["source"]
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray(
            [ffl1.datetime64] * len(dat.capture_time), coords=[dat.capture_time]
        )

        return dat

    with xr.open_mfdataset(
        fnames, combine="nested", concat_dim="capture_time", preprocess=preprocess
    ) as ds:
        dat = ds.load()

    if skipFixes != "all":
        # fix potential integer overflows if necessary
        if ("captureIdOverflows" in config.dataFixes) and (
            "captureIdOverflows" not in skipFixes
        ):
            print("apply fix", "captureIdOverflows")
            dat = fixes.captureIdOverflows(dat, config, dim="capture_time")

        # # beware of unintended implications down the processing chain becuase capture_time is used
        # # in analysis module
        # # fix potential integer overflows if necessary
        # if ("makeCaptureTimeEven" in config.dataFixes) and ("makeCaptureTimeEven" not in skipFixes):
        #     print("apply fix", "makeCaptureTimeEven")
        #     #does not make sense for leader
        #     if "follower" in dat.encoding["source"]:
        #         dat = fixes.makeCaptureTimeEven(dat, config, dim="capture_time")
        #     else:
        #         #make sure follower and leader data are consistent
        #         dat["capture_time_orig"] = dat["capture_time"]

    if start is not None:
        dat = dat.isel(capture_time=(dat.capture_time >= start))
        if len(dat.capture_time) == 0:
            return None

    if end is not None:
        dat = dat.isel(capture_time=(dat.capture_time <= end))
        if len(dat.capture_time) == 0:
            return None

    if len(dat.capture_time) == 0:
        return None

    return dat


def open_mflevel1detect(
    fnamesExt, config, start=None, end=None, skipFixes=[], datVars="all"
):
    """
    Open multiple level1detect files.

    Parameters
    ----------
    fnamesExt : list of str
        File names.
    config : dict
        Configuration settings.
    start : datetime, optional
        Start time, by default None.
    end : datetime, optional
        End time, by default None.
    skipFixes : list of str, optional
        Fixes to skip, by default [].
    datVars : str or list of str, optional
        Variables to include, by default "all".

    Returns
    -------
    xarray.Dataset
        Combined dataset.
    """

    def preprocess(dat):
        # keep trqack of file start time
        fname = dat.encoding["source"]
        # print("open_mflevel1detect",fname)
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray(
            [ffl1.datetime64] * len(dat.pid), coords=[dat.pid]
        )
        if "nThread" not in dat.keys():
            dat["nThread"] = xr.DataArray([-99] * len(dat.pid), coords=[dat.pid])

        if datVars != "all":
            dat = dat[datVars]
        return dat

    if type(fnamesExt) is not list:
        fnamesExt = [fnamesExt]

    fnames = []
    for fname in fnamesExt:
        if fname.endswith("nodata"):
            pass
        elif fname.endswith("broken.txt"):
            pass
        elif fname.endswith("notenoughframes"):
            pass
        else:
            fnames.append(fname)
    if len(fnames) == 0:
        return None

    with xr.open_mfdataset(
        fnames, combine="nested", concat_dim="pid", preprocess=preprocess
    ) as ds:
        dat = ds.load()

    if start is not None:
        dat = dat.isel(pid=(dat.capture_time >= start))
        if len(dat.pid) == 0:
            return None

    if end is not None:
        dat = dat.isel(pid=(dat.capture_time <= end))
        if len(dat.pid) == 0:
            return None

    if skipFixes != "all":
        # fix potential integer overflows if necessary
        if ("captureIdOverflows" in config.dataFixes) and (
            "captureIdOverflows" not in skipFixes
        ):
            dat = fixes.captureIdOverflows(dat, config)

        # moved to meta data
        # # fix potential integer overflows if necessary
        # if( "makeCaptureTimeEven" in config.dataFixes) and ( "makeCaptureTimeEven" not in skipFixes):

        #     #does not make sense for leader
        #     if "follower" in dat.encoding["source"]:
        #         dat = fixes.makeCaptureTimeEven(dat, config, dim="pid")
        #     else:
        #         #make sure follower and leader data are consistent
        #         dat["capture_time_orig"] = dat["capture_time"]

    # replace pid by empty dimesnion to allow concatenating files without jumps in dimension pid
    dat = dat.swap_dims({"pid": "fpid"})

    if len(dat.fpid) == 0:
        return None

    if len(dat.fpid) == 0:
        return None

    dat.load()
    return dat

def globList(fnames, search=None, replace=None):
    if isinstance(fnames, list):
        res = []
        for fname in fnames:
            res += globList(fname, search=search, replace=replace)
    else:
        if (search is not None) and (replace is not None):
            fnames = fnames.replace(search, replace)
        res = sorted(filter(os.path.isfile, glob.glob(fnames)))
    return res



def open_mflevel1match(fnamesExt, config, datVars="all"):
    """
    Open multiple level1match files.

    Parameters
    ----------
    fnamesExt : list of str
        File names.
    config : dict
        Configuration settings.
    datVars : str or list of str, optional
        Variables to include, by default "all".

    Returns
    -------
    xarray.Dataset
        Combined dataset.
    """
    if type(fnamesExt) is not list:
        fnamesExt = [fnamesExt]

    fnames = []
    for fname in fnamesExt:
        if fname.endswith("nodata"):
            pass
        elif fname.endswith("broken.txt"):
            pass
        elif fname.endswith("notenoughframes"):
            pass
        else:
            fnames.append(fname)
    if len(fnames) == 0:
        return None

    def preprocess(dat):
        # keep trqack of file start time
        fname = dat.encoding["source"]
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray(
            [ffl1.datetime64] * len(dat.pair_id), coords=[dat.pair_id]
        )
        if datVars != "all":
            dat = dat[datVars]
        return dat

    with xr.open_mfdataset(
        fnames, combine="nested", concat_dim="pair_id", preprocess=preprocess
    ) as ds:
        dat = ds.load()
    # replace pid by empty dimesnion to allow concatenating files without jumps in dimension pid
    dat = dat.swap_dims({"pair_id": "fpair_id"})

    return dat


def identifyBlockedBlowingSnowData(fnames, config, timeIndex1, sublevel):
    """
    Identify blocked blowing snow data.

    Parameters
    ----------
    fnames : list of str
        File names.
    config : dict
        Configuration settings.
    timeIndex1 : array_like
        Time bins.
    sublevel : str
        Sublevel identifier.

    Returns
    -------
    tuple of xarray.DataArray
        Blowing snow ratio and detected particles.
    """
    # handle blowing snow, estimate ratio of skipped frames

    # print("starting identifyBlowingSnowData", cam)
    movingObjects = []
    for fna in fnames:
        with xr.open_dataset(fna) as ds:
            movingObjects.append(ds.movingObjects.load())
    movingObjects = xr.concat(movingObjects, dim="capture_time")
    # movingObjects = xr.open_mfdataset(fnames,  combine='nested', preprocess=preprocess).movingObjects.load()
    movingObjects = movingObjects.sortby("capture_time")

    tooManyMove = movingObjects > config[f"level1{sublevel}"].maxMovingObjects
    tooManyMove = tooManyMove.groupby_bins(
        "capture_time", timeIndex1, labels=timeIndex1[:-1]
    )

    nFrames = tooManyMove.count()
    # does not make sense for small number of frames
    nFrames = nFrames.where(nFrames > 100)
    blowingSnowRatio = tooManyMove.sum() / nFrames  # now a ratio
    # nan means nothing recorded, so no blowing snow either
    blowingSnowRatio = blowingSnowRatio.fillna(0)
    blowingSnowRatio = blowingSnowRatio.rename(capture_time_bins="time")

    nDetected = (
        movingObjects.groupby_bins("capture_time", timeIndex1, labels=timeIndex1[:-1])
        .sum()
        .fillna(0)
    )
    nDetected = nDetected.rename(capture_time_bins="time")

    # print("done identifyBlowingSnowData")
    return blowingSnowRatio, nDetected


def compareNDetected(nDetectedL, nDetectedF):
    """
    Compare detected particles between two cameras.

    Parameters
    ----------
    nDetectedL : xarray.DataArray
        Detected particles for leader.
    nDetectedF : xarray.DataArray
        Detected particles for follower.

    Returns
    -------
    xarray.DataArray
        Ratio of minimum to maximum detections.
    """
    minParticles = 1000
    nDetected = xr.concat([nDetectedL, nDetectedF], dim="camera")
    ratio = nDetected.min("camera") / nDetected.max("camera")
    ratio = ratio.where((nDetectedL > minParticles) & (nDetectedF > minParticles))
    # ratio.values[(nDetectedL < minParticles) | (nDetectedF < minParticles)] = np.nan
    return ratio


def removeBlockedBlowingData(dat1D, events, config, threshold=0.1):
    """
    Remove blocked or blowing snow data.

    Parameters
    ----------
    dat1D : xarray.Dataset
        Input dataset.
    events : str or xarray.Dataset
        Events file or dataset.
    config : dict
        Configuration settings.
    threshold : float, optional
        Blocking threshold, by default 0.1 i.e. 10%

    Returns
    -------
    xarray.Dataset
        Filtered dataset or None if empty.
    """
    # shortcut
    if dat1D is None:
        return None

    ts, counts = np.unique(dat1D.capture_time, return_counts=True)
    if np.any(counts > config.level1match.maxMovingObjects):
        tsBlowingSnow = ts[counts > config.level1match.maxMovingObjects]
        dat1D = dat1D.isel(fpid=~dat1D.capture_time.isin(tsBlowingSnow))
    if len(dat1D.fpid) == 0:
        print("no data after removing blowing snow data")
        return None

    if type(events) is str:
        events = xr.open_dataset(events)
    blocked = events.blocking.sel(blockingThreshold=50) > threshold

    # interpolate blocking status to observed particles
    isBlocked = blocked.sel(file_starttime=dat1D.capture_time, method="nearest").values
    dat1D = dat1D.isel(fpid=(~isBlocked))
    events.close()

    if len(dat1D.fpid) > 0:
        print(f"{np.sum(isBlocked)/ len(dat1D.fpid)*100}% blocked data removed")

    if len(dat1D.fpid) == 0:
        print("no data after removing blocked data")
        return None
    else:
        return dat1D

def _aggregate(results):
    import pandas as pd
    if not results:
        return results
    first = results[0]
    if isinstance(first, bool):
        return all(results)
    elif isinstance(first, int):
        return sum(results)
    elif isinstance(first, list):
        seen = set()
        out = []
        for lst in results:
            for item in lst:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
        return out
    elif isinstance(first, dict):
        return {key: _aggregate([r[key] for r in results]) for key in first}
    elif isinstance(first, pd.DataFrame):
        return pd.concat(results).sort_index()
    elif isinstance(first, str):
        return results  # ← always a list, even if len==1
    else:
        return results

def estimateCaptureIdDiff(
    leaderFile,
    followerFiles,
    config,
    dim,
    concat_dim="capture_time",
    nPoints=500,
    maxDiffMs=1,
):
    """
    Estimate capture ID difference between cameras.

    Parameters
    ----------
    leaderFile : str
        Leader camera file.
    followerFiles : list of str
        Follower camera files.
    config : dict
        Configuration settings.
    dim : str
        Dimension to use.
    concat_dim : str, optional
        Concatenation dimension, by default "capture_time".
    nPoints : int, optional
        Number of points to sample, by default 500.
    maxDiffMs : int, optional
        Maximum difference in milliseconds, by default 1.

    Returns
    -------
    tuple
        Estimated ID difference and number of samples.
    """
    leaderDat = xr.open_dataset(leaderFile)
    followerDat = xr.open_mfdataset(
        followerFiles, combine="nested", concat_dim=concat_dim
    ).load()

    followerDat = cutFollowerToLeader(
        leaderDat, followerDat, gracePeriod=0, dim=concat_dim
    )

    if "captureIdOverflows" in config.dataFixes:
        leaderDat = fixes.captureIdOverflows(
            leaderDat, config, storeOrig=True, idOffset=0, dim=dim
        )
        followerDat = fixes.captureIdOverflows(
            followerDat, config, storeOrig=True, idOffset=0, dim=dim
        )
        # fix potential integer overflows if necessary

    if "makeCaptureTimeEven" in config.dataFixes:
        # does not make sense for leader
        # redo capture_time based on first time stamp...
        # requires trust in capture_id
        followerDat = fixes.makeCaptureTimeEven(followerDat, config, dim)

    idDiff = estimateCaptureIdDiffCore(
        leaderDat, followerDat, dim, nPoints=nPoints, maxDiffMs=maxDiffMs
    )
    leaderDat.close()
    followerDat.close()

    return idDiff


def estimateCaptureIdDiffCore(
    leaderDat, followerDat, dim, nPoints=500, maxDiffMs=1, timeDim="capture_time"
):
    """
    Core function to estimate capture ID difference.

    Parameters
    ----------
    leaderDat : xarray.Dataset
        Leader data.
    followerDat : xarray.Dataset
        Follower data.
    dim : str
        Dimension to use.
    nPoints : int, optional
        Number of points to sample, by default 500.
    maxDiffMs : int, optional
        Maximum difference in milliseconds, by default 1.
    timeDim : str, optional
        Time dimension, by default "capture_time".

    Returns
    -------
    tuple
        Estimated ID difference and number of samples.

    Raises
    ------
    RuntimeError
        If capture ID varies too much.
    """
    if len(leaderDat[dim]) == 0:
        raise RuntimeError(f"leaderDat has zero length")
    if len(followerDat[dim]) == 0:
        raise RuntimeError(f"followerDat has zero length")

    # check whether correction has been applied and use if present
    if (timeDim == "capture_time") and ("capture_time_even" in followerDat.data_vars):
        timeDimFollower = "capture_time_even"
    else:
        timeDimFollower = timeDim
    if (timeDim == "capture_time") and ("capture_time_even" in leaderDat.data_vars):
        timeDimLeader = "capture_time_even"
    else:
        timeDimLeader = timeDim

    # cut number of investigated points in time if required
    if len(leaderDat[dim]) > nPoints:
        points = np.linspace(0, len(leaderDat[dim]), nPoints, dtype=int, endpoint=False)
    else:
        points = np.arange(len(leaderDat[dim]))

    # Find, for every sampled leader point, the nearest follower
    # timestamp -- vectorized instead of a Python loop that recomputed
    # |t - followerTimes| against the *entire* follower array for every
    # single point (O(nPoints * N_follower), and paying xarray
    # per-element isel/argmin overhead nPoints times on top of that).
    #
    # followerDat[timeDimFollower] is *nearly* sorted (particles are
    # appended in capture order) but not guaranteed to be: the follower
    # camera's onboard clock is resynced to the computer clock close to
    # the start of each 10-minute file, which can make a handful of
    # frames right at a file-rotation boundary compare out of order by a
    # few tens/hundreds of microseconds even though their append order is
    # correct. Checking only the two immediate searchsorted neighbours
    # (as if the array were exactly sorted) can silently return the wrong
    # nearest match right at such a boundary, with no assertion to catch
    # it. Instead, use searchsorted as an approximate locator only, then
    # check a padded window of candidates around it and take the true
    # (order-independent) minimum within that window -- wide enough to
    # comfortably contain the true nearest neighbour even right at a
    # boundary perturbation, while still vectorized and far cheaper than
    # the original per-point full-array search.
    followerTimes = followerDat[timeDimFollower].values
    followerCaptureId = followerDat.capture_id.values

    leaderTimes = leaderDat[timeDimLeader].isel(**{dim: points}).values
    leaderCaptureId = leaderDat.capture_id.isel(**{dim: points}).values

    nFollower = len(followerTimes)
    windowRadius = 25
    idxApprox = np.clip(
        np.searchsorted(followerTimes, leaderTimes, side="left"), 0, nFollower - 1
    )
    offsets = np.arange(-windowRadius, windowRadius + 1)
    # increasing offsets -> increasing (pre-clip) candidate index, so
    # ties within the window resolve to the lowest original index first,
    # matching np.argmin's first-occurrence tie-break on the original
    # full-array search.
    candidateIdx = np.clip(idxApprox[:, None] + offsets[None, :], 0, nFollower - 1)
    candidateDiff = np.abs(followerTimes[candidateIdx] - leaderTimes[:, None])
    bestInWindow = np.argmin(candidateDiff, axis=1)
    rows = np.arange(len(leaderTimes))
    nearestIdx = candidateIdx[rows, bestInWindow]
    pMin = candidateDiff[rows, bestInWindow]

    keep = pMin < np.timedelta64(int(maxDiffMs), "ms")
    idDiffs = followerCaptureId[nearestIdx[keep]] - leaderCaptureId[keep]

    nIdDiffs = len(idDiffs)
    print(f"using {nIdDiffs} of {len(points)}")

    if nIdDiffs > 0:
        (vals, idx, counts) = np.unique(idDiffs, return_index=True, return_counts=True)
        idDiff = vals[np.argmax(counts)]
        ratioSame = np.sum(idDiffs == idDiff) / nIdDiffs
        print(
            "estimateCaptureIdDiff statistic:",
            dict(zip(vals[np.argsort(counts)[::-1]], counts[np.argsort(counts)[::-1]])),
            timeDim,
        )

        if ratioSame > 0.7:
            print(
                f"capture_id determined {idDiff}, {ratioSame*100}% have the same value"
            )
            return idDiff, nIdDiffs
        else:
            raise RuntimeError(
                f"capture_id varies too much, only {ratioSame*100}% of {nIdDiffs} samples have the same value {idDiff}, 2n place: {vals[np.argsort(counts)[-2]]}"
            )

    else:
        print(f"nIdDiffs {nIdDiffs} is too short")
        return None, nIdDiffs


def otherCamera(camera, config):
    """
    Get the other VISSS camera based on configuration.

    Parameters
    ----------
    camera : str
        Camera identifier.
    config : dict
        Configuration settings.

    Returns
    -------
    str
        Other camera identifier.

    Raises
    ------
    ValueError
        If camera is not in instruments list.
    """
    if camera == config["instruments"][0]:
        return config["instruments"][1]
    elif camera == config["instruments"][1]:
        return config["instruments"][0]
    else:
        raise ValueError


def cutFollowerToLeader(leader, follower, gracePeriod=1, dim="fpid"):
    """
    Cut follower data to match leader data with respect to time.

    Parameters
    ----------
    leader : xarray.Dataset
        Leader data.
    follower : xarray.Dataset
        Follower data.
    gracePeriod : int, optional
        Grace period in seconds, by default 1.
    dim : str, optional
        Dimension to use, by default "fpid".

    Returns
    -------
    xarray.Dataset
        Cut follower data.
    """
    start = leader.capture_time[0].values - np.timedelta64(
        int(gracePeriod * 1000), "ms"
    )
    end = leader.capture_time[-1].values + np.timedelta64(int(gracePeriod * 1000), "ms")

    captureTimeValues = follower.capture_time.values

    # Follower particles are appended in capture order, so capture_time is
    # *nearly* monotonic -- but not guaranteed to be: the follower
    # camera's onboard clock is resynced to the computer clock close to
    # the start of each 10-minute file, which can make a handful of
    # frames right at a file-rotation boundary compare out of order by a
    # few tens/hundreds of microseconds even though their append order is
    # correct (the previous file's tail carries drift-inflated
    # timestamps; the new file's head is freshly accurate). Sorting by
    # that faulty comparison would flip those frames into the wrong
    # order, so we must never assume or enforce global sortedness here.
    #
    # Instead, use searchsorted purely as an approximate locator (only
    # exactly correct for genuinely sorted input, but off by no more than
    # a handful of positions for data that's this close to sorted), pad
    # generously on each side to comfortably cover that uncertainty, and
    # then apply an exact, order-independent boolean filter within that
    # local slice. This keeps 66ac0bd's win of not copying/scanning the
    # entire (potentially much larger) follower dataset on every leader
    # chunk in doMatchSlicer -- only the small padded slice is touched --
    # without ever assuming order anywhere a correctness-affecting
    # decision is made.
    nFollower = len(captureTimeValues)
    padFrames = 200
    lo = max(np.searchsorted(captureTimeValues, start, side="left") - padFrames, 0)
    hi = min(np.searchsorted(captureTimeValues, end, side="right") + padFrames, nFollower)

    candidate = follower.isel({dim: slice(lo, hi)})
    mask = (candidate.capture_time.values >= start) & (candidate.capture_time.values <= end)
    return candidate.isel({dim: mask})


def nextCase(case):
    """
    Get the next case identifier.

    Parameters
    ----------
    case : str
        Current case identifier.

    Returns
    -------
    str
        Next case identifier.
    """
    return str(case2timestamp(case) + np.timedelta64(1, "D")).replace("-", "")


def prevCase(case):
    """
    Get the previous case identifier.

    Parameters
    ----------
    case : str
        Current case identifier.

    Returns
    -------
    str
        Previous case identifier.
    """
    return str(case2timestamp(case) - np.timedelta64(1, "D")).replace("-", "")


def timestamp2case(dd):
    """
    Convert timestamp to case identifier.

    Parameters
    ----------
    dd : datetime
        Timestamp.

    Returns
    -------
    str
        Case identifier.
    """
    year = str(dd.year)
    month = "%02i" % dd.month
    day = "%02i" % dd.day
    case = f"{year}{month}{day}"
    return case


def case2timestamp(case):
    """
    Convert case identifier to timestamp.

    Parameters
    ----------
    case : str
        Case identifier.

    Returns
    -------
    numpy.datetime64
        Timestamp.

    Raises
    ------
    NotImplementedError
        If long cases are not implemented.
    """
    if len(case) == 8:
        return np.datetime64(f"{case[:4]}-{case[4:6]}-{case[6:8]}")
    else:
        raise NotImplementedError("long cases not implemented yet")


def rescaleImage(
    frame1, rescale, anti_aliasing=False, anti_aliasing_sigma=None, mode="edge"
):
    """
    Rescale an image.

    Parameters
    ----------
    frame1 : array_like
        Input image.
    rescale : float
        Scaling factor.
    anti_aliasing : bool, optional
        Apply anti-aliasing, by default False.
    anti_aliasing_sigma : float, optional
        Anti-aliasing sigma, by default None.
    mode : str, optional
        Resizing mode, by default "edge".

    Returns
    -------
    array_like
        Rescaled image.
    """
    import skimage

    if len(frame1.shape) == 3:
        newShape = np.array(
            (frame1.shape[0] * rescale, frame1.shape[1] * rescale, frame1.shape[2])
        )
    else:
        newShape = np.array(frame1.shape) * rescale
    frame1 = skimage.transform.resize(
        frame1,
        newShape,
        mode=mode,
        anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma,
        preserve_range=True,
        order=0,
    )
    return frame1


def displayImage(frame, doDisplay=True, rescale=None):
    """
    Display an image.

    Parameters
    ----------
    frame : array_like
        Input image.
    doDisplay : bool, optional
        Whether to display, by default True.
    rescale : float, optional
        Scaling factor, by default None.

    Returns
    -------
    IPython.display.Image or None
        Image object if doDisplay=False, otherwise None.
    """
    # opencv cannot handle grayscale png with alpha channel
    import cv2
    import IPython.display

    if (len(frame.shape) == 3) and (frame.shape[2] == 2):
        fill_color = 0
        frameAlpha = frame[:, :, 1]
        frameData = frame[:, :, 0]
        frameCropped = deepcopy(frameData)
        frameCropped[frameAlpha == 0] = fill_color
        frame1 = np.hstack((frameData, frameCropped))
    else:
        frame1 = frame

    if rescale is not None:
        frame1 = rescaleImage(frame1, rescale)

    _, frame1 = cv2.imencode(".png", frame1)

    if doDisplay:
        IPython.display.display(IPython.display.Image(data=frame1.tobytes()))
    else:
        return IPython.display.Image(data=frame1.tobytes())


class ZipFile(zipfile.ZipFile):
    """
    Extended zip file class with automatic parent directory creation.
    Extra functions to store and retrieve images as numpy arrays
    """

    def __init__(self, file, **kwargs):
        createParentDir(file)
        super().__init__(file, **kwargs)

    def addnpy(self, fname, array):
        # encode
        buf1 = io.BytesIO()
        np.save(buf1, array)
        return self.writestr(fname, buf1.getbuffer())

    def extractnpy(self, fname):
        array = np.load(io.BytesIO(self.read(fname)))
        return array


def imageZipFile(fname, **kwargs):
    r"""
    Create appropriate archive file handler.

    Parameters
    ----------
    fname : str
        File name.
    \*\*kwargs : dict
        Additional arguments for archive creation.

    Returns
    -------
    Archive handler
        ZipFile or BlockImageArchive instance.
    """
    createParentDir(fname)

    if fname.endswith("zip"):
        return ZipFile(fname, **kwargs)
    else:
        return BlockImageArchive(fname, **kwargs)


class BlockImageArchive:
    """
    A high-efficiency binary archive for storing thousands of small numpy arrays.

    This class provides a single-file storage solution specifically optimized for
    small, variable-sized uint8 arrays. It uses block compression to maximize the
    compression ratio and maintains a JSON index for O(1) random access.

    Parameters
    ----------
    filepath : str
        Path to the archive file.
    mode : {'a', 'w', 'r'}, optional
        'a' : Append/Read mode. Loads existing index and allows adding more data.
        'w' : Write mode. Overwrites any existing file.
        'r' : Read-only mode. Prevents any modifications to the file.
        Default is 'a'.
    block_size : int, optional
        Number of images to group into a single compressed block.
        Default is 256.
    level : int, optional
        Zip compression level.
        Default is 8.
    """

    def __init__(self, filepath, mode="r", block_size=256, level=8):
        self.filepath = filepath
        self.mode = mode
        self.level = level
        self.block_size = block_size
        self.index = {}
        self.buffer = []
        self._f = None

        file_exists = os.path.exists(filepath)

        if mode == "r":
            if not file_exists:
                raise FileNotFoundError(f"Archive not found: {filepath}")
            self._f = open(self.filepath, "rb")
        elif mode == "w":
            if file_exists:
                os.remove(filepath)
            self._f = open(self.filepath, "wb+")
        elif mode == "a":
            self._f = open(self.filepath, "rb+" if file_exists else "wb+")
        else:
            raise ValueError("Mode must be 'r', 'w', or 'a'")

        # Load index if the file is not new
        if file_exists and os.path.getsize(self.filepath) > 8:
            self._read_index_from_disk()

    def _read_index_from_disk(self):
        """Reads the index pointer and loads the index dictionary."""

        self._f.seek(-8, os.SEEK_END)
        index_ptr = struct.unpack("<Q", self._f.read(8))[0]
        self._f.seek(index_ptr)
        data = self._f.read(os.path.getsize(self.filepath) - index_ptr - 8)
        self.index = json.loads(data.decode("utf-8"))

        # If appending, seek to the start of the old index to overwrite it.
        # If reading, this seek doesn't hurt.
        self._f.seek(index_ptr)

    def _write_current_block(self):
        """Compresses buffered images and appends them to the file."""
        if self.mode == "r":
            raise IOError("Cannot write to an archive opened in read-only mode.")

        if not self.buffer:
            return

        raw_bytes_list = [img.tobytes() for _, img, _ in self.buffer]
        raw_block = b"".join(raw_bytes_list)
        compressed_block = zlib.compress(raw_block, level=self.level)

        self._f.seek(0, os.SEEK_END)
        block_offset = self._f.tell()
        self._f.write(compressed_block)
        block_len = self._f.tell() - block_offset

        current_inner_offset = 0
        for i, (image_id, array, shape) in enumerate(self.buffer):
            img_len = len(raw_bytes_list[i])
            self.index[str(image_id)] = [
                block_offset,
                block_len,
                current_inner_offset,
                img_len,
                shape,
            ]
            current_inner_offset += img_len

        self.buffer = []

    def addnpy(self, image_id, array):
        """
        Add a numpy array to the archive.

        The array is added to an internal buffer. When the buffer reaches
        `block_size`, the images are compressed and written to disk.

        Parameters
        ----------
        image_id : str or int
            Unique identifier for the image.
        array : numpy.ndarray
            The uint8 array to store.
        """
        if self.mode == "r":
            raise IOError("Archive is in read-only mode.")
        self.buffer.append((image_id, array, list(array.shape)))
        if len(self.buffer) >= self.block_size:
            self._write_current_block()

    def extractnpy(self, image_id):
        """
        Retrieve a numpy array by its ID.

        This method reads only the required compressed block from disk,
        decompresses it in memory, and returns the specific slice.

        Parameters
        ----------
        image_id : str or int
            The identifier of the image to retrieve.

        Returns
        -------
        numpy.ndarray
            The reconstructed uint8 array.

        Raises
        ------
        KeyError
            If the image_id is not found in the archive index.
        """
        if str(image_id) not in self.index:
            raise KeyError(f"ID {image_id} not found.")

        b_offset, b_len, inner_off, img_len, shape = self.index[str(image_id)]

        self._f.seek(b_offset)
        compressed_block = self._f.read(b_len)
        raw_block = zlib.decompress(compressed_block)

        img_bytes = raw_block[inner_off : inner_off + img_len]
        return np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)

    def close(self):
        """
        Closes the file handle. In 'w' or 'a' mode, finalizes the index.
        In 'r' mode, simply closes the file.
        """
        if self._f and not self._f.closed:
            # Only write metadata if we were in a writing mode
            if self.mode in ("w", "a"):
                self._write_current_block()
                self._f.seek(0, os.SEEK_END)
                ptr = self._f.tell()
                self._f.write(json.dumps(self.index).encode("utf-8"))
                self._f.write(struct.pack("<Q", ptr))

            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, "_f") and self._f is not None:
                if not self._f.closed:
                    self.close()
        except:
            pass


def createParentDir(file, mode=None):
    """
    Create parent directory if it doesn't exist.

    Parameters
    ----------
    file : str
        File path.
    mode : int, optional
        Directory mode, by default None.
    """
    parent_dir = os.path.dirname(os.path.abspath(file))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        if mode is not None:
            os.chmod(parent_dir, mode)
    return


def savefig(
    fig, config, filename, fnames=None, addLogo=True, w_pad=None, h_pad=None, **kwargs
):
    r"""
    Save a matplotlib Figure to `filename` with proper permissions.

    This function saves a matplotlib figure to a file with appropriate directory
    creation and file permissions. It also adds status text to the figure and
    optionally includes a logo.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to save.
    config : dict
        Configuration settings containing directory modes and logo information.
    filename : str
        The output file path where the figure will be saved.
    fnames : str or list of str, optional
        File names used to determine creation date for status text.
        If None, no date information is included. Default is None.
    addLogo : bool, optional
        Whether to add a logo to the figure. Default is True.
    w_pad : float, optional
        Width padding for tight layout. Default is None.
    h_pad : float, optional
        Height padding for tight layout. Default is None.
    \*\*kwargs : dict
        Additional keyword arguments passed to matplotlib's savefig function.

    Returns
    -------
    matplotlib.figure.Figure
        The figure that was saved (for chaining operations).

    Notes
    -----
    - Creates parent directories if they don't exist with permissions based on config.dirMode
    - Sets file permissions to config.fileMode after saving
    - Adds status text with VISSSlib version, creation timestamp, and file creation date
    - Optionally adds a logo from config.logo if available
    - Removes existing files with the same name before saving to prevent corruption
    - Uses tight_layout with specified padding for better figure formatting

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> savefig(fig, config, "output.png")
    """

    # Add status text to the figure
    _statusText(fig, fnames, config, addLogo=addLogo)
    fig.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # Ensure parent directory exists
    createParentDir(filename, config.dirMode)

    # sometimes exisiting files make problems
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    fig.savefig(filename, **kwargs)

    # Set file permissions
    try:
        os.chmod(filename, config.fileMode)
    except PermissionError:
        log.warning(f"chmod {config.fileMode} {filename} failed")

    return fig


def _statusText(fig, fnames, config, addLogo=True):
    """
    Add status text to a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add status text to
    fnames : str or list of str
        File names used to determine creation date
    config : dict
        Configuration dictionary containing metadata
    addLogo : bool, optional
        Whether to add logo to the figure, by default True

    Returns
    -------
    matplotlib.figure.Figure
        The figure with added status text
    """
    from PIL import Image, ImageFont

    if not isinstance(fnames, (list, tuple)):
        fnames = [fnames]
    try:
        thisDate = np.max([os.path.getmtime(f) for f in fnames])
    except ValueError:
        thisDate = ""
    except FileNotFoundError:
        thisDate = ""
    except TypeError:
        thisDate = ""
    else:
        thisDate = timestamp2str(thisDate)
    string = f"VISSSlib {__version__}, created  "
    string += f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
    string += f"from files created at {thisDate} "
    fig.text(
        0,
        0,
        string,
        fontsize=8,
        transform=fig.transFigure,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    if addLogo and (config.logo is not None):
        try:
            im = Image.open(config.logo)
        except FileNotFoundError:
            log.error(f"Did not find {config.logo}")
        else:
            fig.figimage(np.asarray(im), 0, fig.bbox.ymax - im.height, zorder=10)

    return fig


_HISTORY_VERSION_RE = re.compile(r"created with VISSSlib (\S+)")


def collectVersionAttrs(level, parentFiles):
    """
    Build version-provenance attrs for `finishNc`'s `extra=`.

    Records this level's own __version__ under "{level}_version",
    plus every "*_version"/"*_versions" attr already recorded on each
    direct parent's files, folded forward under its own key -- so the
    whole ancestor chain (e.g. level1detect -> level1match ->
    level1track -> level2track) accumulates without each level needing
    to know its grandparents. If a parent's underlying files disagree
    (e.g. partial reprocessing), the key becomes the plural
    "*_versions" form with a sorted, comma-joined list of the distinct
    values seen.

    A direct parent file that predates this feature entirely (no
    "*_version" attrs of its own at all -- the case for every existing
    level1detect file, since reprocessing level1detect is not
    feasible and it will never gain a clean "level1detect_version" attr
    on its own) falls back to parsing its "history" attr (every file
    already gets one from `ncAttrs`, of the form "...created with
    VISSSlib X.Y.Z..."), so the chain still records *something* for
    that ancestor instead of silently dropping it forever.

    Parameters
    ----------
    level : str
        This level's name, e.g. "level1match".
    parentFiles : dict of str -> list of str
        Maps each DIRECT parent level name to the already-produced
        file(s) of that level this product was built from, e.g.
        {"level1detect": [leaderFile, *followerFiles]} for level1match,
        or {"level1track": [...], "level2match": [...]} for
        level2track. The level name is only used for the "history"
        fallback above (a file that already carries its own clean
        version attr doesn't need it). Files that don't exist or aren't
        valid netCDF (e.g. broken.txt/nodata sentinels) are silently
        skipped.

    Returns
    -------
    dict
        Attrs to pass as `finishNc(..., extra=...)`.
    """
    merged = {}

    def addVersion(base, value):
        merged.setdefault(base, set()).update(str(value).split(","))

    for parentLevel, files_ in parentFiles.items():
        for f in files_:
            try:
                with xr.open_dataset(f) as pds:
                    attrs = dict(pds.attrs)
            except Exception:
                continue
            foundOwn = False
            for k, v in attrs.items():
                if k.endswith("_versions"):
                    addVersion(k[: -len("_versions")], v)
                    foundOwn = True
                elif k.endswith("_version"):
                    addVersion(k[: -len("_version")], v)
                    foundOwn = True
            if not foundOwn:
                m = _HISTORY_VERSION_RE.search(attrs.get("history", ""))
                if m:
                    addVersion(parentLevel, m.group(1))

    result = {}
    for base, vs in merged.items():
        vs = sorted(vs)
        key = f"{base}_version" if len(vs) == 1 else f"{base}_versions"
        result[key] = ",".join(vs)

    result[f"{level}_version"] = __version__
    return result


# Per-level minimum acceptable output mtime ("breakpoint"). An existing
# output file with an mtime older than this is treated as stale and
# reprocessed regardless of how its parents' mtimes compare -- for code
# changes that alter a level's own logic without changing any parent file
# (the normal mtime-based skipExisting check already handles every other
# case, and does so essentially for free since it's already comparing
# mtimes -- deliberately NOT doing this via a per-file version attr, which
# would mean opening every existing file just to check a string; mtime
# reuses what's already being stat'd). Remove an entry once the fleet has
# caught up.
REPROCESS_AFTER = {
    "level1track": datetime.datetime(2026, 10, 1, 12, 41, 0),  # learned-scale tracker + track_expectedLength (8eaf532)
    "level2detect": datetime.datetime(2026, 9, 2, 17, 0, 0),  # better QC
    "level2match": datetime.datetime(2026, 9, 2, 17, 0, 0),  # better QC
    "level2track": datetime.datetime(2026, 10, 1, 12, 41, 0),  # turn-angle track edges + trackingIncomplete flag (35b9297)
}


def reprocessBreakpoint(level):
    """
    Minimum acceptable mtime (POSIX timestamp) for existing `level`
    output files, or None if there is no pending breakpoint for this
    level. See REPROCESS_AFTER.
    """
    cutoff = REPROCESS_AFTER.get(level)
    if cutoff is None:
        return None
    return cutoff.timestamp()


def ncAttrs(site, visssGen, extra={}):
    """
    Generate NetCDF attributes.

    Parameters
    ----------
    site : str
        Site identifier.
    visssGen : str
        Generator information.
    extra : dict, optional
        Extra attributes, by default {}.

    Returns
    -------
    dict
        NetCDF attributes.
    """
    import cv2
    import psutil

    my_process = psutil.Process(os.getpid())
    myCommand = " ".join(my_process.cmdline())

    if os.environ.get("USER") is not None:
        user = f" by user {os.environ.get('USER')}"
    else:
        user = ""
    attrs = {
        "title": f"Video In Situ Snowfall Sensor (VISSS) observations at {site}",
        "source": f"{visssGen} observations at {site}",
        "history": f"{str(datetime.datetime.now(datetime.UTC))}: created with VISSSlib {__version__} and OpenCV {cv2.__version__} on {socket.getfqdn()}{user}",
        "command": myCommand,
        "references": "Maahn, M., D. Moisseev, I. Steinke, N. Maherndl, and M. D. Shupe, 2024: Introducing the Video In Situ Snowfall Sensor (VISSS). Atmospheric Measurement Techniques, 17, 899–919, https://doi.org/10.5194/amt-17-899-2024.",
    }

    attrs.update(extra)
    return attrs


def finishNc(dat, site, visssGen, extra={}):
    """
    Finalize NetCDF dataset with attributes and encoding.

    Parameters
    ----------
    dat : xarray.Dataset
        Dataset to finalize.
    site : str
        Site identifier.
    visssGen : str
        Generator information.
    extra : dict, optional
        Extra attributes, by default {}.

    Returns
    -------
    xarray.Dataset
        Finalized dataset.
    """
    # todo: add yaml dump of config file

    extra = deepcopy(extra)
    for k in extra.keys():
        extra[k] = str(extra[k])

    dat.attrs.update(ncAttrs(site, visssGen, extra=extra))

    for k in list(dat.data_vars) + list(dat.coords):
        if dat[k].dtype == np.float64:
            dat[k] = dat[k].astype(np.float32)

        # #newest netcdf4 version doe snot like strings or objects:
        if (dat[k].dtype == object) or (dat[k].dtype == str):
            dat[k] = dat[k].astype("U")

        if not str(dat[k].dtype).startswith("<U"):
            dat[k].encoding = {}
            dat[k].encoding["zlib"] = True
            dat[k].encoding["complevel"] = 5
        # need to overwrite units becuase leting xarray handle that might lead to inconsistiencies
        # due to mixing of milli and micro seconds
        if k.endswith("time") or k.endswith("time_orig") or k.endswith("time_even"):
            dat[k].encoding["units"] = "microseconds since 1970-01-01 00:00:00"

    # sort variabels alphabetically
    dat = dat[sorted(dat.data_vars)]
    return dat


def getPrevRotationEstimates(datetime64, config):
    """
    Extract rotation estimates from config.

    Parameters
    ----------
    datetime64 : numpy.datetime64
        Datetime to query.
    config : dict
        Configuration settings.

    Returns
    -------
    tuple
        Rotation matrix, error matrix, and timestamp.
    """
    rotate, rotate_time = getPrevRotationEstimate(datetime64, "transformation", config)
    assert len(rotate) != 0
    rotate_err, rotate_err_time = getPrevRotationEstimate(
        datetime64, "transformation_err", config
    )
    assert len(rotate_err) != 0
    assert rotate_time == rotate_err_time

    return rotate, rotate_err, rotate_time


def getPrevRotationEstimate(datetime64, key, config):
    """
    Extract single rotation estimate from config.

    Parameters
    ----------
    datetime64 : numpy.datetime64
        Datetime to query.
    key : str
        Key for rotation estimate.
    config : dict
        Configuration settings.

    Returns
    -------
    tuple
        Rotation estimate and timestamp.

    Raises
    ------
    RuntimeError
        If no rotation estimate found.
    """

    rotate_all = {
        np.datetime64(datetime.datetime.strptime(d.ljust(15, "0"), "%Y%m%d-%H%M%S")): r[
            key
        ]
        for d, r in config.rotate.items()
    }
    rotTimes = np.array(list(rotate_all.keys()))
    rotDiff = datetime64 - rotTimes
    rotTimes = rotTimes[rotDiff >= np.timedelta64(0)]
    rotDiff = rotDiff[rotDiff >= np.timedelta64(0)]

    try:
        prevTime = rotTimes[np.argmin(rotDiff)]
    except ValueError:
        raise RuntimeError(
            f"datetime64 {datetime64} before earliest rotation estimate {np.min(np.array(list(rotate_all.keys())))}"
        )
    return rotate_all[prevTime], prevTime


def rotXr2dict(dat, config=None):
    """
    Convert rotation xarray data to dictionary.

    Parameters
    ----------
    dat : xarray.Dataset
        Rotation data.
    config : dict, optional
        Configuration settings, by default None.

    Returns
    -------
    dict
        Configuration dictionary with rotation info.
    """
    import pandas as pd

    if config is None:
        config = {}
        config["rotate"] = {}
    elif "rotate" not in config:
        # an explicitly empty `rotate: {}` in the settings yaml does not
        # survive flatten_dict's flatten/unflatten round-trip (an empty
        # dict has no leaves to flatten), so config["rotate"] is simply
        # absent rather than present-and-empty for a deployment that relies
        # entirely on metaRotation-retrieved rotation with no static prior
        config["rotate"] = {}
    for ii, tt in enumerate(dat.file_starttime):
        t1 = pd.to_datetime(str(tt.values)).strftime("%Y%m%d-%H%M%S")
        config["rotate"][t1] = {
            "transformation": dat.isel(file_starttime=ii)
            .sel(camera_rotation="mean")
            .to_pandas()
            .to_dict(),
            "transformation_err": dat.isel(file_starttime=ii)
            .sel(camera_rotation="err")
            .to_pandas()
            .to_dict(),
        }

    return config


def rotDict2Xr(rotate, rotate_err, prevTime):
    """
    Convert rotation dictionary to xarray dataset.

    Parameters
    ----------
    rotate : dict
        Rotation matrix.
    rotate_err : dict
        Rotation error matrix.
    prevTime : numpy.datetime64
        Previous timestamp.

    Returns
    -------
    xarray.Dataset
        Rotation dataset.
    """
    if rotate is np.nan:
        rotate = {"camera_Ofz": np.nan, "camera_phi": np.nan, "camera_theta": np.nan}
    if rotate_err is np.nan:
        rotate_err = {
            "camera_Ofz": np.nan,
            "camera_phi": np.nan,
            "camera_theta": np.nan,
        }

    metaRotationDf = {}
    for k in rotate.keys():
        metaRotationDf[k] = xr.DataArray(
            np.ones((1, 2)) * np.array([rotate[k], rotate_err[k]]),
            dims=["file_starttime", "camera_rotation"],
            coords=[[prevTime], np.array(["mean", "err"])],
        )
    return xr.Dataset(metaRotationDf)


def execute_stdout(command):
    """
    Execute command and print output continuously.

    Parameters
    ----------
    command : str
        Command to execute.

    Returns
    -------
    tuple
        Exit code and output.
    """
    # launch application as subprocess and print output constantly:
    # http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    print(command)
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output = ""

    # Poll process for new output until finished
    for line in process.stdout:
        line = line.decode()
        print(line, end="")
        output += line
        time.sleep(0.1)
    for line in process.stderr:
        line = line.decode()
        print(line, end=" ")
        output += line
        time.sleep(0.1)

    process.wait()
    exitCode = process.returncode

    return exitCode, output


def concat(*strs):
    r"""
    Concatenate strings with spaces.

    Parameters
    ----------
    \*strs : str
        Strings to concatenate.

    Returns
    -------
    str
        Concatenated string.
    """
    concat = " ".join([str(s) for s in strs])
    return concat


def concatImgY(im1, im2, background=0):
    """
    Concatenate images vertically.

    Parameters
    ----------
    im1 : array_like
        Upper image.
    im2 : array_like
        Lower image.
    background : int, optional
        Background color, by default 0.

    Returns
    -------
    array_like
        Concatenated image.
    """
    y1, x1 = im1.shape
    y2, x2 = im2.shape

    y3 = y1 + y2
    x3 = max(x1, x2)
    imT = np.full((y3, x3), background, dtype=np.uint8)
    imT[:y1, :x1] = im1
    imT[y1:, :x2] = im2
    # print("Y",im1.shape, im2.shape, imT.shape)

    return imT


def concatImgX(im1, im2, background=0):
    """
    Concatenate images horizontally.

    Parameters
    ----------
    im1 : array_like
        Left image.
    im2 : array_like
        Right image.
    background : int, optional
        Background color, by default 0.

    Returns
    -------
    array_like
        Concatenated image.
    """
    y1, x1 = im1.shape
    y2, x2 = im2.shape

    y3 = max(y1, y2)
    x3 = x1 + x2
    imT = np.full((y3, x3), background, dtype=np.uint8)
    imT[:y1, :x1] = im1
    imT[:y2, x1:] = im2

    # print("X",im1.shape, im2.shape, imT.shape)

    return imT


def _levelMarkerPaths(file, config):
    """
    Best-effort: resolve the (touch, done) marker paths for the
    DataProduct level+camera+day that `file` belongs to, so a completed
    write can invalidate any cached freshness-check summary for that
    level (see files.FindFiles.markerPath, products.DataProduct).

    Returns (None, None) for anything that isn't a recognized VISSSlib
    level output -- e.g. files written by level3/aux.py's Cloudnet
    downloads, which don't follow the "<level>_V<version>_..." naming
    convention at all -- so marker bookkeeping never blocks or breaks a
    write it doesn't understand, it just skips it. Sentinel writes
    (.nodata/.broken.txt) piggyback on a real level's filename, so those
    are matched too by stripping the suffix first.
    """
    base = file
    for suffix in (".nodata", ".broken.txt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    try:
        ff = files.FilenamesFromLevel(base, config)
    except Exception:
        return None, None
    for level, path in ff.fname.items():
        if path == base:
            return ff.markerPath(level, "touch"), ff.markerPath(level, "done")
    return None, None


def _touchLevelMarker(file, config):
    """
    Bump the "touch" marker for the level+camera+day `file` belongs to
    (creating it if missing), and drop any cached "done" summary for it.

    The marker's fence value is a random token written into the file's
    content, not its mtime: two touches issued close together can land
    in the same tick of the filesystem's (often coarse, ~1-4ms
    resolution) mtime clock and come out with an identical mtime, which
    would let a concurrent write slip past the fence check undetected.
    A fresh random token on every touch has no such resolution limit,
    so any two touches -- however close in time -- are always
    distinguishable.

    Concurrent writers touching the same marker never corrupt it or
    each other: the write is atomic (tmp file + os.replace), whichever
    touch lands last simply wins, and every touch happens right after
    this process's own write completed, so it is always a valid
    stand-in for "most recent write to this level" (see
    files.FindFiles.markerPath and readLevelSummary/writeLevelSummary
    for how a cached summary is fenced against this marker instead of
    trusted on its own -- that's what makes it safe under many
    concurrent SLURM workers writing the same level+camera+day).

    Caveat: this only works if every writer of this level is running
    code that calls this hook. A write from an un-upgraded worker (e.g.
    a mixed-version rollout across the realtime and SLURM-batch
    environments) would not bump the marker, so a cached summary could
    stay trusted past that write. Roll this out to all writers of a
    level together, the same way a `version` bump already implicitly
    requires.
    """
    touchPath, donePath = _levelMarkerPaths(file, config)
    if touchPath is None:
        return
    _invalidateLevelCacheAt(touchPath, donePath, config)


def touchOutputFile(file, config):
    """
    Safely bump an existing VISSSlib output file's mtime to now.

    A bare shell/`os.utime` touch on a managed output file (e.g. to mark
    an already-correct file as "fresh" again after a downstream/upstream
    freshness check started flagging it as stale for no real reason)
    changes the file's real mtime but does *not* bump this level's
    "touch" fence marker, since it bypasses the tools.open2/to_netcdf2
    hooks that normally do that. `_freshnessSummary` then keeps returning
    whatever (n, oldest, newest) was cached before the touch -- fenced
    against a marker that never moved, so the cache never even notices
    there was a write -- so every DataProduct built afterwards keeps
    reporting the pre-touch mtime, silently as if the touch never
    happened. `repairStaleFreshnessCache` can find and fix this after
    the fact, but it has to be told to look, and it's easy to forget: a
    raw touch during an interactive investigation (see
    feedback_raw_touch_bypasses_freshness_cache_fence in project memory)
    cost a long detour before the mismatch was traced back to this.

    Use this instead of a raw `touch`/`os.utime` on any file under a
    level's output directory: it updates the real mtime and invalidates
    this level+camera+day's cached summary in the same step, so there is
    no window where the cache can disagree with the file it's supposed
    to describe.

    Parameters
    ----------
    file : str
        Path to the existing output file to touch (a real `.nc`, or a
        `.nodata`/`.broken.txt` sentinel -- both are recognized the same
        way `_touchLevelMarker` already handles them).
    config : dict
        Settings, as returned by `readSettings`.
    """
    os.utime(file, None)
    _touchLevelMarker(file, config)


def _invalidateLevelCacheAt(touchPath, donePath, config):
    """
    Core of `_touchLevelMarker`, split out so a caller that already has
    the marker paths in hand -- e.g. products.DataProduct.repairStaleCache,
    which derives them from files.FindFiles.markerPath directly -- doesn't
    need a real output filename for `_levelMarkerPaths` to reverse-engineer
    a level+camera+day from (that reverse-engineering requires the file to
    match the full "<level>_V<version>_<site>_<computer>_<visssGen>_<type>_
    <serial>_<timestamp>" naming convention, which not every level actually
    uses, e.g. the "l1" per-file levels' own marker paths are still built
    from the case/camera FindFiles already has on hand rather than from a
    filename at all -- see files.FindFiles.markerPath).

    See `_touchLevelMarker` for what this does and why.
    """
    token = uuid.uuid4().hex
    tmpPath = f"{touchPath}.{os.getpid()}.{token}.tmp"
    try:
        createParentDir(touchPath, mode=config.dirMode)
        with open(tmpPath, "w") as f:
            f.write(token)
        os.chmod(tmpPath, config.fileMode)
        os.replace(tmpPath, touchPath)
    except OSError:
        tryRemovingFile(tmpPath)
        return
    # optimization only, not required for correctness: readLevelSummary
    # already refuses a summary whose fence doesn't match the touch
    # marker's current token, which is now guaranteed to differ from
    # whatever "done" was fenced against before this write.
    tryRemovingFile(donePath)


def getLevelTouchTime(fn, level):
    """
    Current fence token for `level`'s "touch" marker on `fn` (a
    files.FindFiles instance), or 0 if none exists yet -- i.e. nothing
    has ever been written for this level+camera+day through the
    open2/to_netcdf2 hooks. 0 is a legitimate, stable fence value (it
    only changes once the first such write happens), not an error.

    Despite the name, this is no longer a timestamp -- it's an opaque
    per-write token (see `_touchLevelMarker`) that is only ever used
    for equality comparison, never ordering.
    """
    try:
        with open(fn.markerPath(level, "touch")) as f:
            return f.read()
    except OSError:
        return 0


def readLevelSummary(fn, level):
    """
    Read the cached (n, oldest, newest) freshness summary for `level`
    on `fn` (a files.FindFiles instance), if one exists and is still
    valid -- i.e. the "touch" marker it was fenced against has not
    moved since it was written, so nothing has been (re)written for
    this level+camera+day since. Returns None on any cache miss
    (missing, corrupt, or invalidated by a write since it was cached);
    the caller must then fall back to a real scan. This is purely an
    optimization, never the source of truth.
    """
    fence = getLevelTouchTime(fn, level)
    try:
        with open(fn.markerPath(level, "done")) as f:
            summary = json.load(f)
    except (OSError, ValueError):
        return None
    if summary.get("fence") != fence:
        return None
    try:
        return summary["n"], summary["oldest"], summary["newest"]
    except KeyError:
        return None


def writeLevelSummary(fn, level, n, oldest, newest, fenceBefore, config):
    """
    Cache a freshness summary for `level` on `fn`, fenced against the
    level's "touch" marker.

    `fenceBefore` must be the value `getLevelTouchTime` returned right
    before the scan that produced n/oldest/newest started. Only
    publishes if the touch marker's mtime is still exactly that value
    -- i.e. nothing was written for this level+camera+day while the
    scan was running. This is what makes the cache safe under many
    concurrent SLURM workers: a slow scan can never resurrect stale
    data after a fast concurrent write already invalidated it, because
    *publishing* itself is conditional on the fence, not just
    invalidation-on-write. If the fence moved, the freshly computed
    summary is discarded instead of published -- the next reader just
    falls back to a real scan again, same as if nothing had ever been
    cached. A missed optimization, never a correctness gap.
    """
    if getLevelTouchTime(fn, level) != fenceBefore:
        return
    donePath = fn.markerPath(level, "done")
    payload = json.dumps(
        {"fence": fenceBefore, "n": n, "oldest": oldest, "newest": newest}
    )
    createParentDir(donePath, mode=config.dirMode)
    tmpPath = f"{donePath}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(tmpPath, "w") as f:
        f.write(payload)
    os.chmod(tmpPath, config.fileMode)
    # re-check right before publishing: closes the window between the
    # check above and the rename becoming visible to readers.
    if getLevelTouchTime(fn, level) != fenceBefore:
        tryRemovingFile(tmpPath)
        return
    os.rename(tmpPath, donePath)


def open2(file, config, mode="r", cleanUp=True, **kwargs):
    r"""
    Open file with directory creation and permissions.

    Parameters
    ----------
    file : str
        File path.
    config : dict
        Configuration settings.
    mode : str, optional
        File mode, by default "r".
    cleanUp : bool, optional
        Clean up temporary files, by default True.
    \*\*kwargs : dict
        Additional arguments for opening.

    Returns
    -------
    file handle
        Opened file handle.
    """
    createParentDir(file, mode=config.dirMode)

    if cleanUp:
        origFile = file.replace(".nc.nodata", ".nc").replace(".nc.broken.txt", ".nc")
        tryRemovingFile(origFile)
        tryRemovingFile(f"{origFile}.nodata")
        tryRemovingFile(f"{origFile}.broken.txt")
        pass
    f = open(file, mode, **kwargs)
    os.chmod(file, config.fileMode)
    log.warning(f"writing {file}")
    if mode not in ("r", "rb"):
        _touchLevelMarker(file, config)
    return f


def tryRemovingFile(file):
    """
    Try to remove a file.

    Parameters
    ----------
    file : str
        File path.
    """
    try:
        os.remove(file)
    except:
        pass
    else:
        if not file.endswith("processing.txt"):
            log.warning(f"removed {file}")
    return


def to_netcdf2(dat, config, file, **kwargs):
    r"""
    Save dataset to NetCDF with directory creation.
    Write to random file and move to final file to
    avoid errors due to race conditions or exisiting files

    Parameters
    ----------
    dat : xarray.Dataset
        Dataset to save.
    config : dict
        Configuration settings.
    file : str
        Output file name.
    \*\*kwargs : dict
        Additional arguments for saving.

    Returns
    -------
    None
    """

    from dask.diagnostics import ProgressBar

    print(f"saving {file}")

    createParentDir(file, mode=config.dirMode)
    if os.path.isfile(file):
        tryRemovingFile(file)

    #xarray bug
    for var in list(dat.coords) + list(dat.data_vars):
        if hasattr(dat[var].dtype, 'na_value'):
            dat[var] = dat[var].astype(object)

    tmpFile = f"{file}.{os.getpid()}.{uuid.uuid4().hex}.tmp.cdf"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if dat[list(dat.data_vars)[-1]].chunks is not None:
            with ProgressBar():
                res = dat.to_netcdf(tmpFile, **kwargs)
        else:
            res = dat.to_netcdf(tmpFile, **kwargs)
    os.chmod(tmpFile, config.fileMode)
    os.rename(tmpFile, file)
    log.info(f"saved {file}")

    tryRemovingFile(f"{file}.nodata")
    tryRemovingFile(f"{file}.broken.txt")
    _touchLevelMarker(file, config)

    return res


@numba.jit(nopython=True)
def linreg(x, y):
    """
    Perform linear regression.
    Turns out to be a lot faster than scipy and
    numpy code when using numba (1.7 us vs 25 us)


    Parameters
    ----------
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable.

    Returns
    -------
    tuple
        Slope and intercept of regression line.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    n = len(x)
    # formula to calculate m : x-mean_x*y-mean_y/ (x-mean_x)^2
    num = 0
    denom = 0
    for i in range(n):
        num += (x[i] - mean_x) * (y[i] - mean_y)
        denom += (x[i] - mean_x) ** 2
    slope = num / denom

    # calculate c = y_mean - m * mean_x / n
    intercept = mean_y - slope * mean_x
    return slope, intercept


def cart2pol(x, y):
    """
    Convert Cartesian coordinates to polar.

    Parameters
    ----------
    x : array_like
        X coordinates.
    y : array_like
        Y coordinates.

    Returns
    -------
    tuple
        Polar coordinates (rho, phi).
    """

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

# Entfernt ANSI escape codes (Farben, Cursor-Steuerung etc.)
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)

@taskqueue.queueable
def runCommandInQueue(IN, stdout=subprocess.DEVNULL):
    """
    Run command in queue with file locking.

    Parameters
    ----------
    IN : tuple
        Command and output file.
    stdout : file handle, optional
        Standard output, by default subprocess.DEVNULL.

    Returns
    -------
    bool
        Success status.
    """
    import portalocker

    command, fOut = IN
    tmpFile = os.path.basename("%s.processing.txt" % fOut)

    success = True
    running = False
    # with statement extended to avoid race conditions
    try:
        with portalocker.Lock(tmpFile, timeout=0) as f:
            f.write("PID & Host: %i %s\n" % (os.getpid(), socket.gethostname()))
            f.write("Command: %s\n" % command)
            f.write("Outfile: %s\n" % fOut)
            f.write("#########################\n")
            f.flush()
            log.info(f"written {tmpFile} in {os.getcwd()}")
            log.info(command)

            # proc = subprocess.Popen(shlex.split(f'bash -c "{command}"'), stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
            proc = subprocess.Popen(
                command, shell=True, stdout=stdout, stderr=subprocess.PIPE
            )

            # Poll process for new output until finished
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = strip_ansi(line.decode())
                    log.info(line)
                    f.write(line)
                    f.flush()
            for line in proc.stderr:
                line = line.decode()
                log.error(line)
                f.write(line)
                f.flush()

            proc.wait()
            exitCode = proc.returncode
            if exitCode != 0:
                success = False
                log.error(f"BROKEN {fOut} {exitCode}")
            elif not (
                os.path.isfile(fOut)
                or os.path.isfile(f"{fOut}.nodata")
                or os.path.isfile(f"{fOut}.broken.txt")
            ):
                # exited cleanly but produced none of the three recognized
                # terminal artifacts (the real output, a .nodata sentinel, or
                # its own .broken.txt). A clean exit code alone used to be
                # treated as success, so a task that ever hit an edge case
                # exiting 0 without writing anything recognizable would be
                # silently deleted from the queue with zero trace of what
                # happened. Surface it as broken instead so it's visible.
                success = False
                log.error(
                    f"BROKEN {fOut} exited 0 but produced no output/nodata/broken.txt"
                )
            else:
                log.info(f"SUCCESS {fOut} {exitCode}")

            # flush and sync to filesystem
            f.flush()
            os.fsync(f.fileno())

    except portalocker.LockException:
        log.info(f"{fOut} RUNNING")
        success = True
        running = True

    if not success:
        shutil.copy(tmpFile, "%s.broken.txt" % tmpFile)
        tryRemovingFile(fOut)
        tryRemovingFile(f"{fOut}.nodata")
        try:
            createParentDir(fOut)
            shutil.copy(tmpFile, "%s.broken.txt" % fOut)
            # This write bypasses open2/to_netcdf2 (it's a plain
            # shutil.copy, not a real product write), so it would
            # otherwise never bump the level's touch marker -- leaving
            # _freshnessSummary's cached (oldest, newest) permanently
            # stuck at whatever it was before this failure, even though
            # a brand new .broken.txt with today's mtime now exists.
            # That silently reproduces the exact allDone caching bug for
            # every level that ever fails via the task queue (e.g. a
            # known-bad metaRotation day: it succeeds at being marked
            # broken, but keeps looking "not up to date" forever
            # afterwards and gets regenerated -- harmlessly, since it
            # just re-fails and re-writes the same broken.txt, but
            # noisily -- on every single DAG check). Every generated
            # command's settings yaml is always the first positional arg
            # right after `-m VISSSlib <call>` (see
            # products.py's _commandTemplateDaily/_commandTemplateL1), so
            # recover it from the command string to bump the marker here
            # too. Best-effort: ad hoc commands pushed straight into the
            # queue (see reference_task_queue_submission) or the allDone
            # touch command don't match and are silently skipped, same as
            # _levelMarkerPaths already does for anything it doesn't
            # recognize.
            settingsMatch = re.search(r"-m\s+VISSSlib\s+\S+\s+(\S+\.ya?ml)", command)
            if settingsMatch is not None:
                _touchLevelMarker(fOut, readSettings(settingsMatch.group(1)))
        except:
            pass

    if not running:
        tryRemovingFile(tmpFile)

    return success


def _queueHasLeasableTask(tq):
    """
    Whether the queue currently has at least one task with an expired
    (or no) lease, i.e. one `tq.poll` could actually lease right now.

    `tq.is_empty()` only checks whether the queue directory has zero
    files at all, so it stays False as long as a single task is still
    leased (in-flight, or abandoned by a crashed worker) even though
    nothing is actually leasable. Calling `tq.poll` in that state makes
    it busy-spin internally (`QueueEmptyError` never grows its own
    `tries` counter, so its backoff stays ~0-1s) for up to
    `leaseSeconds`, which is exactly what blocks a SLURM node once
    `leaseSeconds` is hours long. Callers should use this instead of
    `tq.is_empty()` to decide whether to poll at all.
    """
    from taskqueue.file_queue_api import get_timestamp, nowfn

    now = nowfn()
    try:
        for entry in os.scandir(tq.api.queue_path):
            if get_timestamp(entry.name) <= now:
                return True
    except FileNotFoundError:
        pass
    return False


def worker1(queue, ww=0, status=None, waitTime=5, leaseSeconds=21600, maxIdleSeconds=60):
    """
    Worker function for processing queue items.

    Parameters
    ----------
    queue : str
        Queue name.
    ww : int, optional
        Worker number, by default 0.
    status : array, optional
        Status array, by default None.
    waitTime : int, optional
        Wait time between checks, by default 5.
    leaseSeconds : int, optional
        Visibility timeout for a leased task, in seconds, by default
        21600 (6h). `tq.poll` leases one task, then runs it fully
        synchronously (`runCommandInQueue` blocks on the subprocess for
        the task's entire runtime) before leasing the next -- the lease
        is never renewed while a task is executing. Once `leaseSeconds`
        elapses, other workers treat the task as abandoned and lease it
        again even though the original worker is still correctly,
        actively running it, causing duplicate concurrent execution of
        the same command (the file lock in `runCommandInQueue` then
        makes the duplicates no-op quickly, which drains the *queue*
        long before the *task* actually finishes -- queue-empty stops
        being a valid "done" signal). Must comfortably exceed the
        slowest realistic single-file runtime for whatever commands
        this queue carries, not just the typical one.
    maxIdleSeconds : int, optional
        Every worker in this job must simultaneously report idle
        (status all 0) for this many seconds, checked every `waitTime`
        seconds, before this worker gives up and exits, by default 60.
        This is deliberately short -- SLURM jobs should free their
        allocation for other cluster users promptly once a queue is
        genuinely empty, rather than sit idle. It does mean a worker
        can exit during a brief real gap between two submission stages
        (e.g. one script finishes draining a batch, then computes what
        to submit next) -- that CPU slot is then gone until a new
        `workers()`/SLURM job is launched. Guarding against that is the
        job of whoever is orchestrating multi-stage submissions (launch
        workers right before submitting each stage), not this worker's
        patience.

    Returns
    -------
    None
    """
    log.info(f"starting worker {ww} for {queue}")
    time.sleep(ww / 5.0)  # to avoid race conditions
    tq = taskqueue.TaskQueue(f"fq://{queue}")
    out = None
    idleRounds = 0
    idleRoundsBeforeExit = max(1, round(maxIdleSeconds / waitTime))
    while True:
        if _queueHasLeasableTask(tq):
            if status is not None:
                status[ww] = 1
            try:
                out = tq.poll(
                    verbose=True,
                    tally=True,
                    stop_fn=lambda: not _queueHasLeasableTask(tq),
                    lease_seconds=leaseSeconds,
                    backoff_exceptions=[BlockingIOError],
                )
            except:
                pass
            finally:
                if status is not None:
                    status[ww] = 0
        else:
            log.warning(f"worker {ww} queue {queue} empty or fully leased elsewhere")
        if status is not None:
            if np.all([ss == 0 for ss in status]):
                idleRounds += 1
                if idleRounds >= idleRoundsBeforeExit:
                    log.warning(
                        f"do not restart worker {ww} because all empty for "
                        f"{idleRounds} consecutive rounds {[status[i] for i in range(len(status))]}"
                    )
                    break
            else:
                idleRounds = 0
            summary = [status[i] for i in range(len(status))]
        else:
            summary = ""
        log.info(f"restart worker {ww} {summary}")
        time.sleep(waitTime)

    return out


def workers(
    queue,
    nJobs=os.cpu_count(),
    waitTime=60,
    join=True,
    leaseSeconds=21600,
    maxIdleSeconds=60,
):
    """
    Start multiple worker processes.

    Parameters
    ----------
    queue : str
        Queue name.
    nJobs : int, optional
        Number of jobs, by default os.cpu_count().
    waitTime : int, optional
        Wait time between checks, by default 60.
    join : bool, optional
        Join processes, by default True.
    leaseSeconds : int, optional
        Passed through to `worker1` -- see its docstring for why this
        must exceed the slowest realistic task runtime, by default
        21600 (6h).
    maxIdleSeconds : int, optional
        Passed through to `worker1` -- see its docstring; by default 60.
        Callers that don't care about freeing a real SLURM allocation
        promptly (e.g. tests walking many DAG stages back to back) can
        pass a much smaller value to avoid paying this wait per stage.

    Returns
    -------
    list
        Worker processes.
    """
    # for communication between subprocesses
    print(f"starting {nJobs} workers")
    status = multiprocessing.Array("i", [0] * nJobs)
    workerList = []
    for ww in range(nJobs):
        x = multiprocessing.Process(
            target=worker1,
            args=(queue,),
            kwargs={
                "ww": ww,
                "status": status,
                "waitTime": waitTime,
                "leaseSeconds": leaseSeconds,
                "maxIdleSeconds": maxIdleSeconds,
            },
        )
        x.start()
        workerList.append(x)
    if join:
        # Reap whichever processes have actually finished, not in launch
        # order: `[x.join() for x in workerList]` blocks on workerList[0]
        # first, so any later-indexed process that already exited sits
        # as a zombie (<defunct>, invisible to `ps` as doing real work)
        # until every earlier process's blocking join() finally returns
        # -- which can be a very long time if an early worker is still
        # genuinely mid-task. Polling is_alive() and joining only the
        # already-dead ones reaps them immediately regardless of exit
        # order.
        remaining = list(workerList)
        while remaining:
            remaining = [x for x in remaining if x.is_alive()]
            for x in workerList:
                if not x.is_alive():
                    x.join(timeout=0)
            if remaining:
                time.sleep(1)
    return workerList


def copyCurrentQuicklook(level, ff):
    """
    Copy the latest quicklook file to the current quicklook location.

    This function copies the most recently generated quicklook file for a given
    processing level to a corresponding "current" location. This is typically used
    to maintain a symlink or copy of the most recent quicklook image for easy access.

    Parameters
    ----------
    level : str
        Processing level for which to copy the quicklook (e.g., 'level1detect').
    ff : object
        File finder object containing file information and quicklook paths.

    Returns
    -------
    None
        This function does not return a value but performs file copying operations.

    Notes
    -----
    The function only copies the quicklook file if the date matches today's date.
    This prevents overwriting previous quicklook files with potentially outdated
    images from previous days.
    """
    fOut = ff.quicklook[level]

    if ff.datetime.date() == datetime.datetime.today().date():
        try:
            shutil.copy(fOut, ff.quicklookCurrent[level])
        except PermissionError:
            log.error(f"No permission to write {fOut}")

    return


def _isLevelGroup(item):
    """True for a (files.FindFiles, level) pair, as opposed to a plain
    file path -- see checkForExisting's events/parents docstring."""
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[1], str)
        and hasattr(item[0], "markerPath")
    )


def _newestMtime(items):
    """
    Newest mtime across `items`, a list that may freely mix plain file
    paths with (files.FindFiles, level) pairs (see checkForExisting).
    A (fn, level) pair is resolved through the on-disk freshness cache
    (readLevelSummary) when available -- no glob, no per-file stat --
    falling back to a real scan of that level's files (fn.listFilesExt)
    only on a cache miss. Empty/None `items` -> 0.
    """
    newest = 0.0
    for item in items or []:
        if _isLevelGroup(item):
            fn, level = item
            cached = readLevelSummary(fn, level)
            if cached is not None:
                newest = max(newest, cached[2])  # newest
                continue
            item = fn.listFilesExt(level)
        else:
            item = [item]
        if item:
            newest = max(newest, max(os.path.getmtime(f) for f in item))
    return newest


def checkForExisting(
    ffOut, level0=None, events=None, parents=None, breakpointLevel=None,
    fL=None, level=None,
):
    """
    Check if file exists and is up-to-date including potential parents.

    Parameters
    ----------
    ffOut : str
        Output file path.
    level0 : list, optional
        Level 0 data files, by default None.
    events, parents : list, optional
        What `ffOut` must not be older than, by default None. Each
        element is either a plain file path (stat'd directly, as
        before), or a (files.FindFiles, level) pair -- e.g.
        `(fL, "level1match")` in place of pre-globbing
        `fL.listFilesExt("level1match")` yourself -- resolved through
        the on-disk freshness-summary cache when possible instead of
        globbing and stat'ing every one of that level's files (see
        _newestMtime/readLevelSummary). The two forms can be mixed
        freely within one list. Combines with `fL`/`level` below if
        both are given (the registry-resolved parents are appended to
        this list, not a replacement for it).
    fL : files.FindFiles, optional
        Given together with `level`, resolve `level`'s declared
        LEVEL_REGISTRY parents (see resolveLevelParents) for `fL`'s own
        case/camera/config and check `ffOut` against those too -- the
        caller doesn't hand-write (and risk drifting out of sync with)
        its own parents/events list for whatever LEVEL_REGISTRY already
        declares. Only makes sense for a level checked at day/whole-file
        granularity; a per-file check against a specific matching
        upstream file (e.g. matchParticles against one particular
        level1detect file) still needs an explicit `parents=`/`events=`
        entry instead, since LEVEL_REGISTRY only knows "day's worth of
        level X", not "this one specific file".
    level : str, optional
        Which level's declared parents to resolve via `fL` (see above).
    breakpointLevel : str, optional
        If given, treat `ffOut` as needing regeneration when its mtime
        is older than `tools.reprocessBreakpoint(breakpointLevel)` --
        see REPROCESS_AFTER. By default None (no such check).

    Returns
    -------
    bool
        True if file should be regenerated.
    """
    if not os.path.isfile(ffOut):
        # file does not exist yet
        return False
    minMtime = reprocessBreakpoint(breakpointLevel) if breakpointLevel else None
    if minMtime is not None and os.path.getmtime(ffOut) < minMtime:
        log.warning(
            f"file exists but predates the {breakpointLevel} reprocessing "
            f"breakpoint (REPROCESS_AFTER), redoing {ffOut}"
        )
        return False
    if level0 is not None:
        if len(level0) == 0:
            log.warning(f"no level0 data {ffOut}")
            return True
    if events is not None:
        if os.path.getmtime(ffOut) < _newestMtime(events):
            log.warning(f"file exists but older than event file, redoing {ffOut}")
            return False
    if fL is not None and level is not None:
        parents = (parents or []) + resolveLevelParents(level, fL.camera, fL.case, fL.config)
    if parents is not None:
        if os.path.getmtime(ffOut) < _newestMtime(parents):
            log.warning(f"file exists but older than parents files, redoing {ffOut}")
            return False
    log.warning(f"output file exists already: {ffOut}")
    return True


def unpackQualityFlags(quality, doubleTimestamps=False):
    """
    Unpack quality flags into boolean array.

    Parameters
    ----------
    quality : xarray.DataArray
        Quality flags.
    doubleTimestamps : bool, optional
        Double timestamps for plotting, by default False.

    Returns
    -------
    xarray.DataArray
        Expanded quality flags.
    """
    flags = xr.DataArray(
        [
            "recordingFailed",
            "processingFailed",
            "cameraBlocked",
            "blowingSnow",
            "obervationsDiffer",
            "trackingIncomplete",
            "zResidualTooWide",
        ],
        dims=["flag"],
        name="flag",
    )
    qualityExpanded = np.unpackbits(
        quality.values[..., np.newaxis], axis=-1, count=len(flags)
    )
    qualityExpanded = xr.DataArray(qualityExpanded, coords=[quality.time, flags])

    # trick for plotting
    if doubleTimestamps:
        qualityExpanded2 = xr.DataArray(
            qualityExpanded.values,
            coords=[qualityExpanded.time + np.timedelta64(5, "m"), flags],
        )
        qualityExpanded = xr.concat(
            (qualityExpanded, qualityExpanded2), dim="time"
        ).sortby("time")
    return qualityExpanded


def timestamp2str(ts):
    """
    Convert timestamp to string.

    Parameters
    ----------
    ts : int
        Timestamp.

    Returns
    -------
    str
        Formatted timestamp string.
    """
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def copyLastMetaRotation(config, fromCase, toCase):
    """
    Copy the last meta rotation file from one case to another.

    This function copies the most recent meta rotation file from a source case
    to a destination case, updating the timestamp in the filename accordingly.

    Parameters
    ----------
    config : dict
        Configuration settings.
    fromCase : str
        Source case identifier (e.g., "20230101").
    toCase : str
        Destination case identifier (e.g., "20230102").

    Returns
    -------
    xarray.Dataset or None
        The copied meta rotation dataset or None if no rotation file exists.

    Notes
    -----
    This function is decorated with @loopify, which means it will automatically
    iterate over all cases if a range is provided. For example, if fromCase is
    "20230101-20230103", it will process all three cases.
    """
    config = readSettings(config)
    ff = files.FindFiles(fromCase, config.leader, config)
    if len(ff.listFiles("metaRotation")) == 0:
        log.error("no rotation file yet")
        return None

    fname = ff.listFiles("metaRotation")[0]
    metaRot = xr.open_dataset(fname)
    metaRotLast = metaRot.isel(file_starttime=-1)
    metaRotLast.attrs = {}

    ffnew = files.FindFiles(toCase, config.leader, config)
    newTime = ffnew.datetime64 + np.timedelta64(1439, "m")
    fnameNew = fname.replace(f"/{ff.year}/", f"/{ffnew.year}/").replace(
        fromCase, toCase
    )
    metaRotLast = metaRotLast.assign_coords(file_starttime=[newTime])
    metaRotLast
    to_netcdf2(metaRotLast, config, fnameNew)


def reportLastFiles(
    settings,
    writeFile=True,
    nameFile=False,
    products=[
        "level0txt",
        "level0",
        "metaFrames",
        "level1detect",
        "metaRotation",
        "level1match",
        "level1track",
        "level2match",
        "level2track",
    ],
):
    """
    report last available files for various processing levels

    Parameters
    ----------
    settings : str
        VISSS settings YAML file
    writeFile : bool, optional
        write output to file (the default is True)
    nameFile : bool, optional
        name last files (the default is False)
    products : list, optional
        list of products to be summarized (the default is [ "level0txt", "level0", "metaFrames", "level1detect", "metaRotation", "level1match", "level1track", "level2match", "level2track", ])

    """
    config = readSettings(settings)
    days = getDateRange(0, config, endYesterday=False)[::-1]

    cameras = [config.follower, config.leader]
    output = ""

    output += "#" * 80
    output += "\n"
    output += (
        f"Last available files for {config.site} at {datetime.datetime.utcnow()} UTC\n"
    )
    output += "#" * 80
    output += "\n"

    for prod in products:
        for camera in cameras:
            if camera == config.follower and (
                (prod in ["level1match", "level1track", "metaRotation"])
                or prod.startswith("level2")
            ):
                continue

            foundLastFile, completeCase, lastFile, lastFileTime, _ = files.findLastFile(
                config, prod, camera
            )

            output += f"{prod.ljust(14)} {(camera.split('_')[0]).ljust(8)} last full day:'{completeCase}' last file:'{lastFileTime}'"
            if nameFile:
                output += f" {lastFile}"
            output += "\n"

    output += "#" * 80
    output += "\n"
    output += f"VISSSlib version {__version__}\n"

    if writeFile:
        fOut = f"{config['pathQuicklooks'].format(version=__version__,site=config['site'], level='')}/{'productReport'}_{config['site']}.html"
        with open2(fOut, config, "w") as f:
            f.write("<html><pre>\n")
            f.write(output)
            f.write("</pre></html>\n")

    return output


def _create_parser():
    """Create the argument parser for VISSS processing pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m VISSSlib",
        description="VISSS data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m VISSSlib metadata.createEvent settings.yaml nDays --camera leader
  python -m VISSSlib detection.detectParticles file.txt settings.yaml --skip-existing
  python -m VISSSlib products.processAll settings.yaml YYYYMMDD-YYYYMMDD

For information about the commands, run
  python -m VISSSlib command --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Processing command")
    subparsers.required = True

    # Helper to add common arguments
    def _add_std_args(
        p, has_fname=False, has_case=True, has_camera=False, has_skip=True
    ):
        """Add standard arguments to a subparser."""
        skip_default = False
        p.add_argument("settings", help="Settings YAML file")
        if has_fname:
            p.add_argument("fname", help="Input file path")
        if has_case:
            p.add_argument(
                "case",
                help="Number of days going back or date string 'YYYYMMDD' or "
                "'YYYYMMDD-YYYYMMDD' or 'YYYYMMDD,YYYYMMDD,YYYYMMDD'",
            )
        if has_camera:
            p.add_argument("--camera", default="all", help="Camera name (default: all)")
        if has_skip:
            p.add_argument(
                "--skip-existing",
                action="store_true",
                default=skip_default,
                help=f"Skip if output already exists (default: {skip_default})",
            )

    # Metadata commands
    p = subparsers.add_parser("metadata.createEvent", help="Create event metadata")
    _add_std_args(p, has_camera=True)

    p = subparsers.add_parser(
        "metadata.createMetaFrames", help="Create metadata frames"
    )
    _add_std_args(p, has_camera=True)

    # Quicklook commands
    p = subparsers.add_parser(
        "quicklooks.createLevel1detectQuicklook",
        help="Create Level 1 detection quicklook",
    )
    _add_std_args(p, has_camera=True)

    p = subparsers.add_parser(
        "quicklooks.createLevel1matchParticlesQuicklook",
        help="Create Level 1 matching quicklook",
    )
    _add_std_args(p)

    p = subparsers.add_parser(
        "quicklooks.metaRotationQuicklook",
        help="Create camera rotation coefficient quicklook",
    )
    _add_std_args(p)

    p = subparsers.add_parser(
        "quicklooks.metaFramesQuicklook", help="Create metaFrames quicklook"
    )
    _add_std_args(p, has_camera=True)

    p = subparsers.add_parser(
        "quicklooks.level0Quicklook", help="Create Level 0 quicklook"
    )
    _add_std_args(p)

    p = subparsers.add_parser(
        "quicklooks.metaRotationYearlyQuicklook",
        help="Create yearly rotation quicklook",
    )
    _add_std_args(p, has_fname=False, has_case=False, has_camera=False, has_skip=False)
    p.add_argument("year", help="Year to process")

    # Detection commands
    p = subparsers.add_parser("detection.detectParticles", help="Detect particles")
    _add_std_args(p, has_fname=True, has_case=False)

    # Matching commands
    p = subparsers.add_parser(
        "matching.createMetaRotation", help="Create metadata rotation"
    )
    _add_std_args(p)

    p = subparsers.add_parser("matching.matchParticles", help="Match particles")
    _add_std_args(p, has_fname=True, has_case=False)

    # Tracking commands
    p = subparsers.add_parser("tracking.trackParticles", help="Track particles")
    _add_std_args(p, has_fname=True, has_case=False)

    # Distribution commands
    p = subparsers.add_parser(
        "distributions.createLevel2detect", help="Create Level 2 detection distribution"
    )
    _add_std_args(p, has_camera=True)

    p = subparsers.add_parser(
        "distributions.createLevel2match", help="Create Level 2 matching distribution"
    )
    _add_std_args(p)

    p = subparsers.add_parser(
        "distributions.createLevel2track", help="Create Level 2 tracking distribution"
    )
    _add_std_args(p)

    # Level 3 commands
    p = subparsers.add_parser(
        "level3.retrieveCombinedRiming", help="Retrieve combined riming"
    )
    _add_std_args(p)

    # Tools commands
    p = subparsers.add_parser("tools.reportLastFiles", help="Report last files")
    _add_std_args(p, has_fname=False, has_case=False, has_camera=False, has_skip=False)
    p = subparsers.add_parser(
        "tools.copyLastMetaRotation", help="Copy last metadata rotation"
    )
    _add_std_args(p, has_fname=False, has_case=False, has_camera=False, has_skip=False)
    p.add_argument("from_case", help="Source case")
    p.add_argument("to_case", help="Destination case")

    # Products commands
    p = subparsers.add_parser("products.submitAll", help="Submit all products")
    _add_std_args(p, has_fname=False, has_case=True, has_camera=False, has_skip=False)
    p.add_argument("task_queue", help="Task queue directory")

    p = subparsers.add_parser("products.processAll", help="Process all products")
    _add_std_args(p)

    p = subparsers.add_parser("products.processRealtime", help="Process realtime")
    _add_std_args(p)

    # Worker
    p = subparsers.add_parser("worker", help="Start task worker")
    p.add_argument("task_queue", help="Task queue directory")
    p.add_argument(
        "--n-jobs", type=int, default=None, help="Number of jobs (default: CPU count)"
    )
    return parser


def ipython_debug(exception):
    from IPython.core.debugger import Pdb

    # exception.__traceback__ is the map back to the actual error
    Pdb().interaction(None, exception.__traceback__)
