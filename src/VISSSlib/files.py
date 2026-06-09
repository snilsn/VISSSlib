# -*- coding: utf-8 -*-

import datetime
import functools
import glob
import json
import logging
import os
import sys
from copy import deepcopy

import numpy as np
import xarray as xr
from addict import Dict
from loguru import logger as log

from . import __version__, metadata
from .tools import (
    DictNoDefault,
    _aggregate,
    getCaseRange,
    globList,
    nicerNames,
    otherCamera,
    readSettings,
)

# to do merge to single class using different constructors with @classmethod?

dailyLevels = [
    "metaEvents",
    "metaRotation",
    "level2detect",
    "level2match",
    "level2track",
    "level3combinedRiming",
    "allDone",
]
hourlyLevels = []
fileLevels = [
    "level1detect",
    "level1match",
    "level1track",
    # "level1shape",
    "metaFrames",
    "metaDetection",
    "imagesL1detect",
]  # , "metaFixedCaptureId"]
quicklookLevelsSep = [
    "level0",
    "metaFrames",
    "metaEvents",
    "level1detect",
    "level1match",
    "level1matchParticles",
    # "level1shape",
    "metaRotation",
    "level2detect",
]
quicklookLevelsComb = [
    "level2match",
    "level2track",
    "level3combinedRiming",
]
imageLevels = ["imagesL1detect"]


def findLastFile(config, prod, camera):
    """
    Find the last available file and related metadata for a given product and camera.

    Parameters
    ----------
    config : dict or str
        Configuration settings or path to configuration file.
    prod : str
        Product name (e.g., 'level1detect').
    camera : str
        Camera identifier (e.g., 'leader' or 'follower').

    Returns
    -------
    tuple
        Tuple containing:
        - foundLastFile: bool indicating if last file was found
        - lastCase: str, the last case found
        - lastFile: str, the path to the last file
        - lastFileTime: datetime obj, the timestamp of the last file

    Notes
    -----
    This function iterates through recent cases to find the most recent file
    of the requested product and camera combination.
    """
    config = readSettings(config)
    cases = getCaseRange(0, config, endYesterday=False)[::-1]

    foundLastFile = False
    foundComplete = False
    lastCase = "n/a"
    lastFileTime = "n/a"
    lastFile = "n/a"
    for case in cases:
        # find files
        ff = FindFiles(case, camera, config)
        if not foundLastFile:
            fnames = ff.listFiles(prod)
            if len(fnames) > 0:
                lastFile = fnames[-1]
                try:
                    f1 = FilenamesFromLevel(fnames[-1], config)
                except ValueError:
                    f1 = Filenames(fnames[-1], config)
                foundLastFile = True
                lastFileTime = f1.datetime

        if not foundComplete:
            foundComplete = ff.isComplete(
                prod, ignoreBrokenFiles=True, requireL0Files=True
            )
        else:
            break
        lastCase = case
    return foundLastFile, lastCase, lastFile, lastFileTime


class FindFiles(object):
    """
    Class to manage and locate files based on case and camera specifications.

    This class handles finding and managing file paths for different processing
    levels of VISSS data, including level0, level1, level2, and level3 products.
    It provides methods to search, validate, and organize file locations based
    on timestamps and camera configurations.

    Attributes
    ----------
    case : str
        The case identifier (YYYYMMDD).
    camera : str
        The camera identifier (e.g., 'leader' or 'follower').
    config : dict
        Configuration dictionary containing paths and settings.
    version : str
        Version string for the VISSS processing.
    year : str
        Year component of the case.
    month : str
        Month component of the case.
    day : str
        Day component of the case.
    hour : str
        Hour component of the case (if applicable).
    minute : str
        Minute component of the case (if applicable).
    datetime : datetime.datetime
        Parsed datetime from the case identifier.
    datetime64 : numpy.datetime64
        Parsed datetime64 from the case identifier.
    logpath : str
        Path to log files for this camera.
    outpath : dict
        Dictionary mapping processing level to output directory path.
    fnamesPattern : dict
        Dictionary mapping processing level to filename pattern.
    fnamesPatternExt : dict
        Dictionary mapping processing level to expanded filename pattern.
    fnamesDaily : dict
        Dictionary mapping processing level to daily filename pattern.
    fnamesHourly : dict
        Dictionary mapping processing level to hourly filename pattern.
    quicklook : dict
        Dictionary mapping processing level to quicklook image filename.
    quicklookCurrent : dict
        Dictionary mapping processing level to current quicklook image filename.
    quicklookPath : dict
        Dictionary mapping processing level to quicklook image path.
    """

    def __init__(self, case, camera, config, version=__version__):
        """
        Initialize FindFiles object with case, camera, and config details.

        Parameters
        ----------
        case : str
            Case identifier (YYYYMMDD format).
        camera : str
            Camera identifier ('leader' or 'follower').
        config : dict or str
            Configuration dictionary or path to configuration file.
        version : str, optional
            Version string, defaults to __version__.

        Notes
        -----
        This constructor builds file paths for different processing levels
        and creates patterns for locating various types of files based on
        the case, camera, and configuration.
        """
        import pandas as pd

        config = readSettings(config)

        if type(case) is not str:
            self.case = pd.to_datetime(case).strftime("%Y%m%d-%H%M%S")
        else:
            self.case = case
        if camera in ["leader", "follower"]:
            self.camera = config[camera]
        else:
            self.camera = camera
        self.config = config
        self.version = version

        computerDict = {}
        for computer1, camera1 in zip(config["computers"], config["instruments"]):
            computerDict[camera1] = computer1
        self.computer = computerDict[self.camera]

        self.year = self.case[:4]
        self.month = self.case[4:6]
        self.day = self.case[6:8]

        self.hour = self.case[9:11]
        self.minute = self.case[11:13]

        try:
            self.timestamps = self.case[9:]
        except IndexError:
            self.timestamps = None

        try:
            self.datetime = datetime.datetime.strptime(
                self.case.ljust(15, "0"), "%Y%m%d-%H%M%S"
            )
        except ValueError:
            self.datetime = datetime.datetime.strptime(self.case, "%Y%m%d")
        self.datetime64 = np.datetime64(self.datetime, "ns")

        self.logpath = "%s/%s_%s_%s/" % (
            config.path.format(level="logs"),
            self.computer,
            config.visssGen,
            self.camera,
        )

        outpath = "%s/%s/%s/%s" % (config["pathOut"], self.year, self.month, self.day)
        outpathDaily = "%s/%s" % (config["pathOut"], self.year)

        self.outpath = DictNoDefault({})
        for dL in fileLevels + hourlyLevels:
            self.outpath[dL] = outpath.format(
                site=config.site, level=dL, version=self.version
            )
        for dL in dailyLevels:
            self.outpath[dL] = outpathDaily.format(
                site=config.site, level=dL, version=self.version
            )
        self.outpath["level0"] = (
            config["path"].format(site=config["site"], level="level0")
            + f'/{self.computer}_{config["visssGen"]}_{self.camera}/{self.year}/{self.month}/{self.day}'
        )

        self.fnamesPattern = DictNoDefault({})
        for dL in fileLevels:
            self.fnamesPattern[dL] = "%s/%s_V%s*%s*%s*.nc" % (
                self.outpath[dL],
                dL,
                version,
                self.camera,
                self.case,
            )
        # overwrite for level0
        if config["nThreads"] is None:
            self.fnamesPattern.level0 = "%s/*%s*.%s" % (
                self.outpath.level0,
                self.case,
                config["movieExtension"],
            )
            self.fnamesPattern.level0txt = "%s/*%s*.%s" % (
                self.outpath.level0,
                self.case,
                "txt",
            )
            self.fnamesPattern.level0jpg = "%s/*%s*.%s" % (
                self.outpath.level0,
                self.case,
                "jpg",
            )
        else:
            self.fnamesPattern.level0 = "%s/*%s*_0.%s" % (
                self.outpath.level0,
                self.case,
                config["movieExtension"],
            )
            self.fnamesPattern.level0txt = "%s/*%s*_0.%s" % (
                self.outpath.level0,
                self.case,
                "txt",
            )
            self.fnamesPattern.level0jpg = "%s/*%s*_0.%s" % (
                self.outpath.level0,
                self.case,
                "jpg",
            )
        self.fnamesPattern.level0status = f"{self.outpath['level0']}/*_{config['visssGen']}_{self.camera}_{self.case}_status.txt"
        for dL in dailyLevels:
            self.fnamesPattern[dL] = "%s/%s_V%s_*%s*%s%s%s.nc" % (
                self.outpath[dL],
                dL,
                version,
                self.camera,
                self.year,
                self.month,
                self.day,
            )
        for dL in hourlyLevels:
            self.fnamesPattern[dL] = "%s/%s_V%s_*%s*%s%s%s-%s%s.nc" % (
                self.outpath[dL],
                dL,
                version,
                self.camera,
                self.year,
                self.month,
                self.day,
                self.hour,
                self.minute,
            )
        self.fnamesPattern["imagesL1detect"] = self.fnamesPattern[
            "imagesL1detect"
        ].replace(".nc", ".bin")
        self.fnamesPattern.level3combinedRiming = (
            self.fnamesPattern.level3combinedRiming.replace(
                "level3combinedRiming",
                f"level3combinedRiming{config.level3.combinedRiming.extraFileStr}",
            )
        )

        self.fnamesPatternExt = DictNoDefault({})
        for dL in fileLevels + dailyLevels + hourlyLevels:
            self.fnamesPatternExt[dL] = "%s/%s_V%s_*%s*%s*nc.[b,n][r,o]*" % (
                self.outpath[dL],
                dL,
                version,
                self.camera,
                self.case,
            )  # finds broken & nodata
        self.fnamesPatternExt.level3combinedRiming = (
            self.fnamesPatternExt.level3combinedRiming.replace(
                "level3combinedRiming",
                f"level3combinedRiming{config.level3.combinedRiming.extraFileStr}",
            )
        )
        self.fnamesPatternExt.level0txt = ""
        self.fnamesPatternExt.level0jpg = ""
        self.fnamesPatternExt.level0 = ""

        self.fnamesDaily = DictNoDefault({})
        for dL in dailyLevels:
            self.fnamesDaily[dL] = "%s/%s_V%s_%s_%s_%s_%s_%s%s%s.nc" % (
                self.outpath[dL],
                dL,
                version,
                config.site,
                self.computer,
                config["visssGen"],
                self.camera,
                self.year,
                self.month,
                self.day,
            )
        self.fnamesDaily.level3combinedRiming = (
            self.fnamesDaily.level3combinedRiming.replace(
                "level3combinedRiming",
                f"level3combinedRiming{config.level3.combinedRiming.extraFileStr}",
            )
        )

        self.fnamesHourly = DictNoDefault({})
        for dL in hourlyLevels:
            self.fnamesHourly[dL] = "%s/%s_V%s_%s_%s_%s_%s_%s%s%s-%s%s.nc" % (
                self.outpath[dL],
                dL,
                version,
                config.site,
                self.computer,
                config["visssGen"],
                self.camera,
                self.year,
                self.month,
                self.day,
                self.hour,
                self.minute,
            )

        self.quicklook = DictNoDefault({})
        self.quicklookCurrent = DictNoDefault({})
        self.quicklookPath = DictNoDefault({})
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            self.quicklookPath[
                qL
            ] = f'{config["pathQuicklooks"].format(version=version, site=config["site"], level=qL)}/{self.year}'

        for qL in quicklookLevelsSep:
            if self.hour == "":
                self.quicklook[
                    qL
                ] = f"{self.quicklookPath[qL]}/{qL}_V{version.split('.')[0]}_{config['site']}_{nicerNames(self.camera).split('_')[0]}_{self.year}{self.month}{self.day}.png"
            else:
                self.quicklook[
                    qL
                ] = f"{self.quicklookPath[qL]}/{qL}_V{version.split('.')[0]}_{config['site']}_{nicerNames(self.camera).split('_')[0]}_{self.year}{self.month}{self.day}T{self.hour}.png"
            self.quicklookCurrent[
                qL
            ] = f"{config['pathQuicklooks'].format(version=version,site=config['site'], level=qL)}/{qL}_{config['site']}_{nicerNames(self.camera).split('_')[0]}_current.png"
        for qL in quicklookLevelsComb:
            if self.hour == "":
                self.quicklook[
                    qL
                ] = f"{self.quicklookPath[qL]}/{qL}_V{version.split('.')[0]}_{config['site']}_{self.year}{self.month}{self.day}.png"
            else:
                self.quicklook[
                    qL
                ] = f"{self.quicklookPath[qL]}/{qL}_V{version.split('.')[0]}_{config['site']}_{self.year}{self.month}{self.day}T{self.hour}.png"
            self.quicklookCurrent[
                qL
            ] = f"{config['pathQuicklooks'].format(version=version,site=config['site'], level=qL)}/{qL}_{config['site']}_current.png"
        self.quicklook.level3combinedRiming = (
            self.quicklook.level3combinedRiming.replace(
                "level3combinedRiming",
                f"level3combinedRiming{config.level3.combinedRiming.extraFileStr}",
            )
        )

    def __repr__(self):
        return json.dumps(self.fnamesPattern, indent=4)

    @property
    def yesterday(self):
        """
        Get the previous day's case identifier.

        Returns
        -------
        str
            Date in YYYYMMDD format for the day before this case.
        """
        return self.yesterdayObject.case.split("-")[0]

    @property
    def yesterdayObject(self):
        """
        Get the FindFiles object for the previous day.

        Returns
        -------
        FindFiles
            FindFiles object for the previous day.
        """
        import pandas as pd

        return FindFiles(
            pd.to_datetime(self.datetime64 - np.timedelta64(24, "h")).strftime(
                "%Y%m%d"
            ),
            self.camera,
            self.config,
        )

    @property
    def tomorrow(self):
        """
        Get the next day's case identifier.

        Returns
        -------
        str
            Date in YYYYMMDD format for the day after this case.
        """
        return self.tomorrowObject.case.split("-")[0]

    @property
    def tomorrowObject(self):
        """
        Get the FindFiles object for the next day.

        Returns
        -------
        FindFiles
            FindFiles object for the next day.
        """
        import pandas as pd

        return FindFiles(
            pd.to_datetime(self.datetime64 + np.timedelta64(24, "h")).strftime(
                "%Y%m%d"
            ),
            self.camera,
            self.config,
        )

    @functools.cache
    def getEvents(self, skipExisting=True):
        """
        Retrieve and optionally create the event dataset for this case and camera.

        Parameters
        ----------
        skipExisting : bool, optional
            Whether to skip processing if files already exist, defaults to True

        Returns
        -------
        tuple(str, xarray.Dataset)
            Tuple of (event_filename, event_dataset) or (None, None) if not found
        """
        # just in case it is missing
        metadata.createEvent(
            self.case,
            self.camera,
            self.config,
            skipExisting=skipExisting,
            quiet=True,
        )
        try:
            eventFile = self.listFiles("metaEvents")[0]
        except IndexError:
            print("no event file")
            return None, None

        eventDat = xr.open_dataset(eventFile).load()
        # opening it several times can cause segfaults
        eventDat.close()

        return eventFile, eventDat

    @functools.cache
    def listFiles(self, level):
        """
        List all files of a specific processing level for this case and camera.

        Parameters
        ----------
        level : str
            Processing level to list files for (e.g., 'level1detect').

        Returns
        -------
        list
            List of full paths to files matching the pattern.

        Raises
        ------
        ValueError
            If the level is not recognized.
        """
        if level not in self.fnamesPattern.keys():
            raise ValueError(
                f"Level not found, level must be in {self.fnamesPattern.keys()}"
            )
        return globList(self.fnamesPattern[level])

    @functools.cache
    def listFilesExt(self, level, ignoreBrokenFiles=False):
        """
        List all files of a specific processing level for this case and camera,
        including broken and nodata files.

        Parameters
        ----------
        level : str
            Processing level to list files for (e.g., 'level1detect').
        ignoreBrokenFiles : bool, optional
            Ignore broken files in the listing (defaults to False).

        Returns
        -------
        list
            List of full paths to files matching the pattern.
        """
        res = globList([self.fnamesPatternExt[level], self.fnamesPattern[level]])
        if ignoreBrokenFiles:
            res = [x for x in res if not x.endswith(".broken.txt")]
        return res

    @functools.cache
    def listBroken(self, level):
        """
        List all broken files of a specific processing level for this case and camera.

        Parameters
        ----------
        level : str
            Processing level to list broken files for (e.g., 'level1detect').

        Returns
        -------
        list
            List of full paths to broken files matching the pattern.
        """
        return globList(
            self.fnamesPatternExt[level],
            search=".[b,n][r,o]*",
            replace=".broken.txt",
        )

    @functools.cache
    def listNoData(self, level):
        """
        List all nodata files of a specific processing level for this case and camera.

        Parameters
        ----------
        level : str
            Processing level to list nodata files for (e.g., 'level1detect').

        Returns
        -------
        list
            List of full paths to nodata files matching the pattern.
        """
        return globList(
            self.fnamesPatternExt[level], search=".[b,n][r,o]*", replace=".nodata"
        )

    @functools.cache
    def listFilesWithNeighbors(self, level):
        """
        List all files of a specific processing level for this case and camera,
        including neighboring files.

        Parameters
        ----------
        level : str
            Processing level to list files for (e.g., 'level1detect').

        Returns
        -------
        list
            List of full paths to files (including neighbors) matching the pattern.
        """
        fnames = self.listFiles(level)
        if len(fnames) > 0:
            ff1 = FilenamesFromLevel(fnames[0], self.config)
            ff2 = FilenamesFromLevel(fnames[-1], self.config)
            fnames = [ff1.prevFile(level=level)] + fnames + [ff2.nextFile(level=level)]

        # remove None
        fnames = [f for f in fnames if f is not None]

        return fnames

    @property
    def isCompleteL0(self):
        """
        Check if all level0 files for this case and camera are complete.

        Returns
        -------
        bool
            True if all level0 files are complete, False otherwise.
        """
        return self.nMissingL0 == 0

    @property
    def isCompleteL1detect(self):
        """
        Check if all level1detect files for this case and camera are complete.

        Returns
        -------
        bool
            True if all level1detect files are complete, False otherwise.
        """
        return self.nMissingL1detect == 0

    @property
    def isCompleteMetaFrames(self):
        """
        Check if all metaFrames files for this case and camera are complete.

        Returns
        -------
        bool
            True if all metaFrames files are complete, False otherwise.
        """
        return self.nMissingMetaFrames == 0

    @property
    def isCompleteL1match(self):
        """
        Check if all level1match files for this case and camera are complete.

        Returns
        -------
        bool
            True if all level1match files are complete, False otherwise.
        """
        return self.nMissingL1match == 0

    @property
    def isCompleteL1track(self):
        """
        Check if all level1track files for this case and camera are complete.

        Returns
        -------
        bool
            True if all level1track files are complete, False otherwise.
        """
        return self.nMissingL1track == 0

    def isComplete(self, level, ignoreBrokenFiles=False, requireL0Files=False):
        """
        Check if all files of a specific processing level for this case and camera are complete.

        Parameters
        ----------
        level : str
            Processing level to check (e.g., 'level1detect').
        ignoreBrokenFiles : bool, optional
            Ignore broken files in the completeness check (defaults to False).
        requireL0Files : bool, optional
            Require the presence of level0 files (defaults to False).

        Returns
        -------
        bool
            True if all files of the specified level are complete, False otherwise.
        """
        return (
            self.nMissing(
                level,
                ignoreBrokenFiles=ignoreBrokenFiles,
                requireL0Files=requireL0Files,
            )
            == 0
        )

    @property
    def nL0(self):
        """
        Get the number of level0 files for this case and camera.

        Returns
        -------
        int
            Number of level0 files.
        """
        return len(self.listFiles("level0txt"))

    @property
    def nMissingL0(self):
        """
        Get the number of missing level0 files for this case and camera.

        Returns
        -------
        int
            Number of missing level0 files.
        """
        return self.nMissing("level0")

    @property
    def nMissingL1detect(self):
        """
        Get the number of missing level1detect files for this case and camera.

        Returns
        -------
        int
            Number of missing level1detect files.
        """
        return self.nMissing("level1detect")

    @property
    def nMissingMetaFrames(self):
        """
        Get the number of missing metaFrames files for this case and camera.

        Returns
        -------
        int
            Number of missing metaFrames files.
        """
        return self.nMissing("metaFrames")

    @property
    def nMissingL1match(self):
        """
        Get the number of missing level1match files for this case and camera.

        Returns
        -------
        int
            Number of missing level1match files.
        """
        return self.nMissing("level1match")

    @property
    def nMissingL1track(self):
        """
        Get the number of missing level1track files for this case and camera.

        Returns
        -------
        int
            Number of missing level1track files.
        """
        return self.nMissing("level1track")

    def nMissing(self, level, ignoreBrokenFiles=False, requireL0Files=False):
        """
        Get the number of missing files of a specific processing level for this case and camera.

        Parameters
        ----------
        level : str
            Processing level to check (e.g., 'level1detect').
        ignoreBrokenFiles : bool, optional
            Ignore broken files in the count (defaults to False).
        requireL0Files : bool, optional
            Require the presence of level0 files (defaults to False).

        Returns
        -------
        int
            Number of missing files.
        """
        if level in dailyLevels:
            try:
                nExpected = len(self.cases)
            except AttributeError:
                # in case we are in FindFiles, not FindFilesRange
                nExpected = 1
            nMissing = nExpected - len(
                self.listFilesExt(level, ignoreBrokenFiles=ignoreBrokenFiles)
            )
        else:
            if requireL0Files and (self.nL0 == 0):
                # use random number larger 0
                nL0 = 1
            else:
                nL0 = self.nL0
            nMissing = nL0 - len(
                self.listFilesExt(level, ignoreBrokenFiles=ignoreBrokenFiles)
            )

        if nMissing < 0:
            log.error(
                f"Likely duplicates for {level} in {self.fnamesPattern[level]}* ."
                f"Or sth wrong in {self.fnamesPattern['level0txt']}"
            )

        return nMissing

    def reportDuplicates(self, level):
        """
        Report duplicate files of a specific processing level for this case and camera.

        Parameters
        ----------
        level : str
            Processing level to check (e.g., 'level1detect').

        Returns
        -------
        list
            List of duplicate file paths.
        """
        duplicates = []
        if self.nMissing(level) < 0:
            if level not in dailyLevels:
                fnamesL0 = self.listFiles("level0txt")
                for fname in fnamesL0:
                    ff = Filenames(fname, self.config, self.version)
                    lfiles = glob.glob(f"{ff.fname[level]}*")
                    if len(lfiles) > 1:
                        duplicates += lfiles
            else:  # for daily levels
                duplicates = self.listFilesExt(level)
        return duplicates

    #
    # def isCompleteL2match(self):
    #     return (len(self.listFiles("level0txt")) == len(self.listFilesExt("level2match")))

    # [f.split("/")[-1].split("-")[-1].split(".")[0] for f in self.listFilesExt("level1detect")]
    # @property
    # def isCompleteL3(self):
    #     return (len(self.listFiles("level2")) == len(self.listFilesExt("level3Ext")))


class FindFilesRange(FindFiles):
    """
    Wraps multiple FindFiles instances for a range of cases, delegating
    all attribute access and method calls with automatic result aggregation.

    Parameters
    ----------
    cases : list of str
        List of case identifiers (YYYYMMDD format).
    camera : str
        Camera identifier.
    config : dict or str
        Configuration dictionary or path to configuration file.

    Examples
    --------
    >>> ffr = FindFilesRange(["20230101", "20230102", "20230103"], "leader", config)
    >>> ffr.listFiles("level1detect")   # returns concatenated list
    >>> ffr.nMissing("level1detect")    # returns sum
    >>> ffr.isComplete("level1detect")  # returns all(...)
    """

    def __init__(self, cases, camera, config, endYesterday=True, version=__version__):
        # Deliberately skip FindFiles.__init__ - we manage instances instead
        self.config = readSettings(config)
        cases = getCaseRange(cases, self.config, endYesterday=endYesterday)
        self._instances = [FindFiles(case, camera, config, version) for case in cases]
        self.cases = cases
        self.camera = camera

    def __getattr__(self, name):
        # Guard against calls during __init__ before _instances is set
        if name == "_instances" or "_instances" not in self.__dict__:
            raise AttributeError(name)
        if not self._instances:
            raise AttributeError(name)
        attr = getattr(self._instances[0], name)
        if callable(attr):

            def multi_method(*args, **kwargs):
                results = [getattr(ff, name)(*args, **kwargs) for ff in self._instances]
                return _aggregate(results)

            return multi_method
        elif name == "config":  # the config is the same for all cases
            return getattr(self._instances[0], name)
        else:
            results = [getattr(ff, name) for ff in self._instances]
            return _aggregate(results)

    def __dir__(self):
        # Merge own attrs with FindFiles instance attrs for full tab completion
        own = set(super().__dir__())
        instance_attrs = set(dir(self._instances[0])) if self._instances else set()
        return sorted(own | instance_attrs)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._instances[self.cases.index(key)]
            except ValueError:
                raise KeyError(f"Case '{key}' not found. Available: {self.cases}")
        return self._instances[key]

    def __iter__(self):
        return iter(self._instances)

    def __len__(self):
        return len(self._instances)

    def __repr__(self):
        return f"FindFilesRange({self.cases}, camera={self.camera})"


class Filenames(object):
    """
    Class to manage file paths and related information for VISSS data processing.

    This class provides methods to construct and manage file paths for different
    processing levels of VISSS data. It handles both level0 (raw video) and higher
    level files (processed data). The class supports operations such as finding
    neighboring files, determining if data is complete, and creating file paths
    for various processing stages.

    Attributes
    ----------
    fname : dict
        Dictionary mapping processing level to full file path.
    config : dict
        Configuration dictionary containing paths and settings.
    version : str
        Version string for the VISSS processing.
    basename : str
        Basename of the file without extension.
    dirname : str
        Directory name of the file.
    case : str
        Case identifier (YYYYMMDD).
    year : str
        Year component of the case.
    month : str
        Month component of the case.
    day : str
        Day component of the case.
    timestamp : str
        Timestamp component of the case (HHMMSS).
    datetime : datetime.datetime
        Parsed datetime from the case identifier.
    datetime64 : numpy.datetime64
        Parsed datetime64 from the case identifier.
    camera : str
        Camera identifier (e.g., 'leader_001').
    visssGen : str
        VISSS generation identifier.
    computer : str
        Computer identifier.
    basenameShort : str
        Short basename for daily files.
    outpath : str
        Output path for the current date.
    outpathDaily : str
        Daily output path for the current date.
    logpath : str
        Log path for this camera.
    quicklookPath : dict
        Dictionary mapping processing level to quicklook image path.
    """

    def __init__(self, fname, config, version=__version__):
        """
        Initialize Filenames object with file path and configuration details.

        Parameters
        ----------
        fname : str
            Full path to the level0 mkv file.
        config : dict or str
            Configuration dictionary or path to configuration file.
        version : str, optional
            Version string, defaults to __version__.

        Notes
        -----
        This constructor extracts all necessary information from the filename
        and builds paths for different processing levels based on the configuration.
        It handles both single-threaded and multi-threaded scenarios.
        """
        config = readSettings(config)
        if fname.endswith("txt"):
            fname = fname.replace("txt", config.movieExtension)

        # double // can mess with checks whether file exist or not
        fname = fname.replace("//", "/")

        self.fname = DictNoDefault(
            {
                "level0": fname,
                "level0txt": fname.replace(config.movieExtension, "txt"),
                "level0jpg": fname.replace(config.movieExtension, "jpg"),
            }
        )

        self.config = config
        self.version = version

        self.basename = os.path.basename(fname).split(".")[0]
        self.dirname = os.path.dirname(fname)

        if config["nThreads"] is None:
            self.case = self.basename.split("_")[-1]
        else:
            self.case = self.basename.split("_")[-2]

        self.year = self.case[:4]
        self.month = self.case[4:6]
        self.day = self.case[6:8]
        self.timestamp = self.case[-6:]

        try:
            self.datetime = datetime.datetime.strptime(self.case, "%Y%m%d-%H%M%S")
        except ValueError:
            self.datetime = datetime.datetime.strptime(self.case, "%Y%m%d")

        self.datetime64 = np.datetime64(self.datetime, "ns")

        if config["nThreads"] is not None:
            self.basename = "_".join(self.basename.split("_")[:-1])

        self.camera = "_".join(self.basename.split("_")[2:4])
        self.visssGen = self.basename.split("_")[1]
        self.computer = self.basename.split("_")[0]
        # basename for daily files
        self.basenameShort = "_".join(
            (
                self.computer,
                self.visssGen,
                self.camera,
                f"{self.year}{self.month}{self.day}",
            )
        )

        self.outpath = "%s/%s/%s/%s" % (
            config["pathOut"],
            self.year,
            self.month,
            self.day,
        )
        self.outpathDaily = "%s/%s" % (config["pathOut"], self.year)
        self.logpath = "%s/%s_%s_%s/" % (
            config.path.format(level="logs"),
            self.computer,
            self.visssGen,
            self.camera,
        )

        for fL in fileLevels:
            self.fname[fL] = "%s/%s_V%s_%s_%s.nc" % (
                self.outpath.format(
                    version=self.version, site=config["site"], level=fL
                ),
                fL,
                version,
                config["site"],
                self.basename,
            )
            self.fname[fL] = self.fname[fL].replace("//", "/")
        for fL in dailyLevels:
            self.fname[fL] = "%s/%s_V%s_%s_%s.nc" % (
                self.outpathDaily.format(
                    version=self.version, site=config["site"], level=fL
                ),
                fL,
                version,
                config["site"],
                self.basenameShort,
            )
            self.fname[fL] = self.fname[fL].replace("//", "/")

        self.fname["imagesL1detect"] = self.fname["imagesL1detect"].replace(
            ".nc", ".bin"
        )
        self.fname.level3combinedRiming = self.fname.level3combinedRiming.replace(
            "level3combinedRiming",
            f"level3combinedRiming{config.level3.combinedRiming.extraFileStr}",
        )

        outpathQuicklooks = "%s/%s/%s/%s" % (
            config["pathQuicklooks"],
            self.year,
            self.month,
            self.day,
        )
        self.quicklookPath = DictNoDefault({})
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            self.quicklookPath[qL] = outpathQuicklooks.format(
                version=version, site=config["site"], level=qL
            )
        self.quicklookPath.level3combinedRiming = (
            self.quicklookPath.level3combinedRiming.replace(
                "level3combinedRiming",
                f"level3combinedRiming{config.level3.combinedRiming.extraFileStr}",
            )
        )
        return

    def __repr__(self):
        return json.dumps(self.fname, indent=4)

    @property
    def yesterday(self):
        """
        Get the previous day's case identifier.

        Returns
        -------
        str
            Date in YYYYMMDD format for the day before this case.
        """
        return self.yesterdayObject.case.split("-")[0]

    @property
    def yesterdayObject(self):
        """
        Get the FindFiles object for the previous day.

        Returns
        -------
        FindFiles
            FindFiles object for the previous day.
        """
        import pandas as pd

        return FindFiles(
            pd.to_datetime(self.datetime64 - np.timedelta64(24, "h")).strftime(
                "%Y%m%d"
            ),
            self.camera,
            self.config,
        )

    @property
    def tomorrow(self):
        """
        Get the next day's case identifier.

        Returns
        -------
        str
            Date in YYYYMMDD format for the day after this case.
        """
        return self.tomorrowObject.case.split("-")[0]

    @property
    def tomorrowObject(self):
        """
        Get the FindFiles object for the next day.

        Returns
        -------
        FindFiles
            FindFiles object for the next day.
        """
        import pandas as pd

        return FindFiles(
            pd.to_datetime(self.datetime64 + np.timedelta64(24, "h")).strftime(
                "%Y%m%d"
            ),
            self.camera,
            self.config,
        )

    def filenamesOtherCamera(self, graceInterval=120, level="level1detect"):
        """
        Find all relevant files of the other camera for a given level.

        Parameters
        ----------
        graceInterval : int, optional
            Grace interval in seconds to account for time offsets, defaults to 120.
        level : str, optional
            Processing level to check, defaults to "level1detect".

        Returns
        -------
        list
            List of filenames from the other camera matching the criteria.

        Notes
        -----
        This method considers time windows and handles cases where files might be
        from adjacent days due to time synchronization issues.
        """
        import pandas as pd

        otherCam = otherCamera(self.camera, self.config)

        case = f"{self.year}{self.month}{self.day}"

        ff = FindFiles(case, otherCam, self.config, self.version)
        # get fnames for correct level
        fnames = ff.listFilesExt(level)

        thisDayStart = self.datetime.replace(hour=0, minute=0, second=0)
        nextDayStart = thisDayStart + datetime.timedelta(days=1)
        prevDayStart = thisDayStart - datetime.timedelta(days=1)

        # check for early files where we need to consider the prev day
        earlyFile = (self.datetime - thisDayStart) <= datetime.timedelta(
            seconds=(self.config["newFileInt"] + graceInterval)
        )
        if earlyFile:
            prevCase = datetime.datetime.strftime(prevDayStart, "%Y%m%d")
            prevFf = FindFiles(prevCase, otherCam, self.config, self.version)
            fnames = prevFf.listFilesExt(level) + fnames

        # same for late files
        lateFile = (nextDayStart - self.datetime) <= datetime.timedelta(
            seconds=self.config["newFileInt"] + graceInterval
        )
        if lateFile:
            nextCase = datetime.datetime.strftime(nextDayStart, "%Y%m%d")
            nextFf = FindFiles(nextCase, otherCam, self.config, self.version)
            fnames += nextFf.listFilesExt(level)

        # get timestamps of surrounding files
        if (self.config.nThreads is not None) and (
            (level == "level0") or (level == "level0txt")
        ):
            ts = np.array([f.split("_")[-2] for f in fnames])
        else:
            ts = np.array([f.split("_")[-1].split(".")[0] for f in fnames])
        try:
            ts = pd.to_datetime(ts, format="%Y%m%d-%H%M%S")
        except ValueError:
            ts = pd.to_datetime(ts, format="%Y%m%d")

        plusDelta = np.timedelta64(self.config["newFileInt"] + graceInterval, "s")
        # grace interval not needed as long as grave less than new file interval
        minusDelta = np.timedelta64(self.config["newFileInt"] + graceInterval, "s")

        windowStart = self.datetime64 - minusDelta
        windowEnd = self.datetime64 + plusDelta

        fnames = np.array(fnames)

        assert fnames.shape == ts.shape

        timeWindow = ts[(ts >= windowStart) & (ts <= windowEnd)]
        fnamesWindow = fnames[(ts >= windowStart) & (ts <= windowEnd)]

        return fnamesWindow.tolist()

    @functools.cached_property
    def fnameTxtAllThreads(self):
        """
        Find level 0 txt filenames of all threads.

        Returns
        -------
        dict
            Dictionary mapping thread number to txt filename.

        Notes
        -----
        For single-threaded setups, this returns a dictionary with only thread 0.
        For multi-threaded setups, it returns file paths for all threads.
        Handles cases where files may have slightly different timestamps.
        """

        if (self.config["nThreads"] is None) or (self.config["nThreads"] == 1):
            return {0: self.fname.level0txt}

        fname0All = dict()
        for nThread in range(self.config["nThreads"]):
            thisFname = self.fname.level0txt.replace("_0.txt", "_%i.txt" % nThread)

            # sometime the second changes while the new thread file is written, fix it:
            if not os.path.isfile(thisFname):
                for tdelta in [1, -1, 2, -2, 3, -3, 4, -4]:
                    thisSplits = thisFname.split("_")
                    thisTime = datetime.datetime.strptime(
                        thisSplits[-2], "%Y%m%d-%H%M%S"
                    )
                    thisTime = (thisTime + datetime.timedelta(seconds=tdelta)).strftime(
                        "%Y%m%d-%H%M%S"
                    )
                    print("looking for file from other thread, looking at", thisTime)
                    thisSplits[-2] = thisTime
                    newFname = "_".join(thisSplits)
                    if os.path.isfile(newFname):
                        thisFname = newFname
                        break
                else:
                    print("did not find file for other thread", thisTime)
                    continue

            fname0All[nThread] = thisFname
        # make sure all filenames are different from each other
        assert len(fname0All) == len(set(fname0All.values()))
        return fname0All

    @functools.cached_property
    def fnameMovAllThreads(self):
        """
        Find level 0 movie filenames of all threads.

        Returns
        -------
        dict
            Dictionary mapping thread number to movie filename.

        Notes
        -----
        Similar to fnameTxtAllThreads but returns movie files instead of txt files.
        Uses the same logic for handling multiple threads and timestamp variations.
        """

        # shortcut
        if (self.config["nThreads"] is None) or (self.config["nThreads"] == 1):
            fname0AllMov = {}
            if os.path.isfile(self.fname.level0):
                fname0AllMov[0] = self.fname.level0
            return fname0AllMov

        fname0AllTxt = self.fnameTxtAllThreads
        fname0AllMov = {}
        for k, v in fname0AllTxt.items():
            fname = v.replace(".txt", f".{self.config.movieExtension}")
            if os.path.isfile(fname):
                fname0AllMov[k] = fname
            # else:
            #     print(f"{fname} not found - most likely no data recorded")
        return fname0AllMov

    @functools.cache
    def nextFile(self, level="level0", debug=False):
        """
        Find the next file in sequence for the specified processing level.

        Parameters
        ----------
        level : str, optional
            Processing level, defaults to "level0".
        debug : bool, optional
            Enable debug output, defaults to False.

        Returns
        -------
        str or None
            Path to the next file, or None if not found.

        Notes
        -----
        If the file does not exist yet, it searches across day boundaries for
        nearby files matching the processing level.
        """
        return self.findNeighborFile(+1, level=level, debug=debug)

    @functools.cache
    def prevFile(self, level="level0", debug=False):
        """
        Find the previous file in sequence for the specified processing level.

        Parameters
        ----------
        level : str, optional
            Processing level, defaults to "level0".
        debug : bool, optional
            Enable debug output, defaults to False.

        Returns
        -------
        str or None
            Path to the previous file, or None if not found.

        Notes
        -----
        If the file does not exist yet, it searches across day boundaries for
        nearby files matching the processing level.
        """
        return self.findNeighborFile(-1, level=level, debug=debug)

    def findNeighborFile(self, offset, level="level0", debug=False):
        """
        Find file at a specific offset from the current file.

        Parameters
        ----------
        offset : int
            Offset to search for (positive for future, negative for past).
        level : str, optional
            Processing level, defaults to "level0".
        debug : bool, optional
            Enable debug output, defaults to False.

        Returns
        -------
        str or None
            Path to the neighbor file, or None if not found.

        Notes
        -----
        Handles boundary conditions when searching across day boundaries.
        """
        if debug:
            print("find a neighbor", offset, level)
        dirname = os.path.dirname(self.fname[level])
        case = self.year + self.month + self.day
        af = FindFiles(case, self.camera, self.config, self.version)
        allFiles = af.listFiles(level)
        if debug:
            print("found", allFiles)
        try:
            thisFileI = allFiles.index(self.fname[level])
        except ValueError:  # self.fname[level] does not exist (yet)
            print(f"findNeighborFile: file {self.fname[level] } do not exist (yet)")
            return None
        neighborFileI = thisFileI + offset
        # neighbor is on a different day
        if neighborFileI >= len(allFiles) or (neighborFileI < 0):
            if debug:
                print("neighbor is on a different day")
            if level in dailyLevels:
                neighborCase = (
                    self.datetime + datetime.timedelta(days=offset)
                ).strftime("%Y%m%d")
                if debug:
                    print("neighborCase", neighborCase)
                allNeighborFiles = FindFiles(
                    neighborCase, self.camera, self.config, self.version
                )
                allNeighborFiles = allNeighborFiles.listFiles(level)

            else:
                dirParts = dirname.split("/")
                dirParts[-3:] = ["*", "*", "*"]
                allDayFolders = sorted(glob.glob("/".join(dirParts)))
                if debug:
                    print("glob", "/".join(dirParts))
                if debug:
                    print("allDayFolders", allDayFolders)
                neighborDayFolderI = allDayFolders.index(dirname) + offset
                if (neighborDayFolderI >= len(allDayFolders)) or (
                    neighborDayFolderI < 0
                ):
                    if debug:
                        print("no neighbor file on a different day")
                    return None
                neighborDayFolder = allDayFolders[neighborDayFolderI]
                year, month, day = [
                    s.split("_")[0] for s in neighborDayFolder.split("/")[-3:]
                ]
                neighborCase = "".join([year, month, day])
                if debug:
                    print("neighborCase", neighborCase)
                allNeighborFiles = FindFiles(
                    neighborCase, self.camera, self.config, self.version
                )
                allNeighborFiles = allNeighborFiles.listFiles(level)
            assert offset in [1, -1], "other offsets than 1, -1 not implemented yet!"
            if offset > 0:
                try:
                    neighborFile = allNeighborFiles[0]
                except IndexError:
                    return None
            else:
                try:
                    neighborFile = allNeighborFiles[-1]
                except IndexError:
                    return None
        else:
            if debug:
                print("neighbor is on same day")
            neighborFile = allFiles[neighborFileI]
        return neighborFile

    @functools.cache
    def _getOffsets(self, level, maxOffset, direction):
        """
        Helper method to compute temporal offsets for neighbor file search.

        Parameters
        ----------
        level : str
            Processing level to search.
        maxOffset : numpy.timedelta64
            Maximum offset to consider.
        direction : int
            Search direction (+1 for future, -1 for past).

        Returns
        -------
        numpy.ndarray
            Array of temporal offsets.
        """

        assert maxOffset <= np.timedelta64(1, "D"), "not supported yet"
        case = self.case.split("-")[0]
        caseClose = "".join(
            str(self.datetime64 + (maxOffset * direction)).split("T")[0].split("-")
        )
        maxOffsetNs = maxOffset / np.timedelta64(1, "ns")
        cases = sorted(set([case, caseClose]))
        allFiles = []
        for case1 in cases:
            ff1 = FindFiles(case1, self.camera, self.config)
            allFiles += ff1.listFiles(level)
        try:
            allTimes = np.array(
                [
                    np.datetime64(
                        datetime.datetime.strptime(
                            a.split("_")[-1].split(".")[0], "%Y%m%d-%H%M%S"
                        )
                    )
                    for a in allFiles
                ]
            )
        except ValueError:
            allTimes = np.array(
                [
                    np.datetime64(
                        datetime.datetime.strptime(
                            a.split("_")[-1].split(".")[0], "%Y%m%d"
                        )
                    )
                    for a in allFiles
                ]
            )

        # take care of daily levels
        if level in ["metaRotation", "level2"]:
            refTime = self.datetime64.astype("datetime64[D]")
        else:
            refTime = self.datetime64

        if len(allTimes) > 0:
            allOffsets = allTimes - (refTime)
        else:
            allOffsets = np.array([])

        return allOffsets

    @functools.cache
    def nextFile2(self, level="level0", maxOffset=np.timedelta64(2, "h")):
        """
        Alternative implementation to find next file using timestamp-based search.

        Parameters
        ----------
        level : str, optional
            Processing level to search, defaults to "level0".
        maxOffset : numpy.timedelta64, optional
            Maximum time offset to search, defaults to 2 hours.

        Returns
        -------
        str or None
            Path to the next file, or None if not found.

        Notes
        -----
        More robust than nextFile when the reference file does not exist yet.
        Uses timestamp comparisons to avoid issues with missing files.
        """
        allOffsets = self._getOffsets(self, level, maxOffset, +1)

        if len(allOffsets) == 0:
            return None
        else:
            # apply boundary conditions
            allOffsets = allOffsets[
                (allOffsets > np.timedelta64(0, "ns")) & (allOffsets <= maxOffset)
            ]
            if len(allOffsets) == 0:
                return None

            neighborOffset = np.min(allOffsets)
            neighborTimestamp = self.datetime64 + neighborOffset
            return FindFiles(neighborTimestamp, self.camera, self.config).listFiles(
                level
            )[0]

    @functools.cache
    def prevFile2(self, level="level0", maxOffset=np.timedelta64(2, "h")):
        """
        Alternative implementation to find previous file using timestamp-based search.

        Parameters
        ----------
        level : str, optional
            Processing level to search, defaults to "level0".
        maxOffset : numpy.timedelta64, optional
            Maximum time offset to search, defaults to 2 hours.

        Returns
        -------
        str or None
            Path to the previous file, or None if not found.

        Notes
        -----
        More robust than prevFile when the reference file does not exist yet.
        Uses timestamp comparisons to avoid issues with missing files.
        """
        allOffsets = self._getOffsets(level, maxOffset, -1)

        if len(allOffsets) == 0:
            return None
        else:
            # apply boundary conditions
            allOffsets = allOffsets[
                (allOffsets < np.timedelta64(0, "ns")) & (allOffsets >= -maxOffset)
            ]
            if len(allOffsets) == 0:
                return None

            neighborOffset = np.min(np.abs(allOffsets))
            neighborTimestamp = self.datetime64 - neighborOffset
            return FindFiles(neighborTimestamp, self.camera, self.config).listFiles(
                level
            )[0]

    @functools.cache
    def getEvents(self, skipExisting=True):
        """
        Get (and create if necessary) event dataset for this case and camera.

        Parameters
        ----------
        skipExisting : bool, optional
            Skip processing existing files (defaults to True).

        Returns
        -------
        tuple(str, xarray.Dataset)
            Tuple containing event filename and event Dataset.

        Notes
        -----
        This method ensures event files are available for data validation
        and processing workflows.
        """

        eventFile = self.fname.metaEvents
        # # just in case it is missing
        # metadata.createEvent(
        #     self.case,
        #     self.camera,
        #     self.config,
        #     skipExisting=skipExisting,
        #     quiet=True,
        # )
        eventDat = xr.open_dataset(eventFile).load()
        # opening it several times can cause segfaults
        eventDat.close()

        return eventFile, eventDat


class FilenamesFromLevel(Filenames):
    """
    Class to manage file paths for VISSS data processing starting from level 1 or level 2 files.

    This class extends the Filenames class and is specifically designed to handle
    file paths when starting from processed level 1 or level 2 files. It extracts
    essential information from these files to reconstruct the full path structure
    including case, camera, and processing level details.

    Attributes
    ----------
    fname : dict
        Dictionary mapping processing level to full file path.
    config : dict
        Configuration dictionary containing paths and settings.
    version : str
        Version string for the VISSS processing.
    basename : str
        Basename of the file without extension.
    dirname : str
        Directory name of the file.
    case : str
        Case identifier (YYYYMMDD).
    year : str
        Year component of the case.
    month : str
        Month component of the case.
    day : str
        Day component of the case.
    timestamp : str
        Timestamp component of the case (HHMMSS).
    datetime : datetime.datetime
        Parsed datetime from the case identifier.
    datetime64 : numpy.datetime64
        Parsed datetime64 from the case identifier.
    camera : str
        Camera identifier (e.g., 'leader_001').
    visssGen : str
        VISSS generation identifier.
    computer : str
        Computer identifier.
    basenameShort : str
        Short basename for daily files.
    outpath : str
        Output path for the current date.
    outpathDaily : str
        Daily output path for the current date.
    logpath : str
        Log path for this camera.
    quicklookPath : dict
        Dictionary mapping processing level to quicklook image path.
    """

    def __init__(self, fname, config):
        """
        Initialize FilenamesFromLevel object with level 1 or level 2 file path and configuration.

        Parameters
        ----------
        fname : str
            Full path to the level 1 or level 2 file.
        config : dict or str
            Configuration dictionary or path to configuration file.

        Notes
        -----
        This constructor parses the filename of level 1 or level 2 files to extract
        case, camera, and other relevant information to build complete path structures.
        """

        config = readSettings(config)
        (
            level,
            version,
            site,
            computer,
            visssGen,
            visssType,
            visssSerial,
            ts,
        ) = fname.split("/")[-1].split("_")
        # remove leading "V"
        version = version[1:]
        case = ts.split(".")[0]
        camera = "_".join((visssType, visssSerial))

        basename = "_".join((computer, visssGen, visssType, visssSerial, case))

        year = case[:4]
        month = case[4:6]
        day = case[6:8]

        outpath0 = "%s/%s_%s_%s/%s/%s/%s" % (
            config["path"].format(level="level0", site=site, version=version),
            computer,
            visssGen,
            camera,
            year,
            month,
            day,
        )
        if config["nThreads"] is None:
            fnameLevel0 = f"{outpath0}/{basename}.{config['movieExtension']}"
        else:
            fnameLevel0 = f"{outpath0}/{basename}_0.{config['movieExtension']}"

        super().__init__(fnameLevel0, config, version)

        return
