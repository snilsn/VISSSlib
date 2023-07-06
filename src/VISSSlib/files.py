# -*- coding: utf-8 -*-


import sys
import glob
import os
import datetime
from copy import deepcopy
import functools
from addict import Dict


import numpy as np
import pandas as pn

import logging
log = logging.getLogger(__name__)

from .tools import nicerNames, otherCamera
from . import __version__


dailyLevels = ["metaEvents", "metaRotation", "level2match"]
fileLevels = ["level1detect", "level1match", "level1track", "metaFrames", "metaDetection", "imagesL1detect"]#, "metaFixedCaptureId"]
quicklookLevelsSep = ["level0", "metaFrames", "metaEvents", "level1detect", "level1match", "level1matchParticles", "metaRotation"]
quicklookLevelsComb = [ "level2match"]
imageLevels = ["imagesL1detect"]

class FindFiles(object):
    def __init__(self, case, camera, config, version=__version__):
        
        '''
        find all files corresponding to certain case

        for level 0, only thread 0 files are returned!
        '''
        assert type(config) is not str

        if type(case) is not str:
            self.case = pn.to_datetime(case).strftime('%Y%m%d-%H%M%S')
        else:
            self.case = case
        self.camera = camera
        self.config = config
        self.version = version


        computerDict = {}
        for computer1, camera1 in zip(config["computers"], config["instruments"]):
            computerDict[camera1] = computer1
        self.computer = computerDict[camera]

        self.year  =self.case[:4]
        self.month  =self.case[4:6]
        self.day  =self.case[6:8]

        self.hour = self.case[9:11]
        self.minute = self.case[11:13]

        try:
            self.timestamps  =self.case[9:]        
        except IndexError:
            self.timestamps = None
            
        try:
            self.datetime = datetime.datetime.strptime(self.case.ljust(15, "0"), "%Y%m%d-%H%M%S")
        except ValueError:
            self.datetime = datetime.datetime.strptime(self.case, "%Y%m%d")
        self.datetime64= np.datetime64(self.datetime, "ns")

        self.logpath = "%s/%s_%s_%s/" % (config.path.format(level="logs"), self.computer, config.visssGen, self.camera)
        
        outpath = "%s/%s/%s/%s" % (config["pathOut"], self.year, self.month, self.day)
        outpathDaily = "%s/%s" % (config["pathOut"], self.year)

        self.outpath = Dict({})
        for dL in fileLevels:
            self.outpath[dL] = outpath.format(site=config.site, level=dL, version=self.version)
        for dL in dailyLevels:
            self.outpath[dL] = outpathDaily.format(site=config.site, level=dL, version=self.version)
        self.outpath["level0"] = config["path"].format(site=config["site"], level='level0')+f'/{self.computer}_{config["visssGen"]}_{camera}/{self.year}/{self.month}/{self.day}'

        # for iL in imageLevels:
        #     self.outpath[iL] = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)

        self.fnamesPattern = Dict({})
        for dL in fileLevels:
            self.fnamesPattern[dL] = "%s/%s_V%s*%s*%s*.nc"%(self.outpath[dL], dL, version, camera, self.case)
        #overwrite for level0
        if config["nThreads"] is None:
            self.fnamesPattern.level0 = '%s/*%s*.%s' % (
                self.outpath.level0, self.case, config["movieExtension"])
            self.fnamesPattern.level0txt = '%s/*%s*.%s' % (
                self.outpath.level0, self.case, "txt")
            self.fnamesPattern.level0jpg = '%s/*%s*.%s' % (
                self.outpath.level0, self.case, "jpg")
        else:
            self.fnamesPattern.level0 = '%s/*%s*_0.%s' % (
                self.outpath.level0, self.case, config["movieExtension"])
            self.fnamesPattern.level0txt = '%s/*%s*_0.%s' % (
                self.outpath.level0, self.case, "txt")
            self.fnamesPattern.level0jpg = '%s/*%s*_0.%s' % (
                self.outpath.level0, self.case, "jpg")
        self.fnamesPattern.level0status = f"{self.outpath['level0']}/*_{config['visssGen']}_{camera}_{self.case}_status.txt"
        for dL in dailyLevels:
            self.fnamesPattern[dL] = "%s/%s_V%s_*%s*%s%s%s.nc"%(self.outpath[dL], dL, version, camera, self.year, self.month, self.day)
        self.fnamesPattern["imagesL1detect"] = self.fnamesPattern["imagesL1detect"].replace(".nc", ".zip")


        self.fnamesPatternExt = Dict({})
        for dL in fileLevels + dailyLevels:
            self.fnamesPatternExt[dL] = "%s/%s_V%s_*%s*%s*nc.[b,n]*"%(self.outpath[dL], dL, version, camera, self.case) #finds broken & nodata
        self.fnamesPatternExt.level0txt = ""
        self.fnamesPatternExt.level0jpg = ""
        self.fnamesPatternExt.level0 = ""

        self.fnamesDaily = Dict({})
        for dL in dailyLevels:
            self.fnamesDaily[dL] = "%s/%s_V%s_%s_%s_%s_%s_%s%s%s.nc"%(self.outpath[dL], dL, version, config.site, self.computer, config['visssGen'], camera, self.year, self.month, self.day)

                
        self.quicklook = Dict({})
        self.quicklookCurrent = Dict({})
        self.quicklookPath = Dict({})
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            self.quicklookPath[qL] =f'{config["pathQuicklooks"].format(version=version, site=config["site"], level=qL)}/{self.year}'

        for qL in quicklookLevelsSep:
            if self.hour == "":
                self.quicklook[qL] = f"{self.quicklookPath[qL]}/{qL}_V{version}_{config['site']}_{nicerNames(camera)}_{self.year}{self.month}{self.day}.png"
            else:
                self.quicklook[qL] = f"{self.quicklookPath[qL]}/{qL}_V{version}_{config['site']}_{nicerNames(camera)}_{self.year}{self.month}{self.day}T{self.hour}.png"
            self.quicklookCurrent[qL] = f"{config['pathQuicklooks'].format(version=version,site=config['site'], level=qL)}/{qL}_{config['site']}_{nicerNames(camera)}_current.png"
        for qL in quicklookLevelsComb:
            if self.hour == "":
                self.quicklook[qL] = f"{self.quicklookPath[qL]}/{qL}_V{version}_{config['site']}_{self.year}{self.month}{self.day}.png"
            else:
                self.quicklook[qL] = f"{self.quicklookPath[qL]}/{qL}_V{version}_{config['site']}_{self.year}{self.month}{self.day}T{self.hour}.png"
            self.quicklookCurrent[qL] = f"{config['pathQuicklooks'].format(version=version,site=config['site'], level=qL)}/{qL}_{config['site']}_current.png"

        
    @functools.cache
    def listFiles(self, level):
        if level not in self.fnamesPattern.keys():
            raise ValueError(f"Level not found, level must be in {self.fnamesPattern.keys()}")
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern[level]) ))
    @functools.cache
    def listFilesExt(self, level):
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPatternExt[level])+glob.glob(self.fnamesPattern[level]) ))
    
    @functools.cache
    def listFilesWithNeighbors(self, level):
        fnames = self.listFiles(level)
        if len(fnames) > 0:
            ff1 = FilenamesFromLevel(fnames[0], self.config)
            ff2 = FilenamesFromLevel(fnames[-1], self.config)
            fnames = [ff1.prevFile(level=level)] + fnames + [ff2.nextFile(level=level)]

        #remove None
        fnames = [f for f in fnames if f is not None]

        return fnames

    @property
    def isCompleteL0(self):
        return self.nMissingL0 == 0
    @property
    def isCompleteL1detect(self):
        return self.nMissingL1detect == 0
    @property
    def isCompleteMetaFrames(self):
        return self.nMissingMetaFrames == 0
    @property
    def isCompleteL1match(self):
        return self.nMissingL1match == 0

    def isComplete(self, level):
        return self.nMissing(level) == 0


    @property
    def nL0(self):
        return len(self.listFiles("level0txt"))

    @property
    def nMissingL0(self):
        return self.nMissing("level0") 
    @property
    def nMissingL1detect(self):
        return self.nMissing("level1detect") 
    @property
    def nMissingMetaFrames(self):
        return self.nMissing("metaFrames") 
    @property
    def nMissingL1match(self):
        return self.nMissing("level1match") 

    def nMissing(self, level):
        if level in dailyLevels:
            return 1 - len(self.listFilesExt(level)) 
        else:
            return self.nL0 - len(self.listFilesExt(level)) 





    # @property
    # def isCompleteL2match(self):
    #     return (len(self.listFiles("level0txt")) == len(self.listFilesExt("level2match")))

# [f.split("/")[-1].split("-")[-1].split(".")[0] for f in self.listFilesExt("level1detect")]
    # @property
    # def isCompleteL3(self):
    #     return (len(self.listFiles("level2")) == len(self.listFilesExt("level3Ext")))


    def createDirs(self):
        res = []
        for fL in dailyLevels + fileLevels:
            #print('mkdir -p %s' %
            #      self.outpath[fL])
            res.append(os.system('mkdir -p %s' %
                  self.outpath[fL]))
        return res

    def createQuicklookDirs(self):
        res = []
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            res.append(os.system('mkdir -p %s' %
                  self.quicklookPath[qL]))
        return res



class Filenames(object):
    def __init__(self, fname, config, version=__version__):
        '''
        create matching filenames based on mov file
        Use always thread 0 file!
        '''
        assert type(config) is not str
        if  fname.endswith("txt"):
            fname = fname.replace("txt", config.movieExtension)

        # double // can mess with checks whether file exist or not
        fname = fname.replace("//", "/")

        self.fname = Dict(
            {
                "level0":fname,
                "level0txt":fname.replace(config.movieExtension,"txt"),
                "level0jpg":fname.replace(config.movieExtension,"jpg"),
            }
            )

        self.config = config
        self.version = version

        self.basename = os.path.basename(fname).split('.')[0]
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

        self.datetime64= np.datetime64(self.datetime, "ns")

        if config["nThreads"] is not None:
            self.basename = '_'.join(self.basename.split('_')[:-1])
            
        self.camera = "_".join(self.basename.split("_")[2:4])
        self.visssGen = self.basename.split("_")[1]
        self.computer = self.basename.split("_")[0]
        #basename for daily files
        self.basenameShort = "_".join((self.computer, self.visssGen, self.camera, f"{self.year}{self.month}{self.day}"))

        self.outpath = "%s/%s/%s/%s" % (config["pathOut"], self.year, self.month, self.day)
        self.outpathDaily = "%s/%s" % (config["pathOut"], self.year)
        self.logpath = "%s/%s_%s_%s/" % (config.path.format(level="logs"), self.computer, self.visssGen, self.camera)

        for fL in fileLevels:
            self.fname[fL] = '%s/%s_V%s_%s_%s.nc' % (
            self.outpath.format(version=self.version, site=config["site"], level=fL), fL, version, config["site"], self.basename)
            self.fname[fL] = self.fname[fL].replace("//", "/")
        for fL in dailyLevels:
            self.fname[fL] = '%s/%s_V%s_%s_%s.nc' % (
            self.outpathDaily.format(version=self.version, site=config["site"], level=fL), fL, version, config["site"], self.basenameShort)
            self.fname[fL] = self.fname[fL].replace("//", "/")

        self.fname["imagesL1detect"] = self.fname["imagesL1detect"].replace(".nc", ".zip")

        # self.outpathImg = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)
        # self.imagepath = Dict({})
        # for iL in imageLevels:
        #     self.imagepath[iL] = "%s/%s/{ppid}"%(self.outpathImg.format(site=config["site"], level=iL),self.fname.level1detect.split("/")[-1])
        
        outpathQuicklooks = "%s/%s/%s/%s" % (config["pathQuicklooks"], self.year, self.month, self.day)
        self.quicklookPath = Dict({})
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            self.quicklookPath[qL] =outpathQuicklooks.format(version=version, site=config['site'], level=qL)

        return

        
    def createDirs(self):
        res = []
        for fL in dailyLevels + fileLevels:
            #print('mkdir -p %s' %
            #      self.outpath.format(version=self.version, site=self.config["site"], level=fL))
            res.append(os.system('mkdir -p %s' %
                  self.outpath.format(version=self.version, site=self.config["site"], level=fL)))
        return res

    def createQuicklookDirs(self):
        res = []
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            res.append(os.system('mkdir -p %s' %
                  self.quicklookPath[qL]))
        return res




    def filenamesOtherCamera(self, graceInterval = 120, level="level1detect"):
        '''
        Find all relevant files of the other camera of ´level´. ´graceinterval´ accounts for 
        potential time offsets
        '''
        

        otherCam = otherCamera(self.camera, self.config)

        case = f"{self.year}{self.month}{self.day}"

        ff = FindFiles(case, otherCam, self.config, self.version)
        # get fnames for correct level
        fnames = ff.listFilesExt(level)
        
        thisDayStart = self.datetime.replace(hour=0, minute=0, second=0)
        nextDayStart = thisDayStart + datetime.timedelta(days=1)
        prevDayStart = thisDayStart - datetime.timedelta(days=1)

        #check for early files where we need to consider the prev day
        earlyFile = (self.datetime - thisDayStart) <= datetime.timedelta(seconds=(self.config["newFileInt"]+ graceInterval))
        if earlyFile:
            prevCase = datetime.datetime.strftime(prevDayStart, "%Y%m%d")
            prevFf = FindFiles(prevCase, otherCam, self.config, self.version)
            fnames = prevFf.listFilesExt(level) + fnames
        
        #same for late files
        lateFile = (nextDayStart - self.datetime) <= datetime.timedelta(seconds=self.config["newFileInt"] + graceInterval)
        if lateFile:
            nextCase = datetime.datetime.strftime(nextDayStart, "%Y%m%d")
            nextFf = FindFiles(nextCase, otherCam, self.config, self.version)
            fnames += nextFf.listFilesExt(level)
        
        # get timestamps of surrounding files
        if (self.config.nThreads is not None) and (( level == "level0") or (level == "level0txt")):
            ts = np.array([f.split("_")[-2] for f in fnames])
        else:
            ts = np.array([f.split("_")[-1].split(".")[0] for f in fnames])
        try:
            ts = pn.to_datetime(ts, format="%Y%m%d-%H%M%S")
        except ValueError:
            ts = pn.to_datetime(ts, format="%Y%m%d")

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
        '''
        find level 0 fnames of other threads
        '''

        if (self.config["nThreads"] is None) or (self.config["nThreads"]==1):
            return {0: self.fname.level0txt}


        fname0All = dict()
        for nThread in range(self.config["nThreads"]):
            
            thisFname = self.fname.level0txt.replace('_0.txt', '_%i.txt'%nThread)

            #sometime the second changes while the new thread file is written, fix it:
            if not os.path.isfile(thisFname):

                for tdelta in [1,-1, 2, -2, 3, -3, 4, -4]:
                    thisSplits = thisFname.split("_")
                    thisTime = datetime.datetime.strptime(thisSplits[-2], "%Y%m%d-%H%M%S")
                    thisTime = (thisTime + datetime.timedelta(seconds=tdelta)).strftime("%Y%m%d-%H%M%S")
                    print("looking for file from other thread, looking at",thisTime)
                    thisSplits[-2] = thisTime
                    newFname = "_".join(thisSplits)
                    if os.path.isfile(newFname):
                        thisFname = newFname
                        break
                else:
                    print("did not find file for other thread",thisTime)
                    continue

            fname0All[nThread] = thisFname
        #make sure all filenames are different from each other
        assert len(fname0All) == len(set(fname0All.values()))
        return fname0All

    @functools.cached_property
    def fnameMovAllThreads(self):
        '''
        find level 0 fnames of other threads
        '''

        #shortcut
        if (self.config["nThreads"] is None) or (self.config["nThreads"]==1):
            fname0AllMov = {}
            if os.path.isfile(self.fname.level0):
                fname0AllMov[0] = self.fname.level0
            return fname0AllMov

        fname0AllTxt = self.fnameTxtAllThreads
        fname0AllMov = {}
        for k, v in fname0AllTxt.items():
            fname = v.replace('.txt', f".{self.config.movieExtension}")
            if os.path.isfile(fname):
                fname0AllMov[k] = fname
            # else:
            #     print(f"{fname} not found - most likely no data recorded")
        return fname0AllMov


    @functools.cache
    def nextFile(self, level="level0", debug=False):
        return self.findNeighborFile(+1, level=level, debug=debug)
    @functools.cache
    def prevFile(self, level="level0", debug=False):
        return self.findNeighborFile(-1, level=level, debug=debug)
    
    def findNeighborFile(self, offset, level="level0", debug=False):
        '''
        find file at difstance of x offsets
        '''
        if debug: print("find a neighbor", offset, level)
        dirname = os.path.dirname(self.fname[level])
        case = self.year+self.month+self.day
        af = FindFiles(case, self.camera, self.config, self.version)
        allFiles = af.listFiles(level)
        if debug: print("found", allFiles)
        try:
            thisFileI = allFiles.index(self.fname[level])
        except ValueError: # self.fname[level] does not exist (yet)
            print(f"findNeighborFile: file {self.fname[level] } do not exist (yet)")
            return None
        neighborFileI = thisFileI + offset
        #neighbor is on a different day
        if (neighborFileI >= len(allFiles) or (neighborFileI < 0)):
            if debug: print("neighbor is on a different day")
            if level in dailyLevels:
                neighborCase = (self.datetime + datetime.timedelta(days=offset)).strftime("%Y%m%d")
                if debug: print("neighborCase", neighborCase)
                allNeighborFiles = FindFiles(neighborCase, self.camera, self.config, self.version)
                allNeighborFiles = allNeighborFiles.listFiles(level)

            else:
                dirParts = dirname.split("/")
                dirParts[-3:] = ["*", "*", "*"]
                allDayFolders = sorted(glob.glob("/".join(dirParts)))
                if debug: print("glob", "/".join(dirParts))
                if debug: print("allDayFolders", allDayFolders)
                neighborDayFolderI = allDayFolders.index(dirname) + offset
                if (neighborDayFolderI >= len(allDayFolders)) or (neighborDayFolderI < 0):
                    if debug: print("no neighbor file on a different day")
                    return None
                neighborDayFolder = allDayFolders[neighborDayFolderI]
                year, month, day = neighborDayFolder.split("/")[-3:]
                neighborCase = "".join([year, month, day])
                if debug: print("neighborCase", neighborCase)
                allNeighborFiles = FindFiles(neighborCase, self.camera, self.config, self.version)
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
            if debug: print("neighbor is on same day")
            neighborFile = allFiles[neighborFileI]
        return neighborFile

    @functools.cache
    def _getOffsets(self, level, maxOffset, direction):
        # helper function nextFile2 and prevFile2

        assert maxOffset <= np.timedelta64(1,"D"), "not supported yet"
        case = self.case.split("-")[0]
        caseClose = "".join(str(self.datetime64 + (maxOffset*direction)).split("T")[0].split("-"))
        maxOffsetNs = maxOffset/np.timedelta64(1,"ns")
        cases = sorted(set([case, caseClose]))
        allFiles = []
        for case1 in cases:
            ff1 = FindFiles(case1, self.camera, self.config)
            allFiles += ff1.listFiles(level)
        try: 
            allTimes = np.array([np.datetime64(datetime.datetime.strptime(a.split("_")[-1].split(".")[0], "%Y%m%d-%H%M%S")) for a in allFiles])
        except ValueError:
            allTimes = np.array([np.datetime64(datetime.datetime.strptime(a.split("_")[-1].split(".")[0], "%Y%m%d")) for a in allFiles])

        #take care of daily levels
        if level in ["metaRotation", "level2"]:
            refTime = self.datetime64.astype('datetime64[D]')
        else:
            refTime = self.datetime64

        if len(allTimes) > 0:
            allOffsets = allTimes-(refTime) 
        else:
            allOffsets = np.array([])

        return allOffsets
        
    @functools.cache
    def nextFile2(self, level="level0", maxOffset=np.timedelta64(2,"h")):
        # alternative implementation based on timestamp. works also when reference file does not exist yet
        allOffsets = _getOffsets(self, level, maxOffset, +1)

        if len(allOffsets) == 0:
            return None
        else:
            #apply boundary conditions
            allOffsets = allOffsets[(allOffsets > np.timedelta64(0,"ns")) & (allOffsets <=maxOffset)]
            if len(allOffsets) == 0:
                return None

            neighborOffset = np.min(allOffsets)
            neighborTimestamp = self.datetime64 + neighborOffset
            return FindFiles(neighborTimestamp, self.camera, self.config).listFiles(level)[0]

    @functools.cache
    def prevFile2(self, level="level0", maxOffset=np.timedelta64(2,"h")):
        # alternative implementation based on timestamp. works also when reference file does not exist yet
        allOffsets = self._getOffsets(level, maxOffset, -1)

        if len(allOffsets) == 0:
            return None
        else:
            #apply boundary conditions
            allOffsets = allOffsets[(allOffsets < np.timedelta64(0,"ns")) & (allOffsets >= -maxOffset)]
            if len(allOffsets) == 0:
                return None

            neighborOffset = np.min(np.abs(allOffsets))
            neighborTimestamp = self.datetime64 - neighborOffset
            return FindFiles(neighborTimestamp, self.camera, self.config).listFiles(level)[0]


class FilenamesFromLevel(Filenames):
    def __init__(self, fname, config):
        '''
        get all filenames from a level 1 or level 2 file
        '''
        
        assert type(config) is not str
        level, version, site, computer, visssGen, visssType, visssSerial, ts = fname.split("/")[-1].split("_")
        #remove leading "V"
        version = version[1:]
        case = ts.split(".")[0]
        camera = "_".join((visssType, visssSerial))
        
        config = config
        basename = "_".join((computer, visssGen, visssType, visssSerial, case))

        year = case[:4]
        month = case[4:6]
        day = case[6:8]

        outpath0 = "%s/%s_visss_%s/%s/%s/%s" % (config["pathOut"].format(level="level0", site=site, version=version), computer, camera, year, month, day)
        if config["nThreads"] is None:
            fnameLevel0 = f"{outpath0}/{basename}.{config['movieExtension']}"
        else:
            fnameLevel0 = f"{outpath0}/{basename}_0.{config['movieExtension']}"

        super().__init__(fnameLevel0, config, version)

        return

    # def __init__(self, fname, config):
    #     '''
    #     get all filenames from a level 1 or level 2 file
    #     '''
        
    #     self.level, self.version, self.site, self.computer, self.visssGen, visssType, visssSerial, ts = fname.split("/")[-1].split("_")
    #     #remove leading "V"
    #     self.version = self.version[1:]
    #     self.case = ts.split(".")[0]
    #     self.camera = "_".join((visssType, visssSerial))
        
    #     self.config = config
    #     self.basename = "_".join((self.computer, self.visssGen, visssType, visssSerial, self.case))

    #     self.year = self.case[:4]
    #     self.month = self.case[4:6]
    #     self.day = self.case[6:8]
    #     self.timestamp = self.case[-6:]

    #     self.datetime = datetime.datetime.strptime(f"{self.year}{self.month}{self.day}-{self.timestamp}", "%Y%m%d-%H%M%S")
    #     self.datetime64= np.datetime64(self.datetime, "ns")

    #     self.outpath0 = "%s/%s_visss_%s/%s/%s/%s" % (config["pathOut"].format(level=0), self.computer, self.camera, self.year, self.month, self.day)
    #     if config["nThreads"] is None:
    #         self.fnameLevel0 = f"{self.outpath0}/{self.basename}.{config['movieExtension']}"
    #     else:
    #         self.fnameLevel0 = f"{self.outpath0}/{self.basename}_0.{config['movieExtension']}"

    #     self.outpath = "%s/%s/%s/%s" % (config["pathOut"], self.year, self.month, self.day)
    #     self.fnameLevel1 = '%s/level1_V%s_%s_%s.nc' % (
    #         self.outpath.format(site=self.site, level='1'), self.version, self.site, self.basename)
    #     self.fnameLevel2 = '%s/level2_V%s_%s_%s.nc' % (
    #         self.outpath.format(site=self.site, level='2'), self.version, self.site, self.basename)
    #     self.fnameLevel3 = '%s/level3_V%s_%s_%s.nc' % (
    #         self.outpath.format(site=self.site, level='3'), self.version, self.site, self.basename)

    #     self.outpathImg = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)
    #     self.fnameLevel2images = "%s/%s/{ppid}"%(self.outpathImg.format(site=self.site, level='2images'),self.fnameLevel2.split("/")[-1])
        
        
    #     return




        
