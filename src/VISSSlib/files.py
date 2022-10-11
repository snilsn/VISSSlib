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
log = logging.getLogger()

from .tools import nicerNames, otherCamera
from . import __version__


dailyLevels = ["metaEvents", "metaMatchCoefficients"]
fileLevels = ["level1detect", "level1match", "level1track", "metaFrames", "metaDetection", "imagesL1detect"]#, "metaFixedCaptureId"]
quicklookLevelsSep = ["metaFrames", "metaEvents", "level1detect", "level1match"]
quicklookLevelsComb = [ "matchCoefficients"]
imageLevels = ["imagesL1detect"]

class FindFiles(object):
    def __init__(self, case, camera, config, version=__version__):
        
        '''
        find all files corresponding to certain case

        for level 0, only thread 0 files are returned!
        '''

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

        self.outpath = Dict({})
        for dL in fileLevels + dailyLevels:
            self.outpath[dL] = outpath.format(site=config.site, level=dL)
        self.outpath["level0"] = config["path"].format(site=config["site"], level='level0')+f'/{self.computer}_{config["visssGen"]}_{camera}/{self.year}/{self.month}/{self.day}'

        # for iL in imageLevels:
        #     self.outpath[iL] = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)

        self.fnamesPattern = Dict({})
        for dL in fileLevels:
            self.fnamesPattern[dL] = "%s/%s_V%s*%s*%s*nc"%(self.outpath[dL], dL, version, camera, self.case)
        #overwrite for level0
        if config["nThreads"] is None:
            self.fnamesPattern.level0 = '%s/*%s*.%s' % (
                self.outpath.level0, self.case, config["movieExtension"])
            self.fnamesPattern.level0txt = '%s/*%s*.%s' % (
                self.outpath.level0, self.case, "txt")
        else:
            self.fnamesPattern.level0 = '%s/*%s*_0.%s' % (
                self.outpath.level0, self.case, config["movieExtension"])
            self.fnamesPattern.level0txt = '%s/*%s*_0.%s' % (
                self.outpath.level0, self.case, "txt")
        self.fnamesPattern.level0status = f"{self.outpath['level0']}/*_{config['visssGen']}_{camera}_{self.case}_status.txt"
        for dL in dailyLevels:
            self.fnamesPattern[dL] = "%s/%s_V%s_*%s*%s%s%s.nc"%(self.outpath[dL], dL, version, camera, self.year, self.month, self.day)

        self.fnamesPatternExt = Dict({})
        for dL in fileLevels + dailyLevels:
            self.fnamesPatternExt[dL] = "%s/%s_V%s_*%s*%s*nc.[b,n]*"%(self.outpath[dL], dL, version, camera, self.case) #finds broken & nodata
        self.fnamesPatternExt.level0txt = ""
        self.fnamesPatternExt.level0 = ""

        self.fnamesDaily = Dict({})
        for dL in dailyLevels:
            self.fnamesDaily[dL] = "%s/%s_V%s_%s_%s_%s_%s_%s%s%s.nc"%(self.outpath[dL], dL, version, config.site, self.computer, config['visssGen'], camera, self.year, self.month, self.day)

                
        self.quicklook = Dict({})
        self.quicklookPath = Dict({})
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            self.quicklookPath[qL] =f'{config["pathQuicklooks"].format(site=config["site"], level=qL)}/{self.year}'

        for qL in quicklookLevelsSep:
            self.quicklook[qL] = f"{self.quicklookPath[qL]}/{qL}_V{version}_{config['site']}_{nicerNames(camera)}_{self.year}{self.month}{self.day}.png"
        for qL in quicklookLevelsComb:
            self.quicklook[qL] = f"{self.quicklookPath[qL]}/{qL}_V{version}_{config['site']}_{self.year}{self.month}{self.day}.png"

        
    @functools.lru_cache
    def listFiles(self, level):
        if level not in self.fnamesPattern.keys():
            raise ValueError(f"Level not found, level must be in {self.fnamesPattern.keys()}")
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern[level]) ))
    @functools.lru_cache
    def listFilesExt(self, level):
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPatternExt[level])+glob.glob(self.fnamesPattern[level]) ))
    
    @functools.lru_cache
    def listFilesWithNeighbors(self, level):
        fnames = self.listFiles(level)
        if len(fnames) > 0:
            ff1 = FilenamesFromLevel(fnames[0], self.config)
            ff2 = FilenamesFromLevel(fnames[-1], self.config)
            fnames = [ff1.prevFile(level=level)] + fnames + [ff2.nextFile(level=level)]

        return fnames

    @property
    def isCompleteL1detect(self):
        return (len(self.listFiles("level0txt")) == len(self.listFilesExt("level1detect")))
    @property
    def isCompleteMetaFrames(self):
        return (len(self.listFiles("level0txt")) == len(self.listFilesExt("metaFrames")))
    @property
    def isCompleteL1match(self):
        return (len(self.listFiles("level0txt")) == len(self.listFilesExt("level1match")))

# [f.split("/")[-1].split("-")[-1].split(".")[0] for f in self.listFilesExt("level1detect")]
    # @property
    # def isCompleteL3(self):
    #     return (len(self.listFiles("level2")) == len(self.listFilesExt("level3Ext")))


    def createQuicklookDirs(self):
        res = []
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            res.append(os.system('mkdir -p %s' %
                  self.quicklookPath[qL]))
        return res

    def createDirs(self):
        res = []
        for fL in dailyLevels + fileLevels:
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
        if  fname.endswith("txt"):
            fname = fname.replace("txt", config.movieExtension)

        # double // can mess with checks whether file exist or not
        fname = fname.replace("//", "/")

        self.fname = Dict(
            {
                "level0":fname,
                "level0txt":fname.replace(config.movieExtension,"txt"),
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
        self.logpath = "%s/%s_%s_%s/" % (config.path.format(level="logs"), self.computer, self.visssGen, self.camera)

        for fL in fileLevels:
            self.fname[fL] = '%s/%s_V%s_%s_%s.nc' % (
            self.outpath.format(site=config["site"], level=fL), fL, version, config["site"], self.basename)
        for fL in dailyLevels:
            self.fname[fL] = '%s/%s_V%s_%s_%s.nc' % (
            self.outpath.format(site=config["site"], level=fL), fL, version, config["site"], self.basenameShort)

        self.fname["imagesL1detect"] = self.fname["imagesL1detect"].replace(".nc", ".tar.gz")

        # self.outpathImg = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)
        # self.imagepath = Dict({})
        # for iL in imageLevels:
        #     self.imagepath[iL] = "%s/%s/{ppid}"%(self.outpathImg.format(site=config["site"], level=iL),self.fname.level1detect.split("/")[-1])
        
        outpathQuicklooks = "%s/%s/%s/%s" % (config["pathQuicklooks"], self.year, self.month, self.day)
        self.quicklookPath = Dict({})
        for qL in quicklookLevelsSep + quicklookLevelsComb:
            self.quicklookPath[qL] =outpathQuicklooks.format(site=config['site'], level=qL)

        return

        
    def createDirs(self):
        res = []
        for fL in dailyLevels + fileLevels:
            res.append(os.system('mkdir -p %s' %
                  self.outpath.format(site=self.config["site"], level=fL)))
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
        ts = pn.to_datetime(ts, format="%Y%m%d-%H%M%S")

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

        if (self.config["nThreads"] is None) or (self.config["nThreads"]==1):
            return {0: self.fname.level0}


        fname0AllTxt = self.fnameTxtAllThreads
        fname0AllMov = {}
        for k, v in fname0AllTxt.items():
            fname = v.replace('.txt', f".{self.config.movieExtension}")
            if os.path.isfile(fname):
                fname0AllMov[k] = fname
            else:
                print(f"{fname} not found - most likely no data recorded")
        return fname0AllMov


    @functools.lru_cache
    def nextFile(self, level="level0", debug=False):
        return self.findNeighborFile(+1, level=level, debug=debug)
    @functools.lru_cache
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



class FilenamesFromLevel(Filenames):
    def __init__(self, fname, config):
        '''
        get all filenames from a level 1 or level 2 file
        '''
        
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

        outpath0 = "%s/%s_visss_%s/%s/%s/%s" % (config["pathOut"].format(level="level0", site=site), computer, camera, year, month, day)
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




        
