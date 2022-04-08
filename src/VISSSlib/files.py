# -*- coding: utf-8 -*-


import sys
import glob
import os
import datetime
from copy import deepcopy
import functools


import numpy as np
import pandas as pn

import logging
log = logging.getLogger()

from .tools import nicerNames, otherCamera



class FindFiles(object):
    def __init__(self, case, camera, config, version):
        
        '''
        find all files corresponding to certain case

        for level 0, only thread 0 files are returned!
        '''

        self.case = case
        self.camera = camera
        self.config = config
        self.version = version

        
        computerDict = {}
        for computer1, camera1 in zip(config["computers"], config["instruments"]):
            computerDict[camera1] = computer1
        self.computer = computerDict[camera]

        self.year  =case[:4]
        self.month  =case[4:6]
        self.day  =case[6:8]        
        try:
            self.timestamps  =case[9:]        
        except IndexError:
            self.timestamps = None
            

        self.outpath0 = config["path"].format(site=config["site"], level='0')+f'/{self.computer}_{config["visssGen"]}_{camera}/{self.year}/{self.month}/{self.day}'
        
        outpath = "%s/%s/%s/%s" % (config["pathOut"], self.year, self.month, self.day)
        self.outpath1 = outpath.format(site=config["site"], level='1')
        self.outpath2 = outpath.format(site=config["site"], level='2')
        self.outpath3 = outpath.format(site=config["site"], level='3')

        if config["nThreads"] is None:
            self.fnamesPattern0 = '%s/*%s*.%s' % (
                self.outpath0, case, config["movieExtension"])
        else:
            self.fnamesPattern0 = '%s/*%s*_0.%s' % (
                self.outpath0, case, config["movieExtension"])

        self.fnamesPattern1 = "%s/level1_V%s*%s*%s*nc"%(self.outpath1, version, camera, case)
        self.fnamesPattern2 = "%s/level2_V%s*%s*%s*nc"%(self.outpath2, version, camera, case)
        self.fnamesPattern3 = "%s/level3_V%s*%s*%s*nc"%(self.outpath3, version, camera, case)

        self.fnamesPattern1Ext = "%s/level1_V%s*%s*%s*nc.[b,n]*"%(self.outpath1, version, camera, case) #finds broken & nodata
        self.fnamesPattern2Ext = "%s/level2_V%s*%s*%s*nc.[b,n]*"%(self.outpath2, version, camera, case) #finds broken & nodata
        self.fnamesPattern3Ext = "%s/level3_V%s*%s*%s*nc.[b,n]*"%(self.outpath3, version, camera, case) #finds broken & nodata

        self.outpathImg = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)
        
        self.quicklookPath1 = f"/projekt4/ag_maahn/quicklooks/{config['site']}/visss/level1/{self.year}/{self.month}/{self.day}"
        self.quicklookPath2 = f"/projekt4/ag_maahn/quicklooks/{config['site']}/visss/level2/{self.year}/{self.month}/{self.day}"
        self.quicklookPath3 = f"/projekt4/ag_maahn/quicklooks/{config['site']}/visss/level3/{self.year}/{self.month}/{self.day}"

        
        self.quicklook1 = f"{self.quicklookPath1}/level1_V{version}_{config['site']}_{self.computer}_{nicerNames(camera)}_{self.year}{self.month}{self.day}.png"
        self.quicklook3 = f"{self.quicklookPath3}/level3_V{version}_{config['site']}_{self.year}{self.month}{self.day}.png"

        
    @functools.cached_property
    def fnames0(self):
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern0) ))
    
    @functools.cached_property
    def fnames1(self):


        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern1) ))
    @functools.cached_property
    def fnames2(self):
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern2) ))

    @functools.cached_property
    def fnames3(self):
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern3) ))


    @functools.cached_property
    def fnames1Ext(self):
        # includes empty and broken data
        return sorted(self.fnames1 + list(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern1Ext) )))

    @functools.cached_property
    def fnames2Ext(self):
        # includes empty and broken data
        return sorted(self.fnames2 + list(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern2Ext) )))


    @functools.cached_property
    def fnames3Ext(self):
        # includes empty and broken data
        return sorted(self.fnames3 + list(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern3Ext) )))


    @functools.cached_property
    def isComplete(self):
        print("replace isComplete with isCompleteL1!")
        return (len(self.fnames0) == len(self.fnames1Ext))

    @functools.cached_property
    def isCompleteL1(self):
        return (len(self.fnames0) == len(self.fnames1Ext))


    @functools.cached_property
    def isCompleteL3(self):
        return (len(self.fnames2) == len(self.fnames3Ext))


        
    def createQuicklookDirs(self):
        res1 = os.system('mkdir -p %s' %
                  self.quicklookPath1)
        res3 = os.system('mkdir -p %s' %
                  self.quicklookPath3)

        return res1, 1, res3


class Filenames(object):
    def __init__(self, fname, config, version):
        '''
        create matching filenames based on mov file
        Use always thread 0 file!
        '''
        self.fname = fname
        self.config = config
        self.version = version

        self.basename = os.path.basename(fname).split('.')[0]
        self.dirname = os.path.dirname(fname)

        if config["nThreads"] is None:
            ts = self.basename.split("_")[-1]
        else:
            ts = self.basename.split("_")[-2]

        self.year = ts[:4]
        self.month = ts[4:6]
        self.day = ts[6:8]
        self.timestamp = ts[-6:]

        self.datetime = datetime.datetime.strptime(ts, "%Y%m%d-%H%M%S")
        self.datetime64= np.datetime64(self.datetime)

        if config["nThreads"] is not None:
            self.basename = '_'.join(self.basename.split('_')[:-1])
            
        self.camera = "_".join(self.basename.split("_")[2:4])

        self.outpath = "%s/%s/%s/%s" % (config["pathOut"], self.year, self.month, self.day)
        self.fnameLevel0 = self.fname
        self.fnameLevel1 = '%s/level1_V%s_%s_%s.nc' % (
            self.outpath.format(site=config["site"], level='1'), version, config["site"], self.basename)
        self.fnameLevel2 = '%s/level2_V%s_%s_%s.nc' % (
            self.outpath.format(site=config["site"], level='2'), version, config["site"], self.basename)
        self.fnameLevel3 = '%s/level3_V%s_%s_%s.nc' % (
            self.outpath.format(site=config["site"], level='3'), version, config["site"], self.basename)

        self.outpathImg = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)
        self.fnameLevel2images = "%s/%s/{ppid}"%(self.outpathImg.format(site=config["site"], level='2images'),self.fnameLevel2.split("/")[-1])
        
        return

        
    def createDirs(self):
        res1 = os.system('mkdir -p %s' %
                  self.outpath.format(site=self.config["site"], level='1'))
        res2 = os.system('mkdir -p %s' %
                  self.outpath.format(site=self.config["site"], level='2'))
        res3 = os.system('mkdir -p %s' %
                  self.outpath.format(site=self.config["site"], level='3'))
        
        return res1, res2, res3


    def filenamesOtherCamera(self, graceInterval = 120, level="2"):
        '''
        Find all relevant files of the other camera of ´level´. ´graceinterval´ accounts for 
        potential time offsets
        '''
        

        otherCam = otherCamera(self.camera, self.config)

        case = f"{self.year}{self.month}{self.day}"

        ff = FindFiles(case, otherCam, self.config, self.version)
        # get fnames for correct level
        fnames = ff.__getattribute__(f"fnames{level[-1]}")
        
        thisDayStart = self.datetime.replace(hour=0, minute=0, second=0)
        nextDayStart = thisDayStart + datetime.timedelta(days=1)
        prevDayStart = thisDayStart - datetime.timedelta(days=1)

        #check for early files where we need to consider the prev day
        earlyFile = (self.datetime - thisDayStart) <= datetime.timedelta(seconds=(self.config["newFileInt"]+ graceInterval))
        if earlyFile:
            prevCase = datetime.datetime.strftime(prevDayStart, "%Y%m%d")
            prevFf = FindFiles(prevCase, otherCam, self.config, self.version)
            fnames = prevFf.__getattribute__(f"fnames{level[-1]}") + fnames
        
        #same for late files
        lateFile = (nextDayStart - self.datetime) <= datetime.timedelta(seconds=self.config["newFileInt"] + graceInterval)
        if lateFile:
            nextCase = datetime.datetime.strftime(nextDayStart, "%Y%m%d")
            nextFf = FindFiles(nextCase, otherCam, self.config, self.version)
            fnames += nextFf.__getattribute__(f"fnames{level[-1]}")
        
        # get timestamps of surrounding files
        ts = np.array([f.split("_")[-1][:-3] for f in fnames])
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
    def fnameAllThreads(self):
        '''
        find level 0 fnames of other threads
        '''
        fname0All = list()
        for nThread in range(self.config["nThreads"]):
            
            thisFname = self.fname.replace(f'_0.{self.config["movieExtension"]}', f'_%i.{self.config["movieExtension"]}'%nThread)
            
            #sometime the second changes while the new thread file is written, fix it:
            if not os.path.isfile(thisFname):
                thisSplits = thisFname.split("_")
                thisTime = datetime.datetime.strptime(thisSplits[-2], "%Y%m%d-%H%M%S")
                thisTime = (thisTime + datetime.timedelta(seconds=1)).strftime("%Y%m%d-%H%M%S")
                thisSplits[-2] = thisTime
                thisFname = "_".join(thisSplits)
                # it can be even BEFORE the _0 file:
                if not os.path.isfile(thisFname):
                    thisSplits = thisFname.split("_")
                    thisTime = datetime.datetime.strptime(thisSplits[-2], "%Y%m%d-%H%M%S")
                    #remove two, becuase 1 has been already added
                    thisTime = (thisTime + datetime.timedelta(seconds=-2)).strftime("%Y%m%d-%H%M%S")
                    thisSplits[-2] = thisTime
                    thisFname = "_".join(thisSplits)
                #now it should work!
                assert os.path.isfile(thisFname)

            fname0All.append(thisFname)
        return fname0All

    @functools.cached_property
    def nextFile(self):
        return self.findNeighborFile(+1)
    @functools.cached_property
    def prevFile(self):
        return self.findNeighborFile(-1)
    
    def findNeighborFile(self, offset):
        '''
        find file at difstance of x offsets
        '''
        dirname = os.path.dirname(self.fname)
        case = self.year+self.month+self.day
        allFiles = FindFiles(case, self.camera, self.config, self.version).fnames0
        thisFileI = allFiles.index(self.fname)
        neighborFileI = thisFileI + offset
        #neighbor is on a different day
        if (neighborFileI >= len(allFiles) or (neighborFileI < 0)):
            dirParts = self.dirname.split("/")
            dirParts[-3:] = ["*", "*", "*"]
            allDayFolders = glob.glob("/".join(dirParts))
            neighborDayFolderI = allDayFolders.index(self.dirname) + offset
            if (neighborDayFolderI >= len(allDayFolders)) or (neighborDayFolderI < 0):
                 # no neighbor file!
                return None
            neighborDayFolder = allDayFolders[neighborDayFolderI]
            year, month, day = neighborDayFolder.split("/")[-3:]
            neighborCase = "".join([year, month, day])
            allNeighborFiles = FindFiles(neighborCase, self.camera, self.config, self.version).fnames0
            assert offset in [1, -1], "other offsets than 1, -1 not implemented yet!"
            if offset > 0:
                neighborFile = allNeighborFiles[0]
            else:
                neighborFile = allNeighborFiles[-1]
        else:
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

        outpath0 = "%s/%s_visss_%s/%s/%s/%s" % (config["pathOut"].format(level=0, site=site), computer, camera, year, month, day)
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
    #     self.datetime64= np.datetime64(self.datetime)

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




        
