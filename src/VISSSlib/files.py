# -*- coding: utf-8 -*-


import sys
import glob
import os
from copy import deepcopy

import logging
log = logging.getLogger()

from .tools import nicerNames



class FindFiles(object):
    def __init__(self, case, camera, config, version):
        
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

        if config["nThreads"] is None:
            self.fnamesPattern0 = '%s/*%s*.%s' % (
                self.outpath0, case, config["movieExtension"])
        else:
            self.fnamesPattern0 = '%s/*%s*_0.%s' % (
                self.outpath0, case, config["movieExtension"])

        self.fnamesPattern1 = "%s/level1_V%s*%s*%s*nc"%(self.outpath1, version, camera, case)
        self.fnamesPattern2 = "%s/level2_V%s*%s*%s*nc"%(self.outpath2, version, camera, case)

        self.fnamesPattern1Ext = "%s/level1_V%s*%s*%s*nc.[b,n]*"%(self.outpath1, version, camera, case) #finds broken & nodata
        self.fnamesPattern2Ext = "%s/level2_V%s*%s*%s*nc.[b,n]*"%(self.outpath2, version, camera, case) #finds broken & nodata

        self.outpathImg = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)
        
        self.quicklookPath1 = f"/projekt4/ag_maahn/quicklooks/{config['site']}/visss/level1/{self.year}/{self.month}/{self.day}"
        self.quicklookPath2 = f"/projekt4/ag_maahn/quicklooks/{config['site']}/visss/level2/{self.year}/{self.month}/{self.day}"
        self.quicklookPath3 = f"/projekt4/ag_maahn/quicklooks/{config['site']}/visss/level3/{self.year}/{self.month}/{self.day}"

        
        self.quicklook1 = f"{self.quicklookPath1}/level1_V{version}_{config['site']}_{self.computer}_{nicerNames(camera)}_{self.year}{self.month}{self.day}.png"

        
    @property
    def fnames0(self):
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern0) ))
    
    @property
    def fnames1(self):
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern1) ))
    @property
    def fnames2(self):
        return sorted(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern2) ))

    @property
    def fnames1Ext(self):
        return sorted(self.fnames1 + list(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern1Ext) )))

    @property
    def fnames2Ext(self):
        return sorted(self.fnames2 + list(filter( os.path.isfile,
                                glob.glob(self.fnamesPattern2Ext) )))

    @property
    def isComplete(self):
        return (len(self.fnames0) == len(self.fnames1Ext))


        
    def createQuicklookDirs(self):
        res1 = os.system('mkdir -p %s' %
                  self.quicklookPath1)

        return res1, 1, 1


class Filenames(object):
    def __init__(self, fname, config, version):
        
        self.config = config
        self.basename = os.path.basename(fname).split('.')[0]
        if config["nThreads"] is None:
            ts = self.basename.split("_")[-1]
        else:
            ts = self.basename.split("_")[-2]

        self.year = ts[:4]
        self.month = ts[4:6]
        self.day = ts[6:8]
        self.timestamp = ts[-6:]

        if config["nThreads"] is not None:
            self.basename = '_'.join(self.basename.split('_')[:-1])
            
        self.camera = "_".join(self.basename.split("_")[2:4])

        self.outpath = "%s/%s/%s/%s" % (config["pathOut"], self.year, self.month, self.day)
        self.out_level1 = '%s/level1_V%s_%s_%s.nc' % (
            self.outpath.format(site=config["site"], level='1'), version, config["site"], self.basename)
        self.out_level2 = '%s/level2_V%s_%s_%s.nc' % (
            self.outpath.format(site=config["site"], level='2'), version, config["site"], self.basename)

        self.outpathImg = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)
        self.out_level2images = "%s/%s/{ppid}"%(self.outpathImg.format(site=config["site"], level='2images'),self.out_level2.split("/")[-1])
        
        
        return

        
    def createDirs(self):
        res1 = os.system('mkdir -p %s' %
                  self.outpath.format(site=self.config["site"], level='1'))
        res2 = os.system('mkdir -p %s' %
                  self.outpath.format(site=self.config["site"], level='2'))
        
        return res1, res2


class FilenamesFromLevel(object):
    def __init__(self, fname, config):
        
        
        _, self.version, self.site, self.computer, self.visssGen, visssType, visssSerial, ts = fname.split("/")[-1].split("_")
        self.case = ts.split(".")[0]
        self.camera = "_".join((visssType, visssSerial))

        
        self.config = config
        self.basename = "_".join((self.computer, self.visssGen, visssType, visssSerial, self.case))

        self.year = self.case[:4]
        self.month = self.case[4:6]
        self.day = self.case[6:8]
        self.timestamp = self.case[-6:]

        self.outpath = "%s/%s/%s/%s" % (config["pathOut"], self.year, self.month, self.day)
        self.out_level1 = '%s/level1_%s_%s_%s.nc' % (
            self.outpath.format(site=self.site, level='1'), self.version, self.site, self.basename)
        self.out_level2 = '%s/level2_%s_%s_%s.nc' % (
            self.outpath.format(site=self.site, level='2'), self.version, self.site, self.basename)

        self.outpathImg = "%s/%s/%s/%s" % (config["pathTmp"], self.year, self.month, self.day)
        self.out_level2images = "%s/%s/{ppid}"%(self.outpathImg.format(site=self.site, level='2images'),self.out_level2.split("/")[-1])
        
        
        return
