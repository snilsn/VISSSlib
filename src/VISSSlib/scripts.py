import sys
import os
import shutil
import socket
import datetime
import multiprocessing
import subprocess
import collections
import logging
import time
import shlex
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr

from . import quicklooks
from . import files
from . import tools
from . import metadata
from . import matching
from . import distributions

from . import __version__

log = logging.getLogger(__name__)



def loopLevel0Quicklook(settings, version=__version__, skipExisting=True, nDays=0):

    config = tools.readSettings(settings)
    instruments = config["instruments"]
    computers = config["computers"]    
    
    days = tools.getDateRange(nDays, config, endYesterday=False)


    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"
        print(case)
        for computer, camera in zip(computers, instruments):
            quicklooks.level0Quicklook(
                case, camera, config, version=version, skipExisting=skipExisting)


def loopMetaFramesQuicklooks(settings, version=__version__, skipExisting=True, nDays=0):

    config = tools.readSettings(settings)
    instruments = config["instruments"]
    computers = config["computers"]    
    
    days = tools.getDateRange(nDays, config, endYesterday=False)


    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"
        print(case)
        for computer, camera in zip(computers, instruments):
            fOut, fig = quicklooks.metaFramesQuicklook(case, camera, config, version=version, skipExisting=skipExisting)
            try:
                fig.close()
            except AttributeError:
                pass

def loopMetaRotationQuicklooks(settings, version=__version__, skipExisting=True, nDays=0):

    config = tools.readSettings(settings)
    instruments = config["instruments"]
    computers = config["computers"]    
    
    days = tools.getDateRange(nDays, config, endYesterday=False)


    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"
        print(case)
        fOut, fig = quicklooks.metaRotationQuicklook(case, config, version=version, skipExisting=skipExisting)
        try:
            fig.close()
        except AttributeError:
            pass

def loopLevel1detectQuicklooks(settings, version=__version__, nDays = 0, skipExisting=True):

    config = tools.readSettings(settings)
    instruments = config["instruments"]
    computers = config["computers"]

    days = tools.getDateRange(nDays, config)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for computer, camera in zip(computers, instruments):
            #         print(case, computer, camera)

            fname, fig = quicklooks.createLevel1detectQuicklook(
                case, 
                camera, 
                config, 
                version=version, 
                skipExisting=skipExisting,
                )
            try:
                fig.close()
            except AttributeError:
                pass
    return


def loopLevel1matchQuicklooks(settings, version=__version__, nDays = 0, skipExisting=True, plotCompleteOnly=True):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config, endYesterday=False)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        #         print(case, computer, camera)

        fname, fig = quicklooks.createLevel1matchQuicklook(
            case, 
            config, 
            version=version, 
            skipExisting=skipExisting,
            plotCompleteOnly=plotCompleteOnly,
            )
        try:
            fig.close()
        except AttributeError:
            pass
    return

def loopLevel1matchParticlesQuicklooks(settings, version=__version__, nDays = 0, skipExisting=True):

    config = tools.readSettings(settings)
    days = tools.getDateRange(nDays, config, endYesterday=False)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        #         print(case, computer, camera)

        fname, fig = quicklooks.createLevel1matchParticlesQuicklook(
            case, 
            config, 
            version=version, 
            skipExisting=skipExisting,
            )
        try:
            fig.close()
        except AttributeError:
            pass
    return


def loopLevel2matchQuicklooks(settings, version=__version__, nDays = 0, skipExisting=True):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config, endYesterday=False)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        #         print(case, computer, camera)

        fname, fig = quicklooks.createLevel2matchQuicklook(
            case, 
            config, 
            version=version, 
            skipExisting=skipExisting,
            )
        try:
            fig.close()
        except AttributeError:
            pass
    return


def loopMetaCoefQuicklooks(settings, version=__version__, skipExisting=True):

    config = tools.readSettings(settings)

    if config["end"] == "today":
        end = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    else:
        end = config["end"]

    for dd in pd.date_range(
        start=config["start"],
        end=end,
        freq="1D",
        tz=None,
        normalize=True,
        name=None,
        inclusive=None
    ):

        # , periods=nDays´´

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        fname, fig = quicklooks.createMetaCoefQuicklook(
            case, config, version=version, skipExisting=skipExisting)
        try:
            fig.close()
        except AttributeError:
            pass


def loopCreateEvents(settings, skipExisting=True, nDays = 0):


    config = tools.readSettings(settings)


    days = tools.getDateRange(nDays, config, endYesterday=False)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for camera in config.instruments:
            metadata.createEvent(case, camera, config, skipExisting=skipExisting)

# def loopCreateLevel2match(settings, skipExisting=True, nDays = 0, doPlot=True):


#     config = tools.readSettings(settings)

#     days = tools.getDateRange(nDays, config, endYesterday=False)

#     for dd in days:

#         year = str(dd.year)
#         month = "%02i" % dd.month
#         day = "%02i" % dd.day
#         case = f"{year}{month}{day}"

#         distributions.createLevel2match(case, config, skipExisting=skipExisting)

#         if doPlot:
#             fname, fig = quicklooks.createLevel2matchQuicklook(
#                 case, 
#                 config, 
#                 skipExisting=skipExisting,
#                 )
#             try:
#                 fig.close()
#             except AttributeError:
#                 pass


def loopCreateLevel2match(settings, skipExisting=True, nDays = 0, useWorker=True, nCPU=None, doPlot=True):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config, endYesterday=False)
    leader = config["leader"]
    follower = config["follower"]
    
    if useWorker and len(days)>1:
        if nCPU is None:
            nCPU = os.cpu_count()//2
        days = [str(day).split(" ")[0].replace("-","") for day in days]

        doWork = partial(loopCreateLevel2matchWorker, settings=settings, skipExisting=skipExisting)
        with multiprocessing.Pool(nCPU) as p:
            for i, r in enumerate(p.imap(doWork, days)):
                log.warning(f'done {i} of {len(days)} files with result {r}')

    else:
        for dd in days:

            year = str(dd.year)
            month = "%02i" % dd.month
            day = "%02i" % dd.day
            case = f"{year}{month}{day}"

            distributions.createLevel2match(case, config, skipExisting=skipExisting)

    if doPlot:
        for dd in days:
            year = str(dd.year)
            month = "%02i" % dd.month
            day = "%02i" % dd.day
            case = f"{year}{month}{day}"
            fname, fig = quicklooks.createLevel2matchQuicklook(
                case, 
                config, 
                skipExisting=skipExisting,
                )
            try:
                fig.close()
            except AttributeError:
                pass
        return



def loopCreateLevel2matchWorker(day, settings=None, skipExisting=True, nCPU=1):
    log = logging.getLogger()

    BIN = os.path.join(sys.exec_prefix, "bin", "python")

    if nCPU is None:
        command = f"{BIN} -m VISSSlib scripts.loopCreateLevel2match {settings} {day} {int(skipExisting)}"
    else:
        command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {BIN} -m VISSSlib scripts.loopCreateLevel2match   {settings} {day} {int(skipExisting)} 0"

    log.info(command)

    config = tools.readSettings(settings)
    fL = files.FindFiles(day, config.leader, config)
    fL.createDirs()
    lv2File = fL.fnamesDaily.level2match
    tmpFile = os.path.basename('%s.processing.txt' % lv2File)

    if skipExisting and ( os.path.isfile(tmpFile)):
            log.info(f'output processing {lv2File}')
            return 0
    if skipExisting and ( os.path.isfile('%s' % lv2File)):
        # log.info('output exists %s %s' % (fname, lv2File))
        return 0
    elif skipExisting and ( os.path.isfile('%s.nodata' % lv2File)):
        # log.info('output exists %s %s' % (fname, lv2File))
        return 0
    elif skipExisting and (os.path.isfile('%s.broken.txt' % lv2File)):
        log.error('output already broken %s.broken.txt' % (lv2File))
        return 1
    elif skipExisting and ( os.path.isfile(tmpFile)):
        log.info('output processing %s' % (lv2File))
        return 0
    else:
        pass

    success = _runCommand(command, tmpFile, lv2File)
    return success


def loopCreateMetaFrames(settings, skipExisting=True, nDays = 0, cameras = "all", doPlot=True):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config, endYesterday=False)

    if cameras == "all":
        cameras = [config.follower, config.leader]

    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for camera in cameras:

            metadata.createMetaFrames(case, camera, config, skipExisting=skipExisting)
            if doPlot:
                fOut, fig = quicklooks.metaFramesQuicklook(case, camera, config, skipExisting=skipExisting)
                try:
                    fig.close()
                except AttributeError:
                    pass

    return

def loopCreateMetaRotation(settings, skipExisting=True, nDays = 0, doPlot=True):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config, endYesterday=False)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        matching.createMetaRotation(case, config, skipExisting=skipExisting)
        if doPlot:
            fOut, fig = quicklooks.metaRotationQuicklook(case, config, skipExisting=skipExisting)
            try:
                fig.close()
            except AttributeError:
                pass

    return




def loopCreateLevel1detectWorker(fname, settings, skipExisting=True, nCPU=1):

    '''
    We need this worker function becuase  we want to do the detection by indipendent 
    processes in parallel
    '''
    config = tools.readSettings(settings)

    errorlines = collections.deque([], 50)
    

    #avoid tmp file race condition
    time.sleep(np.random.random())
            
    fn = files.Filenames(fname, config, __version__)
    camera = fn.camera
    
    fn.createDirs()
    tmpFile = os.path.basename('%s.processing.txt' % fn.fname.level1detect)
    if skipExisting and ( os.path.isfile('%s' % fn.fname.level1detect)):
        # log.info('output exists %s %s' % (fname, fn.fname.level1detect))
        return 0
    elif skipExisting and ( os.path.isfile('%s.nodata' % fn.fname.level1detect)):
        # log.info('output exists %s %s' % (fname, fn.fname.level1detect))
        return 0
    elif skipExisting and (os.path.isfile('%s.broken.txt' % fn.fname.level1detect)):
        log.error('output already broken %s %s.broken.txt' % (fname, fn.fname.level1detect))
        return 1
    elif skipExisting and ( os.path.isfile('%s.nodata' % fn.fname.metaFrames)):
        log.info('metaFrames contains no data %s %s' % (fname, fn.fname.metaFrames))
        with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
            f.write('metaFrames contains no data %s %s' % (fname, fn.fname.metaFrames))
        return 0
    elif skipExisting and ( os.path.isfile(tmpFile)):
        log.info('output processing %s %s' % (fname, fn.fname.level1detect))
        return 0
    else:
        pass

    BIN = os.path.join(sys.exec_prefix, "bin", "python")
    if nCPU is None:
        command = f"{BIN} -m VISSSlib detection.detectParticles  {fname} {settings}"
    else:
        command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {BIN} -m VISSSlib detection.detectParticles  {fname} {settings}"

    success = _runCommand(command, tmpFile, fn.fname.level1detect)
    return 0


def loopCreateLevel1matchWorker(fnameL1detect, settings, skipExisting=True, nCPU=1):

    '''
    We need this worker function becuase  we want to do the matching by indipendent 
    processes in parallel
    '''
    config = tools.readSettings(settings)

    errorlines = collections.deque([], 50)
    

    #avoid tmp file race condition
    time.sleep(np.random.random())
            

    ffl1  = files.FilenamesFromLevel(fnameL1detect, config)
    ffl1.createDirs()

    fname1Match = ffl1.fname["level1match"]
    ffl1.createDirs()
    tmpFile = os.path.basename('%s.processing.txt' % fname1Match)

    if fnameL1detect.endswith("broken.txt") or fnameL1detect.endswith("nodata") or fnameL1detect.endswith("notenoughframes"):
        ffl1.createDirs()
        with open(f"{fname1Match}.nodata", "w") as f:
            f.write("no leader data")
        log.info(f"NO leader DATA {fname1Match}")
        return 0

    elif os.path.isfile(fname1Match) and skipExisting:
        log.info(f"SKIPPING {fname1Match}")
        return 0
    elif os.path.isfile('%s.broken.txt' % fname1Match) and skipExisting:
        log.info(f"SKIPPING BROKEN {fname1Match}")
        return 0
    elif os.path.isfile('%s.nodata' % fname1Match) and skipExisting:
        log.info(f"SKIPPING nodata {fname1Match}")
        return 0
    elif skipExisting and ( os.path.isfile(tmpFile)):
        log.info('output processing %s %s' % (fnameL1detect, ffl1.fname.fname1Match))
        return 0

    BIN = os.path.join(sys.exec_prefix, "bin", "python")
    if nCPU is None:
        command = f"{BIN} -m VISSSlib matching.matchParticles  {fnameL1detect} {settings}"
    else:
        command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {BIN} -m VISSSlib matching.matchParticles  {fnameL1detect} {settings}"

    success = _runCommand(command, tmpFile, fname1Match)

    return 0

def _runCommand(command, tmpFile, fOut, stdout=subprocess.DEVNULL):
    success = True
    # with statement extended to avoid race conditions
    with open(tmpFile, 'w') as f:
        f.write('PID & Host: %i %s\n' % (os.getpid(), socket.gethostname()))
        f.write("Command: %s\n"%command)
        f.write("Outfile: %s\n"%fOut)
        f.write("#########################\n")
        f.flush()
        log.info(f"written {tmpFile} in {os.getcwd()}")
        log.info(command)

        #proc = subprocess.Popen(shlex.split(f'bash -c "{command}"'), stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        proc = subprocess.Popen(command, shell=True, stdout=stdout, stderr=subprocess.PIPE)

        # Poll process for new output until finished
        if proc.stdout is not None:
            for line in proc.stdout: 
                line = line.decode()
                log.info(line)
                f.write(line)
                f.flush()
                output += line
        for line in proc.stderr: 
            line = line.decode()
            log.error(line)
            f.write(line)
            f.flush()
    
        proc.wait()
        exitCode = proc.returncode
        if exitCode != 0:
            success = False
            log.error(f"{fOut} BROKEN {exitCode}")
        else:
            log.info(f"{fOut} SUCCESS {exitCode}")

    if not success:
        shutil.copy(tmpFile, '%s.broken.txt' % tmpFile)
        try:
            shutil.copy(tmpFile, '%s.broken.txt' % fOut)
        except:
            pass
    try:
        os.remove(tmpFile)
    except:
        pass

    return success


def loopCreateLevel1detect(settings, skipExisting=True, nDays = 0, cameras = "all", nCPU=None):
    config = tools.readSettings(settings)
    log = logging.getLogger()


    days = tools.getDateRange(nDays, config, endYesterday=False)
    if cameras == "all":
        cameras = [config.follower, config.leader]

    if nCPU is None:
        nCPU = os.cpu_count()

    fnames = []
    p = None

    for dd in days:
        # create list of files
        year  =str(dd.year)
        month  ="%02i"%(dd.month)
        day  ="%02i"%(dd.day)      
        case = f"{year}{month}{day}"
        for camera in cameras:
            # find files
            ff = files.FindFiles(case, camera, config)
            fname0s =  ff.listFiles("level0txt")
            fnames += filter( os.path.isfile,
                                    fname0s )

        # Sort list of files in directory by size 
        fnames = sorted( fnames,
                                key =  lambda x: os.stat(x).st_size)[::-1]
    if len(fnames) == 0:
        log.error('no files to process %s' % "level0txt")
        # print('no files %s' % fnamesPattern)

    else:
        
        log.info(f"found {len(fnames)} files, lets do it:")
        
        #p.map(partial(loopCreateLevel1detectWorker, settings=settings, skipExisting=skipExisting), fnames)
        doWork = partial(loopCreateLevel1detectWorker, settings=settings, skipExisting=skipExisting)
        with multiprocessing.Pool(nCPU) as p:
            for i, r in enumerate(p.imap(doWork, fnames)):
                log.warning(f'done {i} of {len(fnames)} files with result {r}')

        log.info(f"processed all {len(fnames)} files")
    return


def loopCreateLevel1match(settings, skipExisting=True, nDays = 0,  nCPU=None):
    config = tools.readSettings(settings)
    log = logging.getLogger()


    days = tools.getDateRange(nDays, config, endYesterday=False)

    if nCPU is None:
        nCPU = os.cpu_count()

    fnames = []
    p = None

    for dd in days:
        # create list of files
        year  =str(dd.year)
        month  ="%02i"%(dd.month)
        day  ="%02i"%(dd.day)      
        case = f"{year}{month}{day}"
            # find files
        ff = files.FindFiles(case, config.leader, config)
        if len(ff.listFiles("metaRotation")) == 0:
            log.error(tools.concat("SKIPPING", dd, "no rotation file yet"))
            continue

        fname0s =  ff.listFilesExt("level1detect")
        fnames += filter( os.path.isfile,
                                fname0s )

        # Sort list of files in directory by size 
        fnames = sorted( fnames, key =  lambda x: os.stat(x).st_size)[::-1]
    
    if len(fnames) == 0:
        log.error('no files to process %s' % "level1detect")
        # print('no files %s' % fnamesPattern)
    else:
        log.info(f"found {len(fnames)} files, lets do it:")

        doWork = partial(loopCreateLevel1matchWorker, settings=settings, skipExisting=skipExisting)
        #p.map(partial(loopCreateLevel1matchWorker, settings=settings, skipExisting=skipExisting), fnames)
        with multiprocessing.Pool(nCPU) as p:
            for i, r in enumerate(p.imap(doWork, fnames)):
                log.warning(f'done {i} of {len(fnames)} files with result {r}')

    return




def loopCheckCompleteness(settings, nDays = 0, cameras = "all", listDuplicates=True, listMissing=False, products = ["metaFrames", "level1detect", 'metaRotation',"level1match", "level2match"]):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config)

    if cameras == "all":
        cameras = [config.follower, config.leader]

    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for camera in cameras:
            # find files
            ff = files.FindFiles(case, camera, config)

            nMissing = []
            for prod in products:
                if camera == config.follower and ((prod in ["level1match",'metaRotation']) or prod.startswith("level2")):
                    continue
                nMissing.append(ff.nMissing(prod))
            allDone = np.array(nMissing) == 0

            if np.all(allDone):
                print(camera, case, "all done", np.all(allDone))
            else:
                print(camera, case, "MISSING", nMissing, "of", ff.nL0)

                firstMiss = products[np.where(np.array(allDone) == False)[0][0]]
                recFiles = np.array(ff.listFiles("level0txt"))
                nRec = len(recFiles)
                procFiles = np.array(ff.listFilesExt(firstMiss))
                nProc = len(procFiles)
                print("# level0 has", nRec , "files #", firstMiss, "has only", nProc, "files.")

                processedTimes = np.array([files.FilenamesFromLevel(f, config).datetime64 for f in ff.listFilesExt(firstMiss)])


                if listDuplicates and nProc > nRec:
                    print("too many files processed, check these files:")
                    print("*"*50)
                    seen = set()
                    dupes = [x for x in processedTimes if x in seen or seen.add(x)]    
                    dupeFiles = []
                    for dupe in dupes:
                        dupeFiles.append(procFiles[(dupe == processedTimes)])
                    if len(dupeFiles) > 0:
                        dupeFiles = np.concatenate(dupeFiles)
                        for dupeFile in dupeFiles: print(dupeFile)
                elif listMissing:
                    print("files missing")
                    print("*"*50)

                    recTimes = [files.Filenames(f, config).datetime64 for f in ff.listFiles("level0txt")]
                    missingTimes = set(recTimes).difference( set(processedTimes))
                    for missingTime in missingTimes:
                        print(camera, firstMiss, missingTime)
    return


def reportLastFiles(settings, writeFile=True, products = ["level0txt","level0", "metaFrames", "level1detect", 'metaRotation',"level1match", "level2match"]):

    config = tools.readSettings(settings)
    days = tools.getDateRange(0, config, endYesterday=False)[::-1]

    cameras = [config.follower, config.leader]
    output = ""

    output += "#"*80
    output += "\n"
    output += f"Last available files for {config.site} at {datetime.datetime.utcnow()} UTC\n"
    output += "#"*80
    output += "\n"

    for prod in products:
        for camera in cameras:
            if camera == config.follower and ((prod in ["level1match",'metaRotation']) or prod.startswith("level2")):
                continue

            foundLastFile = False
            foundComplete = False
            completeCase = "nan"
            lastFile = "nan"
            
            for dd in days:
                year = str(dd.year)
                month = "%02i" % dd.month
                day = "%02i" % dd.day
                case = f"{year}{month}{day}"

                # find files
                ff = files.FindFiles(case, camera, config)
                if not foundLastFile:
                    fnames = ff.listFilesExt(prod)
                    if len(fnames) > 0:
                        try:
                            f1 = files.FilenamesFromLevel(fnames[-1], config)
                        except ValueError:
                            f1 = files.Filenames(fnames[-1], config)
                        foundLastFile = True
                        lastFile = f1.datetime
                        
                if not foundComplete:
                    foundComplete = ff.isComplete(prod) 
                    completeCase = case
                        
                
                if foundComplete and foundLastFile:
                    output += f"{prod.ljust(14)} {(camera.split('_')[0]).ljust(8)} last full day:'{completeCase}' last file:'{lastFile}'\n"
                    break
                
                    
    output += "#"*80
    output += "\n"

    if writeFile:
        fOut = f"{config['pathQuicklooks'].format(version=f1.version,site=config['site'], level='')}/{'productReport'}_{config['site']}.html"
        with open(fOut, "w") as f:
            f.write("<html><pre>\n")
            f.write(output)
            f.write("</pre></html>\n")

    return output