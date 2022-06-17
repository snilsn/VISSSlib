import sys
import os
import socket
import datetime
import multiprocessing
import subprocess
import collections
import logging
import shlex
from functools import partial

import pandas as pd


from . import quicklooks
from . import files
from . import tools
from . import metadata
from . import __version__


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

            f, i = quicklooks.createLevel1detectQuicklook(
                case, camera, config, version=version, skipExisting=skipExisting)
            try:
                i.close()
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


    days = tools.getDateRange(nDays, config)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for camera in config.instruments:
            
            fn = files.FindFiles(case, camera, config, __version__)
            fnames0 = fn.listFiles("level0")

            if len(fnames0) == 0:
                print("no data", case )
                continue

            fname0status = fn.listFiles(level="level0status")
            if len(fname0status) > 0:
                fname0status = fname0status[0]  
            else:
                fname0status = None
                
            ff = files.Filenames(fnames0[0], config, __version__)
            outFile = ff.fname.metaEvents

            if skipExisting and os.path.isfile(outFile):
                print("Skipping", case, outFile )
                continue
                
            ff.createDirs()

            print("Running",case, outFile )
            metaDats = metadata.getEvents(fnames0, config, fname0status=fname0status)
            try:
                metaDats.to_netcdf(outFile)
            except AttributeError:
                print("NO DATA",case, outFile )


def loopCreateMetaFrames(settings, skipExisting=True, nDays = 0, cameras = "all"):

    config = tools.readSettings(settings)

    if nDays == 0:
        if config["end"] == "today":
            end = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        else:
            end = config["end"]

        days = pd.date_range(
            start=config["start"],
            end=end,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )
    else:
        end = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        days = pd.date_range(
            end=end,
            periods=nDays,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )

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

            for fname0 in ff.listFiles("level0"):

                fn = files.Filenames(fname0, config)
                if os.path.isfile(fn.fname.metaFrames) and skipExisting:
                    print("%s exists"%fn.fname.metaFrames)
                    continue
                
                if os.path.getsize(fname0.replace(config.movieExtension,"txt")) == 0:
                    print("%s has size 0!"%fname0)
                    continue
                  
                print(fname0)

                fn.createDirs()
                fname0all = list(fn.fnameAllThreads.values())

                metaDat, droppedFrames, beyondRepair = metadata.getMetaData(fname0all, camera, config, idOffset=0)

                if beyondRepair:
                    print(f"{os.path.basename(fname0)}, broken beyond repair, {droppedFrames}, frames dropped, {idOffset}, offset\n")
                
                if metaDat is not None:
                    comp = dict(zlib=True, complevel=5)
                    encoding = {
                        var: comp for var in metaDat.data_vars}
                    metaDat.to_netcdf(fn.fname.metaFrames, encoding=encoding)
                    print("%s written"%fn.fname.metaFrames)
                else:
                    with open(fn.fname.metaFrames+".nodata", "w") as f:
                        f.write("no data recorded")

    return

    nCPU = int(sys.argv[1])
    settings = sys.argv[2]
    nDays = int(sys.argv[3])




def loopCreateLevel1detectWorker(fname, settings, skipExisting=True, stdout=subprocess.DEVNULL, nCPU=1):

    '''
    We need this worker function becuase  we want to do the detection by indiependent 
    processes in parallel
    '''
    config = tools.readSettings(settings)
    logging.config.dictConfig(tools.get_logging_config('detection_run.log'))
    log = logging.getLogger()

    errorlines = collections.deque([], 50)
    
            
    fn = files.Filenames(fname, config, __version__)
    camera = fn.camera
    
    fn.createDirs()
    if skipExisting and (os.path.isfile('%s.broken.txt' % fn.fname.level1detect)):
        log.info('%s, output broken %s' % (fname, fn.fname.level1detect))
        return 1
    elif skipExisting and ( os.path.isfile('%s.nodata' % fn.fname.metaFrames)):
        log.info('%s, output nodata %s' % (fname, fn.fname.metaFrames))
        return 1
    elif skipExisting and ( os.path.isfile('%s' % fn.fname.level1detect)):
        log.info('%s, output exists %s' % (fname, fn.fname.level1detect))
        return 1
    elif skipExisting and ( os.path.isfile('%s.processing.txt' % fn.fname.level1detect)):
        log.info('%s, output processing %s' % (fname, fn.fname.level1detect))
        return 1

    with open('%s.processing.txt' % fn.fname.level1detect, 'w') as f:
        f.write('%i %s' % (os.getpid(), socket.gethostname()))

        
    BIN = os.path.join(sys.exec_prefix, "bin", "python")

    if nCPU is None:
        command = f"{BIN} -m VISSSlib detection.detectParticles  {fname} {settings}"
    else:
        command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {BIN} -m VISSSlib detection.detectParticles  {fname} {settings}"

    log.info(command)
    # sleep to avoid py38 env not found error
    with subprocess.Popen(shlex.split(f'bash -c "{command}"'), stderr=subprocess.PIPE, stdout=stdout, universal_newlines=True) as proc:
        try:
            for line in proc.stdout:
                print(str(line),end='')
        except TypeError:
            pass
        for line in proc.stderr:
            print(str(line),end='')
            errorlines.append(line)
        proc.wait() # required, otherwise returncode can be none is process is too fast
        if proc.returncode != 0:
            with open('%s.broken.txt' % fn.fname.level1detect, 'w') as f:
                
                for errorline in list(errorlines):
                    f.write(errorline)
            log.info(f"{fname} BROKEN {proc.returncode}")
        else:
            log.info(f"{fname} SUCCESS {proc.returncode}")
    try:
        os.remove('%s.processing.txt' % fn.fname.level1detect)
    except:
        pass
    
    
    return 0




def loopCreateLevel1detect(settings, skipExisting=True, nDays = 0, cameras = "all", nCPU=None):
    config = tools.readSettings(settings)

    logging.config.dictConfig(tools.get_logging_config('detection_run.log'))
    log = logging.getLogger()


    if nDays == 0:
        if config["end"] == "today":
            end = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        else:
            end = config["end"]

        days = pd.date_range(
            start=config["start"],
            end=end,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )
    else:
        end = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        days = pd.date_range(
            end=end,
            periods=nDays,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )
    if cameras == "all":
        cameras = [config.follower, config.leader]

    if nCPU is None:
        nCPU = os.cpu_count()

    fnames = []

    for dd in days:
        # create list of files
        year  =str(dd.year)
        month  ="%02i"%(dd.month)
        day  ="%02i"%(dd.day)      
        case = f"{year}{month}{day}"
        for camera in cameras:
            # find files
            ff = files.FindFiles(case, camera, config)
            fname0s =  ff.listFiles("level0")
            fnames += filter( os.path.isfile,
                                    fname0s )

        # Sort list of files in directory by size 
        fnames = sorted( fnames,
                                key =  lambda x: os.stat(x).st_size)[::-1]
    if len(fnames) == 0:
        log.error('no files %s' % fnamesPattern)
        # print('no files %s' % fnamesPattern)

    else:
        
        print(f"found {len(fnames)} files, lets do it:")
        

        p = multiprocessing.Pool(nCPU)
        p.map(partial(loopCreateLevel1detectWorker, settings=settings, skipExisting=skipExisting), fnames)


        p.close()
        p.join()
