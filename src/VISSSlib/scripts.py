import sys
import os
import datetime
import pandas as pd


from . import quicklooks
from . import files
from . import tools
from . import metadata
from . import __version__


def loopLv1Quicklooks(settings, lv2Version, skipExisting=True):

    config = tools.readSettings(settings)
    instruments = config["instruments"]
    computers = config["computers"]

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
            closed=None
    )[::-1]:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for computer, camera in zip(computers, instruments):
            #         print(case, computer, camera)

            f, i = quicklooks.createLv1Quicklook(
                case, camera, config, lv2Version, skipExisting=skipExisting)
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
        closed=None
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
            closed=None
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
            closed=None
        )

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



