import glob
import logging
import warnings

import numpy as np
import xarray as xr
from loguru import logger as log

from .. import __version__, files, tools

warnings.filterwarnings("ignore", category=RuntimeWarning)


def getCloudnet(date, config, path, kind, item):
    """
    Download data from Cloudnet API for a specific date and item.

    Parameters
    ----------
    date : str
        Date in YYYY-MM-DD format.
    config : object
        Configuration object containing auxiliary data settings.
    path : str
        Local path where the downloaded file will be saved.
    kind : str
        Type of data to download (e.g., 'instrument', 'product').
    item : str
        Specific item identifier (e.g., 'weather-station').

    Returns
    -------
    list
        List of downloaded filenames.

    Raises
    ------
    Exception
        If no data is found or download fails.
    """
    import requests

    print(f"downloading {item} for {date}")
    url = "https://cloudnet.fmi.fi/api/files"
    payload = {
        "date": date,
        kind: item,
        "site": config.aux.cloudnet.site,
    }
    metadata = requests.get(url, payload).json()
    if (len(metadata) == 0) or ("status" in metadata[0].keys()):
        log.warning(f"Did not find {url}")
        return []
    fnames = []
    for row in metadata[:1]:
        res = requests.get(row["downloadUrl"])
        fname = f"{path}/{row['filename']}"
        with tools.open2(fname, config, "wb") as f:
            f.write(res.content)
        fnames.append(fname)
    if len(metadata) > 1:
        log.warning("Found more than one file on Cloudnet")
    print(f"done {fnames}")
    return fnames


def getARM(date, config, path, product, user):
    """
    Download data from ARM API for a specific date and product.

    Parameters
    ----------
    date : str
        Date in YYYY-MM-DD format.
    config : object
        Configuration object containing auxiliary data settings.
    path : str
        Local path where the downloaded files will be saved.
    product : str
        Product identifier (e.g., 'met').
    user : str
        User identifier for ARM API access.

    Returns
    -------
    list
        List of downloaded filenames.

    Raises
    ------
    FileNotFoundError
        If no data is found or download fails.
    """
    import requests

    print(f"downloading {product} for {date}")
    url = "https://adc.arm.gov/armlive/data/query"
    payload = {
        "user": user,
        "ds": f"{config.aux.ARM.site}{product}",
        "start": date,
        "end": date,
        "wt": "json",
    }
    metadata = requests.get(url, payload).json()
    fnames = []
    if metadata["status"] == "success":
        for file in metadata["files"]:
            url = "https://adc.arm.gov/armlive/data/saveData"
            payload = {
                "user": user,
                "file": file,
            }
            fname = f"{path}/{file}"
            res = requests.get(url, payload)
            with tools.open2(fname, config, "wb") as f:
                f.write(res.content)
            fnames.append(fname)
    else:
        raise FileNotFoundError(metadata["status"])

    print(f"done {fnames}")
    return fnames


def getMeteoData(case, config):
    """
    Retrieve meteorological data for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object or str
        Configuration object or path to configuration file.

    Returns
    -------
    xarray.Dataset
        Meteorological data for the specified case.

    Raises
    ------
    FileNotFoundError
        If no meteorological data is found.
    """
    if type(config) is str:
        config = tools.readSettings(config)

    fn = files.FindFiles(case, config.leader, config)
    dat = _getMeteoData1(case, config)

    if config.aux.meteo.source == "ARMmet":
        # add data of previous files - new fiels ar enot always reated at 00:00
        fnY = files.FindFiles(fn.yesterday, config.leader, config)
        try:
            datY = _getMeteoData1(fn.yesterday, config)
        except FileNotFoundError:
            log.warning(f"Did not find meteo data for yesterday {fn.yesterday}")
        else:
            # merge data
            dat = xr.concat((datY, dat), dim="time")
            today = (dat.time >= fn.datetime64) & (
                dat.time < (fn.datetime64 + np.timedelta64(1, "D"))
            )
            dat = dat.isel(time=today)
    if dat is not None:
        dat.load()
    return dat


def _getMeteoData1(case, config):
    """
    Internal helper function to retrieve meteorological data based on source.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.

    Returns
    -------
    xarray.Dataset
        Meteorological data for the specified case.

    Raises
    ------
    ValueError
        If the meteorological data source is not recognized.
    """
    if config.aux.meteo.source == "cloudnetMeteo":
        return getMeteoDataCloudnet(case, config)
    elif config.aux.meteo.source == "ARMmet":
        return getMeteoDataARM(case, config)
    elif config.aux.meteo.source == "RPG":
        return getMeteoDataRPG(case, config)
    elif config.aux.meteo.source == "pangaea":
        return getMeteoDataPangaea(case, config)
    else:
        raise ValueError(
            f"Do not understand config.aux.meteo.source:{config.aux.meteo.source}"
        )


def getMeteoDataCloudnet(case, config):
    """
    Retrieve meteorological data from Cloudnet for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.

    Returns
    -------
    xarray.Dataset
        Meteorological data for the specified case.

    Raises
    ------
    FileNotFoundError
        If no Cloudnet data is found.
    """
    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fStr = f"{config.aux.meteo.path.format(year=fn.year, month=fn.month, day=fn.day)}/{case}_{config.aux.cloudnet.site}_weather-station_*.nc"
    fnames = glob.glob(fStr)

    if config.aux.meteo.downloadData and (len(fnames) == 0):
        fnames = getCloudnet(
            date,
            config,
            config.aux.meteo.path.format(year=fn.year, month=fn.month, day=fn.day),
            "instrument",
            "weather-station",
        )

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    print(f"Opening {fStr}")
    dat = xr.open_mfdataset(fnames)
    assert config.level2.freq == "1min"

    dat = dat[
        [
            "air_temperature",
            "relative_humidity",
            "air_pressure",
            "wind_speed",
            "wind_direction",
        ]
    ]
    # timestamps are a couple ns off
    dat = dat.resample(time=config.level2.freq, label="left").nearest()
    return dat


def getMeteoDataARM(case, config):
    """
    Retrieve meteorological data from ARM for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.

    Returns
    -------
    xarray.Dataset
        Meteorological data for the specified case.

    Raises
    ------
    FileNotFoundError
        If no ARM data is found.
    """
    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"
    product = "met"

    path = config.aux.meteo.path.format(
        site=config.aux.ARM.site, product=product, year=fn.year
    )

    fStr = f"{path}/{config.aux.ARM.site}{product}*{case}*.cdf"
    fnames = glob.glob(fStr)

    if config.aux.meteo.downloadData and (len(fnames) == 0):
        fnames = getARM(date, config, path, product, config.aux.ARM.user)

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    print(f"Opening {fnames}")
    dat = xr.open_mfdataset(fStr)
    assert config.level2.freq == "1min"
    dat

    vars = [
        "temp_mean",
        "rh_mean",
        "atmos_pressure",
        "wspd_vec_mean",
        "wdir_vec_mean",
    ]

    for var in vars:
        dat[var] = dat[var].where(dat[f"qc_{var}"] == 0)
    dat = dat[vars]

    dat = dat.rename(
        {
            "temp_mean": "air_temperature",
            "rh_mean": "relative_humidity",
            "atmos_pressure": "air_pressure",
            "wspd_vec_mean": "wind_speed",
            "wdir_vec_mean": "wind_direction",
        }
    )

    dat["air_temperature"] = dat["air_temperature"] + 273.15
    dat["relative_humidity"] = dat["relative_humidity"] / 100
    dat["air_pressure"] = dat["air_pressure"] * 1000

    return dat


def getMeteoDataRPG(case, config):
    """
    Retrieve meteorological data from RPG radar for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.

    Returns
    -------
    xarray.Dataset
        Meteorological data for the specified case.

    Raises
    ------
    FileNotFoundError
        If no RPG data is found.
    """
    import netCDF4

    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"

    path = config.aux.meteo.path.format(year=fn.year, month=fn.month, day=fn.day)
    fStr = f"{path}/*ZEN.LV1.NC"
    fnames = glob.glob(fStr)

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    dat = xr.open_mfdataset(fnames)
    assert config.level2.freq == "1min"

    vars = [
        "SurfTemp",
        "SurfRelHum",
        "SurfPres",
        "SurfWS",
        "SurfWD",
    ]

    dat = dat[vars]

    dat = dat.rename(
        {
            "SurfTemp": "air_temperature",
            "SurfRelHum": "relative_humidity",
            "SurfPres": "air_pressure",
            "SurfWS": "wind_speed",
            "SurfWD": "wind_direction",
            "Time": "time",
        }
    )

    dat["relative_humidity"] = dat["relative_humidity"] / 100
    dat["air_pressure"] = dat["air_pressure"] * 1000
    dat["time"] = netCDF4.num2date(
        dat.time.values,
        "seconds since 2001-01-01T00:00:00",
        only_use_python_datetimes=True,
        only_use_cftime_datetimes=False,
    )

    dat = dat.resample(time=config.level2.freq, label="left").nearest()
    return dat


def getRadarData(
    case,
    config,
):
    """
    Retrieve radar data for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object or str
        Configuration object or path to configuration file.

    Returns
    -------
    tuple
        Tuple containing (xarray.Dataset, float) where the dataset contains
        radar data and the float is the radar frequency.

    Raises
    ------
    FileNotFoundError
        If no radar data is found.
    """
    if type(config) is str:
        config = tools.readSettings(config)

    fn = files.FindFiles(case, config.leader, config)

    dat, frequency = _getRadarData1(case, config, fn)

    # add data of previous files - new files are not always started at 00:00
    fnY = files.FindFiles(fn.yesterday, config.leader, config)
    try:
        datY, frequencyY = _getRadarData1(fn.yesterday, config, fnY)
    except FileNotFoundError:
        log.warning(f"Did not find radar data for yesterday {fn.yesterday}")
    else:
        # merge data
        dat = xr.concat((datY, dat), dim="time")

        today = (dat.time >= fn.datetime64) & (
            dat.time < (fn.datetime64 + np.timedelta64(1, "D"))
        )
        dat = dat.isel(time=today).load()

    return dat, frequency


def _getRadarData1(case, config, fn):
    """
    Internal helper function to retrieve radar data based on source.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.
    fn : object
        FindFiles object for the current case.

    Returns
    -------
    tuple
        Tuple containing (xarray.Dataset, float) where the dataset contains
        radar data and the float is the radar frequency.

    Raises
    ------
    ValueError
        If the radar data source is not recognized.
    """
    if config.aux.radar.source == "cloudnetCategorize":
        dat, frequency = getRadarDataCloudnetCategorize(case, config, fn)

    elif config.aux.radar.source == "cloudnetFMCW94":
        dat, frequency = getRadarDataCloudnetFMCW94(case, config, fn)

    elif config.aux.radar.source == "ARMwcloudradarcel":
        dat, frequency = getRadarDataARMwcloudradarcel(case, config, fn)

    elif config.aux.radar.source == "ARMarsclkazr1kollias":
        dat, frequency = getRadarDataARMarsclkazr1kollias(case, config, fn)

    else:
        raise ValueError(
            f"Do not understand config.aux.radar.source:{config.aux.radar.source}"
        )

    if dat is None:
        return None, None

    h1, h2 = config.aux.radar.heightRange
    heightIndices = (dat.range >= h1) & (dat.range <= h2)
    nBins = np.sum(heightIndices).values
    if nBins <= config.aux.radar.minHeightBins:
        raise ValueError(
            f"found only {nBins} radar range bins wchich is less than config.aux.radar.minHeightBins:{config.aux.radar.minHeightBins}"
        )

    dat = dat.isel(range=heightIndices)
    # dat["Z_error"] = 10 ** (0.1 * dat["Z_error"])

    dat["time"] = dat.time + np.timedelta64(config.aux.radar.timeOffset, "s")
    dat = dat.resample(time=config.level2.freq, label="left").mean()

    dat["Ze"] = 10 * np.log10(dat["Ze"])

    offset, date = tools.getPreviousCalibrationOffset(case, config)
    if offset != 0:
        dat["Ze"] = dat["Ze"] + offset
        log.warning(f"Applied a radar calibration offset of {offset} dB from {date}")

    # dat["Z_error"] = 10 * np.log10(dat["Z_error"])

    # do a linear extrapolation based on the lowest config.aux.radar.heightIndices data points
    fit = dat.polyfit(dim="range", deg=1, full=True)
    hnew = xr.DataArray([0], coords={"range": [0]})
    dat["Ze_ground"] = xr.polyval(coord=hnew, coeffs=fit.Ze_polyfit_coefficients).isel(
        range=0, drop=True
    )
    dat["MDV_ground"] = xr.polyval(
        coord=hnew, coeffs=fit.MDV_polyfit_coefficients
    ).isel(range=0, drop=True)

    dat["Ze_ground_fitResidual"] = fit.Ze_polyfit_residuals
    dat["MDV_ground_fitResidual"] = fit.MDV_polyfit_residuals

    dat["Ze_0"] = dat["Ze"].isel(range=0)
    dat["MDV_0"] = dat["MDV"].isel(range=0)

    dat["Ze_std"] = dat["Ze"].std("range")
    dat["MDV_std"] = dat["MDV"].std("range")

    dat = dat.drop_vars(["Ze", "MDV"])
    return dat, frequency


def getRadarDataCloudnetCategorize(case, config, fn):
    """
    Retrieve radar data from Cloudnet categorize product for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.
    fn : object
        FindFiles object for the current case.

    Returns
    -------
    tuple
        Tuple containing (xarray.Dataset, float) where the dataset contains
        radar data and the float is the radar frequency.

    Raises
    ------
    FileNotFoundError
        If no Cloudnet categorize data is found.
    """
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fStr = f"{config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day)}/{case}_{config.aux.cloudnet.site}_categorize*.nc"
    fnames = glob.glob(fStr)

    if config.aux.radar.downloadData and (len(fnames) == 0):
        fnames = getCloudnet(
            date,
            config,
            config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day),
            "product",
            "categorize",
        )

    if len(fnames) == 0:
        log.warning(f"Did not find {fStr}")
        raise FileNotFoundError(f"Did not find {fStr}")

    print(f"Opening {fStr}")
    dat = xr.open_mfdataset(
        fnames,
        preprocess=lambda dat: dat[
            ["v", "Z", "altitude", "radar_frequency", "radar_melting_atten"]
        ],
    )

    dat = dat.rename(v="MDV", Z="Ze", height="range")
    dat1 = dat[
        [
            "Ze",
            "MDV",
        ]
    ]

    # fix Cloudnet bug - solid precipitation at the ground should never need a melting layer attenuation correction
    # https://github.com/actris-cloudnet/cloudnetpy/issues/121
    dat1["Ze"] = dat1["Ze"] - dat.radar_melting_atten.fillna(0)
    # The correction does not hurt becuase after bugfix, radar_melting_atten is zero!
    # Could be only problematic for perip around 0°C

    try:
        altitude = dat.altitude.values[0]
    except IndexError:
        altitude = dat.altitude.values

    dat1["range"] = dat1.range - altitude
    dat1["Ze"] = 10 ** (0.1 * dat1["Ze"])

    return dat1, float(dat.radar_frequency.values)


def getRadarDataCloudnetFMCW94(case, config, fn):
    """
    Retrieve radar data from Cloudnet FMCW94 instrument for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.
    fn : object
        FindFiles object for the current case.

    Returns
    -------
    tuple
        Tuple containing (xarray.Dataset, float) where the dataset contains
        radar data and the float is the radar frequency.

    Raises
    ------
    FileNotFoundError
        If no Cloudnet FMCW94 data is found.
    """
    fStr = f"{config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day)}/{case}_{config.aux.cloudnet.site}_rpg-fmcw-94*.nc"
    fnames = glob.glob(fStr)

    if config.aux.radar.downloadData and (len(fnames) == 0):
        fnames = getCloudnet(
            case,
            config,
            config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day),
            "instrument",
            "rpg-fmcw-94",
        )

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")

    print(f"Opening {fStr}")
    dat = xr.open_mfdataset(fnames, preprocess=lambda dat: dat[["v", "Zh"]])

    dat = dat.rename(v="MDV", Zh="Ze")
    dat1 = dat[
        [
            "Ze",
            "MDV",
        ]
    ]
    dat1["Ze"] = 10 ** (0.1 * dat1["Ze"])

    return dat1, float(dat.radar_frequency.values)


def getRadarDataARMwcloudradarcel(case, config, fn):
    """
    Retrieve radar data from ARM wcloudradarcel product for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.
    fn : object
        FindFiles object for the current case.

    Returns
    -------
    tuple
        Tuple containing (xarray.Dataset, float) where the dataset contains
        radar data and the float is the radar frequency.

    Raises
    ------
    FileNotFoundError
        If no ARM wcloudradarcel data is found.
    """
    fStr = f"{config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day)}/{config.aux.ARM.site}wcloudradarcel*{case}*.nc"
    fnames = glob.glob(fStr)

    if config.aux.radar.downloadData and (len(fnames) == 0):
        raise FileNotFoundError("IOP data needs to be downloaded from ARM manually")

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    print(f"Opening {fStr}")
    with xr.open_mfdataset(
        fnames, preprocess=lambda dat: dat[["MeanVel", "ZE", "Elv", "Freq"]]
    ) as dat:
        dat = dat.rename(MeanVel="MDV", ZE="Ze")
        dat1 = dat[
            [
                "Ze",
                "MDV",
            ]
        ].where(np.abs(dat.Elv - config.aux.radar.elevation) < 0.5)
        freq = float(dat.Freq.mean().values)

    return dat1, freq


def getRadarDataARMarsclkazr1kollias(case, config, fn):
    """
    Retrieve radar data from ARM arsclkazr1kollias product for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.
    fn : object
        FindFiles object for the current case.

    Returns
    -------
    tuple
        Tuple containing (xarray.Dataset, float) where the dataset contains
        radar data and the float is the radar frequency.

    Raises
    ------
    FileNotFoundError
        If no ARM arsclkazr1kollias data is found.
    """
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fStr = f"{config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day)}/{config.aux.ARM.site}arsclkazr1kollias*{case}*.nc"
    fnames = glob.glob(fStr)

    if config.aux.radar.downloadData and (len(fnames) == 0):
        raise NotImplementedError

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    print(f"Opening {fStr}")
    with xr.open_mfdataset(
        fnames,
        preprocess=lambda dat: dat[
            [
                "mean_doppler_velocity",
                "reflectivity_best_estimate",
                "qc_reflectivity_best_estimate",
                "qc_mean_doppler_velocity",
            ]
        ],
    ) as dat:
        dat = dat.where(dat.qc_reflectivity_best_estimate == 0).where(
            dat.qc_mean_doppler_velocity == 0
        )

        dat = dat.rename(
            mean_doppler_velocity="MDV", reflectivity_best_estimate="Ze", height="range"
        )
        dat1 = dat[
            [
                "Ze",
                "MDV",
            ]
        ]

    frq = float(dat.attrs["radar_operating_frequency_burst"].lstrip().split(" ")[0])
    return dat1, frq


def _downloadPangaea1(config, path, type):
    """
    Download data from Pangaea for a specific type and save it locally.

    Parameters
    ----------
    config : object
        Configuration object.
    path : str
        Local path where the downloaded file will be saved.
    type : str
        Type of data to download (e.g., 'weatherstation').

    Returns
    -------
    str
        Path to the downloaded file.
    """
    from pangaeapy.pandataset import PanDataSet

    doipart = config.aux.meteo.doi.split("/")[-1]
    fnamePart = f"{path}/*_{type}_{config.site}_{doipart}.nc"
    if len(glob.glob(fnamePart)) > 0:
        print(f"{fnamePart} exists")
        return fnamePart

    ds1 = PanDataSet(config.aux.meteo.doi)
    ds1.data = ds1.data.rename(columns={"Date/Time": "time", "TIME": "time"})
    dat = xr.Dataset(ds1.data.set_index("time"))

    for kk in ds1.parameters.keys():
        if kk in ["TIME", "Date/Time"]:
            continue
        if ds1.parameters[kk].name is not None:
            dat[kk].attrs["long_name"] = ds1.parameters[kk].name
        if ds1.parameters[kk].unit is not None:
            dat[kk].attrs["unit"] = ds1.parameters[kk].unit
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
    yearmonth = "".join(str(dat.time[0].values).split("T")[0].split("-")[:2])
    fname = f"{path}/{yearmonth}_{type}_{config.site}_{doipart}.nc"

    tools.to_netcdf2(dat, config, fname)
    return fname


def downloadPangaea(config, path, type):
    """
    Download all available data from Pangaea for a specific type.

    Parameters
    ----------
    config : object
        Configuration object.
    path : str
        Local path where the downloaded files will be saved.
    type : str
        Type of data to download (e.g., 'weatherstation').
    """
    from pangaeapy.pandataset import PanDataSet

    ds = PanDataSet(config.aux.meteo.doi)
    for doi in ds.collection_members:
        fname = _downloadPangaea1(config, path, type)
    return


def getMeteoDataPangaea(case, config):
    """
    Retrieve meteorological data from Pangaea for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : object
        Configuration object.

    Returns
    -------
    xarray.Dataset
        Meteorological data for the specified case.

    Raises
    ------
    FileNotFoundError
        If no Pangaea data is found.
    """
    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"
    product = "met"

    path = config.aux.meteo.path.format(site=config.site, year=fn.year)

    fStr = f"{path}/{fn.year}{fn.month}_weatherstation_{config.site}*.nc"
    fnames = glob.glob(fStr)

    if config.aux.meteo.downloadData and (len(fnames) == 0):
        # for pangaea, we do not know the doi of the monthly dataset
        # therefore download everything which is available
        downloadPangaea(config, path, "weatherstation")
        fnames = glob.glob(fStr)

    if len(fnames) == 0:
        print(f"No Pangaea meteo data yet, check http://doi.org/{config.aux.meteo.doi}")
        return None

    print(f"Opening {fnames}")
    dat = xr.open_mfdataset(fStr)
    assert config.level2.freq == "1min"
    dat

    vars = [
        "T2",
        "RH_2",
        "PoPoPoPo",
        "FF10",
        "DD10",
    ]

    dat = dat[vars]

    dat = dat.rename(
        {
            "T2": "air_temperature",
            "RH_2": "relative_humidity",
            "PoPoPoPo": "air_pressure",
            "FF10": "wind_speed",
            "DD10": "wind_direction",
        }
    )

    today = (dat.time >= fn.datetime64) & (
        dat.time < (fn.datetime64 + np.timedelta64(1, "D"))
    )
    dat = dat.isel(time=today).load()

    dat["air_temperature"] = dat["air_temperature"] + 273.15
    dat["relative_humidity"] = dat["relative_humidity"] / 100
    dat["air_pressure"] = dat["air_pressure"] * 100

    return dat


def loopDownloadAux(settings, nDays=0):
    """
    helper script to download aux data


    Parameters
    ----------
    settings : str
        VISSS settings YAML file
    nDays : number or str, optional
        number of days N`` to go back or date ``str(YYYYMMDD)`` or date range ``str(YYYYMMDD-YYYYMMDD)`` (the default is 0)
    """
    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config, endYesterday=False)

    if "source" in config.aux.meteo:
        for dd in days:
            year = str(dd.year)
            month = "%02i" % dd.month
            day = "%02i" % dd.day
            case = f"{year}{month}{day}"
            try:
                getMeteoData(case, config)
            except Exception as e:
                log.error(f"failed for {case} {settings}")
                print(e)
    else:
        log.warning(f"source not in config.aux.meteo")

    if "source" in config.aux.radar:
        for dd in days:
            year = str(dd.year)
            month = "%02i" % dd.month
            day = "%02i" % dd.day
            case = f"{year}{month}{day}"
            try:
                getRadarData(case, config)
            except Exception as e:
                log.error(f"failed for {case} {settings}")
                print(e)
    else:
        log.warning(f"source not in config.aux.radar")

    return
