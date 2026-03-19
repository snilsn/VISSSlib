import glob
import os
import warnings

import numpy as np
import xarray as xr
from loguru import logger as log

from .. import __version__, files, quicklooks, tools
from . import aux

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)


def retrieveM(y_obs, psd, air_temperature, Dmean, Dbound, frequency, config):
    """
    Perform optimal estimation retrieval for riming mass parameter.

    This function uses the pyOptimalEstimation library to retrieve the riming
    mass parameter M from radar reflectivity observations and particle size
    distribution data.

    Parameters
    ----------
    y_obs : array-like
        Observed radar reflectivity values
    psd : array-like
        Particle size distribution data
    air_temperature : float
        Air temperature in Kelvin
    Dmean : array-like
        Mean diameter values
    Dbound : array-like
        Diameter bin boundaries
    frequency : float
        Radar frequency in GHz
    config : object
        Configuration object containing retrieval settings

    Returns
    -------
    tuple
        A tuple containing (M_oe, M_err, Ze_combinedRetrieval) where:
        - M_oe: Optimally estimated riming mass parameter
        - M_err: Error in the estimated riming mass
        - Ze_combinedRetrieval: Combined radar reflectivity retrieval

    Notes
    -----
    This function implements a Bayesian optimal estimation approach to retrieve
    the riming mass parameter from radar measurements and PSD data.
    """
    import pyOptimalEstimation as pyOE

    x_vars = ["M"]
    y_vars = ["Ze"]
    nBins = len(psd)
    x_a = np.float64(-1.0)  # a priori
    S_a = np.array([1.0**2])  # a priori uncertainty

    S_y = np.array([1.5**2])  # ([0.5**2]) # measurement uncertainty

    psd_data = {
        "ice": psd,
        "alt": 5,
        "temp": air_temperature,
        "bins": nBins,
        "dmean": Dmean,
        "dbound": Dbound,
        "shape": config.level3.combinedRiming.habit,
        "frequency": frequency,
        "elevation": config.aux.radar.elevation,
    }

    # try:
    # print(x_vars, x_a, S_a, y_vars, y_obs, S_y)
    oe = pyOE.optimalEstimation(
        x_vars,  # state variable names
        x_a,  # a priori
        S_a,  # a priori uncertainty
        y_vars,  # measurement variable names
        y_obs,  # observations
        S_y,  # observation uncertainty
        reflec_logM,  # forward Operator
        forwardKwArgs=psd_data,  # additonal function arguments
    )

    oe.doRetrieval(maxIter=10, maxTime=1.0)

    # how many successes

    try:
        # x
        M_oe = oe.x_op.iloc[0]
        # y
        Ze_combinedRetrieval = oe.y_op.iloc[0]
        # errors
        M_err = oe.S_op.iloc[0]
    except AttributeError:
        M_oe = np.nan
        Ze_combinedRetrieval = np.nan
        M_err = np.nan

    return M_oe, M_err, Ze_combinedRetrieval


def ssrga_parameter(M, elevation):
    """
    Calculate SSRGA parameters for given riming mass and elevation.

    This function computes the SSRGA (Self-Similar Rayleigh-Gans Approximation)
    parameters for snowflake scattering based on riming mass and radar elevation.

    Parameters
    ----------
    M : float or array-like
        Riming mass parameter
    elevation : float
        Radar elevation angle in degrees

    Returns
    -------
    tuple
        A tuple containing (kappa, beta, gamma, zeta1, alpha_eff) parameters
        for SSRGA scattering calculations

    Raises
    ------
    ValueError
        If elevation is not 90 or around 50 degrees

    Notes
    -----
    The SSRGA parameters are used in the scattering calculations for snowflakes
    with different riming characteristics at different radar elevations.
    """
    import pyPamtra

    if elevation == 90:
        return pyPamtra.descriptorFile.ssrga_parameter(M)
    elif np.abs(elevation - 50) < 11:
        p0 = 0.5035
        p1 = np.array([0.0168, 0.117, -2.648, -0.8126, 0.1125])
        p2 = np.array([0.1609, -0.0022, 0.6949, 1.6618, -0.1316])
        p3 = np.array([0.7234, 0.0429, 2.8542, 2.4369, 0.1158])

        alpha_eff = p1[0] * M ** (2 * p0) + p2[0] * M**p0 + p3[0]
        kappa = p1[1] * M ** (2 * p0) + p2[1] * M**p0 + p3[1]
        beta = p1[2] * M ** (2 * p0) + p2[2] * M**p0 + p3[2]
        gamma = p1[3] * M ** (2 * p0) + p2[3] * M**p0 + p3[3]
        zeta1 = p1[4] * M ** (2 * p0) + p2[4] * M**p0 + p3[4]

        return kappa, beta, gamma, zeta1, alpha_eff
    else:
        ValueError(f"elevation must be 90 or around 50. Got {elevation}")


def reflec_logM(
    X, ice, alt, temp, bins, dmean, dbound, shape, frequency, elevation
):  # ice
    """
    Forward model for radar reflectivity based on riming mass.

    This function calculates the radar reflectivity using a forward model
    that incorporates riming mass, particle size distribution, and atmospheric
    conditions.

    Parameters
    ----------
    X : float
        Logarithm of riming mass parameter
    ice : array-like
        Ice particle size distribution
    alt : float
        Altitude in meters
    temp : float
        Temperature in Kelvin
    bins : int
        Number of size bins
    dmean : array-like
        Mean diameter values
    dbound : array-like
        Diameter bin boundaries
    shape : str
        Snowflake habit shape
    frequency : float
        Radar frequency in GHz
    elevation : float
        Radar elevation angle in degrees

    Returns
    -------
    float
        Calculated radar reflectivity (Ze)

    Notes
    -----
    This function implements a forward model using pyPamtra to simulate
    radar reflectivity based on the physical properties of snowflakes
    with varying riming characteristics.
    """
    import pyPamtra

    M = 10**X

    a = pyPamtra.descriptorFile.riming_dependent_mass_size(M, shape)[
        0
    ]  # needle, column, rosette, plate, mean
    b = pyPamtra.descriptorFile.riming_dependent_mass_size(M, shape)[1]

    scattering = "ss-rayleigh-gans"

    temp_lev = np.array([temp, temp])
    hgt_lev = np.array([alt - 5, alt + 5])

    dsd_i = ice
    nBins = bins
    Dmean = dmean
    Dbound = dbound
    alt6 = alt

    pam = pyPamtra.pyPamtra()

    pam.df.addHydrometeor(
        (
            "ice",  # name
            -99.0,  # aspect ratio (NOT RELEVANT)
            -1,  # liquid - ice flag
            -99.0,  # density (NOT RELEVANT)
            -99.0,  # mass size relation prefactor a (NOT RELEVANT)
            -99.0,  # mass size relation exponent b (NOT RELEVANT)
            -99.0,  # area size relation prefactor alpha (NOT RELEVANT)
            -99.0,  # area size relation exponent beta (NOT RELEVANT)
            0,  # moment provided later (NOT RELEVANT)
            nBins,  # number of bins
            "fullBin",  # distribution name (NOT RELEVANT)
            -99.0,  # distribution parameter 1 (NOT RELEVANT)
            -99.0,  # distribution parameter 2 (NOT RELEVANT)
            -99.0,  # distribution parameter 3 (NOT RELEVANT)
            -99.0,  # distribution parameter 4 (NOT RELEVANT)
            -99.0,  # minimum diameter (NOT RELEVANT)
            -99.0,  # maximum diameter (NOT RELEVANT)
            scattering,  # scattering model
            "heymsfield10_particles",  # fall velocity relation  (NOT RELEVANT)
            0.0,  # canting angle  (NOT RELEVANT)
        )
    )

    pam = pyPamtra.importer.createUsStandardProfile(
        pam, hgt_lev=list(hgt_lev), temp_lev=list(temp_lev)
    )

    pam.p["turb_edr"][:] = 1e-4

    pam.nmlSet["passive"] = False  # passive mode
    pam.nmlSet["active"] = True  # active mode

    # 0 is real noise, -1 means that the seed is created from latitude and longitude, other value gives always the same random numbers
    pam.nmlSet["randomseed"] = 0
    # Use “simple” radar simulator provides only Z_e by integrating over D. The advanced “spectrum” simulator simulates the complete radar Doppler spectrum and estimates all moments from the spectrum. “moments” is identical to “spectrum” but the full Doppler spectrum is discarded to save memory.
    pam.nmlSet["radar_mode"] = "simple"

    pam.nmlSet[
        "hydro_fullspec"
    ] = True  # pass values directly from python to PAMTRA using numpy arrays.

    pam.p["sfc_type"] = np.zeros(pam._shape2D)
    pam.p["sfc_model"] = np.zeros(pam._shape2D)
    pam.p["sfc_refl"] = np.chararray(pam._shape2D)
    pam.p["sfc_refl"][pam.p["sfc_type"] == 0] = "F"
    pam.p["obs_height"][:, 0] = 0.0

    pam.nmlSet[
        "radar_attenuation"
    ] = "top-down"  # include attenuation by gas and hydrometeors
    pam.set["verbose"] = 0
    pam.set["pyVerbose"] = 0

    pam.df.addFullSpectra()

    pam.df.dataFullSpec["d_bound_ds"][:] = Dbound
    pam.df.dataFullSpec["d_ds"][:] = Dmean
    pam.df.dataFullSpec["n_ds"][0, 0, :, 0, :] = (dsd_i) * np.diff(Dbound)

    # snow
    pam.df.dataFullSpec["area_ds"][0, 0, :, 0, :] = (
        0.3898 * pam.df.dataFullSpec["d_ds"][0, 0, :, 0, :] ** 1.977
    )
    pam.df.dataFullSpec["mass_ds"][0, 0, :, 0, :] = (
        a * pam.df.dataFullSpec["d_ds"][0, 0, :, 0, :] ** b
    )

    kappa, beta, gamma, zeta, alpha = ssrga_parameter(M, elevation)

    pam.df.dataFullSpec["as_ratio"][0, 0, :, 0, :] = alpha
    pam.df.dataFullSpec["rg_kappa_ds"][0, 0, :, 0, :] = kappa
    pam.df.dataFullSpec["rg_beta_ds"][0, 0, :, 0, :] = beta
    pam.df.dataFullSpec["rg_gamma_ds"][0, 0, :, 0, :] = gamma
    pam.df.dataFullSpec["rg_zeta_ds"][0, 0, :, 0, :] = zeta
    pam.df.dataFullSpec["rho_ds"][0, 0, :, 0, :] = (
        6.0 * pam.df.dataFullSpec["mass_ds"][0, 0, :, 0, :]
    ) / (np.pi * pam.df.dataFullSpec["d_ds"][0, 0, :, 0, :] ** 3.0 * alpha.values)

    pam.runPamtra([frequency])

    Ze = pam.r["Ze"].squeeze()
    att_hydro = pam.r["Att_hydro"].squeeze()
    att_atmo = pam.r["Att_atmo"].squeeze()

    Ze[Ze == -9.99900000e03] = np.nan

    return Ze


def mass_size(M):
    """
    Interpolate mass-size relationship for riming parameter.

    This function interpolates the mass-size relationship parameters (a and b)
    as a function of the riming mass parameter M using cubic spline interpolation.

    Parameters
    ----------
    M : float or array-like
        Riming mass parameter

    Returns
    -------
    tuple
        A tuple containing (a_m, b_m) mass-size relationship parameters

    Notes
    -----
    This function implements a cubic spline interpolation based on lookup tables
    from Maherndl et al. (2023) for riming-dependent mass-size relationships.
    """
    import scipy.interpolate

    M_list = np.array(
        [
            0.0,
            0.0129,
            0.02045,
            0.03245,
            0.05145,
            0.08155,
            0.129,
            0.2045,
            0.3245,
            0.5145,
            0.8155,
        ]
    )
    if hasattr(M, "__len__"):
        M[M > M_list[-1]] = M_list[-1]

    else:
        if M > M_list[-1]:
            M = M_list[-1]

    a_m_list = np.array(
        [0.0324, 0.224, 0.537, 1.54, 4.27, 10.1, 22.2, 43.3, 79.0, 157.0, 173.0]
    )
    b_m_list = np.array(
        [2.10, 2.35, 2.45, 2.57, 2.69, 2.77, 2.85, 2.89, 2.93, 2.97, 2.93]
    )

    a_int = scipy.interpolate.interp1d(
        M_list, a_m_list, kind="cubic", fill_value="extrapolate"
    )
    b_int = scipy.interpolate.interp1d(
        M_list, b_m_list, kind="cubic", fill_value="extrapolate"
    )

    a_m = a_int(M)
    b_m = b_int(M)

    return a_m, b_m


def dynamic_viscosity_air(temperature):
    """
    Calculate dynamic viscosity of dry air using Sutherland's law.

    This function computes the dynamic viscosity of dry air at a given
    temperature using the Sutherland's law formula.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    float
        Dynamic viscosity of dry air in Pascal-seconds (Pa·s)

    Notes
    -----
    The calculation uses coefficients from F. M. White, Viscous Fluid Flow,
    2nd ed., McGraw-Hill, (1991) and Kim et al., arXiv:physics/0410237v1.
    """
    mu0 = 1.716e-5  # Pas
    T0 = 273.0
    C = 111.0  # K

    eta = mu0 * ((T0 + C) / (temperature + C)) * (temperature / T0) ** 1.5

    return eta


def dry_density_air(temperature, press):
    """
    Calculate dry air density using ideal gas law.

    This function computes the density of dry air at given temperature
    and pressure using the ideal gas law.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    press : float
        Atmospheric pressure in Pascals

    Returns
    -------
    float
        Dry air density in kg/m³

    Notes
    -----
    Uses the specific gas constant for dry air (R_s = 287.0500676 J/kg·K).
    """
    R_s = 287.0500676
    rho = press / (R_s * temperature)

    return rho


def heymsfield10_particles_M(Dmax, M, temperature, press, shape):
    """
    Calculate fall velocity for snowflakes using Heymsfield et al. (2010) parameterization.

    This function computes the fall velocity of snowflakes using the Heymsfield et al.
    (2010) parameterization that depends on riming mass, particle size, and atmospheric
    conditions.

    Parameters
    ----------
    Dmax : float or array-like
        Maximum particle diameter
    M : float or array-like
        Normalized rime mass
    temperature : float
        Temperature in Kelvin
    press : float
        Atmospheric pressure in Pascals
    shape : str
        Snowflake habit shape

    Returns
    -------
    float or array-like
        Fall velocity in m/s

    Notes
    -----
    This implementation follows the Heymsfield et al. (2010) approach for calculating
    fall velocities of rimed snowflakes based on their mass and size characteristics.
    """
    import pyPamtra

    dynamicViscosity = dynamic_viscosity_air(temperature)
    dryAirDensity = dry_density_air(temperature, press)

    a, b = pyPamtra.descriptorFile.riming_dependent_mass_size(M, shape)
    aa, ba = pyPamtra.descriptorFile.riming_dependent_area_size(M, shape)

    a = xr.DataArray(a, [M.time])
    b = xr.DataArray(b, [M.time])
    aa = xr.DataArray(aa, [M.time])
    ba = xr.DataArray(ba, [M.time])

    mass = a * Dmax**b
    crossSectionArea = aa * Dmax**ba

    k = 0.5  # defined in the paper
    delta_0 = 8.0
    C_0 = 0.35
    g = 9.81

    area_proj = crossSectionArea / ((np.pi / 4.0) * Dmax**2)

    # eq 9
    Xstar = (
        8.0
        * dryAirDensity
        * mass
        * g
        / (np.pi * area_proj ** (1.0 - k) * dynamicViscosity**2)
    )
    # eq10
    Re = (
        delta_0**2
        / 4.0
        * ((1.0 + ((4.0 * np.sqrt(Xstar)) / (delta_0**2 * np.sqrt(C_0)))) ** 0.5 - 1)
        ** 2
    )

    velSpec = dynamicViscosity * Re / (dryAirDensity * Dmax)
    return velSpec


def retrieveCombinedRiming(
    case, config, skipExisting=True, writeNc=True, doQuicklook=True
):
    """
    Apply combined riming retrieval to VISSS data.

    This function performs a combined riming retrieval following the methodology
    described in Maherndl et al. (2023) to determine riming mass and related
    microphysical parameters from radar and meteorological data.

    Parameters
    ----------
    case : str
        Case identifier for data processing
    config : object or str
        Configuration object or path to configuration file
    skipExisting : bool, optional
        Skip processing if output file exists, default is True
    writeNc : bool, optional
        Write NetCDF output file, default is True
    doQuicklook : bool, optional
        Generate quicklook plots, default is True

    Returns
    -------
    tuple
        A tuple containing (lv3Dat, lv3File) where:
        - lv3Dat: Processed level 3 dataset
        - lv3File: Output file path

    Notes
    -----
    This function implements the combined riming retrieval algorithm that:
    1. Retrieves riming mass parameter using optimal estimation
    2. Derives microphysical parameters from the retrieved mass
    3. Calculates ice water content and snowfall rates
    4. Generates NetCDF output and quicklook plots

    References
    ----------
    Maherndl, N., M. Maahn, F. Tridon, J. Leinonen, D. Ori, and S. Kneifel, 2023:
    A riming-dependent parameterization of scattering by snowflakes using the
    self-similar rayleigh–gans approximation. Q. J. R. Meteorolog. Soc., 149,
    3562–3581, doi:10.1002/qj.4573.
    """
    import pyPamtra

    if type(config) is str:
        config = tools.readSettings(config)
    fL = files.FindFiles(case, config.leader, config)

    lv3File = fL.fnamesDaily[f"level3combinedRiming"]

    log.info(f"Processing {lv3File}")

    if (
        writeNc
        and skipExisting
        and tools.checkForExisting(
            lv3File,
            parents=fL.listFilesExt(f"level2track"),
        )
    ):
        return None, None

    if (
        writeNc
        and skipExisting
        and tools.checkForExisting(
            "%s.nodata" % lv3File,
            parents=fL.listFilesExt(f"level2track"),
        )
    ):
        return None, None

    if np.all([f.endswith("broken.txt") for f in fL.listFilesExt(f"level2track")]):
        raise RuntimeError(
            f"All level2track in {fL.fnamesPatternExt[f'level2track']} are broken."
        )

    isEmpty = fL.listFilesExt(f"level2track")[0].endswith("nodata")
    if isEmpty:
        with tools.open2("%s.nodata" % lv3File, config, "w") as f:
            f.write("no data for %s" % case)
        log.warning("no data for %s" % case)
        log.warning("written: %s.nodata" % lv3File)
        return None, lv3File

    radarDat, frequency = aux.getRadarData(case, config)
    meteoDat = aux.getMeteoData(case, config)
    if meteoDat is None:
        return None, None

    lv2DatA = xr.open_dataset(fL.listFilesExt(f"level2track")[0])

    lv2Dat = lv2DatA.sel(cameratrack="max", drop=True)
    lv2Dat = lv2Dat.sel(size_definition="Dmax")
    lv2Dat["velocity_dist"] = lv2DatA["velocity_dist"].sel(
        size_definition="Dmax", cameratrack="mean", dim3D="z", drop=True
    )

    lv2Dat = xr.merge((lv2Dat, meteoDat, radarDat))

    coldEnough = lv2Dat.air_temperature < config.level3.combinedRiming.maxTemp
    isPrecip = (
        lv2Dat[config.level3.combinedRiming.Zvar] >= config.level3.combinedRiming.minZe
    )
    goodQuality = lv2Dat.qualityFlags == 0
    enoughParticles = lv2Dat.nParticles >= config.level3.combinedRiming.minNParticles

    if np.all(~coldEnough | ~isPrecip | ~goodQuality | ~enoughParticles):
        with tools.open2("%s.nodata" % lv3File, config, "w") as f:
            f.write("no snowfall for %s\r" % case)
            f.write("coldEnough %i\r" % np.sum(coldEnough))
            f.write("isPrecip %i\r" % np.sum(isPrecip))
            f.write("goodQuality %i\r" % np.sum(goodQuality))
            f.write("enoughParticles %i\r" % np.sum(enoughParticles))
        log.warning("coldEnough %i" % np.sum(coldEnough))
        log.warning("isPrecip %i" % np.sum(isPrecip))
        log.warning("goodQuality %i" % np.sum(goodQuality))
        log.warning("enoughParticles %i" % np.sum(enoughParticles))
        log.warning("written: %s.nodata" % lv3File)
        return None, lv3File

    goodData = coldEnough & isPrecip & goodQuality & enoughParticles
    lv3Dat = lv2Dat[
        [
            "Ze_0",
            "MDV_0",
            "Ze_ground",
            "MDV_ground",
            "Ze_std",
            "MDV_std",
            "Ze_ground_fitResidual",
            "MDV_ground_fitResidual",
            "air_temperature",
            "air_pressure",
            "relative_humidity",
        ]
    ]
    lv3Dat = lv3Dat.sel(time=lv2Dat.time)

    #### do retrieval ####

    Dbound = np.append(
        lv2Dat.D_bins_left.mean("time").values,
        lv2Dat.D_bins_right.mean("time").values[-1],
    )
    Dmean = lv2Dat.D_bins.values
    psd = np.ma.masked_invalid(lv2Dat.PSD.values).filled(0.0)

    M_oe = np.empty(lv2Dat.time.size) * np.nan
    Ze_combinedRetrieval = np.empty(lv2Dat.time.size) * np.nan
    M_err = np.empty(lv2Dat.time.size) * np.nan

    for j in np.where(goodData)[0]:
        Ze_obs = lv2Dat[config.level3.combinedRiming.Zvar].isel(time=j).values

        print(j, Ze_obs)

        M_oe[j], M_err[j], Ze_combinedRetrieval[j] = retrieveM(
            Ze_obs,
            psd[j],
            lv2Dat.air_temperature.values[j],
            Dmean,
            Dbound,
            frequency,
            config,
        )

    Mlog = xr.DataArray(M_oe, coords=[lv2Dat.time])
    M_err = xr.DataArray(M_err, coords=[lv2Dat.time])
    lv3Dat["Ze_combinedRetrieval"] = xr.DataArray(
        Ze_combinedRetrieval, coords=[lv2Dat.time]
    )
    lv3Dat["combinedNormalizedRimeMass"] = 10**Mlog

    ### derive microphysical parameters
    a, b = pyPamtra.descriptorFile.riming_dependent_mass_size(
        lv3Dat["combinedNormalizedRimeMass"], config.level3.combinedRiming.habit
    )

    lv3Dat["massSizeA"] = ("time", a)
    lv3Dat["massSizeB"] = ("time", b)

    deltaD = lv2Dat.D_bins_right - lv2Dat.D_bins_left

    lv3Dat["IWC"] = (
        lv2Dat.PSD * lv3Dat.massSizeA * (lv2Dat.D_bins) ** lv3Dat.massSizeB * deltaD
    ).sum("D_bins")

    lv3Dat["velocity_dist_heymsfield10"] = heymsfield10_particles_M(
        lv2Dat.D_bins,
        lv3Dat.combinedNormalizedRimeMass,
        lv2Dat.air_temperature,
        lv2Dat.air_pressure,
        config.level3.combinedRiming.habit,
    )

    velocity_dist = lv2Dat.velocity_dist.where(
        lv2Dat.velocity_dist > 0
    )  # negative werte rausschmeißen
    velocity_dist = velocity_dist.interpolate_na(
        dim="D_bins", method="nearest", fill_value="extrapolate"
    )

    lv3Dat["SR_M_dist"] = (
        velocity_dist
        * lv2Dat.PSD
        * lv3Dat.massSizeA
        * (lv2Dat.D_bins) ** lv3Dat.massSizeB  # .max('D_bins')
        * deltaD
    )  # kg/m2/s
    lv3Dat["SR_M"] = lv3Dat["SR_M_dist"].sum("D_bins") * 3600  # mm/h

    lv3Dat["SR_M_heymsfield10_dist"] = (
        lv3Dat["velocity_dist_heymsfield10"]
        * lv2Dat.PSD
        * lv3Dat.massSizeA
        * (lv2Dat.D_bins) ** lv3Dat.massSizeB
        * deltaD
    )
    lv3Dat["SR_M_heymsfield10"] = (
        lv3Dat["SR_M_heymsfield10_dist"].sum("D_bins") * 3600
    )  # mm/h

    # some variables are zero when unknown
    for var in [
        "IWC",
        "SR_M",
        "SR_M_dist",
        "SR_M_heymsfield10",
        "SR_M_heymsfield10_dist",
        "Ze_combinedRetrieval",
        "massSizeA",
        "massSizeB",
        "velocity_dist_heymsfield10",
    ]:
        lv3Dat[var] = lv3Dat[var].where(lv3Dat.combinedNormalizedRimeMass.notnull())

    lv3Dat = tools.finishNc(
        lv3Dat,
        config.site,
        config.visssGen,
        extra={"settings": str(config.level3.combinedRiming)},
    )
    h1, h2 = config.aux.radar.heightRange

    lv3Dat.combinedNormalizedRimeMass.attrs.update(
        dict(
            units="-",
            long_name="normalized rime mass distribution (based on combined retrieval)",
        )
    )
    lv3Dat.Ze_combinedRetrieval.attrs.update(
        dict(
            units="dBz",
            long_name="retrieved radar reflectivity",
        )
    )
    lv3Dat.Ze_0.attrs.update(
        dict(
            units="dBz",
            long_name=f"measured radar reflectivity at lowest used altitude at {h1} m",
        )
    )
    lv3Dat.Ze_ground.attrs.update(
        dict(
            units="dBz",
            long_name=f"measured radar reflectivity extrapolated to 0 m AGL using data from {h1} to {h2} m",
        )
    )
    lv3Dat.Ze_std.attrs.update(
        dict(
            units="dBz",
            long_name=f"standard deviation of measured radar reflectivity using data from {h1} to {h2} m",
        )
    )

    lv3Dat.Ze_ground_fitResidual.attrs.update(
        dict(
            units="dBz",
            long_name=f"residual of the linear fit using measured radar reflectivity using data from {h1} to {h2} m",
        )
    )

    lv3Dat.MDV_0.attrs.update(
        dict(
            units="m/s",
            long_name=f"measured mean Doppler velocity at lowest used altitude at {h1} m",
        )
    )
    lv3Dat.MDV_ground.attrs.update(
        dict(
            units="m/s",
            long_name=f"measured mean Doppler velocity extrapolated to 0 m AGL using data from {h1} to {h2} m",
        )
    )
    lv3Dat.MDV_ground_fitResidual.attrs.update(
        dict(
            units="m/s",
            long_name=f"residual of the linear fit using measured mean Doppler velocity using data from {h1} to {h2} m",
        )
    )

    lv3Dat.MDV_std.attrs.update(
        dict(
            units="m/s",
            long_name=f"standard deviation of measured mean Doppler velocity using data from {h1} to {h2} m",
        )
    )
    lv3Dat.massSizeA.attrs.update(
        dict(
            units="SI",
            long_name="prefactor of the mass size relation",
        )
    )
    lv3Dat.massSizeB.attrs.update(
        dict(
            units="SI",
            long_name="exponent of the mass size relation",
        )
    )
    lv3Dat.IWC.attrs.update(
        dict(
            units="kg/m^3",
            long_name="ice water content",
        )
    )
    lv3Dat.SR_M_dist.attrs.update(
        dict(
            units="kg/m^2/s",
            long_name="spectral snowfall rate using observed fall velocity",
        )
    )
    lv3Dat.SR_M_heymsfield10_dist.attrs.update(
        dict(
            units="kg/m^2/s",
            long_name="spectral snowfall rate using Heymsfield10 fall velocity",
        )
    )
    lv3Dat.velocity_dist_heymsfield10.attrs.update(
        dict(
            units="m/s",
            long_name="spectral fall velocity using Heymsfield10",
        )
    )

    lv3Dat.SR_M.attrs.update(
        dict(
            units="mm/h water equivalent",
            long_name="snowfall rate using observed fall velocity",
        )
    )
    lv3Dat.SR_M_heymsfield10.attrs.update(
        dict(
            units="mm/h water equivalent",
            long_name="snowfall rate using Heymsfield10 fall velocity",
        )
    )
    lv3Dat.relative_humidity.attrs.update(
        dict(
            units="-",
            long_name="2m relative humidity by weather station",
        )
    )
    lv3Dat.air_pressure.attrs.update(
        dict(
            units="Pa",
            long_name="surface air pressure",
        )
    )
    lv3Dat.air_temperature.attrs.update(
        dict(
            units="K",
            long_name="2m air temperature",
        )
    )

    if writeNc:
        tools.to_netcdf2(lv3Dat, config, lv3File)

    if doQuicklook:
        quicklooks.createLevel3RimingQuicklook(case, config)

    lv2DatA.close()

    return lv3Dat, lv3File
