# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pn
import bottleneck as bn

import logging
log = logging.getLogger()

from copy import deepcopy


#various tools to fix tiem stamps

def fixMosaicTimeL1(dat1, config):

    '''
    attempt to fix drift of capture time with record_time
    '''

    datS = dat1[["capture_time", "record_time"]]
    datS = datS.isel(capture_time=slice(None, None, config["fps"]))
    diff = (datS.capture_time - datS.record_time)

    # no estiamte the drift
    drifts = []
    #group netcdf into 1 minute chunks
    index1min = diff.capture_time.resample(capture_time="1T", label="right").first().capture_time.values
    if len(index1min) <= 2:
        index1min = diff.capture_time.resample(capture_time="30s", label="right").first().capture_time.values
        if len(index1min) <= 2:
            index1min = diff.capture_time.resample(capture_time="10s", label="right").first().capture_time.values
            if len(index1min) <= 2:
                index1min = diff.capture_time.resample(capture_time="1s", label="right").first().capture_time.values

    grps = diff.groupby_bins("capture_time", bins=index1min)

    #find max. difference in each chunk
    #this is the one were we assume it is the true dirft
    #also time stamp or max.  is needed, this is why resample cannot be used directly
    for ii, grp in grps:
        drifts.append(grp.isel(capture_time=grp.argmax()))
    drifts = xr.concat(drifts, dim="capture_time")

    # interpolate to original resolution
    # extrapolation required for beginning or end - works usually very good!
    driftsInt = drifts.astype(int).interp_like(dat1.capture_time, kwargs={"fill_value": "extrapolate"}).astype('timedelta64[ns]')

    #get best time estimate
    bestestimate = dat1.capture_time.values - driftsInt.values

#                 plt.figure()
#                 driftsInt.plot(marker="x")
#                 diff.plot()

    # replace time in nc file
    dat1["capture_time_orig"] = deepcopy(dat1["capture_time"])
    dat1 = dat1.assign_coords(capture_time=bestestimate)

    #the difference between bestestimate and capture time must jump more than 0.1% of the measurement interval
    assert np.all((np.abs(((dat1.capture_time-dat1.capture_time_orig).diff("capture_time")/dat1.capture_time_orig.diff("capture_time")))) < 0.001)

    return dat1
