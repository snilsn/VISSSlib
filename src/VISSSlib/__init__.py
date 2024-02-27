# __version__ = '20210315' # detection only, based on netcdf data
# __version__ = '20210809' # detection only, based on netcdf data
# __version__ = '20210829' # including Cx and Cy, capture_id is int
# __version__ = '20210830' # add record_id
# __version__ = '20210905' # custom movement detection
# __version__ = '20210907a' # back to CV2 movement detection, reading meta data on the fly.
# __version__ = '20210908' # new variable: touches border, no longer checking for blur, 20 frames history
# __version__ = '20211014' # adapting to multiple frames
# __version__ = '20211022' # minor adaptions for VISSS2
# __version__ = '20211029' # maxDarkestPoint=130 instead 100
# __version__ = '20212016' # movingObjects variable to detect blowing snow better, maxNParticle=60
# __version__ = '20220308' # fixed time stamp for mosaic!!
# __version__ = '20220521' # reversed timestamp fix, applied capture_id fix instead!
# __version__ = '20220521' # Aug 3rd: added background image and nThread to level1detect without new version number
# __version__ = '20221024' # bug fixes: blur, angle. more blocking bins 4 event file. using canny for particle detection, added cnt, perimeterEroded to nc file.
# images in tar file, alpha channel shows applied mask, particles touching border noct procesed
# __version__ = '20230106' # Dmax bugfix, clean up of variables, position_centroid missing until jan 31st!!
# __version__ = '20231116' #adapted detection (lower thresholds), first workign tracking version
# __version__ = '1.0.0' # new  version number scheme to allow better control about reprocessin, added FFT contour


# Version is pulled from git tag!!
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = ".".join(version("VISSSlib").split(".")[:2])
    __versionFull__ = version("VISSSlib")
except PackageNotFoundError:
    # package is not installed
    __version__ = "NotAvailable"
    __versionFull__ = "NotAvailable"
    pass

import logging
import logging.config
import os
import socket
from copy import deepcopy

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "format": "%(asctime)s: %(levelname)s: %(name)s.%(funcName)s: %(message)s"
        },
    },
    "handlers": {
        "stream": {
            "level": "WARNING",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # stream is stdout
        },
        "err": {
            "level": "ERROR",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",  # stream is stderr
        },
        # 'file': {
        #     'level': 'WARNING',
        #     'formatter': 'standard',
        #     'class': 'logging.FileHandler',
        #     'filename': None,  # stream is stderr
        # },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["stream", "err"],  # , 'file'
            "level": "DEBUG",
            "propagate": False,
        },
    },
}


def get_logging_config(fname):
    lc = deepcopy(LOGGING_CONFIG)
    # lc['handlers']['file']['filename'] = fname
    return lc


# logging.config.dictConfig(get_logging_config(f'VISSS_{socket.gethostname()}_{os.getpid()}.log'))
logging.config.dictConfig(get_logging_config(f"VISSS_{socket.gethostname()}.log"))
log = logging.getLogger(__name__)


from . import (
    analysis,
    av,
    detection,
    distributions,
    files,
    fixes,
    matching,
    metadata,
    quicklooks,
    scripts,
    tools,
    tracking,
)
