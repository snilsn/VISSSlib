# -*- coding: utf-8 -*-

import yaml
import warnings
import datetime

from addict import Dict
from copy import deepcopy

import pandas as pd


LOGGING_CONFIG = { 
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': { 
        'standard': { 
        'format': "'%(asctime)s: %(levelname)s: %(name)s.%(funcName)s: %(message)s'"
        },
    },
    'handlers': { 
        'stream': { 
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # stream is stderr
        },
        'file': { 
            'level': 'WARNING',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': None,  # stream is stderr
        },    },
    'loggers': { 
        '': {  # root logger
            'handlers': ['stream', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
    } 
}

def get_logging_config(fname):
    lc = deepcopy(LOGGING_CONFIG)
    lc['handlers']['file']['filename'] = fname

    return lc

    
niceNames= (
    ('master', 'leader'),
    ('trigger', 'leader'),
    ('slave', 'follower'),
)

def nicerNames(string):
    for i in range(len(niceNames)):
        string=string.replace(*niceNames[i]) 
    return string

def readSettings(fname):
    with open(fname, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return Dict(config)

def getDateRange(nDays, config):
    if config["end"] == "today":
        end = datetime.datetime.utcnow() 
        end2 = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    else:
        end = end2 = config["end"]
    
    if nDays == 0:
        days = pd.date_range(
            start=config["start"],
            end=end2,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )
    elif type(nDays) is str:
        days = pd.date_range(
            start=nDays,
            periods=1,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )
    else:
        days = pd.date_range(
            end=end2,
            periods=nDays,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )
    return days

def otherCamera(camera, config):
    if camera == config["instruments"][0]:
        return config["instruments"][1]
    elif camera == config["instruments"][1]:
        return config["instruments"][0]
    else:
        raise ValueError
