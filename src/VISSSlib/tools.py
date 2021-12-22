# -*- coding: utf-8 -*-

import yaml

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
    return config
