[![Documentation Status](https://readthedocs.org/projects/vissslib/badge/?version=latest)](https://vissslib.readthedocs.io/en/latest/?badge=latest)


# VISSS (Video In Situ Snowfall Sensor) processing library

This repository contains the VISSS data acquisition software. Please see also
* VISSS data acquisition software https://github.com/maahn/VISSS
* VISSS2 hardware plans https://zenodo.org/doi/10.5281/zenodo.7640820
* VISSS3 hardware plans https://zenodo.org/doi/10.5281/zenodo.10526897


## Installation

Install conda/mamba dependencies

    conda install numpy  scipy  xarray  dask[complete]  pandas pyyaml matplotlib bottleneck pillow  addict opencv Pillow netcdf4 ipywidgets trimesh=4.0.5 scikit-image tqdm filterpy flox portalocker numba xarray-extras

Install PIP dependencies

    pip install image-packer flatten_dict pyOptimalEstimation vg manifold3d==2.2.2

Clone the library with 

    git clone https://github.com/maahn/VISSSlib

and install with

    cd VISSSlib
    pip install -e .
