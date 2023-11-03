# VISSS processing library

Install conda/mamba dependencies

    conda install numpy  scipy  xarray  dask[complete]  pandas pyyaml matplotlib bottleneck pillow  addict opencv Pillow netcdf4 ipywidgets trimesh scikit-image tqdm filterpy flox portalocker numba

Note that trimesh also requires Blender or OpenSCAD which is not available through conda. 

Install PIP dependencies

    pip install image-packer flatten_dict pyOptimalEstimation vg

Clone the library with 

    git clone https://github.com/maahn/VISSSlib

and install with

    cd VISSSlib
    pip install -e .
