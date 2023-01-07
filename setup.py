import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VISSSlib",
    use_scm_version={
        "version_scheme": "post-release",
    },
    author="Maximilian Maahn",
    description="VISSS processing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maahn/VISSSlib",
    project_urls={
        "Bug Tracker": "https://github.com/maahn/VISSSlib/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy', 'scipy', 'xarray', 'dask[complete]', 'pandas', "pyyaml", "trimesh",
                   "matplotlib", "ipywidgets", "bottleneck", "pillow", "image-packer", "addict", "pyOptimalEstimation"],
    setup_requires=['setuptools_scm'],
)
 # "opencv-python" -> conda version is not recongnized
 # https://stackoverflow.com/questions/57821903/setup-py-with-dependecies-installed-by-conda-not-pip