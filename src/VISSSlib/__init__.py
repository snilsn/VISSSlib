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
# __version__ = '20220520' # reversed timestamp fix, applied capture_id fix instead!

# Version is pulled from git tag!!



from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("VISSSlib").split(".")[0]
    __versionFull__ = version("VISSSlib")
except PackageNotFoundError:
    # package is not installed
    pass


from . import files
from . import quicklooks
from . import av
from . import tools
from . import detection
from . import metadata
from . import matching
from . import tracking
from . import fixes
from . import scripts

