import numpy as np
import pytest
from VISSSlib.tracking import *

from helpers import get_test_data_path, get_test_path, readTestSettings


class TestTracking(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield

    def testL1Track(self):
        fname = f"{self.testPath}/test_0.6/products/level1detect/2026/01/10/level1detect_V1.2_test_visss11gb_visss_leader_S1145792_20260110-083000.nc"
        dat, _ = trackParticles(fname, self.config, writeNc=False, skipExisting=False)

        for var in [
            "Dfit",
            "Dmax",
            "Droi",
            "angle",
            "area",
            "areaConsideringHoles",
            "aspectRatio",
            "blur",
            "camera_Ofz",
            "camera_phi",
            "camera_theta",
            "capture_id",
            "capture_time",
            "contourFFT",
            "contourFFTstd",
            "contourFFTsum",
            "extent",
            "extentConsideringHoles",
            "file_starttime",
            "matchScore",
            "nThread",
            "perimeter",
            "perimeterConsideringHoles",
            "perimeterEroded",
            "pixCenter",
            "pixKurtosis",
            "pixMax",
            "pixMean",
            "pixMin",
            "pixPercentiles",
            "pixSkew",
            "pixStd",
            "pixSum",
            "position3D_center",
            "position3D_centroid",
            "position_centroid",
            "position_circle",
            "position_fit",
            "position_upperLeft",
            "record_id",
            "record_time",
            "solidity",
            "solidityConsideringHoles",
            "track_angleGuess",
            "track_id",
            "track_step",
            "track_velocityGuess",
        ]:
            assert var in dat.data_vars
        assert np.isclose(dat.Dmax.mean(), 4.98294973)
        assert np.isclose(dat.area.mean(), 15.86436176)
        assert np.isclose(dat.perimeter.mean(), 13.49942)
        assert np.isclose(dat.contourFFT.mean(), 1.2981159)
