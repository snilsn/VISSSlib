import numpy as np
import VISSSlib
from VISSSlib.distributions import *

from helpers import get_test_data_path, get_test_path, readTestSettings

nSample = 100
seed = 0


import numpy as np
import pytest
from VISSSlib.metadata import *

from helpers import get_test_data_path, get_test_path, readTestSettings


class TestL2(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield

    def testL2Detect(self):
        case = "20260110"
        dat, _ = createLevel2detect(
            case,
            "leader",
            self.config,
            skipExisting=False,
            writeNc=False,
            doPlot=False,
            doParticlePlot=False,
        )
        assert np.isclose(dat.PSD.mean(), 438.8838501)
        assert np.isclose(dat.M6.mean(), 5.05424415e-20)
        assert np.isclose(dat.angle_mean.mean(), 75.70424652)
        for var in [
            "D32",
            "D43",
            "D_bins_left",
            "D_bins_right",
            "Dequiv_mean",
            "Dequiv_std",
            "Dmax_mean",
            "Dmax_std",
            "M1",
            "M2",
            "M3",
            "M4",
            "M6",
            "N0_star_32",
            "N0_star_43",
            "Ntot",
            "PSD",
            "angle_dist",
            "angle_mean",
            "angle_std",
            "areaConsideringHoles_dist",
            "areaConsideringHoles_mean",
            "areaConsideringHoles_std",
            "area_dist",
            "area_mean",
            "area_std",
            "aspectRatio_dist",
            "aspectRatio_mean",
            "aspectRatio_std",
            "blockedPixelRatio",
            "blowingSnowRatio",
            "complexityBW_dist",
            "complexityBW_mean",
            "complexityBW_std",
            "counts",
            "extent_dist",
            "extent_mean",
            "extent_std",
            "nParticles",
            "normalizedRimeMass_dist",
            "normalizedRimeMass_mean",
            "normalizedRimeMass_std",
            "obs_volume",
            "perimeterConsideringHoles_dist",
            "perimeterConsideringHoles_mean",
            "perimeterConsideringHoles_std",
            "perimeter_dist",
            "perimeter_mean",
            "perimeter_std",
            "qualityFlags",
            "solidity_dist",
            "solidity_mean",
            "solidity_std",
        ]:
            assert var in dat.data_vars

    def testL2Match(self):
        case = "20260110"
        dat, _ = createLevel2match(
            case,
            self.config,
            skipExisting=False,
            writeNc=False,
            doPlot=False,
            doParticlePlot=False,
        )
        assert np.isclose(dat.PSD.mean(), 4228.98486328)
        assert np.isclose(dat.M6.mean(), 3.32397295e-20)
        assert np.isclose(dat.angle_mean.mean(), 78.70565796)
        for var in [
            "D32",
            "D43",
            "D_bins_left",
            "D_bins_right",
            "Dequiv_mean",
            "Dequiv_std",
            "Dmax_mean",
            "Dmax_std",
            "M1",
            "M2",
            "M3",
            "M4",
            "M6",
            "N0_star_32",
            "N0_star_43",
            "Ntot",
            "PSD",
            "angle_dist",
            "angle_mean",
            "angle_std",
            "areaConsideringHoles_dist",
            "areaConsideringHoles_mean",
            "areaConsideringHoles_std",
            "area_dist",
            "area_mean",
            "area_std",
            "aspectRatio_dist",
            "aspectRatio_mean",
            "aspectRatio_std",
            "blockedPixelRatio",
            "blowingSnowRatio",
            "complexityBW_dist",
            "complexityBW_mean",
            "complexityBW_std",
            "counts",
            "extent_dist",
            "extent_mean",
            "extent_std",
            "matchScore_mean",
            "matchScore_std",
            "nParticles",
            "normalizedRimeMass_dist",
            "normalizedRimeMass_mean",
            "normalizedRimeMass_std",
            "obs_volume",
            "observationsRatio",
            "perimeterConsideringHoles_dist",
            "perimeterConsideringHoles_mean",
            "perimeterConsideringHoles_std",
            "perimeter_dist",
            "perimeter_mean",
            "perimeter_std",
            "qualityFlags",
            "solidity_dist",
            "solidity_mean",
            "solidity_std",
        ]:
            assert var in dat.data_vars

    def testL2Track(self):
        case = "20260110"
        dat, _ = createLevel2track(
            case,
            self.config,
            skipExisting=False,
            writeNc=False,
            doPlot=False,
        )
        assert np.isclose(dat.PSD.mean(), 4219.70556641)
        assert np.isclose(dat.M6.mean(), 2.45204412e-20)
        assert np.isclose(dat.angle_mean.mean(), 67.9868927)
        for var in [
            "D32",
            "D43",
            "D_bins_left",
            "D_bins_right",
            "Dequiv_mean",
            "Dequiv_std",
            "Dmax_mean",
            "Dmax_std",
            "M1",
            "M2",
            "M3",
            "M4",
            "M6",
            "N0_star_32",
            "N0_star_43",
            "Ntot",
            "PSD",
            "angle_dist",
            "angle_mean",
            "angle_std",
            "areaConsideringHoles_dist",
            "areaConsideringHoles_mean",
            "areaConsideringHoles_std",
            "area_dist",
            "area_mean",
            "area_std",
            "aspectRatio_dist",
            "aspectRatio_mean",
            "aspectRatio_std",
            "blockedPixelRatio",
            "blowingSnowRatio",
            "complexityBW_dist",
            "complexityBW_mean",
            "complexityBW_std",
            "counts",
            "extent_dist",
            "extent_mean",
            "extent_std",
            "matchScore_mean",
            "matchScore_std",
            "nParticles",
            "normalizedRimeMass_dist",
            "normalizedRimeMass_mean",
            "normalizedRimeMass_std",
            "obs_volume",
            "observationsRatio",
            "perimeterConsideringHoles_dist",
            "perimeterConsideringHoles_mean",
            "perimeterConsideringHoles_std",
            "perimeter_dist",
            "perimeter_mean",
            "perimeter_std",
            "qualityFlags",
            "solidity_dist",
            "solidity_mean",
            "solidity_std",
            "track_angle_dist",
            "track_angle_mean",
            "track_angle_std",
            "track_length_mean",
            "track_length_std",
            "velocity_dist",
            "velocity_mean",
            "velocity_std",
        ]:
            assert var in dat.data_vars


class TestVolume(object):
    def test_VolumeInterpolation(self):
        width = 1280
        height = 1024

        maxSharpnessSizes = tuple()
        maxSharpnessLeader = tuple()
        maxSharpnessFollower = tuple()
        correctForSmallOnes = False

        np.random.seed(seed)
        phi, theta, Of_z = np.random.random(3)
        minDmax, maxDmax = 0, 20
        sizeBins = tuple(np.linspace(minDmax, maxDmax))
        D_highRes, V_highRes = VISSSlib.distributions._estimateVolumes(
            width,
            height,
            correctForSmallOnes,
            phi,
            theta,
            Of_z,
            sizeBins,
            maxSharpnessSizes,
            maxSharpnessLeader,
            maxSharpnessFollower,
            nSteps=21,
            interpolate=True,
        )
        D_lowRes, V_lowRes = VISSSlib.distributions._estimateVolumes(
            width,
            height,
            correctForSmallOnes,
            phi,
            theta,
            Of_z,
            sizeBins,
            maxSharpnessSizes,
            maxSharpnessLeader,
            maxSharpnessFollower,
            nSteps=2,
            interpolate=True,
        )

        assert np.all(np.abs(V_lowRes - V_highRes) / V_highRes < 1e-2)

    def test_volumeEstimate(self):
        width = 1280
        height = 1024

        # no rotation!
        phi, theta, Of_z = 0, 0, 0

        V = VISSSlib.distributions._estimateVolume(width, height, phi, theta, Of_z)

        assert np.isclose(V, width * width * height)
