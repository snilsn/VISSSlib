import types

import numpy as np
import pytest
import VISSSlib
import xarray as xr
from VISSSlib.distributions import *
from VISSSlib.distributions import _applyBlurThreshold, _blurThreshold

from helpers import get_test_data_path, get_test_path, readTestSettings

nSample = 100
seed = 0


class TestBlurThreshold:
    """Unit tests for distributions._blurThreshold/_applyBlurThreshold,
    split out of _createLevel2part so this filtering logic (pure function
    of an already-loaded level1dat) can be tested without real
    level1detect files.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("visssGen", ["visss", "visss2", "visss3"])
    def test_blurThreshold_known_generations(self, visssGen):
        thresh = _blurThreshold(visssGen)
        assert np.isnan(thresh[0])
        assert len(thresh) > 350  # must cover the Dmax<=350 filter range
        assert np.all(np.isfinite(thresh[1:]))

    @pytest.mark.unit
    def test_blurThreshold_unknown_generation_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            _blurThreshold("visss99")

    def _makeLevel1dat(self, dmax, blur):
        n = len(dmax)
        return xr.Dataset(
            {
                "Dmax": ("pair_id", np.asarray(dmax, dtype=float)),
                "blur": ("pair_id", np.asarray(blur, dtype=float)),
            },
            coords={"pair_id": np.arange(n)},
        )

    @pytest.mark.unit
    def test_applyBlurThreshold_keeps_sharp_particles(self):
        config = types.SimpleNamespace(visssGen="visss")
        thresh = _blurThreshold("visss")
        # Dmax=10 -> threshold from the table; well above it is "sharp"
        level1dat = self._makeLevel1dat(
            dmax=[10, 10], blur=[thresh[10] + 100, thresh[10] + 200]
        )
        result = _applyBlurThreshold(level1dat, config, "dummy.nc")
        assert result is not None
        assert len(result.pair_id) == 2
        assert "blur" not in result.data_vars

    @pytest.mark.unit
    def test_applyBlurThreshold_drops_blurry_particles(self):
        config = types.SimpleNamespace(visssGen="visss")
        thresh = _blurThreshold("visss")
        level1dat = self._makeLevel1dat(
            dmax=[10, 10], blur=[thresh[10] - 100, thresh[10] + 100]
        )
        result = _applyBlurThreshold(level1dat, config, "dummy.nc")
        assert len(result.pair_id) == 1

    @pytest.mark.unit
    def test_applyBlurThreshold_drops_oversized_particles(self):
        config = types.SimpleNamespace(visssGen="visss")
        thresh = _blurThreshold("visss")
        # Dmax > 350 is dropped outright, regardless of blur
        level1dat = self._makeLevel1dat(
            dmax=[10, 400], blur=[thresh[10] + 100, 1e9]
        )
        result = _applyBlurThreshold(level1dat, config, "dummy.nc")
        assert len(result.pair_id) == 1
        assert result.Dmax.item() == 10

    @pytest.mark.unit
    def test_applyBlurThreshold_returns_none_when_all_filtered(self):
        config = types.SimpleNamespace(visssGen="visss")
        thresh = _blurThreshold("visss")
        level1dat = self._makeLevel1dat(dmax=[10], blur=[thresh[10] - 100])
        result = _applyBlurThreshold(level1dat, config, "dummy.nc")
        assert result is None


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

        # unlike match/track, _createLevel2part's detect branch explicitly
        # deletes the size-definition-within-its-own-bin column (see
        # distributions.py _binTimeAndSize) instead of leaving a half-NaN
        # Dmax_dist/Dequiv_dist behind - assert it stays gone rather than
        # reappearing as an accidental side effect of a future refactor.
        assert "Dmax_dist" not in dat.data_vars
        assert "Dequiv_dist" not in dat.data_vars

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
        assert np.isclose(dat.PSD.mean(), 4228.98486328, rtol=0.02)
        # M6 (~D^6) is dominated by the single largest particle in this
        # tiny test fixture (only ~60 nonzero bins spanning ~10 orders
        # of magnitude), so the few-percent cross-platform Dmax noise
        # already tolerated in testL1Detect gets raised to the 6th power
        assert np.isclose(dat.M6.mean(), 3.32397295e-20, rtol=0.25)
        # angle_mean has only ~90 non-NaN samples out of 25920 in this
        # fixture, and cv2's ellipse-fit angle is inherently noisy for
        # near-circular/small particles, so its mean over so few samples
        # is highly sensitive to the same contour-fit drift as contourFFT
        assert np.isclose(dat.angle_mean.mean(), 78.70565796, rtol=0.2)
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

        # Dmax_dist/Dequiv_dist are a side effect of the size-definition
        # loop in _createLevel2part (distributions.py _binTimeAndSize):
        # only the size_definition slice matching their own name is
        # populated, the other is left NaN. Assert the structure explicitly
        # so a refactor that drops or over-populates these fields gets
        # caught here instead of only showing up as a diff against real
        # production data.
        assert "Dmax_dist" in dat.data_vars
        assert "Dequiv_dist" in dat.data_vars
        assert np.isfinite(dat.Dmax_dist.sel(size_definition="Dmax")).any()
        assert np.isnan(dat.Dmax_dist.sel(size_definition="Dequiv")).all()
        assert np.isfinite(dat.Dequiv_dist.sel(size_definition="Dequiv")).any()
        assert np.isnan(dat.Dequiv_dist.sel(size_definition="Dmax")).all()

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

        # same Dmax_dist/Dequiv_dist half-NaN structure as testL2Match, see
        # comment there.
        assert "Dmax_dist" in dat.data_vars
        assert "Dequiv_dist" in dat.data_vars
        assert np.isfinite(dat.Dmax_dist.sel(size_definition="Dmax")).any()
        assert np.isnan(dat.Dmax_dist.sel(size_definition="Dequiv")).all()
        assert np.isfinite(dat.Dequiv_dist.sel(size_definition="Dequiv")).any()
        assert np.isnan(dat.Dequiv_dist.sel(size_definition="Dmax")).all()


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
