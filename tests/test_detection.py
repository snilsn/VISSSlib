import types

import numpy as np
import pytest
from VISSSlib.detection import *

from helpers import get_test_data_path, get_test_path, makeSyntheticConfig, readTestSettings


class TestRoi(object):
    @pytest.mark.unit
    def test_roi(self):
        img = np.random.random((100, 100))

        for xr in range(0, 40):
            for yr in range(0, 40):
                roi = (xr, yr, 40, 40)
                imgE, xo, yo, _ = extractRoi(roi, img)

                imgE1, xo, yo, extraROI = extractRoi(roi, img, extra=20)
                imgE2, _, _, _ = extractRoi(extraROI, imgE1)

                assert np.all(imgE2 == imgE), (xr, yr)


class TestCheckMotion:
    """Unit tests for detection.checkMotion, a pure function of two frames
    and threshold values (no cv2 pipeline/video files needed).
    """

    @pytest.mark.unit
    def test_counts_pixels_above_each_threshold(self):
        sub = np.zeros((5, 5), dtype=np.uint8)
        old = np.zeros((5, 5), dtype=np.uint8)
        sub[2, 2] = 100
        result = checkMotion(sub, old, [10, 50, 150])
        assert list(result) == [1, 1, 0]

    @pytest.mark.unit
    def test_no_oldFrame_defaults_to_zeros(self):
        sub = np.zeros((5, 5), dtype=np.uint8)
        sub[2, 2] = 100
        assert list(checkMotion(sub, None, [10, 50, 150])) == list(
            checkMotion(sub, np.zeros((5, 5), dtype=np.uint8), [10, 50, 150])
        )

    @pytest.mark.unit
    def test_no_change_below_threshold(self):
        sub = np.full((4, 4), 5, dtype=np.uint8)
        old = np.zeros((4, 4), dtype=np.uint8)
        assert list(checkMotion(sub, old, [10])) == [0]


class TestJoinEdges:
    """Unit tests for detection.joinEdges, a pure function of a binary
    mask array that bridges small gaps between edge pixels.
    """

    @pytest.mark.unit
    def test_all_zero_mask_unchanged(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = joinEdges(mask)
        assert np.all(result == 0)

    @pytest.mark.unit
    def test_input_not_mutated(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3, 3] = 255
        mask[3, 4] = 255
        original = mask.copy()
        joinEdges(mask)
        assert np.array_equal(mask, original)

    @pytest.mark.unit
    def test_bridges_small_gap_between_edge_endpoints(self):
        # two short diagonal edge segments whose endpoints are 3px apart
        # along the same row; joinEdges should fill the gap between them
        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[3, 3] = 255
        mask[4, 4] = 255
        mask[5, 5] = 255
        mask[5, 8] = 255
        mask[6, 9] = 255
        mask[7, 10] = 255

        result = joinEdges(mask)

        before = set(map(tuple, np.argwhere(mask)))
        after = set(map(tuple, np.argwhere(result)))
        assert after - before == {(5, 6), (5, 7)}


class TestSplitUpContours:
    """Unit tests for detection.splitUpConours, a pure function that
    regroups a cv2.findContours-style (contours, hierarchy) pair into
    top-level contours and their descendants, using synthetic
    placeholder "contours" and hand-built hierarchy arrays.
    """

    @pytest.mark.unit
    def test_all_top_level_contours_pass_through_unchanged(self):
        cntsTmp = ["A", "B", "C"]
        # hierarchy columns are [next, previous, first_child, parent]
        hierarchy = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]])
        cnts, cntChildren = splitUpConours(cntsTmp, hierarchy)
        assert cnts == ["A", "B", "C"]
        assert cntChildren == [[], [], []]

    @pytest.mark.unit
    def test_single_child_grouped_under_its_parent(self):
        cntsTmp = ["A", "B"]
        # B's parent is index 0 (A)
        hierarchy = np.array([[[-1, -1, -1, -1], [-1, -1, -1, 0]]])
        cnts, cntChildren = splitUpConours(cntsTmp, hierarchy)
        assert cnts == ["A"]
        assert cntChildren == [["B"]]

    @pytest.mark.unit
    def test_grandchild_attributed_to_topmost_parent(self):
        cntsTmp = ["A", "B", "C"]
        # B's parent is A (index 0); C's parent is B (index 1)
        hierarchy = np.array(
            [[[-1, -1, -1, -1], [-1, -1, -1, 0], [-1, -1, -1, 1]]]
        )
        cnts, cntChildren = splitUpConours(cntsTmp, hierarchy)
        assert cnts == ["A"]
        assert cntChildren == [["B", "C"]]


def _makeFrame(shape=(80, 80), background=200, particleValue=50):
    """Bright background with a dark filled region -- matches VISSS's
    shadowgraph/silhouette imaging (particles are dark against a bright
    background), so particleContrast (background - pixMin) stays
    positive and doesn't hit uint8 wraparound.
    """
    frame = np.full(shape, background, dtype=np.uint8)
    frame[15:45, 15:55] = particleValue
    return frame


def _rectCnt(x0=20, y0=20, x1=50, y1=40):
    return np.array(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32
    )


class TestSingleParticle:
    """Unit tests for detection.singleParticle, the per-particle geometry
    class. Contours here are hand-built polygons (not the output of
    cv2.findContours on a real video frame), so these exercise VISSSlib's
    own downstream math (contourArea/arcLength/moments/minEnclosingCircle
    -- all stable, version-independent computational geometry) without
    depending on cv2's contour-tracing algorithm, which changed across
    OpenCV releases (see distributions.py/detection.py findContours
    changelog investigation) and is not expected to be bit-reproducible
    across versions.
    """

    @pytest.fixture
    def config(self, tmp_path):
        return makeSyntheticConfig(tmp_path)

    @pytest.fixture
    def parent(self):
        return types.SimpleNamespace(testing=[], brightnessBackground=200)

    def _makeParticle(self, config, parent, cnt, cntChild=[], frame=None, pp1=0):
        frame = _makeFrame() if frame is None else frame
        return singleParticle(
            parent,
            config,
            capture_id=1,
            record_id=1,
            capture_time=np.datetime64("2026-01-01"),
            record_time=np.datetime64("2026-01-01"),
            nThread=0,
            pp1=pp1,
            frame1=frame.copy(),
            mask1=None,
            cnt=cnt.copy(),
            cntChild=[c.copy() for c in cntChild],
            xOffset=0,
            yOffset=0,
            testing=[],
        )

    @pytest.mark.unit
    def test_rectangle_geometry_matches_analytic_values(self, config, parent):
        # 30x20 axis-aligned rectangle at (20,20)-(50,40)
        cnt = _rectCnt()
        sp = self._makeParticle(config, parent, cnt)

        assert sp.success
        assert sp.area == pytest.approx(30 * 20)
        assert sp.perimeter == pytest.approx(2 * (30 + 20))
        # cv2.boundingRect is pixel-inclusive, hence the +1 on each dim
        assert list(sp.roi) == [20, 20, 31, 21]
        # min enclosing circle of a rectangle's diagonal
        assert sp.Dmax == pytest.approx((30**2 + 20**2) ** 0.5, rel=1e-3)
        assert sp.position_centroid == (35, 30)
        # rect-based aspect ratio (short side / long side)
        assert sp.aspectRatio[-1] == pytest.approx(20 / 30)
        assert sp.extent == pytest.approx(sp.area / (sp.roi[2] * sp.roi[3]))
        assert sp.solidity == pytest.approx(1.0)  # convex hull of a rectangle is itself
        assert sp.pixMin == sp.pixMax == sp.pixMean == 50
        assert sp.blur == 0.0  # uniform fill -> zero-variance Laplacian
        assert sp.particleContrast == pytest.approx(200 - 50)

    @pytest.mark.unit
    def test_particleContrast_goes_negative_for_reflective_particles(
        self, config, parent
    ):
        # a particle brighter than the background (e.g. a sunlight
        # reflection) must give a negative contrast, not wrap around
        # under uint8 arithmetic
        frame = _makeFrame(background=50, particleValue=200)
        parent.brightnessBackground = 50
        sp = self._makeParticle(config, parent, _rectCnt(), frame=frame)

        assert sp.pixMin == 200
        assert sp.particleContrast == pytest.approx(50 - 200)

    @pytest.mark.unit
    def test_hole_is_subtracted_from_area_and_added_to_perimeter(self, config, parent):
        cnt = _rectCnt()
        hole = _rectCnt(30, 25, 40, 35)  # 10x10 hole inside the particle
        sp = self._makeParticle(config, parent, cnt, cntChild=[hole])

        assert sp.area == pytest.approx(600)  # unaffected by the hole
        assert sp.areaConsideringHoles == pytest.approx(600 - 100)
        assert sp.perimeterConsideringHoles == pytest.approx(100 + 40)
        assert len(sp.cntChild) == 1

    @pytest.mark.unit
    def test_tiny_hole_ignored_when_check4childCntLength(self, config, parent):
        # holes with area <= 4 are discarded regardless of check4childCntLength
        config.level1detect.check4childCntLength = True
        cnt = _rectCnt()
        tinyHole = _rectCnt(30, 25, 32, 27)  # 2x2 -> area 4
        sp = self._makeParticle(config, parent, cnt, cntChild=[tinyHole])

        assert sp.areaConsideringHoles == pytest.approx(sp.area)
        assert len(sp.cntChild) == 0

    @pytest.mark.unit
    def test_ellipse_fits_require_more_than_four_points(self, config, parent):
        # a 4-point rectangle can't be ellipse-fit -- NaN by design
        sp4 = self._makeParticle(config, parent, _rectCnt())
        assert np.isnan(sp4.angle_ellipse)
        assert np.isnan(sp4.angle_ellipseDirect)

        # a 5+ point contour does get fit
        pentagon = np.array(
            [[[35, 15]], [[55, 25]], [[47, 42]], [[23, 42]], [[15, 25]]],
            dtype=np.int32,
        )
        sp5 = self._makeParticle(config, parent, pentagon)
        assert not np.isnan(sp5.angle_ellipse)
        assert not np.isnan(sp5.angle_ellipseDirect)


class TestDetectedParticlesAdd:
    """Unit tests for detectedParticles.add()'s bookkeeping and rejection
    logic, with applyCanny2Particle disabled so the given contour is used
    directly (bypassing detectedParticles' internal cv2.findContours
    call, keeping this test independent of cv2's contour-tracing
    version/algorithm).
    """

    @pytest.fixture
    def dp(self, tmp_path):
        config = makeSyntheticConfig(tmp_path)
        config.level1detect.applyCanny2Particle = False
        config.level1detect.minBlur = -999
        config.level1detect.minContrast = -999
        config.level1detect.erosionTestThreshold = -1

        dp = detectedParticles(config, testing=[])
        dp.capture_id, dp.record_id = 1, 1
        dp.capture_time = np.datetime64("2026-01-01")
        dp.record_time = np.datetime64("2026-01-01")
        dp.nThread = 0
        dp.brightnessBackground = 200
        dp.frame4drawing = None
        return dp

    @pytest.fixture
    def frame(self):
        return _makeFrame()

    @pytest.fixture
    def fgMask(self):
        return np.zeros((80, 80), dtype=np.uint8)

    @pytest.mark.unit
    def test_accepts_valid_particle_and_tracks_it(self, dp, frame, fgMask):
        added = dp.add(frame.copy(), fgMask.copy(), _rectCnt())
        assert added is True
        assert dp.N == 1
        assert dp.pids == [0]
        assert dp.pp == 1

    @pytest.mark.unit
    def test_rejects_particle_touching_border(self, dp, frame, fgMask):
        cntBorder = _rectCnt(0, 20, 20, 40)  # x0=0 touches the left edge
        added = dp.add(frame.copy(), fgMask.copy(), cntBorder)
        assert added is False
        assert dp.N == 0
        assert dp.pp == 0

    @pytest.mark.unit
    def test_rejects_particle_below_minDmax(self, dp, frame, fgMask):
        dp.config.level1detect.minDmax = 9999
        added = dp.add(frame.copy(), fgMask.copy(), _rectCnt())
        assert added is False
        assert dp.N == 0

    @pytest.mark.unit
    def test_reflective_particle_survives_abs_contrast_filter(self, dp, fgMask):
        # a particle brighter than the background (e.g. a sunlight
        # reflection) must still pass the minContrast filter, which
        # compares abs(particleContrast) against the threshold -- not
        # just particleContrast, which would always be negative here
        dp.config.level1detect.minContrast = 20  # the real default
        dp.brightnessBackground = 50  # dark background
        reflectiveFrame = _makeFrame(background=50, particleValue=200)

        added = dp.add(reflectiveFrame.copy(), fgMask.copy(), _rectCnt())
        assert added is True
        assert dp.N == 1

    @pytest.mark.unit
    def test_skips_contour_below_minCntSize(self, dp, frame, fgMask):
        tinyCnt = np.array([[[5, 5]], [[6, 5]]], dtype=np.int32)  # 2 points
        added = dp.add(frame.copy(), fgMask.copy(), tinyCnt)
        assert added is False
        assert dp.N == 0


class TestDetection(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield

    def testL1Detect(self):
        fname = f"{self.testPath}/test_0.6/rawdata/level0/visss11gb_visss_leader_S1145792/2026/01/10/visss11gb_visss_leader_S1145792_20260110-083000_0.txt"
        dat = detectParticles(fname, self.config, writeNc=False, skipExisting=False)

        for var in [
            "Dfit",
            "Dmax",
            "Droi",
            "angle",
            "area",
            "areaConsideringHoles",
            "aspectRatio",
            "blur",
            "capture_id",
            "capture_time",
            "contourFFT",
            "contourFFTstd",
            "contourFFTsum",
            "extent",
            "extentConsideringHoles",
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
            "position_centroid",
            "position_circle",
            "position_fit",
            "position_upperLeft",
            "record_id",
            "record_time",
            "solidity",
            "solidityConsideringHoles",
        ]:
            assert var in dat.data_vars
        # these come straight out of cv2.findContours -> contourArea/
        # arcLength/moments, and findContours' own tracing algorithm
        # changed across OpenCV releases in this range (4.11's contour
        # approximation fix, 4.12's findContours optimization, 4.13's
        # new findTRUContours) -- rtol=1e-3 isn't reproducible across
        # OpenCV versions, observed drift is up to ~2%
        assert np.isclose(dat.Dmax.mean(), 6.44612646, rtol=0.02)
        assert np.isclose(dat.area.mean(), 41.22136307, rtol=0.02)
        assert np.isclose(dat.perimeter.mean(), 18.46730042, rtol=0.02)
        # contourFFT resamples the contour onto a fixed angular grid and
        # FFTs it, which turned out more sensitive still: observed drift
        # up to ~4.2% even between two machines on the *same* OpenCV
        # version, so this isn't purely a findContours-version effect --
        # some of it is platform-level floating point noise in the
        # resampling/FFT step
        assert np.isclose(dat.contourFFT.mean(), 1.45889866, rtol=0.06)
