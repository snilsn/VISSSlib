import types

import numpy as np
import pytest
import xarray as xr
from VISSSlib.fixes import (
    captureIdOverflows,
    isDroppableTrailingFrameShortfall,
    removeFlippedCaptureTimeFrames,
    revertIdOverflowFix,
)


def _makeDat(ids, fps=100):
    n = len(ids)
    times = np.datetime64("2026-01-01T00:00:00") + np.arange(n) * np.timedelta64(
        round(1e6 / fps), "us"
    )
    return xr.Dataset(
        {"capture_id": ("pid", np.asarray(ids))},
        coords={"pid": np.arange(n), "capture_time": ("pid", times)},
    )


class TestCaptureIdOverflows:
    """Unit tests for fixes.captureIdOverflows/revertIdOverflowFix, pure
    functions of an in-memory xr.Dataset with capture_id/capture_time
    (no file I/O). The exact overflow-wrap boundary value is a hardware
    detail this suite doesn't assert on; these cover the surrounding,
    well-defined behavior instead.
    """

    @pytest.mark.unit
    def test_no_overflow_leaves_ids_unchanged(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        out = captureIdOverflows(dat, config)
        assert list(out.capture_id.values) == list(range(100, 110))

    @pytest.mark.unit
    def test_storeOrig_preserves_original_values(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        out = captureIdOverflows(dat, config, storeOrig=True)
        assert "capture_id_orig" in out.keys()
        assert list(out.capture_id_orig.values) == list(range(100, 110))

    @pytest.mark.unit
    def test_storeOrig_false_omits_orig_variable(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        out = captureIdOverflows(dat, config, storeOrig=False)
        assert "capture_id_orig" not in out.keys()

    @pytest.mark.unit
    def test_idOffset_is_added_to_all_ids(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        out = captureIdOverflows(dat, config, idOffset=5)
        assert list(out.capture_id.values) == list(range(105, 115))

    @pytest.mark.unit
    def test_existing_capture_id_orig_is_reverted_before_refixing(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        dat["capture_id_orig"] = dat["capture_id"].copy()
        dat["capture_id"] = dat["capture_id"] * 0  # corrupt capture_id
        out = captureIdOverflows(dat, config, storeOrig=False)
        # corrupted capture_id must have been restored from capture_id_orig first
        assert list(out.capture_id.values) == list(range(100, 110))

    @pytest.mark.unit
    def test_more_observed_jumps_than_expected_raises(self):
        config = types.SimpleNamespace(fps=100)
        # two apparent wraps within a time span that only accounts for one
        dat = _makeDat([65533, 65534, 0, 1, 65533, 0, 1])
        with pytest.raises(RuntimeError):
            captureIdOverflows(dat, config)

    @pytest.mark.unit
    def test_revertIdOverflowFix_restores_original_capture_id(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        fixed = captureIdOverflows(dat, config, storeOrig=True)
        reverted = revertIdOverflowFix(fixed)
        assert list(reverted.capture_id.values) == list(range(100, 110))
        assert "capture_id_fixed" in reverted.keys()


def _makeMetaDat1(captureTimes):
    """A single source's metadata in its own original recording order,
    matching the shape fixes.removeFlippedCaptureTimeFrames expects."""
    n = len(captureTimes)
    return xr.Dataset(
        {"dummy": ("capture_time", np.arange(n))},
        coords={"capture_time": np.asarray(captureTimes, dtype="int64")},
    )


class TestRemoveFlippedCaptureTimeFrames:
    """Unit tests for fixes.removeFlippedCaptureTimeFrames, a pure function
    of a single source's in-memory xr.Dataset (no file I/O), reflecting the
    real M2050/M1280 hardware glitch shapes measured directly against
    production hyytiala2_v4 and SAIL ascii logs: one or more isolated,
    microsecond-scale backwards flips between two consecutive frames,
    scattered independently through an otherwise clean file.
    """

    @pytest.mark.unit
    def test_no_flips_leaves_data_untouched(self):
        metaDat = _makeMetaDat1([0, 1, 2, 3, 4])
        result, nDropped = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert nDropped == 0
        assert list(result.capture_time.values) == [0, 1, 2, 3, 4]

    @pytest.mark.unit
    def test_single_isolated_flip_drops_neighbours(self):
        # index 2 (value 5) and index 3 (value 1) are flipped, matching the
        # real single-swap shape measured on the SAIL/M1280 case
        metaDat = _makeMetaDat1([0, 3, 5, 1, 6, 7])
        result, nDropped = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert nDropped == 3
        assert list(result.capture_time.values) == [0, 6, 7]

    @pytest.mark.unit
    def test_multiple_isolated_flips_each_repaired_independently(self):
        # several well-separated single-sample flips in one file -- the
        # exact pattern measured on the real hyytiala2_v4 case (5 isolated
        # flips per thread, far apart from each other). The old algorithm
        # this replaces asserted all jumps had to form one contiguous
        # group and would have raised here; this must repair each
        # independently instead.
        metaDat = _makeMetaDat1([0, 5, 1, 6, 10, 15, 11, 16, 20, 25, 21, 26])
        result, nDropped = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert nDropped == 9  # 3 clusters x 3 dropped each
        assert np.all(np.diff(result.capture_time.values) >= 0)

    @pytest.mark.unit
    def test_grouped_adjacent_flips_dropped_together(self):
        # two adjacent backwards steps form one cluster and are repaired
        # as a single unit, not two overlapping ones
        metaDat = _makeMetaDat1([0, 3, 5, 2, 1, 6])
        result, nDropped = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert nDropped > 0
        assert np.all(np.diff(result.capture_time.values) >= 0)

    @pytest.mark.unit
    def test_too_many_flips_raises(self):
        metaDat = _makeMetaDat1(list(range(21, 0, -1)))
        with pytest.raises(AssertionError, match="more than 20"):
            removeFlippedCaptureTimeFrames(metaDat, "f.txt")

    @pytest.mark.unit
    def test_surviving_values_are_never_mutated(self):
        # every timestamp that isn't dropped must be bit-for-bit identical
        # to its original value -- this fix must only ever remove frames
        # whose own timestamp is already self-contradictory, never shift,
        # average, or interpolate a value for a frame that is kept
        original = [0, 5, 1, 6, 10, 15, 11, 16]
        metaDat = _makeMetaDat1(original)
        result, _ = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert set(result.capture_time.values.tolist()) <= set(original)


class TestIsDroppableTrailingFrameShortfall:
    """Unit tests for fixes.isDroppableTrailingFrameShortfall, reflecting
    the observed real-world shortfall sizes (1-6 frames, always at the
    tail of a file) on hyytiala2_v3 and nyaalesund.
    """

    @pytest.mark.unit
    def test_small_shortfall_is_droppable(self):
        assert isDroppableTrailingFrameShortfall(1) is True
        assert isDroppableTrailingFrameShortfall(6) is True

    @pytest.mark.unit
    def test_shortfall_at_cap_is_droppable(self):
        assert isDroppableTrailingFrameShortfall(25, maxFrames=25) is True

    @pytest.mark.unit
    def test_large_shortfall_is_not_droppable(self):
        # a shortfall spanning a large chunk of a file indicates real
        # corruption, not a benign encoder-flush tail loss, and must not
        # be silently swallowed
        assert isDroppableTrailingFrameShortfall(26, maxFrames=25) is False
        assert isDroppableTrailingFrameShortfall(5000) is False
