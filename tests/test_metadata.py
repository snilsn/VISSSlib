import types

import numpy as np
import pytest
import xarray as xr
from VISSSlib.metadata import *
from VISSSlib.metadata import _repairTimeJumps

from helpers import get_test_data_path, get_test_path, readTestSettings


def _makeMetaDat(captureTimes):
    n = len(captureTimes)
    return xr.Dataset(
        {"dummy": ("capture_time", np.arange(n))},
        coords={"capture_time": np.asarray(captureTimes, dtype="int64")},
    )


class TestRepairTimeJumps:
    """Unit tests for metadata._repairTimeJumps, split out of getMetaData()
    so this control-flow logic can be tested without real video/csv files.
    """

    @pytest.mark.unit
    def test_no_jumps_leaves_data_untouched(self):
        config = types.SimpleNamespace(visssGen="visss")
        metaDat = _makeMetaDat([0, 1, 2, 3, 4])
        result, nDropped = _repairTimeJumps(metaDat, ["f.txt"], config)
        assert nDropped == 0
        assert len(result.capture_time) == 5

    @pytest.mark.unit
    def test_single_jump_drops_frame_and_neighbours(self):
        config = types.SimpleNamespace(visssGen="visss")
        # index 3 (value 1) is lower than index 2 (value 5) -> jump at position 2
        metaDat = _makeMetaDat([0, 3, 5, 1, 6, 7])
        result, nDropped = _repairTimeJumps(metaDat, ["f.txt"], config)
        assert nDropped == 3
        assert list(result.capture_time.values) == [0, 6, 7]

    @pytest.mark.unit
    def test_grouped_jumps_are_allowed(self):
        config = types.SimpleNamespace(visssGen="visss")
        # two adjacent backwards jumps (positions 2 and 3) form one group
        metaDat = _makeMetaDat([0, 3, 5, 2, 1, 6])
        result, nDropped = _repairTimeJumps(metaDat, ["f.txt"], config)
        assert nDropped > 0
        assert np.all(np.diff(result.capture_time.values) >= 0)

    @pytest.mark.unit
    def test_nonadjacent_jumps_raise(self):
        config = types.SimpleNamespace(visssGen="visss")
        # backwards jumps at two non-adjacent positions
        metaDat = _makeMetaDat([0, 5, 1, 6, 2, 7])
        with pytest.raises(AssertionError):
            _repairTimeJumps(metaDat, ["f.txt"], config)

    @pytest.mark.unit
    def test_too_many_jumps_raise(self):
        config = types.SimpleNamespace(visssGen="visss")
        # 20 consecutive backwards steps -> nJumps >= 20
        metaDat = _makeMetaDat(list(range(21, 0, -1)))
        with pytest.raises(AssertionError, match="more than 20"):
            _repairTimeJumps(metaDat, ["f.txt"], config)

    @pytest.mark.unit
    def test_visss2_single_jump_not_yet_implemented(self):
        config = types.SimpleNamespace(visssGen="visss2")
        metaDat = _makeMetaDat([0, 5, 1, 6])
        with pytest.raises(RuntimeError, match="develop fix"):
            _repairTimeJumps(metaDat, ["f.txt"], config)

    @pytest.mark.unit
    def test_visss2_multiple_jumps_raise_not_implemented(self):
        config = types.SimpleNamespace(visssGen="visss2")
        metaDat = _makeMetaDat([0, 5, 1, 6, 2, 7])
        with pytest.raises(NotImplementedError):
            _repairTimeJumps(metaDat, ["f.txt"], config)

    @pytest.mark.unit
    def test_unknown_visssGen_raises(self):
        config = types.SimpleNamespace(visssGen="somethingElse")
        metaDat = _makeMetaDat([0, 5, 1, 6])
        with pytest.raises(RuntimeError, match="unknown VISSS generation"):
            _repairTimeJumps(metaDat, ["f.txt"], config)


class TestMeta(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield

    def testEvents(self):
        case = "20260110"
        dat = createEvent(
            case, "leader", self.config, skipExisting=False, writeNc=False
        )
        assert np.isclose(dat.cameraTemperature.mean(), 35.8633728)
        for var in [
            "blocking",
            "brightnessMean",
            "brightnessStd",
            "cameraTemperature",
            "capture_firsttime",
            "capture_lasttime",
            "capture_starttime",
            "configuration",
            "event",
            "filename",
            "gitBranch",
            "gitTag",
            "hostname",
            "ptpStatus",
            "serialnumber",
            "transferMaxBlockSize",
            "transferQueueCurrentBlockCount",
        ]:
            assert var in dat.data_vars

    def testMetaFrames(self):
        case = "20260110"
        dat = createMetaFrames(
            case, "leader", self.config, skipExisting=False, writeNc=False
        )

        assert np.isclose(dat.nMovingPixel.mean(), 61.05675239)
        for var in [
            "capture_id",
            "nMovingPixel",
            "nThread",
            "queue_size",
            "record_id",
            "record_time",
        ]:
            assert var in dat.data_vars
