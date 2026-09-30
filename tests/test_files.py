import pytest
import VISSSlib

from helpers import makeSyntheticConfig


class TestFindFilesDayArithmetic:
    """Unit tests for FindFiles.yesterday/tomorrow, pure date arithmetic
    that rebuilds a FindFiles for the adjacent day -- exercised here
    across month/year boundaries with a synthetic, network-free config.
    """

    @pytest.fixture
    def config(self, tmp_path):
        return makeSyntheticConfig(tmp_path)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "case, expectedYesterday, expectedTomorrow",
        [
            ("20260115", "20260114", "20260116"),
            ("20260101", "20251231", "20260102"),  # year boundary
            ("20260301", "20260228", "20260302"),  # non-leap month boundary
            ("20240301", "20240229", "20240302"),  # leap-year month boundary
        ],
    )
    def test_yesterday_tomorrow(
        self, config, case, expectedYesterday, expectedTomorrow
    ):
        fn = VISSSlib.files.FindFiles(case, "leader", config)
        assert fn.yesterday == expectedYesterday
        assert fn.tomorrow == expectedTomorrow
