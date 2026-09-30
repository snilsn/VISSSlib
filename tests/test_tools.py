import os

import numpy as np
import pytest
import VISSSlib

from helpers import downloadData, get_test_data_path


def test_readConfig():
    testRoot = get_test_data_path()
    settings = f"{testRoot}/test_0.6/testtmp_0.6.yaml"
    if not os.path.exists(settings):
        downloadData()
    config = VISSSlib.tools.readSettings(settings)
    assert config.visssGen == "visss"
    assert config.path.startswith("/")
    assert "$HOSTNAME" not in config.pathQuicklooks


def test_block_archive(tmp_path):
    """Test block archive operations"""
    archive_file = tmp_path / "test.block"

    # Create and write to archive
    with VISSSlib.tools.BlockImageArchive(archive_file, mode="w") as archive:
        img1 = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        archive.addnpy(f"image_1", img1)

    # Read from archive
    with VISSSlib.tools.BlockImageArchive(archive_file, mode="r") as archive:
        for i in range(10):
            img2 = archive.extractnpy(f"image_1")
            assert img2.shape == (10, 10)
            assert img2.dtype == np.uint8
            assert np.all(img1 == img2)


class TestDataFixesMerge:
    """DEFAULT_SETTINGS['dataFixes'] holds fixes meant to apply to every
    deployment (see tools.readSettings) regardless of what a specific
    yaml lists -- unlike every other setting, it must be unioned with the
    yaml's own value instead of being silently overridden by it.
    """

    @pytest.mark.unit
    def test_omitted_dataFixes_gets_the_defaults(self, tmp_path):
        from helpers import makeSyntheticConfig

        config = makeSyntheticConfig(tmp_path)
        assert set(config.dataFixes) == set(VISSSlib.tools.DEFAULT_SETTINGS["dataFixes"])

    @pytest.mark.unit
    def test_empty_dataFixes_still_gets_the_defaults(self, tmp_path):
        from helpers import makeSyntheticConfig

        config = makeSyntheticConfig(tmp_path, dataFixes=[])
        assert set(config.dataFixes) == set(VISSSlib.tools.DEFAULT_SETTINGS["dataFixes"])

    @pytest.mark.unit
    def test_yamls_own_fixes_are_added_not_replaced(self, tmp_path):
        from helpers import makeSyntheticConfig

        config = makeSyntheticConfig(tmp_path, dataFixes=["makeCaptureTimeEvenBothCameras"])
        assert set(config.dataFixes) == set(
            VISSSlib.tools.DEFAULT_SETTINGS["dataFixes"] + ["makeCaptureTimeEvenBothCameras"]
        )

    @pytest.mark.unit
    def test_duplicate_entry_is_not_repeated(self, tmp_path):
        from helpers import makeSyntheticConfig

        default = VISSSlib.tools.DEFAULT_SETTINGS["dataFixes"][0]
        config = makeSyntheticConfig(tmp_path, dataFixes=[default])
        assert config.dataFixes.count(default) == 1


class TestIsBadPeriod:
    """isBadPeriod's dailyLevels branch turns a date-only `end` into an
    end-of-day exclusive bound by adding one day, then (before the fix)
    compared it with `<=` -- so the day right after the configured `end`
    parsed to that same added-day midnight and was incorrectly flagged
    bad too. Guard against regressing that boundary.
    """

    @pytest.mark.unit
    def test_day_after_end_is_not_flagged(self, tmp_path):
        from helpers import makeSyntheticConfig

        config = makeSyntheticConfig(
            tmp_path,
            badData=[
                {
                    "start": "20260628",
                    "end": "20260629",
                    "reason": "test",
                    "products": ["metaRotation"],
                }
            ],
        )
        assert VISSSlib.tools.isBadPeriod("20260627", config, product="metaRotation")[0] is False
        assert VISSSlib.tools.isBadPeriod("20260628", config, product="metaRotation")[0] is True
        assert VISSSlib.tools.isBadPeriod("20260629", config, product="metaRotation")[0] is True
        assert VISSSlib.tools.isBadPeriod("20260630", config, product="metaRotation")[0] is False


class TestRunCommandInQueueTerminalArtifact:
    """runCommandInQueue used to treat a clean subprocess exit code alone as
    success, even if the command produced none of the three artifacts that
    actually mark a VISSSlib task as finished (the real output file, a
    .nodata sentinel, or its own .broken.txt) -- see
    [[project-level1detect-sync-fixes]]. That let a task silently vanish
    from the task queue (delete()'d as "done") with zero trace if it ever
    hit an edge case exiting 0 without writing anything recognizable.
    """

    @pytest.mark.unit
    def test_success_when_real_output_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fOut = str(tmp_path / "result.nc")
        cmd = f"touch {fOut}"
        assert VISSSlib.tools.runCommandInQueue((cmd, fOut)) is True
        assert os.path.isfile(fOut)
        assert not os.path.isfile(f"{fOut}.broken.txt")

    @pytest.mark.unit
    def test_success_when_nodata_sentinel_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fOut = str(tmp_path / "result.nc")
        cmd = f"touch {fOut}.nodata"
        assert VISSSlib.tools.runCommandInQueue((cmd, fOut)) is True
        assert not os.path.isfile(f"{fOut}.broken.txt")

    @pytest.mark.unit
    def test_success_when_command_writes_its_own_broken_txt(self, tmp_path, monkeypatch):
        # a command that already did its own failure bookkeeping (wrote its
        # own .broken.txt) but still exits 0 shouldn't get a second,
        # redundant outer .broken.txt piled on top
        monkeypatch.chdir(tmp_path)
        fOut = str(tmp_path / "result.nc")
        cmd = f"touch {fOut}.broken.txt"
        assert VISSSlib.tools.runCommandInQueue((cmd, fOut)) is True

    @pytest.mark.unit
    def test_failure_when_exit_code_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fOut = str(tmp_path / "result.nc")
        cmd = "exit 1"
        assert VISSSlib.tools.runCommandInQueue((cmd, fOut)) is False
        assert os.path.isfile(f"{fOut}.broken.txt")

    @pytest.mark.unit
    def test_failure_when_exit_zero_but_nothing_written(self, tmp_path, monkeypatch):
        # the bug this fix addresses: a clean exit with no recognized
        # terminal artifact must not be silently treated as success
        monkeypatch.chdir(tmp_path)
        fOut = str(tmp_path / "result.nc")
        cmd = "true"
        assert VISSSlib.tools.runCommandInQueue((cmd, fOut)) is False
        assert os.path.isfile(f"{fOut}.broken.txt")
        assert not os.path.isfile(fOut)
