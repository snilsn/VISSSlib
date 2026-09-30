import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pytest
import VISSSlib
from VISSSlib.products import *

from helpers import makeSyntheticConfig, readTestSettings


class TestProducts:
    @pytest.mark.parametrize(
        "config_subpath, case",
        [
            ("test_0.6/testtmp_0.6.yaml", "20260110"),
            #            ("test_0.4/testtmp_0.4.yaml", "20260111"),
        ],
    )
    def test_processAll(self, config_subpath, case):
        # Read the config for this test case
        config = readTestSettings(config_subpath)
        # Ensure tmpPath is clean
        if os.path.exists(config.tmpPath):
            shutil.rmtree(config.tmpPath)
        try:
            processAll(
                case,
                config,
                ignoreErrors=False,
                nJobs=2,
                fileQueue=config.fileQueue,
                # No real SLURM allocation to free promptly here -- skip
                # most of worker1's idle-exit wait between the DAG's ~9
                # stages (default 60s each) to keep this test fast.
                maxIdleSeconds=2,
            )
        finally:
            # Clean up tmpPath after test
            if os.path.exists(config.tmpPath):
                shutil.rmtree(config.tmpPath)


class TestDataProductDAG:
    """Unit tests for DataProduct's dependency-graph wiring (parentNames,
    the processL1match guard, isComplete). These use a fully synthetic,
    network-free config (see helpers.makeSyntheticConfig) rather than the
    downloaded sample dataset, since they only exercise control flow, not
    the actual science code.
    """

    @pytest.fixture
    def config(self, tmp_path):
        return makeSyntheticConfig(tmp_path)

    @pytest.fixture
    def queue(self, tmp_path):
        return str(tmp_path / "fileQueue")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "level, camera, expectedParents",
        [
            ("level0", "leader", []),
            ("level0txt", "leader", []),
            ("metaEvents", "leader", ["leader_level0txt"]),
            ("metaFrames", "follower", ["follower_level0txt"]),
            ("level1detect", "leader", []),
            (
                "metaRotation",
                "leader",
                [
                    "leader_level1detect",
                    "follower_level1detect",
                    "leader_metaEvents",
                    "follower_metaEvents",
                ],
            ),
            ("level1match", "leader", ["leader_metaRotation"]),
            ("level1track", "leader", ["leader_level1match"]),
            (
                "level2detect",
                "follower",
                ["follower_level1detect", "follower_metaEvents"],
            ),
            (
                "level2match",
                "leader",
                ["leader_level1match", "leader_metaEvents", "follower_metaEvents"],
            ),
            (
                "level2track",
                "leader",
                [
                    "leader_level1track",
                    "leader_level2match",
                    "leader_metaEvents",
                    "follower_metaEvents",
                ],
            ),
            (
                "level3combinedRiming",
                "leader",
                ["leader_level2track", "leader_metaEvents", "follower_metaEvents"],
            ),
        ],
    )
    def test_parentNames(self, config, queue, level, camera, expectedParents):
        p = DataProduct(level, "20260101", config, queue, camera, addRelatives=False)
        assert p.parentNames == expectedParents

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "processL1match, processL2detect, processRetrieval, expectedExtra",
        [
            (
                True,
                True,
                True,
                [
                    "leader_level2track",
                    "leader_level2match",
                    "leader_level2detect",
                    "follower_level2detect",
                    "leader_level3combinedRiming",
                ],
            ),
            (False, False, False, []),
            (True, False, False, ["leader_level2track", "leader_level2match"]),
            (
                # combinedRiming depends on level2track, so a stray
                # processRetrieval: true must not be able to pull in
                # level2track (and thus fail) when the matching branch
                # itself is disabled.
                False,
                True,
                True,
                ["leader_level2detect", "follower_level2detect"],
            ),
        ],
    )
    def test_allDone_parentNames_respects_toggles(
        self,
        config,
        queue,
        processL1match,
        processL2detect,
        processRetrieval,
        expectedExtra,
    ):
        config.level1match.processL1match = processL1match
        config.level2.processL2detect = processL2detect
        config.level3.combinedRiming.processRetrieval = processRetrieval
        p = DataProduct("allDone", "20260101", config, queue, "leader", addRelatives=False)
        assert p.parentNames == [
            "leader_metaEvents",
            "follower_metaEvents",
        ] + expectedExtra

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "level",
        ["metaRotation", "level1match", "level1track", "level2match", "level2track"],
    )
    def test_processL1match_false_blocks_matching_levels(self, config, queue, level):
        config.level1match.processL1match = False
        with pytest.raises(ValueError, match="processL1match"):
            DataProduct(level, "20260101", config, queue, "leader", addRelatives=False)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "level, camera",
        [
            ("level1detect", "leader"),
            ("level2detect", "leader"),
            ("metaEvents", "leader"),
            ("allDone", "leader"),
        ],
    )
    def test_processL1match_false_does_not_block_other_levels(
        self, config, queue, level, camera
    ):
        config.level1match.processL1match = False
        # should not raise
        DataProduct(level, "20260101", config, queue, camera, addRelatives=False)

    @pytest.mark.unit
    def test_isComplete_false_when_no_files_exist(self, config, queue):
        p = DataProduct(
            "metaEvents", "20260101", config, queue, "leader", addRelatives=False
        )
        assert p.isComplete is False

    @pytest.mark.unit
    def test_isComplete_true_once_output_file_exists(self, config, queue):
        p = DataProduct(
            "metaEvents", "20260101", config, queue, "leader", addRelatives=False
        )
        outFile = p.fn.fnamesDaily["metaEvents"]
        os.makedirs(os.path.dirname(outFile), exist_ok=True)
        open(outFile, "w").close()
        assert p.isComplete is True

    @pytest.mark.unit
    def test_camera_must_be_leader_or_follower(self, config, queue):
        with pytest.raises(ValueError, match="camera"):
            DataProduct(
                "level1detect", "20260101", config, queue, "sideways", addRelatives=False
            )

    @pytest.mark.unit
    def test_generateCommands_metaFrames_no_level0_files(self, config, queue):
        # metaFrames is a files.fileLevel (one output per level0 input, not
        # one per day) but is invoked via a daily/looping CLI call, so it
        # has no files.fnamesDaily entry. With zero level0 files nMissing
        # is trivially 0, so this should just skip, not raise KeyError.
        p = DataProduct(
            "metaFrames", "20260101", config, queue, "leader", addRelatives=False
        )
        assert p.generateCommands(skipExisting=True) == []

    @pytest.mark.unit
    def test_generateCommands_metaFrames_with_pending_level0_file(self, config, queue):
        p = DataProduct(
            "metaFrames", "20260101", config, queue, "leader", addRelatives=False
        )
        l0dir = os.path.dirname(p.fn.fnamesPattern.level0txt)
        os.makedirs(l0dir, exist_ok=True)
        open(
            os.path.join(l0dir, "testcomputer_visss_leader_test_20260101-000000_0.txt"),
            "w",
        ).close()

        # re-create so nL0/nMissing pick up the new file (cached_property)
        p = DataProduct(
            "metaFrames", "20260101", config, queue, "leader", addRelatives=False
        )
        commands = p.generateCommands(skipExisting=True)
        assert len(commands) == 1
        command, outFile = commands[0]
        assert "metadata.createMetaFrames" in command
        assert "--camera=leader" in command
        assert outFile.endswith(".nc")

    @pytest.mark.unit
    def test_generateAllCommands_survives_stale_day_level_false_positive(
        self, config, queue, monkeypatch
    ):
        """
        Regression test for a real bug found on hyytiala2_v3: iterative
        reprocessing of individual level1match files (across several
        self-heal fix rounds) left level2track permanently stuck for 68
        days, because generateAllCommands refused to generate its own
        command whenever _parentsUpToDateWithGrandparents was False --
        without checking whether that flag was actually a false positive.

        _upToDateWithParentsDict deliberately compares a parent's overall
        NEWEST file mtime against this product's overall OLDEST file
        mtime (day-wide, not matched per-file pairs -- see its own
        docstring for why that's the right comparison for THIS product's
        own up-to-date-ness). But when used to decide whether a
        *grandchild* may even attempt to generate a command, that coarse
        day-wide comparison can flag a day as stale even though every
        individual parent/self file pair is already fully consistent:
        here, level1match/level1track pair B is reprocessed (both files
        touched) strictly after pair A's original run, so the day's
        overall newest level1match mtime (from B) is newer than the
        day's overall oldest level1track mtime (from A), even though
        every pair individually has its level1track file newer than its
        own level1match file.
        """
        case = "20260101"
        camera = "leader"
        # This test is purely about the day-level false-positive heuristic
        # below, not about tools.REPROCESS_AFTER's reprocessing breakpoint
        # (which compares against real wall-clock time, while every mtime
        # here is a tiny literal offset for relative-ordering purposes only)
        # -- neutralize it so a future breakpoint entry can't turn every
        # file below "stale" for an unrelated reason.
        monkeypatch.setattr(VISSSlib.tools, "REPROCESS_AFTER", {})

        def touch(dp, level, suffix, mtime):
            # FilenamesFromLevel strictly parses this as 8 "_"-separated
            # fields (level, Vversion, site, computer, visssGen, camera
            # [itself 2 fields, e.g. "leader_test"], timestamp) -- has to
            # match the real naming convention exactly, a loose glob-match
            # isn't enough once generateCommands parses the file back.
            d = dp.fn.outpath[level]
            os.makedirs(d, exist_ok=True)
            f = os.path.join(
                d,
                f"{level}_V{dp.fn.versionShort}_{config.site}_{dp.fn.computer}_"
                f"{config.visssGen}_{dp.fn.camera}_{dp.fn.case}-{suffix}.nc",
            )
            open(f, "w").close()
            os.utime(f, (mtime, mtime))
            return f

        def touchDaily(dp, level, mtime):
            f = dp.fn.fnamesDaily[level]
            os.makedirs(os.path.dirname(f), exist_ok=True)
            open(f, "w").close()
            os.utime(f, (mtime, mtime))
            return f

        # Two level0txt windows per camera, with matching level1detect
        # output for each -- level1match/level1track's own isComplete is
        # gated by the LEADER's level0txt count (files.FindFiles.nL0), so
        # this has to line up with the two level1match/level1track window
        # files created below, or isComplete would be False for an
        # unrelated reason (file-count mismatch) and mask the thing this
        # test actually checks.
        for cam in ("leader", "follower"):
            l0 = DataProduct("level0txt", case, config, queue, cam, addRelatives=False)
            l0dir = os.path.dirname(l0.fn.fnamesPattern.level0txt)
            os.makedirs(l0dir, exist_ok=True)
            detect = DataProduct(
                "level1detect", case, config, queue, cam, addRelatives=False
            )
            for suffix in ("000000", "001000"):
                l0file = os.path.join(
                    l0dir, f"testcomputer_visss_{cam}_test_{case}-{suffix}_0.txt"
                )
                open(l0file, "w").close()
                os.utime(l0file, (1, 1))
                touch(detect, "level1detect", suffix, 5)

        rotation = DataProduct(
            "metaRotation", case, config, queue, "leader", addRelatives=False
        )
        touchDaily(rotation, "metaRotation", 10)

        match = DataProduct(
            "level1match", case, config, queue, camera, addRelatives=False
        )
        track = DataProduct(
            "level1track", case, config, queue, camera, addRelatives=False
        )
        # pair A: original batch, fully consistent (track newer than match)
        touch(match, "level1match", "000000", 100)
        touch(track, "level1track", "000000", 200)
        # pair B: reprocessed later, ALSO fully consistent per-file
        touch(match, "level1match", "001000", 500)
        touch(track, "level1track", "001000", 600)

        leaderEvents = DataProduct(
            "metaEvents", case, config, queue, "leader", addRelatives=False
        )
        followerEvents = DataProduct(
            "metaEvents", case, config, queue, "follower", addRelatives=False
        )
        touchDaily(leaderEvents, "metaEvents", 8)
        touchDaily(followerEvents, "metaEvents", 8)

        # level2track also depends on level2match (it reuses level2match's
        # own zResidualTooWide flag rather than recomputing one from
        # level1track). Give it a comfortably up-to-date mtime -- this test
        # is about the level1track false-positive specifically, not about
        # level2match's own staleness.
        match2 = DataProduct(
            "level2match", case, config, queue, "leader", addRelatives=False
        )
        touchDaily(match2, "level2match", 700)

        level2track = DataProduct(
            "level2track", case, config, queue, camera, addRelatives=True
        )
        # level2track's own output predates everything -- genuinely stale,
        # it really should get a command generated.
        touchDaily(level2track, "level2track", 50)

        # Sanity check: the coarse day-level heuristic IS a false positive
        # here (level1track has nothing real pending per-file) ...
        level1track_dp = level2track.parents["leader_level1track"]
        assert not level1track_dp._upToDateWithParents
        assert (
            len(
                level1track_dp.generateAllCommands(
                    skipExisting=True, withParents=False
                )
            )
            == 0
        )

        # ... so level2track's own genuinely-needed command must still be
        # generated, not swallowed by "grandparents older".
        commands = level2track.generateAllCommands(
            skipExisting=True, withParents=False
        )
        assert len(commands) == 1
        assert "distributions.createLevel2track" in commands[0][0]
