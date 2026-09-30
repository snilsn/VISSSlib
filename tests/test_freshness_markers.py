"""Tests for the on-disk freshness-summary cache (files.FindFiles.markerPath,
tools.readLevelSummary/writeLevelSummary/_touchLevelMarker, and their use in
products.DataProduct.newestFileCreation/oldestFileCreation and
tools.checkForExisting's parentsSummary fast path).

These are network-free unit tests against a synthetic config (see
helpers.makeSyntheticConfig), since they only exercise the caching mechanics,
not real VISSS science code.
"""

import os

import numpy as np
import pytest
import xarray as xr
import VISSSlib
from VISSSlib import files, tools
from VISSSlib.products import DataProduct

from helpers import makeSyntheticConfig


@pytest.fixture
def config(tmp_path):
    # nThreads=None keeps output basenames simple (no trailing "_<thread>"
    # token to account for), matching most of the fixture filenames below.
    return makeSyntheticConfig(tmp_path, nThreads=None)


def _level1matchPath(config, case="20260101-120000"):
    """Build a realistic level1match output path for `case`, the same way
    matching.matchParticles derives it from a level0 input filename."""
    camera = config.leader
    basename = f"{config.computers[0]}_{config.visssGen}_{camera}_{case}"
    level0Fname = f"/dummy/{basename}.{config.movieExtension}"
    ff = files.Filenames(level0Fname, config)
    return ff, ff.fname["level1match"]


class TestMarkerPathConsistency:
    @pytest.mark.unit
    def test_findFiles_and_filenames_agree_on_marker_path(self, config):
        """A marker written from a Filenames instance (as tools.open2/
        to_netcdf2 do, starting from one specific output file) must resolve
        to the exact same path a FindFiles instance (as products.DataProduct
        does, starting from case="20260101") computes for that
        level+camera+day -- otherwise touch/read never actually meet."""
        ff, path = _level1matchPath(config)
        fn = files.FindFiles("20260101", "leader", config)

        assert ff.markerPath("level1match", "touch") == fn.markerPath(
            "level1match", "touch"
        )
        assert ff.markerPath("level1match", "done") == fn.markerPath(
            "level1match", "done"
        )

    @pytest.mark.unit
    def test_marker_path_scoped_to_day_not_subday_timestamp(self, config):
        """Two 10-minute files on the same day must share one marker."""
        ff1, _ = _level1matchPath(config, case="20260101-000000")
        ff2, _ = _level1matchPath(config, case="20260101-235900")
        assert ff1.markerPath("level1match", "touch") == ff2.markerPath(
            "level1match", "touch"
        )


class TestTouchAndSummary:
    @pytest.mark.unit
    def test_open2_write_touches_marker(self, config):
        ff, path = _level1matchPath(config)
        touchPath = ff.markerPath("level1match", "touch")
        assert not os.path.exists(touchPath)
        with tools.open2(f"{path}.nodata", config, "w") as f:
            f.write("no data")
        assert os.path.exists(touchPath)

    @pytest.mark.unit
    def test_to_netcdf2_write_touches_marker(self, config):
        ff, path = _level1matchPath(config)
        touchPath = ff.markerPath("level1match", "touch")
        dat = xr.Dataset({"x": 1})
        tools.to_netcdf2(dat, config, path)
        assert os.path.exists(touchPath)

    @pytest.mark.unit
    def test_write_invalidates_existing_done_marker(self, config):
        ff, path = _level1matchPath(config)
        donePath = ff.markerPath("level1match", "done")
        os.makedirs(os.path.dirname(donePath), exist_ok=True)
        with open(donePath, "w") as f:
            f.write('{"fence": 0, "n": 1, "oldest": 0, "newest": 0}')

        dat = xr.Dataset({"x": 1})
        tools.to_netcdf2(dat, config, path)

        assert not os.path.exists(donePath)

    @pytest.mark.unit
    def test_summary_round_trips_when_fence_matches(self, config):
        fn = files.FindFiles("20260101", "leader", config)
        fence = tools.getLevelTouchTime(fn, "level1match")
        assert fence == 0  # nothing written yet

        tools.writeLevelSummary(fn, "level1match", 3, 100.0, 200.0, fence, config)
        assert tools.readLevelSummary(fn, "level1match") == (3, 100.0, 200.0)

    @pytest.mark.unit
    def test_summary_miss_when_no_marker_written(self, config):
        fn = files.FindFiles("20260101", "leader", config)
        assert tools.readLevelSummary(fn, "level1match") is None


class TestConcurrentWriterRace:
    """Reproduces the race raised in review: worker A scans a level's files
    while worker B concurrently (re)writes one of them. A's summary must
    never get published once it's stale relative to B's write, since a
    same-file rewrite doesn't change the file count and so can't be caught
    by a naive count check -- only the fence can catch it."""

    @pytest.mark.unit
    def test_stale_scan_is_not_published_after_concurrent_write(self, config):
        fn = files.FindFiles("20260101", "leader", config)
        ff, path = _level1matchPath(config)

        # simulate the level already having one file, written earlier
        dat = xr.Dataset({"x": 1})
        tools.to_netcdf2(dat, config, path)

        # worker A begins its scan: captures the fence and computes a
        # summary based on what it currently sees
        fenceA = tools.getLevelTouchTime(fn, "level1match")
        staleSummary = (1, 111.0, 111.0)

        # worker B concurrently rewrites the same file (same count, new
        # mtime) while A's scan is still in flight
        tools.to_netcdf2(dat, config, path)

        # worker A finishes and tries to publish what it saw -- this must
        # be rejected, since fenceA is now stale
        tools.writeLevelSummary(fn, "level1match", *staleSummary, fenceA, config)

        cached = tools.readLevelSummary(fn, "level1match")
        assert cached is None or cached != staleSummary

    @pytest.mark.unit
    def test_scan_publishes_when_no_concurrent_write_happened(self, config):
        """Sanity check on the other side of the same test: with no
        concurrent write, a scan's summary does get published."""
        fn = files.FindFiles("20260101", "leader", config)
        ff, path = _level1matchPath(config)

        dat = xr.Dataset({"x": 1})
        tools.to_netcdf2(dat, config, path)

        fenceA = tools.getLevelTouchTime(fn, "level1match")
        summary = (1, 111.0, 111.0)
        tools.writeLevelSummary(fn, "level1match", *summary, fenceA, config)

        assert tools.readLevelSummary(fn, "level1match") == summary


class TestDataProductIntegration:
    @pytest.fixture
    def queue(self, tmp_path):
        return str(tmp_path / "fileQueue")

    @pytest.mark.unit
    def test_newestFileCreation_matches_real_scan_on_cache_miss(self, config, queue):
        ff, path = _level1matchPath(config)
        dat = xr.Dataset({"x": 1})
        tools.to_netcdf2(dat, config, path)
        realMtime = os.path.getmtime(path)

        p = DataProduct(
            "level1match", "20260101", config, queue, "leader", addRelatives=False
        )
        assert p.newestFileCreation == pytest.approx(realMtime)
        assert p.oldestFileCreation == pytest.approx(realMtime)

    @pytest.mark.unit
    def test_newestFileCreation_works_for_raw_passthrough_levels(self, config, queue):
        """level0/level0txt/level0jpg have no per-level entry in
        FindFiles.outpath (they share level0's directory via
        fnamesPattern overrides instead), so markerPath can't build a
        path for them. _freshnessSummary must fall back to a plain scan
        for these rather than raising a KeyError."""
        p = DataProduct(
            "level0txt", "20260101", config, queue, "leader", addRelatives=False
        )
        assert p.newestFileCreation == 0
        assert p.oldestFileCreation == 0

    @pytest.mark.unit
    def test_newestFileCreation_uses_cache_instead_of_rescanning(self, config, queue):
        """Proves the cache is actually consulted (not just correct by
        coincidence): after a first DataProduct caches a summary, silently
        changing the real file's mtime without going through
        open2/to_netcdf2 (so the touch marker doesn't move) must not be
        picked up by a second, independent DataProduct instance."""
        ff, path = _level1matchPath(config)
        dat = xr.Dataset({"x": 1})
        tools.to_netcdf2(dat, config, path)

        p1 = DataProduct(
            "level1match", "20260101", config, queue, "leader", addRelatives=False
        )
        cachedNewest = p1.newestFileCreation  # triggers scan + publish

        # bypass the marker hooks entirely, as e.g. a manual `touch` or an
        # out-of-band file copy would
        os.utime(path, (cachedNewest + 1000, cachedNewest + 1000))

        p2 = DataProduct(
            "level1match", "20260101", config, queue, "leader", addRelatives=False
        )
        assert p2.newestFileCreation == pytest.approx(cachedNewest)

    @pytest.mark.unit
    def test_touchOutputFile_reflects_in_newestFileCreation(self, config, queue):
        """Mirror image of test_newestFileCreation_uses_cache_instead_of_
        rescanning: tools.touchOutputFile is the sanctioned way to bump an
        existing output's mtime without going through a full open2/
        to_netcdf2 write, and it must invalidate the cache exactly like a
        real write does -- unlike a bare os.utime, which that other test
        proves gets silently ignored."""
        ff, path = _level1matchPath(config)
        dat = xr.Dataset({"x": 1})
        tools.to_netcdf2(dat, config, path)

        p1 = DataProduct(
            "level1match", "20260101", config, queue, "leader", addRelatives=False
        )
        p1.newestFileCreation  # triggers scan + publish

        tools.touchOutputFile(path, config)
        realNewest = os.path.getmtime(path)

        p2 = DataProduct(
            "level1match", "20260101", config, queue, "leader", addRelatives=False
        )
        assert p2.newestFileCreation == pytest.approx(realNewest)

    @pytest.mark.unit
    def test_newestFileCreation_reflects_new_write_through_hooks(self, config, queue):
        """The mirror image: a write that *does* go through to_netcdf2
        (and so bumps the touch marker) must invalidate the cache."""
        ff, path = _level1matchPath(config)
        dat = xr.Dataset({"x": 1})
        tools.to_netcdf2(dat, config, path)

        p1 = DataProduct(
            "level1match", "20260101", config, queue, "leader", addRelatives=False
        )
        firstNewest = p1.newestFileCreation

        tools.to_netcdf2(dat, config, path)
        secondMtime = os.path.getmtime(path)
        # Not asserting secondMtime > firstNewest here: Linux stamps inode
        # mtimes off a coarse clock (tick resolution, ~1-4ms), so two writes
        # this close together can land in the same tick and get an
        # identical mtime. Cache invalidation itself doesn't depend on the
        # mtime advancing -- every to_netcdf2 write unconditionally drops
        # the cached "done" summary (tools._touchLevelMarker) -- so what
        # matters below is that p2 picks up whatever the real mtime is now.

        p2 = DataProduct(
            "level1match", "20260101", config, queue, "leader", addRelatives=False
        )
        assert p2.newestFileCreation == pytest.approx(secondMtime)


class TestCheckForExistingUsesCache:
    """tools.checkForExisting's events/parents accept either plain file
    paths (unchanged legacy behavior) or (files.FindFiles, level) pairs
    -- passing the latter must never change the answer relative to the
    caller pre-globbing the same files itself, it should only change how
    that answer gets computed (cached "newest" vs. stat'ing every file)."""

    @pytest.mark.unit
    def test_group_form_matches_plain_list_form(self, config, tmp_path):
        fn = files.FindFiles("20260101", "leader", config)
        ff, path = _level1matchPath(config)
        tools.to_netcdf2(xr.Dataset({"x": 1}), config, path)
        parents = fn.listFilesExt("level1match")
        assert len(parents) == 1

        outFile = str(tmp_path / "out.nc")
        open(outFile, "w").close()

        withPaths = tools.checkForExisting(outFile, parents=parents)
        withGroup = tools.checkForExisting(outFile, parents=[(fn, "level1match")])
        assert withGroup == withPaths

    @pytest.mark.unit
    def test_uses_cached_newest_instead_of_rescanning(self, config, tmp_path):
        """Proves the fast path is actually taken (not just correct by
        coincidence): caches a deliberately wrong "newest" timestamp,
        validly fenced as a real scan would produce it, and checks that
        checkForExisting follows the cached value rather than the real
        file's mtime."""
        fn = files.FindFiles("20260101", "leader", config)
        ff, path = _level1matchPath(config)
        tools.to_netcdf2(xr.Dataset({"x": 1}), config, path)

        fence = tools.getLevelTouchTime(fn, "level1match")
        realNewest = os.path.getmtime(path)
        fakeNewest = realNewest + 10_000
        tools.writeLevelSummary(
            fn, "level1match", 1, realNewest, fakeNewest, fence, config
        )
        assert tools.readLevelSummary(fn, "level1match") == (1, realNewest, fakeNewest)

        # ffOut is newer than the real parent file, but older than the
        # (fake) cached "newest" -- only the cache-driven answer treats
        # it as stale
        outFile = str(tmp_path / "out.nc")
        open(outFile, "w").close()
        os.utime(outFile, (realNewest + 1, realNewest + 1))

        assert tools.checkForExisting(outFile, parents=[(fn, "level1match")]) is False
        assert (
            tools.checkForExisting(outFile, parents=fn.listFilesExt("level1match"))
            is True
        )

    @pytest.mark.unit
    def test_falls_back_when_any_group_uncached(self, config, tmp_path):
        """parents/events can list several (fn, level) groups (e.g.
        metaRotation's leader+follower level1detect parents); a group
        with no cached summary yet must still fall back to a real scan
        of just that group, not be silently skipped or poison the whole
        check."""
        fn1 = files.FindFiles("20260101", "leader", config)
        fn2 = files.FindFiles("20260101", "follower", config)
        ff, path = _level1matchPath(config)
        tools.to_netcdf2(xr.Dataset({"x": 1}), config, path)
        parents = fn1.listFilesExt("level1match")

        fence = tools.getLevelTouchTime(fn1, "level1match")
        tools.writeLevelSummary(fn1, "level1match", 1, 0, 0, fence, config)

        outFile = str(tmp_path / "out.nc")
        open(outFile, "w").close()

        withPaths = tools.checkForExisting(outFile, parents=parents)
        withGroups = tools.checkForExisting(
            outFile, parents=[(fn1, "level1match"), (fn2, "level1track")]
        )
        assert withGroups == withPaths

    @pytest.mark.unit
    def test_events_and_parents_groups_can_be_mixed_with_plain_paths(
        self, config, tmp_path
    ):
        fn = files.FindFiles("20260101", "leader", config)
        ff, path = _level1matchPath(config)
        tools.to_netcdf2(xr.Dataset({"x": 1}), config, path)

        extraFile = str(tmp_path / "extra.nc")
        open(extraFile, "w").close()

        outFile = str(tmp_path / "out.nc")
        open(outFile, "w").close()

        # must not raise, and must agree with the fully-plain-list answer
        mixed = tools.checkForExisting(
            outFile, parents=[(fn, "level1match"), extraFile]
        )
        plain = tools.checkForExisting(
            outFile, parents=fn.listFilesExt("level1match") + [extraFile]
        )
        assert mixed == plain
