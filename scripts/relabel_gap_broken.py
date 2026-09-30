#!/usr/bin/env python
"""
Relabel metaEvents/level2{detect,match,track} ".nodata" sentinels that were
actually a confirmed raw-data gap (zero level0 files for that case/camera,
not "no precipitation") as ".broken.txt" instead, using the same fixed
logic now in metadata.createEvent / distributions._createLevel2.

Before this fix, both code paths wrote ".nodata" whenever there were no
level0 files at all for a case/camera. That is indistinguishable from a
genuine no-precipitation day (which is what ".nodata" is supposed to mean
everywhere else), even though metaEvents/level2 have no real "quiet
weather" concept of their own -- they only see raw file listings, so zero
of those is always a raw-data problem (instrument offline, transfer
broken, ...).

For metaEvents this self-heals: metadata.createEvent's own skip-check only
looks at whether the real .nc exists (not the .nodata sentinel), so it
always reprocesses a .nodata day regardless of --skip-existing, and the
now-fixed code will relabel it correctly and clean up the old sentinel via
open2's own cleanup. So this script just re-invokes metadata.createEvent
for every candidate day/camera it finds -- no direct file surgery needed.

For level2{detect,match,track}, distributions._createLevel2's own
skipExisting check explicitly treats an existing ".nodata" sentinel as
"already done, skip" -- so the stale sentinel has to be removed first (via
tools.tryRemovingFile + tools._touchLevelMarker, the same primitives
scripts/relabel_missing_movies_broken.py already uses for the analogous
level1detect problem) before re-running the level2 command regenerates it
under the fixed logic.

Every candidate is re-verified against live level0 file listings right
before acting -- this is not a blind text-match relabel.

Usage
-----
    # dry run: just list what would happen
    python scripts/relabel_gap_broken.py settings.yaml

    # only check specific levels / a date range
    python scripts/relabel_gap_broken.py settings.yaml --levels metaEvents level2track --case 20260101-20261231

    # actually fix it
    python scripts/relabel_gap_broken.py settings.yaml --apply
"""

import argparse
import sys
from pathlib import Path

from VISSSlib import __version__, distributions, files, metadata, tools

LEVEL2_MESSAGE_PREFIX = "no level 0 data for "
LEVEL2_SUBLEVELS = {
    "level2detect": "detect",
    "level2match": "match",
    "level2track": "track",
}
VERSION_SHORT = ".".join(__version__.split(".")[:2])


def _levelRoot(config, level):
    # mirrors files.FindFiles' own outpath resolution (pathOut.format with
    # site/level/version), without needing a concrete case/camera
    return Path(
        config["pathOut"].format(site=config.site, level=level, version=VERSION_SHORT)
    )


def find_metaevents_candidates(config):
    root = _levelRoot(config, "metaEvents")
    out = []
    for fname in sorted(root.rglob("metaEvents_*.nc.nodata")):
        try:
            fn = files.FilenamesFromLevel(str(fname), config)
            case = fn.case.split("-")[0]
            fL = files.FindFiles(case, fn.camera, config)
            isGap = len(fL.listFilesExt("level0txt")) == 0
        except Exception as e:
            # pre-existing unrelated filename anomalies (e.g. a handful of
            # 2023/2024 hyytiala2_v5 metaEvents files with a mismatched
            # camera/serial combination) shouldn't abort the whole scan
            print(f"skipping {fname}, cannot parse/check: {e}", file=sys.stderr)
            continue
        if isGap:
            out.append((fname, case, fn.camera))
    return out


def find_level2_candidates(config, level):
    root = _levelRoot(config, level)
    out = []
    for fname in sorted(root.rglob(f"{level}_*.nc.nodata")):
        try:
            content = fname.read_text()
        except OSError as e:
            print(f"cannot read {fname}: {e}", file=sys.stderr)
            continue
        if not content.startswith(LEVEL2_MESSAGE_PREFIX):
            continue
        try:
            fn = files.FilenamesFromLevel(str(fname), config)
            case = fn.case.split("-")[0]
            fL = files.FindFiles(case, fn.camera, config)
            isGap = len(fL.listFilesExt("level0txt")) == 0
        except Exception as e:
            print(f"skipping {fname}, cannot parse/check: {e}", file=sys.stderr)
            continue
        if isGap:
            out.append((fname, case, fn.camera))
    return out


def fix_metaevents(case, camera, config):
    metadata.createEvent(case, camera, config, skipExisting=True)


def fix_level2(fname, case, level, config):
    tools.tryRemovingFile(str(fname))
    tools._touchLevelMarker(str(fname)[: -len(".nodata")], config)
    sublevel = LEVEL2_SUBLEVELS[level]
    fn = getattr(distributions, f"createLevel2{sublevel}")
    if level == "level2detect":
        fn(case, files.FilenamesFromLevel(str(fname), config).camera, config)
    else:
        fn(case, config)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("settings", help="Path to the VISSSlib settings YAML")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["metaEvents", "level2detect", "level2match", "level2track"],
        choices=["metaEvents", "level2detect", "level2match", "level2track"],
    )
    parser.add_argument(
        "--case",
        default=None,
        help="restrict to a case/date range (YYYYMMDD, YYYYMMDD-YYYYMMDD); "
        "default: check every candidate found on disk",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually relabel (default: dry run, only lists candidates)",
    )
    args = parser.parse_args()

    config = tools.readSettings(args.settings)
    caseSet = None
    if args.case is not None:
        caseSet = set(tools.getCaseRange(args.case, config))

    for level in args.levels:
        if level == "metaEvents":
            candidates = find_metaevents_candidates(config)
        else:
            candidates = find_level2_candidates(config, level)

        if caseSet is not None:
            candidates = [c for c in candidates if c[1] in caseSet]

        print(f"{level}: {len(candidates)} confirmed-gap candidate(s)", file=sys.stderr)
        for fname, case, camera in candidates:
            print(f"{level} {case} {camera} {fname}")

        if not args.apply:
            continue

        for fname, case, camera in candidates:
            try:
                if level == "metaEvents":
                    fix_metaevents(case, camera, config)
                else:
                    fix_level2(fname, case, level, config)
            except Exception as e:
                print(f"FAILED to fix {fname}: {e}", file=sys.stderr)

    if not args.apply:
        print("\ndry run -- pass --apply to relabel", file=sys.stderr)


if __name__ == "__main__":
    main()
