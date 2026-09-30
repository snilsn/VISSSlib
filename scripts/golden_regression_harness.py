#!/usr/bin/env python
"""
Golden-output regression harness for the rotation/matching core shared by
matching.createMetaRotation, matching.matchParticles, and
matching.manualRotationEstimate (all three ultimately call
matching._matchSegments).

Purpose: let _matchSegments (and friends) be refactored for readability
without silently changing what any of its three callers compute. Record a
snapshot of real outputs for a curated set of cases *before* refactoring,
then verify a refactored version reproduces the same outputs within a
numeric tolerance.

None of this writes to production data -- every entry point is called with
writeNc=False, so only the in-memory return value is captured; the golden
snapshots themselves live under scripts/golden_outputs/ (git-ignored,
regenerate with `record` after checking out a commit you trust).

Usage
-----
    python scripts/golden_regression_harness.py record [--case NAME]
    python scripts/golden_regression_harness.py verify [--case NAME]
    python scripts/golden_regression_harness.py verify --fast   # skip slow cases

Cases are deliberately picked as small/fast files where the code path
allows it (see notes per case for why a given file was chosen); a few
(marked slow=True) are inherently expensive because the behavior they
cover only shows up with enough real particles, and shouldn't be run on
every iteration of a refactor -- use --fast for quick turnaround and run
the full set before finalizing/committing a refactor step.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

# Deliberately *not* sys.path-inserting src/ -- that bypasses the properly
# `pip install -e .`-installed package and makes VISSSlib.__version__
# resolve dynamically off the live (possibly dirty) git state via
# setuptools_scm instead of the installed "1.2", which then breaks every
# {version}-templated production path (visss_1.2/... vs
# visss_<git-describe>/...). Rely on the editable install like every other
# script in this repo.
import VISSSlib
from loguru import logger

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_outputs"

CASES = [
    dict(
        name="matchParticles_captureid_drop",
        entrypoint="matchParticles",
        settings="nyaalesund_v3",
        case="20211204-125000",
        camera="leader",
        slow=False,
        notes=(
            "Pre-PTP file with a silently-dropped-frame capture_id "
            "ambiguity (idDiff cleanly splits 7/8 at one timestamp); "
            "exercises fixes.detectCaptureIdDropTimes + the timeBlocks "
            "split in matchParticles. Known good: ~18571 matched particles."
        ),
    ),
    dict(
        name="matchParticles_follower_reset",
        entrypoint="matchParticles",
        settings="nyaalesund_v5",
        case="20241103-090000",
        camera="leader",
        slow=False,
        expect_exception="RuntimeError",
        notes=(
            "Follower capture_id goes backward within the window -- "
            "exercises the 'follower camera reset detected' continue "
            "branch in _matchSegments (a real broken level1match example)."
        ),
    ),
    dict(
        name="matchParticles_matchscore_failure",
        entrypoint="matchParticles",
        settings="hyytiala2_v4",
        case="20250418-052000",
        camera="leader",
        slow=True,
        expect_exception="RuntimeError",
        notes=(
            "Genuine 'median matchScore below minMatchScore' failure "
            "(not fixable by the capture_id-drop logic) -- exercises the "
            "quality-gate RuntimeError at the end of matchParticles. "
            "Smallest known real example that still reproduces it "
            "(~2.5MB level1detect file); expect a few minutes."
        ),
    ),
    dict(
        name="matchParticles_ptp_active",
        entrypoint="matchParticles",
        settings="nyaalesund_v5",
        case="20250204-124000",
        camera="leader",
        slow=False,
        notes=(
            "PTP-active file (post 2025-06-27 rollout) with confirmed "
            "real data on both cameras that day -- exercises the "
            "ptpTime=True branch in _matchSegments, which skips "
            "capture_id-offset estimation entirely and uses synchronized "
            "timestamps directly. Same day used for createMetaRotation_ptp_active. "
            "(20251002, tried first, turned out to be a genuine full-day "
            "follower gap -- avoid dates already known to be all-nodata.)"
        ),
    ),
    dict(
        name="createMetaRotation_plain",
        entrypoint="createMetaRotation",
        settings="nyaalesund_v4",
        case="20221213",
        camera="leader",
        slow=True,
        notes=(
            "Ordinary non-PTP day with a confirmed-working level1match "
            "chain (verified earlier in this session) -- baseline happy "
            "path for createMetaRotation's daily loop and the "
            "config.rotate carry-forward logic. Inherently slow: every "
            "10-min file in the day gets its own rotationOnly "
            "matchParticles pass (~10s of minutes)."
        ),
    ),
    dict(
        name="createMetaRotation_ptp_active",
        entrypoint="createMetaRotation",
        settings="nyaalesund_v5",
        case="20250204",
        camera="leader",
        slow=True,
        notes="Post-PTP day, mirrors createMetaRotation_plain but with ptpTime=True. Same slow-for-real-reasons caveat.",
    ),
    dict(
        name="manualRotationEstimate_correctly_rejects_blocked",
        entrypoint="manualRotationEstimate",
        settings="nyaalesund_v3",
        case="20220122-005000",
        camera="leader",
        slow=True,
        notes=(
            "Directly exercises manualRotationEstimate's from-scratch "
            "fit (blur>100, wide-open prior, 4-iteration refine) on a "
            "file from a day whose carried-forward rotation default was "
            "stuck (matching.createMetaRotation's tryFixFromScratch calls "
            "exactly this). This particular day turned out to be a "
            "genuinely blocked-camera stretch, not a recoverable frame "
            "shift -- the fit correctly fails its own validation and "
            "returns None (pinned result: {'20220122-005000': None}), "
            "which is the behavior worth protecting here: a refactor must "
            "not start silently accepting a bad fit for an unrecoverable "
            "file. Does NOT cover the success path (no confirmed-successful "
            "from-scratch fit has been found yet this session -- TODO: add "
            "one from a real frame-shift cluster once found). ~1-2 min."
        ),
    ),
]


def _resolve_config(settings_name):
    logger.disable("VISSSlib")
    config = VISSSlib.tools.readSettings(
        f"/projekt1/ag_maahn/VISSS_config/{settings_name}.yaml"
    )
    logger.enable("VISSSlib")
    return config


def _resolve_level1detect_file(config, case, camera):
    # FindFiles wants a day, not a "YYYYMMDD-HHMMSS" file-level case --
    # resolve the day's files and pick the one matching the full case.
    day = case.split("-")[0]
    logger.disable("VISSSlib")
    fn = VISSSlib.files.FindFiles(day, getattr(config, camera), config)
    files_ = [
        f for f in fn.listFilesExt("level1detect") if f.endswith(".nc") and case in f
    ]
    logger.enable("VISSSlib")
    if not files_:
        raise FileNotFoundError(f"no level1detect file for {case} ({camera})")
    return files_[0]


def _clean_attrs(ds):
    """Strip attrs/encoding that legitimately differ run to run (e.g.
    creation timestamps) so they don't cause spurious diffs."""
    ds = ds.copy()
    for key in ("created", "history", "date_created"):
        ds.attrs.pop(key, None)
    for var in ds.variables:
        ds[var].encoding = {}
    return ds


def _series_to_dict(s):
    if s is None:
        return None
    try:
        return {k: float(v) for k, v in dict(s).items()}
    except TypeError:
        return None


def run_case(case_def):
    """Run one case, return (dataset_or_None, meta_dict)."""
    entrypoint = case_def["entrypoint"]
    config = _resolve_config(case_def["settings"])
    meta = {"exception": None, "exception_message": None}

    logger.disable("VISSSlib")
    try:
        if entrypoint == "matchParticles":
            fname = _resolve_level1detect_file(
                config, case_def["case"], case_def["camera"]
            )
            (
                fname1Match,
                matchedDats,
                rotate_final,
                rotate_err_final,
                nLeader,
                nFollower,
                nParticles,
                errors,
            ) = VISSSlib.matching.matchParticles(
                fname, config, writeNc=False, skipExisting=False
            )
            meta.update(
                nLeader=nLeader,
                nFollower=nFollower,
                nParticles=nParticles,
                rotate_final=_series_to_dict(rotate_final),
                rotate_err_final=_series_to_dict(rotate_err_final),
            )
            ds = matchedDats if isinstance(matchedDats, xr.Dataset) else None

        elif entrypoint == "createMetaRotation":
            metaRotation, fnameMetaRotation = VISSSlib.matching.createMetaRotation(
                case_def["case"],
                config,
                writeNc=False,
                skipExisting=False,
                doPlots=False,
            )
            ds = metaRotation if isinstance(metaRotation, xr.Dataset) else None

        elif entrypoint == "manualRotationEstimate":
            results = VISSSlib.matching.manualRotationEstimate(
                case_def["case"], config, returnResultOnly=True
            )
            fixed = results.get(case_def["case"])
            meta["result"] = fixed
            ds = None

        else:
            raise ValueError(f"unknown entrypoint {entrypoint}")

    except Exception as e:  # noqa: BLE001 -- deliberately broad, this is a harness
        meta["exception"] = type(e).__name__
        meta["exception_message"] = str(e)
        meta["traceback"] = traceback.format_exc()
        ds = None
    finally:
        logger.enable("VISSSlib")

    return ds, meta


def record(cases):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for case_def in cases:
        name = case_def["name"]
        print(f"=== recording {name} ===", flush=True)
        ds, meta = run_case(case_def)

        if case_def.get("expect_exception"):
            if meta["exception"] != case_def["expect_exception"]:
                print(
                    f"  WARNING: expected {case_def['expect_exception']}, "
                    f"got {meta['exception']}: {meta['exception_message']}"
                )
            else:
                print(f"  got expected {meta['exception']}: {meta['exception_message']}")
        elif meta["exception"] is not None:
            print(f"  WARNING: unexpected exception: {meta['exception_message']}")
            print(meta.get("traceback", ""))

        meta_path = GOLDEN_DIR / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump({k: v for k, v in meta.items() if k != "traceback"}, f, indent=2, default=str)

        nc_path = GOLDEN_DIR / f"{name}.nc"
        if ds is not None:
            _clean_attrs(ds).to_netcdf(nc_path)
            print(f"  saved {nc_path} ({dict(ds.sizes)})")
        elif nc_path.exists():
            nc_path.unlink()
        print(f"  saved {meta_path}")


def _compare_meta(name, saved, current):
    ok = True
    for key in set(saved) | set(current):
        if key in ("exception_message", "traceback"):
            continue  # message text isn't load-bearing, only the type is
        sv, cv = saved.get(key), current.get(key)
        if isinstance(sv, dict) and isinstance(cv, dict):
            for k in set(sv) | set(cv):
                if not np.isclose(sv.get(k, np.nan), cv.get(k, np.nan), rtol=1e-4, equal_nan=True):
                    print(f"  MISMATCH {key}.{k}: saved={sv.get(k)} current={cv.get(k)}")
                    ok = False
        elif isinstance(sv, (int, float)) and isinstance(cv, (int, float)):
            if not np.isclose(sv, cv, rtol=1e-4, equal_nan=True):
                print(f"  MISMATCH {key}: saved={sv} current={cv}")
                ok = False
        elif sv != cv:
            print(f"  MISMATCH {key}: saved={sv} current={cv}")
            ok = False
    return ok


def verify(cases):
    failed = []
    for case_def in cases:
        name = case_def["name"]
        print(f"=== verifying {name} ===", flush=True)
        meta_path = GOLDEN_DIR / f"{name}.json"
        nc_path = GOLDEN_DIR / f"{name}.nc"
        if not meta_path.exists():
            print(f"  SKIP: no golden snapshot at {meta_path} (run `record` first)")
            failed.append(name)
            continue

        with open(meta_path) as f:
            saved_meta = json.load(f)

        ds, meta = run_case(case_def)
        ok = _compare_meta(name, saved_meta, meta)

        if nc_path.exists():
            if ds is None:
                print("  MISMATCH: golden has a dataset, current run returned None")
                ok = False
            else:
                saved_ds = xr.open_dataset(nc_path)
                try:
                    xr.testing.assert_allclose(
                        _clean_attrs(saved_ds), _clean_attrs(ds), rtol=1e-4
                    )
                except AssertionError as e:
                    print(f"  MISMATCH in dataset contents:\n{e}")
                    ok = False
                saved_ds.close()
        elif ds is not None:
            print("  MISMATCH: golden has no dataset, current run returned one")
            ok = False

        print("  OK" if ok else "  FAILED")
        if not ok:
            failed.append(name)

    print()
    if failed:
        print(f"{len(failed)}/{len(cases)} case(s) FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"all {len(cases)} case(s) passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "verify"])
    parser.add_argument("--case", help="run only the named case")
    parser.add_argument("--fast", action="store_true", help="skip cases marked slow=True")
    args = parser.parse_args()

    cases = CASES
    if args.case:
        cases = [c for c in cases if c["name"] == args.case]
        if not cases:
            sys.exit(f"no such case: {args.case}")
    if args.fast:
        cases = [c for c in cases if not c.get("slow")]

    if args.mode == "record":
        record(cases)
    else:
        verify(cases)


if __name__ == "__main__":
    main()
