#!/usr/bin/env python
"""
Reprocess one or more VISSSlib levels, one level at a time, via the
task-queue/SLURM-worker mechanism in products.py.

For each level (in the order given): submit commands for every requested
camera, make sure enough SLURM workers are running to drain the queue,
then wait for it to fully drain before moving on to the next level. This
"consecutive levels" ordering matters because a level's own staleness
check (isComplete/_upToDateWithParents in products.py) looks at its
parents' file mtimes -- reprocessing level N and level N+1 at the same
time can make N+1's work list computed before N is actually finished.

SLURM workers auto-exit once a queue is fully drained and idle
(tools.worker1: all worker slots idle -> "do not restart" -> break), so
this script actively keeps checking (every --poll-seconds, for as long as
a level still has pending/in-flight work) that WORKERS worker jobs are
running for this queue, and relaunches any that are missing -- whether
because a level's queue emptied and its workers exited, or because a
worker crashed/got preempted mid-run.

After the first forward pass through --levels, this script sweeps forward
through them again (up to --max-passes times) and resubmits anything that
now has pending work. This matters because a concurrent process outside
this script's own view -- most notably a separate serial metaRotation
backfill running for the same config -- can complete parent data for a
level this run already declared drained and moved past; a single
one-shot-per-level pass would silently never notice or resubmit that
newly-unlocked work. The sweep ends early once a full pass resubmits
nothing anywhere.

Usage
-----
    # normal incremental run: only what's missing/stale, canonical level order
    python scripts/reprocess_levels.py \\
        --settings /projekt1/ag_maahn/VISSS_config/hyytiala2_v3.yaml \\
        --queue /projekt6/ag_maahn/visss_task_queue_1.2_EXPRESS/ \\
        --levels all

    # force-reprocess ALL level1track files, then catch up level2 normally
    python scripts/reprocess_levels.py \\
        --settings hyytiala2_v3.yaml --queue .../visss_task_queue_1.2_EXPRESS/ \\
        --levels level1track --skip-existing false
    python scripts/reprocess_levels.py \\
        --settings hyytiala2_v3.yaml --queue .../visss_task_queue_1.2_EXPRESS/ \\
        --levels level2detect level2match level2track --skip-existing true

    # both of the above in one run, one flag each is not possible since
    # skip-existing is one setting for the whole invocation -- just call
    # the script twice as shown, or pass everything with skip-existing
    # true if you don't need to force level1track.

    # dry run: just report how many commands each level would generate
    python scripts/reprocess_levels.py --settings hyytiala2_v3.yaml \\
        --queue .../visss_task_queue_1.2_EXPRESS/ --levels all --dry-run

Note on the case argument: "0" (the default) is the full configured
deployment range (config.start .. config.end), same as
tools.getCaseRange's special-cased 0. See --case.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

from VISSSlib import products, tools

# canonical processing order, mirrors products.qc_report's default
# `products` list -- level2detect is left out of "all" by default (also
# commented out there) since most deployments care about level2match/track;
# pass it explicitly via --levels if you want it.
ALL_LEVELS = [
    "metaFrames",
    "level1detect",
    "metaRotation",
    "level1match",
    "level1track",
    "level2match",
    "level2track",
]

LAUNCH_SCRIPT = "/home/mmaahn/slurm_launch_workers.sh"


def ts():
    return time.strftime("%F %T")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def camerasForLevel(level, config):
    """Leader-only levels (metaRotation, level1match, level1track,
    level2match, level2track, level3combinedRiming, allDone -- see
    tools.LEVEL_REGISTRY) produce one combined file keyed by the
    leader camera; everything else is processed per camera."""
    if tools.LEVEL_REGISTRY[level].get("leaderOnly", False):
        return ["leader"]
    return ["leader", "follower"]


def queueDepth(queue):
    return len(glob.glob(os.path.join(queue, "queue", "*")))


def workerCwd(queue):
    return f"/home/mmaahn/slurm_{os.path.basename(queue.rstrip('/'))}"


def processingCount(queue):
    return len(glob.glob(os.path.join(workerCwd(queue), "*.processing.txt")))


def runningWorkerJobs():
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "mmaahn"), "-h", "-o", "%j"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as e:
        log(f"squeue check failed ({e}), assuming 0 running")
        return 0
    return sum(1 for line in out.splitlines() if "slurm_launch_workers" in line)


def ensureWorkers(target, queue, condaEnv):
    running = runningWorkerJobs()
    toLaunch = max(0, target - running)
    if toLaunch == 0:
        return
    log(f"{running}/{target} worker job(s) running, launching {toLaunch} more")
    for _ in range(toLaunch):
        cmd = ["sbatch", "--nice=10000", LAUNCH_SCRIPT, condaEnv, queue]
        log(f"launching worker: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def waitForDrain(label, queue, workers, condaEnv, pollSeconds):
    log(f"waiting for {label} to drain...")
    stableEmpty = 0
    while True:
        qd, pc = queueDepth(queue), processingCount(queue)
        log(f"{label}: queue_depth={qd} processing={pc}")
        if qd == 0 and pc == 0:
            stableEmpty += 1
            # two consecutive empty polls, to dodge the race right as the
            # last task finishes and before a new one would be leased
            if stableEmpty >= 2:
                break
        else:
            stableEmpty = 0
            ensureWorkers(workers, queue, condaEnv)
        time.sleep(pollSeconds)
    log(f"{label} drained.")


def processLevel(level, settings, case, queue, camera, skipExisting, workers,
                  condaEnv, pollSeconds, withParents, dryRun):
    config = tools.readSettings(settings)
    cameras = [camera] if camera else camerasForLevel(level, config)

    total = 0
    anySubmitted = False
    for cam in cameras:
        p = products.DataProductRange(level, case, settings, queue, camera=cam)
        cmds = tools._aggregate(
            [dp.generateAllCommands(skipExisting=skipExisting, withParents=withParents)
             for dp in p]
        )
        total += len(cmds)
        if dryRun:
            log(f"{level} ({cam}): {len(cmds)} command(s) [dry-run, nothing submitted]")
            continue
        if len(cmds) == 0:
            continue
        p.submitCommands(skipExisting=skipExisting, withParents=withParents)
        anySubmitted = True

    if dryRun:
        return total

    if anySubmitted:
        ensureWorkers(workers, queue, condaEnv)
        waitForDrain(level, queue, workers, condaEnv, pollSeconds)
    return total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--settings", required=True, help="VISSSlib settings YAML file")
    parser.add_argument(
        "--levels", nargs="+", required=True,
        help=f"'all' for the canonical order ({', '.join(ALL_LEVELS)}), "
        "or an explicit list of levels to do consecutively, e.g. "
        "level1track level2match level2track",
    )
    parser.add_argument(
        "--camera", choices=["leader", "follower"], default=None,
        help="Restrict to one camera. Default: both for per-camera levels, "
        "leader-only for combined levels (metaRotation/level1match/level1track/"
        "level2match/level2track/...).",
    )
    parser.add_argument(
        "--case", default="0",
        help="'0' (default) for the full configured deployment range, or "
        "'YYYYMMDD'/'YYYYMMDD-YYYYMMDD'/comma-separated dates, or an integer "
        "number of days back.",
    )
    parser.add_argument("--queue", required=True, help="Task queue directory")
    parser.add_argument(
        "--skip-existing", choices=["true", "false"], default="true",
        help="true (default): only (re)generate outputs not up to date with "
        "their parents (the normal products.py staleness check). false: "
        "force-regenerate every output for the given level(s)/case regardless "
        "of what already exists. Applies to every level in this run.",
    )
    parser.add_argument(
        "--with-parents", action="store_true",
        help="Also (re)generate missing/stale parent-level commands (with "
        "skip-existing forced true for those) alongside each requested level. "
        "Off by default so each level's work stays isolated to what you asked "
        "for.",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Target number of SLURM worker jobs to keep running while a "
        "level drains (default: 2). Each worker job runs "
        "`python -m VISSSlib worker <queue>`, which itself fans out to "
        "os.cpu_count() parallel processes using the job's allocated CPUs. "
        "Only run this against a queue/cluster you're allowed to use this "
        "many concurrent SLURM jobs on.",
    )
    parser.add_argument(
        "--conda-env", default="py313",
        help="Conda env slurm_launch_workers.sh should activate (default: py313).",
    )
    parser.add_argument(
        "--poll-seconds", type=int, default=300,
        help="How often to check drain progress / worker health (default: 300).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report how many commands each level would generate; "
        "submit nothing and launch no workers.",
    )
    parser.add_argument(
        "--max-passes", type=int, default=5,
        help="After the first forward pass through --levels, re-check every "
        "level in order and resubmit any that now has pending work, up to "
        "this many additional passes (default: 5). Guards against a level "
        "that was already declared drained having new work unlocked "
        "afterward by a concurrent process outside this script (e.g. a "
        "separate serial metaRotation backfill completing days this run's "
        "own metaRotation step had already passed) -- see "
        "feedback_reprocess_levels_misses_concurrent_upstream_unlock in "
        "project memory. A pass that resubmits nothing for every level ends "
        "the loop early.",
    )
    args = parser.parse_args()

    if [l.lower() for l in args.levels] == ["all"]:
        levels = ALL_LEVELS
    else:
        levels = args.levels
        unknown = [l for l in levels if l not in tools.LEVEL_REGISTRY]
        if unknown:
            parser.error(f"unknown level(s): {unknown}; known: {list(tools.LEVEL_REGISTRY)}")

    skipExisting = args.skip_existing == "true"

    log(f"levels (consecutive): {levels}")
    log(f"case={args.case} skipExisting={skipExisting} workers={args.workers} "
        f"queue={args.queue} dryRun={args.dry_run}")

    for level in levels:
        processLevel(
            level, args.settings, args.case, args.queue, args.camera,
            skipExisting, args.workers, args.conda_env, args.poll_seconds,
            args.with_parents, args.dry_run,
        )

    if args.dry_run:
        log("=== All levels complete ===")
        return

    # First pass above submitted each level once, in order. A concurrent
    # process outside this script (e.g. a serial metaRotation backfill)
    # can unlock new work for a level this run already passed, which the
    # single forward pass would otherwise never notice or resubmit. Sweep
    # forward through --levels again, resubmitting anything now pending,
    # until a full sweep finds nothing left anywhere, or --max-passes is
    # exhausted.
    for passNum in range(1, args.max_passes + 1):
        passTotal = 0
        for level in levels:
            passTotal += processLevel(
                level, args.settings, args.case, args.queue, args.camera,
                skipExisting, args.workers, args.conda_env, args.poll_seconds,
                args.with_parents, args.dry_run,
            )
        if passTotal == 0:
            break
        log(f"settle pass {passNum}/{args.max_passes} resubmitted "
            f"{passTotal} command(s) across {levels} -- checking again")
    else:
        log(f"WARNING: still finding new work after {args.max_passes} settle "
            f"passes -- either a concurrent process keeps unlocking work "
            f"faster than this script drains it, or something upstream "
            f"needs a look before this can converge")

    log("=== All levels complete ===")


if __name__ == "__main__":
    sys.exit(main())
