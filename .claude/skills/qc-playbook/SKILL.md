---
name: qc-playbook
description: Run the VISSS quality-control playbook against a deployment settings YAML — structural scan, resubmit broken files via the task queue, re-scan, sample-then-escalate matchScore/Z-residual checks, classify remaining findings as accepted-unfixable vs. genuinely new, loop submit/re-check against `allDone` (the full DAG, not just level1track/level2match/level2track) until it actually settles, and deliver a broken-periods table. Use this whenever the user asks to "run QC", "QC scan", "quality control", "check for broken/missing files", or "reprocess broken" against a VISSSlib config/settings YAML, or names a deployment (e.g. "run the QC playbook on hyytiala2_v3") — even if they don't spell out the individual steps, since this skill already encodes the ordering and the repo-specific idioms (task queue, DataProductRange, memory-keyed accepted-unfixable list) that make doing it ad hoc error-prone. Never declare a pass complete while `allDone`'s `generateAllCommands()` is still non-empty and unsubmitted.
---

# VISSS QC playbook

A repeatable audit-and-reprocess pass over one deployment's settings YAML. The
goal is to separate three outcomes for every finding: fixed by reprocessing,
already-known-and-accepted as unfixable, or genuinely new (needs a human to
look at it). Nothing here is a black box — every step calls into
`scripts/qc_report.py` / `scripts/reprocess_broken.py` / `products.py`, which
are the actual source of truth and may have grown flags or categories since
this was written. Skim `python scripts/qc_report.py --help` and that script's
own module docstring before a run if it's been a while — don't rely on this
file's paraphrase of it.

## Step 0 — load prior status from memory, don't re-derive it

Before touching any scripts, look in the user's memory directory
(`/Users/mmaahn/.claude/projects/-Users-mmaahn-projectsSrv-VISSSlib/memory/`)
for `project_<config-basename>_qc_status.md`, where `<config-basename>` is the
settings YAML's filename without `.yaml` (e.g. `hyytiala2_v3.yaml` ->
`project_hyytiala2_v3_qc_status.md`). If it exists, it already records this
deployment's list of accepted-unfixable periods and any deployment-specific
reprocessing quirks — read it and use it as ground truth for Step 5's
classification instead of re-deriving what "accepted unfixable" means from
scratch.

If it doesn't exist, this is the first QC pass for this config: proceed with
the general playbook below, and write that memory file at the end of the run
(Step 7) so future passes don't repeat this analysis.

Also check `vissslib_production_deployment.md` in memory for the current task
queue directory and conda env — these vary and shouldn't be hardcoded.

## Step 0.5 — repair stale freshness caches before scanning

Always run this before Step 1, on every pass, not just the first one for a
config. `products.py`'s per-level `_freshnessSummary` cache (the small
`.done` files under each level's output dir) is only invalidated by writes
that go through `tools.open2`/`to_netcdf2`'s fence-bump hook. A handful of
write paths that bypass that hook have already been found and fixed
(`allDone`'s bare `touch` command, `runCommandInQueue`'s `.broken.txt` write
on task failure, `cleanUpBroken`/`cleanUpDuplicates`' plain `os.remove`), but
there's no guarantee that list is exhaustive — and a long-running worker
process holding pre-fix code in memory reproduces the exact same stale-cache
symptom regardless of how many such bugs get fixed on disk, since it never
re-imports a fix until it restarts. A stale cache masks real staleness
(`_upToDateWithParents` reports true when it shouldn't) and can also make a
DAG check spuriously never converge — either way it silently corrupts what
Step 1's `dag_stale` category and Step 6's settle-check are measuring, so
don't skip this even on a config that "should" already be clean:

```python
from VISSSlib import products
p = products.DataProductRange("allDone", case, "<config>.yaml", queue, camera="leader")
repaired = p.repairStaleFreshnessCache(withParents=True)
print("repaired any stale cache:", repaired)  # check the log for which ones, if True
```

This is read-mostly and safe to run every time — it only touches a cache
file when it actually disagrees with a live scan of the real files, never
the real data itself.

## Step 1 — structural scan

```bash
python scripts/qc_report.py <config>.yaml --out /tmp/qc_<config-basename>.csv
```

This is fast (no file-opening) and reports: `missing`, `broken`,
`nodata_suspect`, `duplicate`, `reduced_coverage` (see the script's docstring
for exactly what each means — it's detailed and deliberately not repeated
here). Known-bad periods already listed in `config.badData` are excluded by
default; don't pass `--include-known-bad` unless the user specifically wants
those re-surfaced.

## Step 2 — resubmit everything broken via the task queue

For each affected `<camera>_<product>` pair the scan turned up (e.g.
`leader_level1match`, `follower_level1detect`):

```bash
python scripts/reprocess_broken.py \
    --settings <config>.yaml \
    --products <camera>_<product> [<camera>_<product> ...] \
    --case <case-range-or-0-for-full-range> \
    --queue <task-queue-dir-from-memory>
```

This clears `.broken.txt` sentinels and resubmits via
`products.DataProductRange(...).cleanUpBroken()` +
`.submitCommands()` — cheap and safe because `skipExisting=True` means only
what's actually missing gets regenerated. It deliberately does **not** touch
SLURM directly (no `sbatch`/`squeue`/`scancel`) — existing workers drain the
queue on their own.

**Check `tools.REPROCESS_AFTER` before including `level1track` (or
anything downstream of it) in `--products`.** This dict can mark an
entire level as stale fleet-wide regardless of individual `.broken.txt`
status (e.g. a `level1track` entry from a 2026-09-02 Dmax/dropped-frame
fix) — `generateAllCommands`'s DAG walk picks this up silently and can
inflate a submission meant to cover a few hundred QC findings into tens
of thousands of commands, absorbing a much larger, possibly
deliberately-deferred backlog (see
`project_detectphasejump_fleetwide_scan.md` for a case where this
happened on nyaalesund_v5). Check `tools.REPROCESS_AFTER` and any memory
noting a deferred backlog for the target config before submitting; if in
doubt, confirm scope with the user rather than assuming a QC-scoped
resubmit stays QC-scoped.

**SLURM is a shared cluster resource — no artificial worker cap.** An
earlier version of this playbook capped concurrent QC workers at 2; that
was corrected 2026-09-04 (see
`feedback_do_not_touch_slurm.md`/`[[feedback-do-not-touch-slurm]]`): the
queue and its workers are shared infrastructure other jobs/users legitimately
use too, so submitting QC work onto a queue that already has unrelated
traffic (e.g. another deployment's reprocessing) is expected and fine, not
something to hold back on or ask permission for.

**Order matters: submit to the queue first, then start a worker if one is
needed.** A worker launched against an empty queue can idle-exit within
~60s (see commit 55898f1, "Bound SLURM worker idle-exit to ~60s"), so
starting a worker before there's anything queued risks it exiting before
your `reprocess_broken.py` submission lands. If workers are already
actively draining the target queue (check recent
`~/slurm_<queue-basename>/*.processing.txt` mtimes), submitting is enough
— no need to launch another.

Don't wait inline for the queue to drain by polling tightly — check back
after a reasonable interval, or ask the user to confirm workers have caught
up, before moving to Step 3.

## Step 3 — re-scan

Once the queue has drained, re-run the Step 1 scan. Anything that dropped out
of the `broken`/`missing` categories was fixed by reprocessing. Anything
still there survived a fresh reprocessing attempt with current code — that's
the real remainder to classify.

## Step 4 — matchScore / Z-residual check: sample first, escalate only if non-trivial

```bash
python scripts/qc_report.py <config>.yaml --matchscore-check sample --out /tmp/qc_<config-basename>_matchscore.csv
```

This opens a random sample of real `level1match` files per level/camera/day
and flags `matchscore_suspect` (median matchScore below threshold, often
because the file had too few pairs for matchParticles' own quality gate to
ever run) and `z_sigma_suspect` (Z-consistency residual too wide — see Step 5).

Only escalate to `--matchscore-check all` (which opens every level1match
file — slow) if the sample result looks non-trivial: findings spread across
many distinct days/periods rather than one or two isolated files, or a rate
that suggests the sample undercounts a real systemic problem. A sample
turning up nothing, or one or two isolated one-off files, does not warrant
the full scan.

## Step 5 — classify the remainder

For everything still outstanding after Steps 3–4, sort into:

- **Accepted-unfixable**: a `z_sigma_suspect` finding — a real, non-broken
  `level1match` file whose Z-consistency residual is *wide* rather than
  *biased*. Per `matching.zResidualSigma`'s empirical basis, a rotation
  refit can correct a biased residual but cannot shrink a genuinely wide
  one, so a wide sigma that survives reprocessing is the expected signature
  of real correspondence ambiguity, not a new bug — it does not need further
  investigation. Cross-check against the prior periods already listed in
  `project_<config-basename>_qc_status.md` (Step 0) if that file existed.

- **Genuinely new — investigate**: anything else that didn't resolve:
  persistent `broken`/`missing`/`matchscore_suspect`/`nodata_suspect`
  findings, or a `z_sigma_suspect` in a period/pattern not already covered
  by the memory file. Before treating it as a mystery, check whether it
  matches a known bug class these narrower scripts already handle:
  - `scripts/relabel_missing_movies_broken.py` — a `.nodata` sentinel that
    actually describes a raw-data-availability problem ("movie file... not
    found", transfer/corruption language) rather than confirmed
    no-precipitation.
  - `scripts/fix_level1match_rotation_seeds.py` — level1match failing on
    "matchScore smaller than minMatchScore" because metaRotation never got
    a good seed for that day.
  - `metaRotation` itself logging "no previous data found ... Nh old which
    is more than 48h" and suggesting `tools.copyLastMetaRotation <settings>
    <fromCase> <toCase>` — a genuine gap exceeding `maxAgeDaysPrevFile`
    (48h) in the sequential retrieval's day-to-day seeding, not a caching
    bug, even though it looks like one (the command exits 0 without
    writing anything when this happens, and if an old output file with
    that name already exists `runCommandInQueue` reads that as success, so
    the same command keeps getting regenerated by the DAG and re-skipped
    identically forever). The automatic self-heal path only fires for
    gaps under 8 days with confirmed instrument-offline evidence (missing
    prior-day metaEvents) — anything larger falls through to needing this
    manual seed. Treat the suggested command as the correct first remedy
    (see [[feedback_copylastmetarotation_first_try]]), confirm with the
    user before running it (it fabricates a synthetic seed by copying the
    last real calibration forward across the gap), then re-run
    `createMetaRotation` for the target date to confirm it actually
    retrieves (can take a couple of minutes per day). Multiple
    independent gaps in one deployment's history are normal, not
    surprising — check whether the target dates are consecutive before
    deciding serial vs. parallel re-run (see
    [[feedback_metarotation_run_serial_not_slurm]]).
  Only escalate to genuinely open-ended investigation if it matches none of these.

## Step 6 — confirm the dependent levels settle

**Check `allDone`, not just level1track/level2match/level2track.** A prior
run of this playbook (2026-09-17) checked only those three levels, called
the pass complete once their non-empty pending lists were rationalized as
"just known accepted-unfixable retries" (true for some configs, never
verified for others), and never noticed that `allDone`'s own declared
parents (`tools.LEVEL_REGISTRY["allDone"]`: `metaEvents`, `level2detect`,
`level2match`, `level2track`, and — if `level3.combinedRiming.processRetrieval`
is enabled — `level3combinedRiming`) include levels that check never
touched at all. The user caught real outstanding work (e.g. eriswil_v1 had
5 un-submitted `createLevel2track` commands) that this narrower check
structurally could not see. Always check `allDone` itself:

```python
from VISSSlib import products
p = products.DataProductRange("allDone", case, "<config>.yaml", queue, camera="leader")
print(p.generateAllCommands())  # empty list == settled
```

**This must be a loop, not a one-shot check.** `generateAllCommands()` is
read-only — finding a non-empty list and then writing up the pass as
"complete" (rationalizing the count from the log messages alone, without
submitting anything) is exactly the mistake to avoid; the skill's own
point of this step is "don't call the pass done until these settle," which
means actually closing the loop:

1. `p.generateAllCommands()` — if empty, done for this config.
2. If non-empty, `p.submitCommands()` (submits to the shared queue, cheap
   and safe per Step 2's reasoning) and wait for the queue to drain
   (Step 2's queue-mechanics caveats — lease timing, `.processing.txt` as
   ground truth, worker liveness — all apply here too).
3. Re-run `generateAllCommands()`. Repeat until either empty, or every
   remaining command is confirmed (not assumed) to be a cascade of an
   already-classified accepted-unfixable failure (e.g. by sampling a few
   of the returned `(command, outfile)` tuples and checking they target
   dates already in that config's accepted-unfixable table) — a large
   count "probably being retries" is not the same as checking.

If checking every one of a large `generateAllCommands()` result individually
isn't practical, at minimum submit it and re-check after draining — don't
skip straight from "found a non-empty list" to "wrote the deliverable."

## Deliverable — broken-periods table

Report a concise table, not the raw CSV dump:

| level | camera | case/period | category | status | note |
|---|---|---|---|---|---|

Where `status` is one of `reprocessed-fixed`, `accepted-unfixable`, or
`new-investigated`, and `note` is a one-line reason (e.g. "z-sigma 5.8 after
refit, matches known-unfixable pattern" or "seed missing, see
fix_level1match_rotation_seeds.py output").

## Deliverable — LaTeX broken-periods table

Alongside the markdown table above, always also emit a LaTeX version
covering the **accepted-unfixable and new-investigated periods only**
(skip `reprocessed-fixed` rows — those are resolved, not broken periods
to report). Columns: period, affected camera(s), reason. Collapse
consecutive per-file windows within the same day/period and cause into
one row (the markdown table's per-window detail is for this chat; the
LaTeX table is for a document, so it should read as a period list, not a
file dump). Use `booktabs`; escape LaTeX special characters (`_`, `%`,
`&`, `#`) in period strings and reasons before inserting them (e.g.
`level1match_V1.2_...` → escape every `_`).

```latex
\begin{table}[htbp]
  \centering
  \caption{Broken/unfixable periods for <config-basename>}
  \label{tab:<config-basename>-broken-periods}
  \begin{tabular}{lll}
    \toprule
    Period & Camera(s) & Reason \\
    \midrule
    2023-12-07 & leader & Z-residual sigma too wide (correspondence ambiguity) \\
    2024-01-02 & leader, follower & Corrupted video, 0 decodable frames \\
    \bottomrule
  \end{tabular}
\end{table}
```

Render this table directly in the chat response (inside a fenced ```latex
block) after the markdown table — don't write it to a file unless the
user asks.

## Step 7 — memory update policy

Only write or update `project_<config-basename>_qc_status.md` if this pass
found something **not already recorded there** — a newly-confirmed
accepted-unfixable period, a reprocessing quirk specific to this deployment,
or (if the file didn't exist yet) the full first-pass summary. Don't restate
findings that memory file already covers; that just adds noise for the next
read.
