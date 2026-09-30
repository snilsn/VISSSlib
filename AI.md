# AI.md

This file provides guidance to AI coding assistants (Claude Code, and others that read
this convention) when working with code in this repository.

## What this is

VISSSlib is the processing library for the VISSS (Video In Situ Snowfall Sensor), a
dual-camera instrument that images falling snow. Two stereo cameras ("leader" and
"follower") record video continuously; this library turns the raw per-camera video/CSV
files into calibrated particle-size distributions and derived microphysical products
(riming, aspect ratio, fall velocity, etc.). Companion repos: the data-acquisition
software (github.com/maahn/VISSS) and the VISSS2/VISSS3 hardware plans (Zenodo).

Background on the processing chain and the physics is in [docs/source/processing.rst](docs/source/processing.rst),
[docs/source/matching.rst](docs/source/matching.rst), and [docs/source/metaRotation.rst](docs/source/metaRotation.rst) —
read these before touching `matching.py` or the rotation/calibration retrieval code.

## Development commands

Install in editable mode (Python >= 3.11 required):

```bash
pip install -e .
```

Level 3 products additionally require PAMTRA (https://github.com/igmk/pamtra), which is
not a pip dependency and must be installed separately.

Run the full test suite:

```bash
pytest tests/
```

Run a single test file / test:

```bash
pytest tests/test_matching.py
pytest tests/test_products.py::TestProducts::test_processAll
```

There is no `pytest.ini`/`pyproject` test config and no CI workflow in this repo — plain
`pytest` defaults apply. Most tests (see `tests/helpers.py`) work against real sample data
that is lazily downloaded from a Uni Leipzig cloud share into `tests/data/` the first time
it's needed, so the first test run requires network access and takes noticeably longer.
Each test reads its config via `readTestSettings("test_X.Y/testtmp_X.Y.yaml")`, which points
`tmpPath`/`fileQueue` at a per-hostname scratch dir under `tests/data/`; tests generally
`shutil.rmtree` that dir before running.

Formatting uses `black` (line-length 88, see `pyproject.toml`); there's no separate lint
config.

Build the docs (Sphinx, autodoc + `sphinx-argparse` for the CLI reference):

```bash
cd docs && make html
```

## Running the pipeline

The library is normally driven either as a Python API or via `python -m VISSSlib` (entry
point defined in `src/VISSSlib/__main__.py`, argument parser built by
`tools._create_parser()`). Every subcommand takes a YAML **settings** file (see `sample.yaml`
and `docs/source/config_files.rst`) and, for most levels, a **case** — either an integer
number of days to look back, or `"YYYYMMDD"`, `"YYYYMMDD-YYYYMMDD"`, or a comma-separated
list of dates (parsed by `tools.getCaseRange`).

```bash
python -m VISSSlib products.processAll settings.yaml YYYYMMDD-YYYYMMDD
python -m VISSSlib metadata.createEvent settings.yaml nDays --camera leader
python -m VISSSlib detection.detectParticles file.txt settings.yaml --skip-existing
python -m VISSSlib <command> --help
```

`products.submitAll` / the `worker` subcommand enqueue jobs into a `task-queue`
(`taskqueue.TaskQueue`, file-backed queue directory) that one or more `worker` processes
drain; `scripts/VISSSlib_slurm.sh` shows this pattern on a SLURM cluster. Note
`scripts/*.sh` reference an older `scripts.*` module layout and may not match the current
`__main__.py` command names — check `tools._create_parser()` for the authoritative command
list before relying on the shell scripts.

## Architecture

### Data levels and the processing DAG

Processing is organized into named **levels**, each with its own file layout and netCDF
schema. Roughly, in dependency order:

```
level0 (raw video/csv/jpg from acquisition, read-only input)
  -> metaEvents (per-camera daily status/blocking, from level0 csv+jpg)
  -> metaFrames (per-camera netCDF version of the frame timestamp csv)
  -> level1detect (per-camera, per-particle properties in pixel units; detection.py)
  -> metaRotation (leader/follower misalignment retrieval, daily; matching.py)
  -> level1match (particles matched between leader & follower cameras; matching.py)
  -> level1track (particles tracked across frames for fall velocity; tracking.py)
  -> level2detect / level2match / level2track (daily, calibrated to metric units,
     aggregated into size/shape distributions; distributions.py)
  -> level3combinedRiming (derived microphysics, e.g. riming; level3/combined_riming.py)
  -> allDone (sentinel marking a case fully processed)
```

`level1detect/level1match/level1track` are per-10-minute-file products (square corners in
the processing flowchart figure); most others are daily aggregates (rounded corners). This
dependency graph is encoded explicitly as `tools.LEVEL_REGISTRY` (`src/VISSSlib/tools.py`) —
that dict is the authoritative source of "what depends on what", not the prose above.
`products.DataProduct.__init__` resolves each level's `parentNames` from it, and each
processing function's own `tools.checkForExisting(..., fL=..., level=...)` skip-check
resolves the same declared parents (see `tools.resolveLevelParents`) rather than
hand-writing its own list — the two used to drift apart (see the `level2track`/`level2match`
case below) before they shared this one source.

One exception to the otherwise-parallel level2 products: `level2track` also depends on
`level2match` (not just `level1track`) — it reuses level2match's `zResidualTooWide` quality
flag rather than recomputing an equivalent from level1track data, since tracking's own
frame-to-frame continuity requirement already filters out most particles that would show a
wide Z-consistency residual, making an independent per-track computation largely
ineffective (see `distributions.getZResidualQuality`'s docstring). This means level2match
must be (re)built before level2track when reprocessing.

`products.DataProduct` is the central orchestration class: given a `level`, `case`,
`config`, and `camera`, it recursively builds its parent products, checks whether output
files already exist/are newer than their parents (`isComplete`, `_youngerThanParents`), and
if not, generates the shell commands needed to (re)produce them
(`generateCommands`/`generateAllCommands`). `products.processAll` / `processRealtime` /
`submitAll` are the entry points that walk this DAG for a case and either run commands
directly or push them onto the task queue. When adding a new product/level, you must wire
it into this `parentNames` block and into the module dispatch table in `__main__.py`.

### Module map

- `files.py` — filename patterns and path resolution per level (`FindFiles`,
  `Filenames`); `dailyLevels`/`fileLevels`/`quicklookLevels*`/`imageLevels` in this module
  classify which levels are daily vs per-file vs quicklook-only vs image-producing.
- `metadata.py` — builds `metaEvents`/`metaFrames` from raw level0 csv/jpg.
- `detection.py` — single-camera particle detection (level1detect), OpenCV-based contour
  detection on video frames.
- `matching.py` — camera misalignment retrieval (`metaRotation`, Optimal-Estimation-based,
  see `docs/source/metaRotation.rst`) and stereo particle matching (`level1match`, see
  `docs/source/matching.rst`).
- `tracking.py` — frame-to-frame particle tracking (`level1track`) for fall velocity.
- `distributions.py` — daily calibration + aggregation into level2 distributions (imports
  `matching.py` via `from .matching import *`).
- `level3/` — derived level3 microphysical products; `combined_riming.py` implements the
  riming retrieval, `aux.py` holds shared helpers; `level3.AVAILABLE_PRODUCTS` registers
  which products exist.
- `av.py` — video (`VideoReader`)/metadata reading wrapper around the raw mov/mkv files.
- `quicklooks.py` — plotting/quicklook image generation for each level.
- `analysis.py` — higher-level/offline analysis helpers built on top of the products.
- `fixes.py` — targeted workarounds for known bugs/artifacts in specific historical data
  periods (campaign-specific patches), keyed by date/period.
- `tools.py` — grab-bag of shared infrastructure: settings loading (`readSettings`,
  `DEFAULT_SETTINGS`), the `DictNoDefault` (addict-based, KeyErrors on missing keys) config
  type, case/date-range parsing, the CLI parser (`_create_parser`), task-queue helpers
  (`runCommandInQueue`, `workers`).

### Configuration

Deployments are described by YAML settings files (see `sample.yaml`,
`docs/source/config_files.rst`) with keys like `leader`/`follower` (camera IDs),
`fps`/`resolution`/`frame_height`/`frame_width`, `path`/`pathOut`/`pathQuicklooks`
(templated with `{level}`), `site`, `start`/`end`, `rotate` (per-period camera
transformation priors), and `calibration`. `tools.readSettings` merges a settings file over
`DEFAULT_SETTINGS` (via `flatten_dict`, since settings are nested), warns on unknown keys
(except under `rotate`), resolves relative paths against the YAML file's directory, and
expands `$HOSTNAME`. The resulting config is an `addict`-based `DictNoDefault` — accessing
an undefined key raises `KeyError` rather than silently returning `{}`.

### Logging

Logging goes through `loguru` (`from loguru import logger as log`), configured once in
`src/VISSSlib/__init__.py` (INFO+ to stdout, ERROR+ to stderr). Many top-level entry points
are wrapped in `@log.catch(reraise=True)`.
