``tools`` - shared infrastructure
==================================

Motivation
----------

``tools.py`` is the module every other part of VISSSlib depends on: settings
loading, case/date-range parsing, the CLI parser, netCDF read/write helpers,
and the task-queue plumbing all live here. There is no single unifying theme
beyond "infrastructure that many modules need" — this page groups the
non-obvious parts by concern rather than walking through every function
(see :doc:`api` / the autodoc block below for the full member list).

Settings and configuration
---------------------------

``DEFAULT_SETTINGS`` is the authoritative schema for a deployment's YAML
settings file. :func:`VISSSlib.tools.readSettings` merges a settings file
over ``DEFAULT_SETTINGS`` (flattening both dicts first via ``flatten_dict``
so nested keys merge correctly instead of being replaced wholesale), warns
about unknown top-level keys (except everything under ``rotate``, since
those keys are per-timestamp deployment periods, not schema), resolves
``path``/``pathOut``/``pathQuicklooks`` relative to the YAML file's own
directory, and expands ``$HOSTNAME``. The result is a
:class:`VISSSlib.tools.DictNoDefault` (an ``addict.Dict`` subclass) —
accessing an undefined key raises ``KeyError`` instead of silently
returning ``{}``, which is deliberate: a typo'd config key should fail loud,
not produce an empty dict that gets used downstream.

``readSettings`` is idempotent: it accepts either a YAML path or an
already-parsed config and returns the config unchanged in the latter case.
This is why almost every public function's ``config`` parameter can be
either a path or a live config object — they all call ``readSettings``
first.

The ``@loopify`` / ``@loopify_with_camera`` decorators
--------------------------------------------------------

Most per-case processing entry points (e.g.
:func:`VISSSlib.matching.createMetaRotation`) are decorated with
:func:`VISSSlib.tools.loopify`, and per-case-and-camera ones (e.g.
:func:`VISSSlib.metadata.createEvent`) with
:func:`VISSSlib.tools.loopify_with_camera`. The decorator turns a function
that processes a single ``case`` into one that accepts the same ``case``
argument styles :func:`VISSSlib.tools.getCaseRange` understands (a number of
days, ``YYYYMMDD``, ``YYYYMMDD-YYYYMMDD``, or a comma-separated list) and
loops over the resulting case range, collecting results. Both decorators
also wrap the function in ``@log.catch(reraise=True)`` so a failure in one
case is logged with a full traceback before propagating. This is also where
the CLI's ``case`` positional argument (see :doc:`command_line`) gets its
uniform meaning across almost all subcommands.

File I/O conventions
---------------------

A handful of small helpers encode conventions used throughout the codebase,
not just in this module:

- :func:`VISSSlib.tools.checkForExisting` — the ``skipExisting`` check used
  by nearly every processing function: a file "counts" as already done only
  if it is newer than all of its parent files (or event/level0 files). This
  is the mtime-based half of the DAG completeness logic; the other half
  (counting expected vs. present files) lives in
  :class:`VISSSlib.files.FindFiles.nMissing`.
- :func:`VISSSlib.tools.open2` — opens a file for writing after creating
  parent directories and, on ``cleanUp``, removing any stale
  ``.nodata``/``.broken.txt`` sentinel siblings of the *real* output file.
  Those two sentinel suffixes are VISSSlib's way of recording "this file was
  attempted but produced no data" / "this file failed" without leaving a
  half-written netCDF around — several modules check for them with plain
  ``os.path.isfile(f"{fname}.nodata")`` rather than a shared helper, so the
  convention is implicit rather than enforced by a type.
- :func:`VISSSlib.tools.to_netcdf2` — writes to a randomly-suffixed temp
  file and renames it into place with ``os.rename``, to avoid partial files
  being picked up by a concurrent reader or another worker process; also
  clears any stale sentinel files for the same output on success.
- :func:`VISSSlib.tools.finishNc` — the common last step before writing any
  product: attaches standard attributes (:func:`VISSSlib.tools.ncAttrs`,
  including the paper citation and the exact command line that produced the
  file), downcasts ``float64`` to ``float32`` and strings/objects to fixed
  ``<U`` dtypes (newer netCDF4 dislikes Python object arrays), enables zlib
  compression, and pins time-variable encoding to microseconds since epoch
  (left to xarray's default, mixed ms/us units across files have caused
  inconsistencies).

The ``open_mf*`` family (:func:`~VISSSlib.tools.open_mfmetaFrames`,
:func:`~VISSSlib.tools.open_mflevel1detect`,
:func:`~VISSSlib.tools.open_mflevel1match`) are thin wrappers around
``xarray.open_mfdataset`` that additionally: attach a ``file_starttime``
coordinate per record (via a ``preprocess`` callback using
:class:`VISSSlib.files.FilenamesFromLevel` on each file's own path), filter
out ``.nodata``/``.broken.txt``/``.notenoughframes`` sibling paths before
opening, and apply the subset of :mod:`VISSSlib.fixes` selected by
``config.dataFixes`` unless explicitly skipped. This is the layer where the
campaign-specific patches in ``fixes.py`` actually get invoked.

``BlockImageArchive``
----------------------

:class:`VISSSlib.tools.BlockImageArchive` is a bespoke single-file archive
format for the ``imagesL1detect`` product (thousands of small per-particle
uint8 image crops), used instead of a real zip file
(:class:`VISSSlib.tools.ZipFile` is kept as a fallback, selected by
:func:`VISSSlib.tools.imageZipFile` based on file extension). Images are
buffered and grouped into ``block_size``-sized ``zlib``-compressed blocks
(compressing many small images together beats compressing them individually
for both size and per-image overhead), with a JSON index appended at the end
of the file recording each image's block offset/length and in-block byte
range for O(1) random access without decompressing the whole archive.

Task queue
----------

:func:`VISSSlib.tools.runCommandInQueue` (registered with
``@taskqueue.queueable`` for the `python-task-queue
<https://github.com/seung-lab/python-task-queue>`_ package used by
``products.submitAll``) runs one shell command, using a
``portalocker.Lock`` on a ``.processing.txt`` file to detect and skip a
command that's already running elsewhere (important since many worker
processes/nodes can pull from the same queue) rather than to serialize
access in general. A non-zero exit code copies the log to
``<output>.broken.txt`` next to the intended output file.
:func:`VISSSlib.tools.workers` / :func:`VISSSlib.tools.worker1` spin up
``multiprocessing`` worker processes that poll a queue and self-terminate
once all sibling workers report an empty queue (via a shared
``multiprocessing.Array`` status flag), so a fixed-size worker pool doesn't
have to be told explicitly when a batch is finished.

Rotation config plumbing
--------------------------

:func:`VISSSlib.tools.rotXr2dict` / :func:`VISSSlib.tools.rotDict2Xr` /
:func:`VISSSlib.tools.getPrevRotationEstimate(s)` convert between the
``metaRotation`` netCDF representation (an ``xr.Dataset`` indexed by
``file_starttime`` with a ``camera_rotation`` dim of ``["mean", "err"]``)
and the settings-file ``rotate:`` block's dict-of-dicts representation (see
``sample.yaml`` and :doc:`metaRotation`). This is the glue that lets
:func:`VISSSlib.matching.matchParticles` fall back to a manually
pre-computed rotation from the config file when no ``metaRotation`` product
exists yet for a period.

``VISSSlib.tools`` API
------------------------

.. automodule:: VISSSlib.tools
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
