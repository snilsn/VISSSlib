``metadata`` - metaFrames and metaEvents
===========================================

Motivation
----------

Raw level0 data per camera consists of a video file, a per-frame CSV
(``capture_time``, ``record_time``, capture/record IDs, ...), a first-frame
JPEG (for a quick camera-blocking check without decoding video), and a daily
status text log of software start/stop events. ``metadata.py`` turns these
into the two netCDF metadata products everything downstream depends on:

- **metaFrames** — one netCDF per level0 file (a ``fileLevel``, see
  :doc:`files`), essentially a cleaned-up netCDF version of the per-frame
  CSV, produced by :func:`VISSSlib.metadata.createMetaFrames`.
- **metaEvents** — one netCDF per case/day (a ``dailyLevel``), one record
  per level0 file plus the day's status-log events, produced by
  :func:`VISSSlib.metadata.createEvent`. This is what
  :func:`VISSSlib.tools.removeBlockedBlowingData` and the ``ptpStatus``
  checks in :func:`VISSSlib.matching.matchParticles` read.

Timestamp repair
-----------------

The module-level docstring-comment at the top of ``metadata.py`` lists the
known raw-data pathologies (mostly from the MOSAiC deployment, where data
acquisition computers occasionally couldn't keep up with heavy snowfall):
``capture_id`` overflow at 65535, ``capture_time`` drift (only reset on
camera restart), ``record_time`` jitter/drift, flipped/swapped consecutive
timestamps, and "invented" frames where an extra frame appears with
compressed inter-frame spacing. :func:`VISSSlib.metadata.getMetaData` (via
:func:`VISSSlib.metadata._getMetaData1` per file) is where the >0
generation-specific handling for these lives — e.g. for ``visssGen ==
"visss"``, a detected time-jump drops the 3 frames around the jump rather
than attempting to reorder them, while ``visss2`` intentionally has no
generic fix yet (``raise NotImplementedError``) beyond a first-frame-only
case. This is a good first place to look when a new deployment/generation
surfaces a new timestamp anomaly.

metaEvents / ``createEvent``
-------------------------------

:func:`VISSSlib.metadata.getEvents` builds one record per level0 file from
:func:`VISSSlib.metadata._readHeaderData` (which parses the small text
header block at the top of each per-file CSV: VISSS file-format version,
git tag/branch of the acquisition software, capture start/first/last time,
camera serial number, hostname, PTP sync status, camera temperature, and
transfer-queue block counts) — recording an ``event="brokenfile"`` record
with NaN fields rather than dropping the file entirely when the header
can't be parsed, so file *counts* stay meaningful for completeness checks
even when content is missing.

The **camera-blocking estimate** is computed here too, from the JPEG
thumbnail rather than the video: a cumulative brightness histogram (fixed
thresholds ``[0, 11, 21, ..., 251]``, matching
``config.level1detect.threshs``) normalized by pixel count, stored per file
as the ``blocking`` variable indexed by ``blockingThreshold``. Downstream,
:func:`VISSSlib.tools.removeBlockedBlowingData` thresholds this at
``blockingThreshold=50`` against ``config.quality.blockedPixThresh`` to
decide whether a period should be excluded as camera-blocked (snow/frost on
the window) — the raw video is never touched for this check.

:func:`VISSSlib.metadata.createEvent` (per case+camera, decorated with
:func:`VISSSlib.tools.loopify_with_camera`) is more than a thin wrapper:
its ``skipExisting`` check is unusually careful for a daily product,
because metaEvents for *today* keeps changing throughout the day as new
level0 files land. It re-generates (rather than skips) when: the existing
file is empty, the daily status file was modified within the last 6 hours
(a generous buffer for delayed data transfer), or the recorded
``noLevel0Files`` count in the existing file's attrs no longer matches the
number of level0 files currently on disk. If building the event dataset
fails entirely (``ValueError``/``AssertionError``, i.e. zero files), it
distinguishes "not synced yet" from "genuine data gap" by checking whether
*newer* level0 data already exists for a later case — only in the latter
case does it write the ``.nodata`` sentinel (see :doc:`files`).

``VISSSlib.metadata`` API
----------------------------

.. automodule:: VISSSlib.metadata
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
