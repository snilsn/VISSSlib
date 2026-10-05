``distributions`` - calibration and Level 2 distributions
==============================================================

Motivation
----------

``distributions.py`` (the largest core module, ~3000 lines) turns the
per-particle, pixel-unit Level 1 products (``level1detect``/
``level1match``/``level1track``) into the Level 2 products described in
the paper's "Time-resolved particle properties" section: calibrated
(metric-unit), one-minute-resolved particle size distributions (PSDs) plus
aggregate statistics. It imports ``from .matching import *`` — it reuses
the rotation-transform functions (:func:`VISSSlib.matching.shiftRotate_F2L`
etc.) for the observation-volume geometry below, not just for particle
matching.

Entry points and chunking
-----------------------------

:func:`VISSSlib.distributions.createLevel2detect` (per-camera, decorated
with :func:`VISSSlib.tools.loopify_with_camera`),
:func:`VISSSlib.distributions.createLevel2match`, and
:func:`VISSSlib.distributions.createLevel2track` (both cameras combined,
:func:`VISSSlib.tools.loopify`) are thin dispatchers into
:func:`VISSSlib.distributions._createLevel2` with ``sublevel="detect"/
"match"/"track"``, followed by triggering the corresponding quicklook.

:func:`VISSSlib.distributions._createLevel2` does the completeness checks
(via :class:`VISSSlib.files.FindFiles`'s ``isCompleteL1*`` properties —
refuses to produce a Level 2 day until every expected Level 1 file for that
day exists) and then — for performance — **splits a full-day case into 24
hourly sub-cases** and calls
:func:`VISSSlib.distributions._createLevel2part` once per hour, concatenating
the results along ``time`` afterward. This chunking exists purely for
memory/runtime reasons; it is invisible in the output (a case parameter like
``"20260110-08"`` handed to ``_createLevel2part`` directly skips the
hourly split and processes just that hour, which is also how tests keep
Level 2 tests fast).

``applyFilters`` — the quality-selection DSL
-------------------------------------------------

Both ``createLevel2*`` functions accept an ``applyFilters`` list, each
entry a 4-tuple: ``(variable, operator, value, cameraSelector, extraDimSel)``.
``operator`` is one of ``>``/``<``/``>=``/``<=``/``==``
(:data:`VISSSlib.distributions._operators`); ``cameraSelector`` is
``"min"``/``"max"``/``"mean"`` (:data:`VISSSlib.distributions._select`) —
since a matched/tracked particle has one value per camera (or per track
step), filters must first decide how to reduce that to a single number;
``value`` can also be a two-element ``[intercept, slope]`` pair, in which
case the threshold is itself linear in ``Dmax`` rather than a constant
(e.g. a size-dependent aspect-ratio cutoff). This mechanism is what lets
callers implement ad-hoc quality selections without a code change.

Single-camera ("detect") volume correction
-----------------------------------------------

For ``sublevel="detect"`` (single camera, no stereo constraint on the
observation volume), :func:`VISSSlib.distributions._createLevel2part`
applies a **hardcoded, empirically derived per-``Dmax``-bin blur
threshold** (for ``visssGen == "visss"``) instead of a geometric volume
cutoff — the comment above the table explains it was derived by comparing
cumulative detect-vs-match PSDs in Hyytiälä winter 2021/22 to find, per
size bin, the blur threshold that makes the single-camera distribution
agree with the (volume-constrained) matched one. This is the concrete
implementation of the paper's aside that a single-camera product "would
also be possible... using a threshold based on particle blur to define the
observation volume, similar to the PIP" — it exists, and is size-dependent
rather than a single global blur cutoff.

Binning
--------

The core of ``_createLevel2part`` groups particles into one-minute time
bins (``groupby_bins`` on ``time``) and, within each, into ``Dmax``/
``Dequiv`` pixel bins (``DbinsPixel``, default 1px steps 0–300) via
``groupby_bins`` again — directly matching the paper's PSD binning
description. Both ``level2match``/``level2track`` use camera- or
track-reduced values (see ``applyFilters`` above and
:func:`VISSSlib.distributions.getPerTrackStatistics` for the track-based
min/max/mean/std reduction along a track) as the basis for statistics.

Observation volume: mesh intersection, not OpenSCAD
---------------------------------------------------------

The paper states the leader/follower observation-volume intersection is
computed "using the OpenSCAD library"; the actual implementation
(:func:`VISSSlib.distributions.createLeaderBox`,
:func:`VISSSlib.distributions.createFollowerBox`,
:func:`VISSSlib.distributions._createBox`,
:func:`VISSSlib.distributions._estimateVolume`) instead builds each
camera's observation volume as an 8-vertex box with :mod:`trimesh`
(boolean-intersection backend ``manifold3d``, an ``installation.rst``
dependency) and computes ``leader.intersection(follower).volume`` directly
— conceptually the same "rotate follower's box into the leader frame, then
intersect" approach the paper describes (follower vertices are transformed
via :func:`VISSSlib.matching.shiftRotate_F2L` using the retrieved
``camera_phi``/``camera_theta``/``camera_Ofz``), just a different concrete
library than what's written up. Worth knowing if you're trying to reproduce
the paper's numbers from the library alone. :func:`VISSSlib.distributions._estimateVolumes`
(``@functools.cache``d, since the same camera geometry is reused across many
size bins/cases) computes this per size bin, shrinking each box's edges to
account for :math:`D_\text{max}`-dependent partial-particle exclusion, per
the paper's ``effective observation volume reduced by`` :math:`D_\text{max}/2`
description.

``tests/test_distributions.py::TestVolume`` is a good sanity check to know
about if you're touching this geometry: ``test_volumeEstimate`` asserts
that with **zero** rotation/offset, the intersection volume collapses to
exactly ``width * width * height`` (the cameras observe the same cuboid
directly), and ``test_VolumeInterpolation`` asserts that
``_estimateVolumes``' size-bin interpolation (computing the expensive
mesh intersection at only ``nSteps`` bins and interpolating the rest, for
performance) stays within 1% of computing every bin exactly — i.e. the
interpolation shortcut is verified numerically, not just assumed safe.

Track time and the border filter
-------------------------------------

Particles too close to the image border are removed before the track
statistics are estimated (see ``farEnoughFromBorder``). This can remove the
*first* step of a track. The time of a track (used to assign it to a one-minute
bin) is the time of its first **remaining** step,
:func:`VISSSlib.distributions.getPerTrackStatistics`. Previously, the time of
``track_step=0`` was used, which was missing for such tracks whenever other
tracks in the same data had a ``track_step=0``; these tracks were then silently
dropped. Whether a track survived thus depended on the other tracks processed
together with it, which mainly showed for rare classes (few tracks) and made
the results of :func:`~VISSSlib.distributions.createLevel2_multiple_classes`
depend on how files were grouped. The fix increases the number of tracks
in ``level2track`` (about 3.5% for the day tested in Hyytiälä, 2023-12-25).

Optional shape variables
----------------------------

Newer Level 1 files contain the additional shape variables
``areaConsideringHoles``, ``perimeterConsideringHoles``, ``solidity`` and
``extent``. They are processed like ``area`` and ``perimeter`` (giving
``*_dist``, ``*_mean`` and ``*_std`` in Level 2) **if they are present**
(:func:`VISSSlib.distributions._optionalShapeVars`); older Level 1 files
(e.g. ``V1.0``) without them are processed without these variables. If
``areaConsideringHoles`` is available, ``Dequiv`` is derived from it,
otherwise from ``area``.

Per-class Level 2 distributions
-------------------------------------

Level 2 can also be created for Level 1 data that has been split into
particle classes (e.g. riming classes from a classifier), so that PSDs and
all other statistics are available separately for each class. In contrast
to :func:`~VISSSlib.distributions.createLevel2match` and
:func:`~VISSSlib.distributions.createLevel2track`, these functions work on
an already loaded Level 1 dataset that is handed in by the caller: no files
are searched, read or written, and the data quality variables (which
require the complete Level 1 record) are not added.

Workflow:

1. Load the Level 1 data of the period of interest (``match`` or ``track``)
   and add a per-particle variable ``category`` with the class name for
   every ``pair_id``. If the classes are available per track, e.g. as a
   ``track_id -> class`` dictionary as stored in the ``level1class`` files,
   :func:`VISSSlib.distributions.attachTrackCategories` does this; tracks
   without a class are put in the class ``"unclassified"``.
2. Call :func:`VISSSlib.distributions.createLevel2_multiple_classes`. It
   splits the data by ``category``, processes every class with
   :func:`VISSSlib.distributions.createLevel2_single_class` and combines the
   results along the additional dimension ``particle_class``.

.. code-block:: python

    import xarray as xr
    import VISSSlib
    from VISSSlib.distributions import (
        attachTrackCategories,
        createLevel2_multiple_classes,
    )

    config = VISSSlib.tools.readSettings("settings.yaml")
    level1 = xr.load_dataset(level1trackFile)
    level1 = attachTrackCategories(level1, trackClasses)

    level2 = createLevel2_multiple_classes(level1, config, sublevel="track")
    level2.PSD.sel(particle_class="stellar aggregate")

Multiple Level 1 files
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both functions also accept a **list** of datasets and/or file names (e.g. all
10 minute ``level1track`` files of an hour or a day) and create a single
Level 2 dataset from them. The data is combined by
:func:`VISSSlib.distributions._combineLevel1`, which

* makes ``pair_id`` unique, and adds a per-file offset to ``track_id``,
  because ``track_id`` restarts at zero in every Level 1 file (otherwise
  different particles of different files would be merged into one track);
* drops variables that are not available in all files;
* raises a ``ValueError`` if the files do not fit together, e.g. because they
  were observed by different cameras.

Because ``track_id`` is only unique within a file, the classes have to be
attached to every file *before* combining them:

.. code-block:: python

    level1 = [
        attachTrackCategories(xr.load_dataset(f), trackClasses[f])
        for f in level1trackFiles
    ]
    level2 = createLevel2_multiple_classes(level1, config, sublevel="track")

The result is identical to processing the files one by one and joining the
results in time (checked for the counts and the number of tracks of all
classes for a full day of 142 files). It is, however, considerably faster
for many files (31 s instead of 120 s in that test). All data is held in
memory at once, so use hourly or daily chunks rather than months.

Things to know:

* **Common time grid**: the time steps are derived once from *all*
  particles and used for every class. A class without particles in a
  minute has zero ``counts`` (and ``nParticles``) there, and NaN for the
  mean and standard deviation variables, rather than a shorter time axis.
  Alternatively, pass ``timeIndex`` (left edges of the time steps) to
  :func:`~VISSSlib.distributions.createLevel2_single_class` to control the
  grid yourself.
* **Same processing as the standard products**: both functions use the same
  filtering, binning and calibration code as ``createLevel2match``/
  ``createLevel2track`` (:func:`VISSSlib.distributions._level2FromLevel1`),
  and accept the same ``freq``, ``DbinsPixel``, ``sizeDefinitions``,
  ``camera`` and ``applyFilters`` arguments.
* **Empty classes** (no particles left after filtering) are omitted with a
  warning; ``None`` is returned if no class has data.
* **Tracking completeness**: if the ``level1track`` files contain
  ``track_expectedLength`` (newer tracker), the result also has
  ``track_completeness``, as in ``level2track``. Older files without it can
  still be used; the result then simply has no ``track_completeness``. The
  quality flags themselves (``qualityFlags``) are not part of the per-class
  products.
* ``D_bins_left`` and ``D_bins_right`` are identical for all classes and do
  not have the ``particle_class`` dimension.
* Missing values in ``category`` are treated as one class named by the
  argument ``unclassified``.

``VISSSlib.distributions`` API
----------------------------------

.. automodule:: VISSSlib.distributions
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
