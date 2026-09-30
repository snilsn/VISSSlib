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

``VISSSlib.distributions`` API
----------------------------------

.. automodule:: VISSSlib.distributions
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
