Calibration
===========

Where the numbers come from
----------------------------------

``config.calibration.slope`` (and ``slope_err``, see :doc:`config_files`
and ``sample.yaml``) converts pixel units to metric ones and is obtained
empirically per instrument, not computed from the lens specification
alone: reference steel/ceramic spheres of known diameter are dropped
through the observation volume, processed with the normal VISSS routines,
and a linear least-squares fit of observed vs. expected :math:`D_\text{max}`
gives the slope. `Maahn et al. (2024) <https://amt.copernicus.org/articles/17/899/2024/>`_
report, from 604 reference-sphere observations at Hyytiälä (SAIL), a
VISSS1 slope of :math:`0.01700\pm0.00001` px/μm (inverse: 58.83 μm/px,
close to the 58.75 μm/px manufacturer spec) and, from 372 samples at
Ny-Ålesund, a VISSS2 slope of :math:`0.02311\pm0.00003` px/μm (inverse:
43.27 μm/px vs. 43.125 μm/px spec).

**Why the intercept is dropped:** the raw sphere-calibration fit has a
non-zero intercept, but testing against artificial images showed spheres
are *overestimated* and squares are *underestimated* by the detection
routine (Gaussian-blur-induced rounding of corners on squares vs.
apparent enlargement of spheres) — real particle shapes fall anywhere
between the two, so forcing a real, shape-dependent intercept would bias
one shape family in favor of the other. VISSSlib instead calibrates with
the intercept fixed to 0, accepting a shape-dependent bias of roughly
±4–6% (somewhat larger below 10 px due to discretization).

**Perimeter uses the same slope as Dmax, not its own.** The sphere-derived
perimeter slope is about 5% steeper than the :math:`D_\text{max}` slope,
but that steepening is an artifact of the Canny-edge blur (also present
for artificial spheres) rather than something expected to hold for real,
complex particle shapes, where perimeter increases with decreasing scale
at fixed area (compare to the coastline paradox). Applying a
perimeter-specific slope would therefore likely overcorrect, so VISSSlib
reuses the :math:`D_\text{max}` slope for perimeter too, at the cost of a
larger uncertainty on perimeter than on :math:`D_\text{max}`/:math:`D_\text{eq}`.

:math:`D_\text{eq}` (and consequently area :math:`A`) calibrate almost
identically to :math:`D_\text{max}`, so no separate slope is needed for
those either.

Applying the calibration
------------------------------

Level 1 data (``level1detect``/``level1match``/``level1track``) is still
in pixel units. To calibrate manually: divide pixel quantities
(``Dmax``, ``perimeter``, etc.) by ``config.calibration.slope``, and
pixel\ :sup:`2` quantities (``area``) by ``config.calibration.slope**2``.
Level 2/3 products (see :doc:`distributions`) are already calibrated to
metric units by :func:`VISSSlib.distributions.calibrateData` as part of
processing — no manual conversion needed there.
