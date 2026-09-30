``tracking`` - frame-to-frame particle tracking
===================================================

Motivation
----------

``tracking.py`` implements the "Particle Tracking" step from
`Maahn et al. (2024) <https://amt.copernicus.org/articles/17/899/2024/>`_:
linking matched (``level1match``) particle
observations across consecutive frames into ``level1track`` tracks, from
which sedimentation velocity is derived and per-particle property estimates
can be improved by combining multiple observations of the same particle. As
the paper describes, this is a Kalman-filter-predicted-position plus
Hungarian-algorithm assignment problem, but the actual implementation
generalizes the cost function and the velocity first-guess mechanism
somewhat beyond what the paper's high-level description covers (see below).

Kalman filter (``myKF`` / ``Track``)
----------------------------------------

:func:`VISSSlib.tracking.myKF` sets up a standard constant-velocity Kalman
filter with state vector ``[x, vx, y, vy, z, vz]`` (``dt=1`` frame),
measuring only position. :class:`VISSSlib.tracking.Track` wraps one such
filter per tracked particle, keeping the full position trace
(``self._trace``, with ``[nan, nan, nan]`` appended for frames where no
detection was assigned rather than dropping the frame index) plus a
feature-vector and size history used by the cost function below.

Assignment (``Tracker.update``)
------------------------------------

:meth:`VISSSlib.tracking.Tracker.update` runs once per frame:

1. Predict each active track's next position via its Kalman filter.
2. Build a cost matrix from **all** configured features, not just position:
   for each ``(track, detection)`` pair, the squared difference is computed
   per feature in ``featureVariance`` (position distance plus, by default,
   ``Dmax``; :func:`~VISSSlib.tracking.trackParticles`'s default is
   ``{"distance": 200**2, "Dmax": 1}``), each normalized by its configured
   variance, then averaged. This generalizes the paper's "cost derived from
   the product of :math:`\delta l` and :math:`\delta A`" description — the
   actual implementation is an inverse-variance-weighted mean over an
   arbitrary feature set, of which position and one size-like variable are
   the defaults. Costs above ``dist_thresh`` are set to a very large value
   before assignment rather than left as-is, because ``scipy``'s Hungarian
   solver (``linear_sum_assignment``) can otherwise pick a globally cheaper
   but individually nonsensical assignment (see the worked example in a
   code comment in ``Tracker.update``).
3. ``scipy.optimize.linear_sum_assignment`` solves the assignment; pairs
   whose actual cost still exceeds ``dist_thresh`` are un-assigned again
   after the fact.
4. Unassigned tracks accumulate ``skipped_frames`` and are archived once
   that exceeds ``max_frames_to_skip``; unassigned detections start new
   tracks.

``costExperiencePenalty`` inflates the cost for longer-established tracks
(index into the array is track length, e.g. default
``[1, 1, 6, 9, 9, 9, 9, ...]``) — a track that has proven itself over
several frames is held to a *stricter* matching tolerance, not a looser
one, presumably to avoid a well-established, confidently-predicted track
drifting onto a nearby but different particle.

Velocity first guess
----------------------

The paper describes deriving a first-guess velocity from ~200 previously
tracked particles (or running the algorithm twice on the first 400 if none
are available yet). The actual mechanism in
:meth:`VISSSlib.tracking.Tracker.updateVelocityFirstGuess` is a live-updated
**power-law fit** between particle size and fall speed,
``log10(v) = slope * log10(size) + intercept``, refit periodically (every
500 ms, or whenever too little recent history exists) from the archive of
recently completed tracks that meet a minimum length/recency/positive-
velocity filter. Module-level ``_reference_slopes`` /
``_reference_intercepts`` dicts (per ``visssGen``, keyed by the size
variable used — ``area`` or ``pixSum``) provide the fallback when there
isn't yet enough archived data to fit; ``costGuessFactor`` correspondingly
loosens the assignment cost while running on defaults and tightens once a
live fit is available.

Entry point
------------

:func:`VISSSlib.tracking.trackParticles` is the per-``level1match``-file
entry point. It filters to ``matchScore >= minMatchScore`` before tracking
(quality threshold from the paper's match-score cutoff discussion), and can
optionally trigger :func:`VISSSlib.matching.matchParticles` itself via
``doMatchIfRequired`` if the ``level1match`` file doesn't exist yet — see
the tuple-unpacking fix in this repository's git history for a cautionary
note about keeping that call site's unpacking in sync with
``matchParticles``'s return signature.

``VISSSlib.tracking`` API
----------------------------

.. automodule:: VISSSlib.tracking
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
