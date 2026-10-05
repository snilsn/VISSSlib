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
The initial covariance uses the measurement variance for the (measured)
position and the learned first-guess uncertainty (``velocitySigma``, see below)
for the velocity, so the first update moves the whole first-step innovation
into the velocity. The process noise has one ``(position, velocity)`` block per
axis. A track that goes undetected coasts on its prediction without a KF
update, so its uncertainty keeps growing. ``track_step`` counts real
observations only.

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
   that exceeds ``max_frames_to_skip`` (default 1, i.e. a single missed
   detection does not end a track); unassigned detections start new tracks.
   A frame that is missing from the data altogether (no particle in the
   whole volume) is bridged the same way; the Kalman filters are propagated
   over it before coasting. Links across a missed detection are the most
   error-prone ones (a track whose particle left the volume would grab any
   particle near its prediction), so a track only coasts while its
   prediction is inside the observation volume (``coastOnlyIfVisible``) and
   a coasted track gets an extra cost (``coastedExtraCost``, 1), so it loses
   against tracks observed in the last frame.

Gates learned from the data
----------------------------

All distances the tracker compares against are learned from recently
finished tracks (within ``maxAge4training``, 120 s in
:func:`~VISSSlib.tracking.trackParticles`) instead of being fixed pixel
values, so the same code works for VISSS1/2/3 with their different frame
rates and resolutions, and follows changes in turbulence:

* ``dvScale``: median frame-to-frame velocity change of tracks with at least
  4 observations. The gate for tracks with 2 (>= 3) observations is
  ``gate2Factor`` (``gate3Factor``) times ``dvScale``; it replaces the former
  fixed ``costExperiencePenalty``.
* ``velocitySigma``: robust spread (per axis) of the difference between the
  velocity of kinematically clean tracks and the first-guess velocity they
  started with. Only clean tracks are used because wrong links would inflate
  it, widen the gates and cause even more wrong links.

Until enough tracks were seen the class defaults are used; the training pass
(see below) only ends when both scales were learned.

First links (look-ahead)
--------------------------

The first link of a new track is the ambiguous one: from a single detection
the velocity is only known from the first guess, whose error is dominated by
turbulence. Instead of trusting it, a candidate ``b`` for a track that started
at ``a`` is *confirmed* if the next frame contains a detection within
``lookaheadRadiusFactor * dvScale`` of ``2b - a`` (or the frame after, for a
missed detection). A confirmed candidate may be far from the first guess (up
to ``wideFirstLinkFactor * |velocitySigma|``) because a wrong partner is very
unlikely to be confirmed too. If the extrapolated position would not be
observed (leaves the observation volume, or the next frame is not in the
data) the first-guess gate ``firstLinkFactor * |velocitySigma|`` applies,
widened at low particle concentration to the radius within which only
``falseCandidateProbability`` (1 %) random particles are expected (capped at
``maxFirstLinkGateFraction`` of the frame width), because without competitors
a wide gate costs nothing; this keeps fast, gust-driven particles that are
seen only twice. A
checkable but unconfirmed candidate is still allowed with a tighter gate and
an extra cost, so any confirmed alternative wins.

The observation volume (:meth:`VISSSlib.tracking.Tracker.visible`) is the
intersection of both camera images with the retrieved follower rotation, as
in :func:`~VISSSlib.distributions.estimateObservationVolume`, shrunk by half
the particle size, plus depth limits along both viewing directions learned
from where particles are actually observed (focus, illumination and partial
blocking of a window are not part of the camera boxes).

New tracks start with the *local flow* as first guess when at least two
established tracks are active: their median horizontal velocity, and their
median deviation from the size-velocity relation added to it.

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

Tracking completeness
----------------------

When a track ends, :meth:`VISSSlib.tracking.Tracker._finishTracks` writes
``track_expectedLength`` to all its particles: the observed span plus the
number of frames the particle would still have been visible before its first
and after its last observation, extrapolated with its mean velocity (the first
guess for single observations) through the observation volume
(:meth:`~VISSSlib.tracking.Tracker.visible`). level2track sums observed and
expected observations of all tracks starting in a time bin;
``track_completeness`` is their ratio and the ``trackingIncomplete`` quality
flag is set below ``config.quality.minTrackCompleteness`` (default 0.25). It
replaces ``tracksTooShort`` (same bit), which compared the mean track length
with a fixed threshold and could not tell fast particles that cross the volume
in a few frames from poorly tracked ones, nor detect wrongly merged tracks.

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
