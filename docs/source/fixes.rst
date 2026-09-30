``fixes`` - campaign-specific data patches
==============================================

Motivation
----------

``fixes.py`` collects targeted workarounds for known artifacts in specific
historical data periods (mostly MOSAiC-era hardware/timing issues, see the
top-of-file comment block in :doc:`metadata`). It's a lookup table more
than an algorithm module — treat it as reference material to check when a
new deployment surfaces a *new* raw-data anomaly, not something to
understand line-by-line up front.

Which fixes are actually live
-----------------------------------

Most fixes are opt-in per deployment via the settings file's ``dataFixes``
list (see :doc:`config_files`), checked with e.g. ``if "captureIdOverflows"
in config.dataFixes:`` at the call sites (in :mod:`VISSSlib.tools`'s
``open_mf*`` family and in :mod:`VISSSlib.matching`) — cross-checking which
fix names are actually referenced anywhere outside this file:

- **Live, opt-in**: :func:`VISSSlib.fixes.captureIdOverflows` (16-bit
  ``capture_id`` wraparound on M1280 cameras),
  :func:`VISSSlib.fixes.makeCaptureTimeEven` (follower-only clock drift),
  and :func:`VISSSlib.fixes.makeCaptureTimeEvenBothCameras`
  (leader-*and*-follower clock drift ahead of leader-follower capture_id
  offset estimation in :func:`VISSSlib.matching._resolveMatchingOffset`
  -- for deployments where the *leader's* own onboard clock drifts too,
  not just the follower's), all gated on ``config.dataFixes``.
  :func:`VISSSlib.fixes.revertIdOverflowFix` undoes the id-overflow
  correction again right before :func:`VISSSlib.matching.matchParticles`
  writes its output, so the *shifted* ids used internally for matching
  don't leak into the product and confuse anyone comparing against the
  raw ``capture_id``.
- **Live, unconditional**: :func:`VISSSlib.fixes.delayedClockReset` is
  called directly from :mod:`VISSSlib.metadata` without a ``dataFixes``
  check — it always runs.
- **Dead code**: :func:`VISSSlib.fixes.fixMosaicTimeL1` has zero callers
  anywhere in the codebase, and its own docstring says so explicitly:
  *"This is a poor attempt at fixing drift and is not used anymore."*
  :func:`VISSSlib.fixes.removeGhostFrames` is referenced only inside a
  commented-out block in :func:`VISSSlib.metadata.getMetaData` ("unclear
  whether it works"). Both are candidates for removal if you're doing
  cleanup, but are left as-is here since neither was in scope for a
  documentation pass.

``VISSSlib.fixes`` API
--------------------------

.. automodule:: VISSSlib.fixes
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
