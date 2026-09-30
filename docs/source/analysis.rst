``analysis`` - interactive stereo-view inspection
======================================================

Motivation
----------

Unlike every other module covered so far, ``analysis.py`` is not part of
the batch processing DAG (:doc:`products`) — it has no CLI subcommand and
:func:`VISSSlib.products.DataProduct.generateCommands` never references
it. It is a set of ``ipywidgets``-based interactive viewers meant to be
used from a Jupyter notebook (see ``notebooks/match_estimate_rotation.ipynb``)
for visually inspecting and — where automatic matching fails — manually
correcting leader/follower particle correspondence for a single case
(one 10-minute file). Four classes, two GUI/backing pairs:

- :class:`VISSSlib.analysis._stereoViewMatch` opens both cameras'
  video (via :class:`VISSSlib.av.VideoReaderMeta`) plus level1detect for one
  case and steps through frames, showing matched leader/follower particle
  pairs side by side (optionally with detection boxes, contrast
  enhancement, or track overlays). :class:`VISSSlib.analysis.matchGUI` is
  the ``ipywidgets`` wrapper around it — this is the tool for visually
  confirming the automatic matching/rotation retrieval actually looks
  right for a given period, e.g. after installing or adjusting an
  instrument.
- :class:`VISSSlib.analysis._stereoViewDetect` is the equivalent for
  *unmatched* per-camera detections (no assumption that automatic matching
  succeeded), and :class:`VISSSlib.analysis.manualMatchGUI` wraps it for
  **manually** pairing up particles by hand when the automatic capture-id
  offset estimation fails — the paper notes this was sometimes needed for
  MOSAiC data (drifting clocks, acquisition bugs); see also
  :func:`VISSSlib.matching.manualRotationEstimate` and the "Manual
  adjustments" walkthrough in :doc:`metaRotation` for the corresponding
  non-interactive/scriptable recovery path.

If you're debugging why matching or rotation retrieval looks wrong for a
specific period, this module — run interactively against that one case —
is usually a faster way to see what's going on than reading netCDF output
numerically.

``VISSSlib.analysis`` API
-----------------------------

.. automodule:: VISSSlib.analysis
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
