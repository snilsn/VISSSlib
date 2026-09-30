``detection`` - single-camera particle detection
====================================================

Motivation
----------

``detection.py`` implements the CV pipeline described in the "Particle
Detection" section of `Maahn et al. (2024) <https://amt.copernicus.org/articles/17/899/2024/>`_:
turning raw per-camera video
frames into the ``level1detect`` product (per-particle geometric/brightness
properties in pixel units, one camera at a time — stereo combination
happens later in :doc:`matching`). Two classes do the work:
:class:`VISSSlib.detection.detectedParticles` is the per-frame, stateful
detector (owns the OpenCV background subtractor and accumulates results
across a whole file); :class:`VISSSlib.detection.singleParticle` computes
the actual geometric/brightness properties for one detected contour.

Per-frame detection (``detectedParticles``)
----------------------------------------------

:meth:`~VISSSlib.detection.detectedParticles.update` runs once per video
frame and follows the same steps as the paper: apply
``cv2.BackgroundSubtractorKNN`` (``config.level1detect.backSub`` /
``backSubKW``) to get a moving-pixel mask, bail out early if nothing moved,
optionally dilate/erode the mask to close small internal gaps, then dilate
again specifically to make ``cv2.findContours`` more robust before iterating
over the found contours via
:meth:`~VISSSlib.detection.detectedParticles.add`.

:meth:`~VISSSlib.detection.detectedParticles.add` first rejects contours
whose bounding box touches the frame border (a partially-observed particle
can't be sized correctly). For the rest, when
``config.level1detect.applyCanny2Particle`` is set (the default),
:meth:`~VISSSlib.detection.detectedParticles.applyCannyFilter` re-runs edge
detection on just a padded ROI around the particle (Gaussian blur, then
``cv2.Canny``, then dilate/fill-contour/erode to close small gaps in the
detected edges) — this is the refinement step that lets VISSS resolve
finer particle structure than the raw background-subtraction mask alone
would give, matching the paper's description of the 1px dilation being
enough. Child contours (holes inside the dilate/fill/erode result) are kept
as real particle holes rather than filled in, unless
``check4childCntLength`` decides they're small enough to be noise.

Per-particle properties (``singleParticle``)
------------------------------------------------

Each accepted contour becomes one :class:`VISSSlib.detection.singleParticle`,
which computes — directly mirroring the paper's equations —
:math:`D_\text{max}` via ``cv2.minEnclosingCircle``, angle/aspect ratio
three different ways (``cv2.minAreaRect``, ``cv2.fitEllipse``, and the more
numerically stable ``cv2.fitEllipseDirect``, kept as separate variables
rather than one "winning" method since downstream products can choose),
perimeter/area from the contour, and brightness statistics (min/max/mean/
percentiles/std/skew/kurtosis) over the masked pixel values.
``perimeterEroded`` is a diagnostic: eroding the particle mask by one pixel
and re-measuring perimeter distinguishes a properly-filled 2D particle
shape from a mask that only describes a thin line (i.e. detection failure),
since a line's perimeter collapses under erosion while a filled shape's
doesn't.

``checkMotion`` — parity with the acquisition-side C code
-------------------------------------------------------------

:func:`VISSSlib.detection.checkMotion` is explicitly documented as
"identical to VISSS C code" — it re-implements, in Python, the same
simple absolute-frame-difference motion threshold that the *data
acquisition* software (a separate C codebase, see :doc:`data_acquisition`)
uses in real time to decide whether a frame is worth saving at all. Its
presence here is for consistency checking / offline reprocessing logic
that needs to agree with what the acquisition side already decided, not
part of the main detection pipeline itself.
``config.level1detect.minMovingPixels`` / the module-level ``threshs``
array (``[20, 30, 40, 60, 80, 100, 120]`` by default) are the corresponding
per-brightness-threshold pixel-count cutoffs.

Entry point
------------

:func:`VISSSlib.detection.detectParticles` is the per-level0-file entry
point (one file in, one ``level1detect`` netCDF out), decoded from the raw
video via :mod:`VISSSlib.av`. Like :func:`VISSSlib.matching.matchParticles`,
it resolves its own output path via :class:`VISSSlib.files.Filenames`,
honors ``skipExisting`` via :func:`VISSSlib.tools.checkForExisting`, and
writes through :func:`VISSSlib.tools.finishNc` /
:func:`VISSSlib.tools.to_netcdf2`.

``VISSSlib.detection`` API
-----------------------------

.. automodule:: VISSSlib.detection
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
