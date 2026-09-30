``av`` - video reading
==========================

Motivation
----------

Raw level0 video is split across multiple files per camera when
``config.nThreads`` is set (the acquisition side runs several parallel
encoding threads for the higher frame rates of VISSS2/3, see
:doc:`data_acquisition`), and per-frame metadata (``capture_time``,
``record_id``, which thread) lives separately in ``metaFrames``, not in the
video container. ``av.py``'s job is to hide both of these facts and let
callers read "the frame observed at capture_time X" without knowing which
physical video file or frame index that maps to.

``VideoReaderMeta`` — the class actually used elsewhere
--------------------------------------------------------------

:class:`VISSSlib.av.VideoReaderMeta` (used throughout :doc:`quicklooks`
and :doc:`analysis`) opens **one low-level reader per thread**
(``self.video[thread]``) on construction, then
:meth:`~VISSSlib.av.VideoReaderMeta.getFrameByCaptureTime` looks up the
requested ``capture_time`` in ``metaFrames`` to find which thread and
``record_id`` (frame index within that thread's own video file) it
corresponds to, and delegates to that thread's reader. It optionally also
holds a ``level1detect``/``level1match`` dataset and the ``imagesL1detect``
:class:`VISSSlib.tools.BlockImageArchive`, so
:meth:`~VISSSlib.av.VideoReaderMeta.getFrameByCaptureTimeWithParticles` can
overlay detected/matched particle boxes on a frame (or fetch a pre-cropped
particle image straight from the archive instead of re-cropping from the
full frame) — this is the machinery behind the particle-annotated
quicklooks.

The low-level reader (``create_VideoReader``)
--------------------------------------------------

The actual ``cv2.VideoCapture``-based reader is not a module-level class
but built by :func:`VISSSlib.av.create_VideoReader()`, called lazily inside
:meth:`VideoReaderMeta._openVideo` rather than at import time — deferring
the ``cv2`` import and avoiding constructing the class before it's needed.
Its ``getFrameByIndex`` is ``functools.lru_cache``d and supports a
``safeMode`` that only ever seeks forward (fast-forwarding by reading
frames rather than seeking) — a comment right above
``VideoReaderMeta`` in the source (**"can cause segfaults!"**) is a live
warning about caching bound methods of a ``cv2.VideoCapture`` subclass;
keep that in mind before changing the caching strategy here.

Small helpers
--------------

:func:`VISSSlib.av.doubleDynamicRange` (used by
:meth:`VISSSlib.detection.detectedParticles.applyCannyFilter` when
``config.level1detect.doubleDynamicRange`` is set) doubles image contrast
by subtracting an estimated offset and multiplying by 2 — the factor of 2
specifically so integer pixel gradients scale exactly, and the offset is
chosen so the brightest pixel still saturates at 255 rather than
overflowing. :func:`VISSSlib.av.cvtColor` / :func:`VISSSlib.av.cvtGray`
are the grayscale-conversion helpers used when reading frames.

``VISSSlib.av`` API
-----------------------

.. automodule:: VISSSlib.av
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
