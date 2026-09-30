Configuration files
===================

Every deployment (one site, for one period) is described by a single YAML
settings file (see ``sample.yaml`` in the repository root for a complete,
annotated example). How the file is loaded and merged is covered in
:doc:`tools` (:func:`VISSSlib.tools.readSettings`) — this page is about
what to put *in* it.

Required top-level keys
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Key
     - Description
   * - computers
     - List of computer hostnames where the data was collected (one per
       camera, same order as ``leader``/``follower``)
   * - fps
     - Camera frame rate (frames per second)
   * - frame_height
     - Image height in pixels
   * - frame_width
     - Image width in pixels
   * - leader
     - Camera ID for the leader camera (e.g. ``leader_S1145792``)
   * - follower
     - Camera ID for the follower camera
   * - nThreads
     - Number of parallel acquisition threads per camera, or ``null`` for
       single-threaded acquisition (changes the level0 filename pattern,
       see :doc:`files`)
   * - path
     - Base path template for level0 (raw) input data; contains a
       ``{level}`` placeholder
   * - pathOut
     - Base path template for processed output data; also templated with
       ``{level}`` (and ``{version}``, ``{site}``)
   * - pathQuicklooks
     - Base path template for quicklook images; same templating
   * - visssGen
     - Instrument generation (``visss``, ``visss2``, ``visss3``) — several
       modules (e.g. :doc:`tracking`'s size-velocity reference table)
       branch on this
   * - site
     - Deployment site identifier, conventionally three letters
   * - start
     - Start date of the deployment period (inclusive)
   * - end
     - End date of the deployment period (inclusive), or the literal
       string ``"today"`` for an ongoing deployment
   * - name
     - Human-readable deployment name
   * - model
     - Camera model name, e.g. ``M1280``

``path``/``pathOut``/``pathQuicklooks`` are resolved relative to the
settings file's own directory if not already absolute, and ``$HOSTNAME``
in any of them is expanded to the current machine's hostname (useful when
the same settings file is shared across acquisition and processing
machines with different local data roots).

.. note::

   ``sample.yaml`` also sets a top-level ``resolution`` key, but no code
   in VISSSlib actually reads it — :func:`~VISSSlib.tools.readSettings`
   warns about it on every load ("not in the default settings and might
   be unused"). The real pixel-to-metric conversion is
   ``calibration.slope`` below, not ``resolution``. Likely a leftover from
   an earlier config layout; safe to omit in a new settings file.

Optional settings groups
------------------------------

These nest under their own top-level key and default to the values in
:data:`VISSSlib.tools.DEFAULT_SETTINGS` if omitted — only override what a
given deployment actually needs to change.

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Key
     - Description
   * - calibration
     - ``slope``/``slope_err``: the pixel-to-metric conversion factor,
       obtained empirically per instrument — see :doc:`calibration`.
   * - rotate
     - Per-period camera misalignment priors
       (``camera_phi``/``camera_theta``/``camera_Ofz`` and their errors,
       keyed by ``YYYYMMDD-HHMMSS``), used as a fallback before a
       ``metaRotation`` product exists for a period — see :doc:`metaRotation`.
   * - dataFixes
     - List of named workarounds to enable for this deployment's known raw
       data issues (e.g. ``captureIdOverflows``, ``makeCaptureTimeEven``) —
       see :doc:`fixes` for which names actually do something.
   * - quality
     - Thresholds used across Level 1/2 processing to flag/exclude data:
       blocked-camera and blowing-snow pixel ratios, the minimum
       :doc:`matching` match score, minimum track length, etc.
   * - level1detect
     - Detailed knobs for the particle-detection CV pipeline (background
       subtractor parameters, size/blur/contrast thresholds, ...) — see
       :doc:`detection`. Rarely needs changing per deployment beyond the
       defaults.
   * - level1match / level2
     - ``processL1match``/``processL2detect`` toggle whether the
       stereo-matching branch and the single-camera Level 2 product are
       produced at all for this deployment (e.g. skip matching entirely
       for a single-camera setup); ``level2.freq`` sets the Level 2 time
       resolution (default ``"1min"``).
   * - level3.combinedRiming
     - ``processRetrieval`` toggle, plus the quality gate for running it
       (``maxTemp``, ``minZe``, ``minNParticles``) — see :doc:`level3`.
   * - aux
     - Where to fetch the external radar/meteo data the riming retrieval
       needs (``aux.radar.source``, ``aux.meteo.source``, and
       per-source paths/credentials) — see :doc:`level3`.
   * - logo
     - Path to an image overlaid on quicklook figures
       (:func:`VISSSlib.tools.savefig`); omit for no logo.
   * - movieExtension, newFileInt, dirMode, fileMode, goodFiles, badData
     - Lower-level knobs (raw video container format, expected seconds
       per level0 file, created-file permissions, a manual
       known-good-files/bad-data-period override) — see the docstrings in
       :doc:`tools` (:data:`VISSSlib.tools.DEFAULT_SETTINGS`) for exact
       defaults and semantics.
