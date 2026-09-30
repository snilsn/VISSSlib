``files`` - filename patterns and path resolution
====================================================

Motivation
----------

Every level in the processing DAG (see :doc:`processing`) has its own
directory layout and filename convention. ``files.py`` is the single place
that encodes those conventions, in both directions:

- **Forward** — given a case, camera, and level, where should output files
  go / be searched for? (:class:`VISSSlib.files.FindFiles`)
- **Backward** — given an existing file (typically a level0 raw video file,
  or a level1/level2 product), what case/camera does it belong to, and what
  are the corresponding paths for every *other* level?
  (:class:`VISSSlib.files.Filenames` / :class:`VISSSlib.files.FilenamesFromLevel`)

Almost nothing outside ``files.py`` constructs a VISSS file path by hand;
callers ask one of these classes instead. This is what makes it possible to
change the on-disk layout (or add a new level) in one place.

Level classification
---------------------

Four module-level lists classify how a level's files are organized, and
several methods across ``files.py``/``tools.py`` branch on membership in
these lists rather than hardcoding level names:

- ``fileLevels`` — per-10-minute-file products (``level1detect``,
  ``level1match``, ``level1track``, ``metaFrames``, ``metaDetection``,
  ``imagesL1detect``): one output file per level0 input file.
- ``dailyLevels`` — daily aggregate products (``metaEvents``,
  ``metaRotation``, ``level2detect``, ``level2match``, ``level2track``,
  ``level3combinedRiming``, ``allDone``): one output file per case (day),
  independent of how many level0 files exist.
- ``quicklookLevelsSep`` / ``quicklookLevelsComb`` — which levels get a
  per-camera quicklook image vs. a single combined one.
- ``imageLevels`` — currently just ``imagesL1detect``, the one level stored
  as a binary image archive (:class:`VISSSlib.tools.BlockImageArchive`)
  rather than netCDF.

This distinction matters for completeness checking
(:meth:`VISSSlib.files.FindFiles.nMissing`): for a ``fileLevel``, the
expected count is the number of level0 files for that case+camera; for a
``dailyLevel``, the expected count is just 1 (or the number of cases, for a
:class:`~VISSSlib.files.FindFilesRange`).

``FindFiles``
--------------

:class:`VISSSlib.files.FindFiles` takes a ``case`` (``YYYYMMDD`` or
``YYYYMMDD-HHMMSS``), ``camera`` (``"leader"``/``"follower"`` or an explicit
camera id), and ``config``, and builds glob patterns
(``self.fnamesPattern[level]``) for every level by string-formatting
``config.path``/``config.pathOut``/``config.pathQuicklooks`` (which
themselves contain a ``{level}`` placeholder). ``listFiles(level)`` /
``listFilesExt(level)`` (the latter also matching ``.broken.txt``/``.nodata``
sentinel files) glob these patterns. ``isComplete``/``nMissing`` compare the
count of present files against the expected count (see above) to answer "has
this level been fully processed for this case yet" — this is the
file-existence half of what :class:`VISSSlib.products.DataProduct` uses to
decide whether a product needs (re)generating; the other half is the mtime
check in :func:`VISSSlib.tools.checkForExisting`.

:class:`VISSSlib.files.FindFilesRange` wraps a list of per-case
``FindFiles`` instances and forwards attribute/method access to all of them
via ``__getattr__``, aggregating results with :func:`VISSSlib.tools._aggregate`
(booleans AND together, ints sum, lists dedupe-concatenate, dicts recurse
per-key) — so e.g. ``FindFilesRange(cases, camera, config).isCompleteL1detect``
transparently answers "complete across the whole range", not just one case.

``Filenames`` / ``FilenamesFromLevel``
-----------------------------------------

:class:`VISSSlib.files.Filenames` does the inverse: given an existing
level0 file path, it parses the embedded ``computer_visssGen_camera_case``
basename to recover case/camera/datetime, then rebuilds the *same* set of
per-level paths as ``FindFiles`` would for that case+camera — stored as a
dict, ``self.fname[level]``, rather than glob patterns, since here the
concrete case is already known.

:class:`VISSSlib.files.FilenamesFromLevel` is the far more commonly used
entry point in practice (e.g. throughout :mod:`VISSSlib.matching`): it
accepts a level1/level2 *product* filename instead of a level0 one, parses
the ``{level}_V{version}_{site}_{computer}_{visssGen}_{camera}_{case}.nc``
naming convention to reconstruct the corresponding level0 path, and then
delegates to ``Filenames.__init__`` with that reconstructed path — i.e. it
round-trips through the level0 naming scheme rather than duplicating the
per-level path logic.

:meth:`VISSSlib.files.Filenames.filenamesOtherCamera` is the piece that
makes stereo matching possible: given one camera's file, it finds the other
camera's files (of a given level) whose timestamps fall within
``newFileInt + graceInterval`` seconds of this file's start time —
including files from the *previous or next day* when this file is near a
day boundary, since level0 files are chunked every 10 minutes regardless of
midnight.

The ``.nodata`` / ``.broken.txt`` sentinel convention
---------------------------------------------------------

Processing functions across the codebase (not just here) write a
``<realfile>.nodata`` file when a case legitimately produced no data, and
``<realfile>.broken.txt`` when processing failed, instead of writing (or
leaving behind a partial) ``<realfile>``. ``files.py`` and ``tools.py``
both know about this convention (``listFilesExt``/``listBroken``/
``listNoData``, ``open2``'s cleanup), but it is not enforced by a shared
type — callers create these paths by string concatenation
(``f"{fname}.nodata"``) at the call site. Keep this in mind when adding a
new processing step: matching the existing sentinel filenames exactly is
what lets downstream ``isComplete``/``checkForExisting`` calls recognize a
"done, but empty" case.

``VISSSlib.files`` API
------------------------

.. automodule:: VISSSlib.files
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
