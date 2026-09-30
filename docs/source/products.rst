``products`` - the processing DAG
=====================================

Motivation
----------

``products.py`` is where the whole per-level pipeline (:doc:`metadata`,
:doc:`detection`, :doc:`matching`, :doc:`tracking`, :doc:`distributions`,
:doc:`level3`) gets tied together into one dependency graph and driven
end-to-end, either directly or via a task queue for cluster/multi-worker
processing. :class:`VISSSlib.products.DataProduct` is the whole module's
center of gravity; everything else (the per-level subclasses,
:func:`~VISSSlib.products.processAll`,
:func:`~VISSSlib.products.submitAll`,
:func:`~VISSSlib.products.processRealtime`) is a thin layer on top of it.

The dependency graph lives in code, not config
----------------------------------------------------

:data:`VISSSlib.tools.LEVEL_REGISTRY` is the single authoritative
statement of "what depends on what, and how is it built" for the whole
library — one dict entry per level, keyed on ``level``, used by both
:meth:`VISSSlib.products.DataProduct.__init__` (to resolve ``parentNames``)
and :meth:`VISSSlib.products.DataProduct.generateCommands` (to build the
shell command). E.g. ``level1match``'s only parent is
``{camera}_metaRotation``, ``level2match``'s parents are
``{camera}_level1match`` plus **both** cameras' ``metaEvents`` (added
explicitly so a Level 2 product regenerates whenever *more level0 data has
been transferred*, since ``metaEvents`` is the record of that, not just to
satisfy a strict data dependency).
**Adding a new level requires editing this registry** (and
:data:`VISSSlib.files.fileLevels`/``dailyLevels``) — there is no single
declarative place spanning the whole library, but within ``products.py``
itself it is now one dict entry, not several call sites kept in sync by
hand.

``level1match.processL1match: false`` (see :doc:`config_files`) disables
the whole stereo-matching branch, but only at the ``allDone``/
:func:`~VISSSlib.products.processAll` entry points — constructing a
``DataProduct``/``DataProductRange`` for ``metaRotation``, ``level1match``,
``level1track``, ``level2match``, or ``level2track`` **directly** checks
the same flag and raises ``ValueError`` rather than silently proceeding as
if matching were enabled. If you need to run one of these levels
one-off despite the deployment having matching disabled, call the
underlying :mod:`VISSSlib.matching`/:mod:`VISSSlib.tracking` function
directly instead of going through ``DataProduct``.

Construction is recursive and eager
-----------------------------------------

Constructing a ``DataProduct`` for one level **recursively constructs a
``DataProduct`` for every ancestor**, all the way down to ``level0`` — the
whole relevant subgraph is built up front, not lazily. ``childrensRelatives``
is a dict threaded through the recursion so a parent shared by two branches
(e.g. both ``leader_metaEvents`` and ``follower_metaEvents`` are parents of
several levels) is only constructed once and then reused, rather than
rebuilt per branch.

Commands, not direct calls
------------------------------

:meth:`VISSSlib.products.DataProduct.generateCommands` does not call
e.g. :func:`VISSSlib.detection.detectParticles` directly — it builds a
**shell command string** that re-invokes the same
``python -m VISSSlib <module.function> ...`` CLI documented in
:doc:`command_line`, one command per level0 input file (via
:meth:`~VISSSlib.products.DataProduct._commandTemplateL1`) or one per case
(via :meth:`~VISSSlib.products.DataProduct._commandTemplateDaily`). This is
what makes the same DAG usable both for direct in-process execution
(:func:`VISSSlib.products.processAll`) and for distributed processing: the
commands are exactly what gets pushed onto the
:func:`VISSSlib.tools.runCommandInQueue` task queue for
:func:`VISSSlib.tools.workers` (potentially on other machines/SLURM nodes,
see ``scripts/VISSSlib_slurm.sh``) to execute.
:meth:`~VISSSlib.products.DataProduct.generateAllCommands` decides, for
*each* level in the ancestor chain independently, whether it's worth
generating commands at all — only once its own parents are already
complete and not younger than *their* parents — and otherwise just recurses
further down. This is why :func:`VISSSlib.products.submitAll` (and
:meth:`VISSSlib.products.DataProduct.process`) are documented as sometimes
needing to be called repeatedly: a single call only submits work for
whichever part of the chain is *currently* ready, not the full pipeline in
one shot, when several levels are simultaneously missing.

Entry points
-------------

- :func:`VISSSlib.products.processAll` walks a fixed, explicit list of
  levels (not the general parent-recursion above) and, for each, submits
  commands and drains the queue with :func:`VISSSlib.tools.workers` before
  moving to the next level — a synchronous, one-level-at-a-time barrier.
  Its own docstring calls this "a rather unefficient way of processing the
  data and mostly for testing".
- :func:`VISSSlib.products.submitAll` is the production path: it builds one
  ``DataProductRange("allDone", ...)`` (a
  :class:`~VISSSlib.products.DataProductRange`, the
  :class:`VISSSlib.files.FindFilesRange`-style wrapper for a whole case
  range) and pushes *all* currently-generatable commands across the whole
  DAG into the queue at once, for however many :func:`VISSSlib.tools.workers`
  (potentially remote) to drain in parallel — see ``worker`` in
  :doc:`command_line`.
- :func:`VISSSlib.products.processRealtime` is the "cheap, run-often"
  subset (metadata/quicklooks) meant to be invoked frequently on the
  acquisition/near-real-time side without the expensive Level 1/2/3
  processing.

Typical deployment: two tiers
-----------------------------------

A production deployment typically splits processing into two independently
scheduled tiers, matching the cheap/expensive split above:

1. **Realtime tier** — run frequently (e.g. from cron), close to the data
   acquisition side, with no cluster/queue involved. Loops over one
   settings file per monitored site:

   .. code:: console

       python -m VISSSlib products.processRealtime <settings.yaml> <days> --skip-existing

   This only ever runs :func:`VISSSlib.products.processRealtime`'s cheap
   subset (metaEvents, level0 quicklook, metaFrames, last-file report) — it
   deliberately never triggers Level 1/2/3 processing, so it stays fast
   enough to run every few minutes without competing for cluster resources.

2. **Batch tier** — run on a cluster for the expensive Level 1/2/3 work.
   One job submits work into a task queue, a separate (typically
   ``sbatch``-submitted, low-priority) job runs one or more
   :func:`VISSSlib.tools.workers` processes to drain it:

   .. code:: console

       # submit currently-generatable commands into the queue
       python -m VISSSlib products.submitAll <settings.yaml> <case> <task_queue_dir>

       # separately, on the cluster, drain the queue (see also scripts/VISSSlib_slurm.sh)
       python -m VISSSlib worker <task_queue_dir> --n-jobs <N>

   Because :func:`~VISSSlib.products.submitAll` only submits whatever part
   of the DAG is *currently* ready (see above), it is typically scheduled
   to run repeatedly (e.g. once a day) rather than once, so that a level
   whose parents weren't ready yet on one run gets picked up on the next.

A useful, easy-to-miss operational detail: if
``config.level3.combinedRiming.processRetrieval`` is enabled, workers need
the ``PAMTRA_DATADIR`` environment variable set to PAMTRA's data directory
(scattering databases etc.) *before* they start — this isn't validated or
documented anywhere inside VISSSlib itself, so a missing/wrong value fails
silently deep inside :mod:`VISSSlib.level3.combined_riming`'s retrieval
rather than at startup.

It is also common to run two separate environments for the two tiers — a
"current/dev" environment for the realtime tier and a separate, pinned
environment for the batch tier — so that a bug introduced on the
development branch can't affect an in-progress campaign's batch processing
until it has been vetted.

``tests/test_products.py::TestProducts::test_processAll`` is the closest
thing this repo has to a true end-to-end integration test: it runs
:func:`VISSSlib.products.processAll` for a whole case against the
downloaded sample dataset, exercising every level from ``level0`` through
``allDone`` in one go (cited in ``AI.md`` as the reference single-test
invocation). ``tests/test_products.py::TestDataProductDAG`` covers the same
class's dependency-graph wiring (``parentNames`` per level, the
``processL1match`` guard, ``isComplete``) as fast, network-free unit tests
(``pytest -m unit``) against a synthetic config built by
``tests/helpers.py::makeSyntheticConfig``, rather than the downloaded sample
dataset.

``VISSSlib.products`` API
-----------------------------

.. automodule:: VISSSlib.products
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
