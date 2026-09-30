``level3`` - derived microphysical products
================================================

Motivation
----------

``level3`` is the only package (not flat module) under ``VISSSlib``, and
the only part of the processing chain with an optional dependency: PAMTRA
(https://github.com/igmk/pamtra, imported as ``pyPamtra``), which is not a
pip dependency and must be installed separately (see the top-level
``AI.md``). ``level3/__init__.py`` registers available products in a
single dict:

.. code:: python

    AVAILABLE_PRODUCTS = {
        "combined_riming": retrieveCombinedRiming,
    }

— currently just the one product. Adding a new level3 product means adding
an entry here (and to :data:`VISSSlib.files.dailyLevels` /
:class:`VISSSlib.products.DataProduct`'s ``parentNames``, see
:doc:`products`).

``aux.py`` — fetching ancillary observations
-------------------------------------------------

Despite the generic name, ``aux.py`` is not shared level3 infrastructure —
it is entirely about **downloading/reading the external ancillary data**
(cloud radar reflectivity, temperature/pressure meteo data) that
``combined_riming.py`` needs as retrieval inputs, since VISSS itself only
observes particle shape/size/velocity, not the radar reflectivity or
atmospheric state needed to constrain a riming retrieval. It supports
several source backends selected by ``config.aux.radar.source`` /
``config.aux.meteo.source``: Cloudnet (categorize product or raw FMCW94),
ARM (``wcloudradarcel``/``arsclkazr1kollias``), RPG cloud radar files, and
PANGAEA-archived datasets (with a download-and-cache helper,
:func:`VISSSlib.level3.aux.downloadPangaea`). If you're adding a new
deployment/campaign that isn't Cloudnet/ARM/Pangaea, this is the module to
extend.

Riming retrieval (``combined_riming.py``)
----------------------------------------------

:func:`VISSSlib.level3.combined_riming.retrieveM` implements the actual
retrieval, following Maherndl et al. (2023) — and structurally mirrors
:func:`VISSSlib.matching.retrieveRotation`: both use the same Bayesian
inverse **Optimal Estimation** framework (``pyOptimalEstimation``), just
retrieving a different state variable against a different forward model.
Here the state is a single riming mass parameter ``M`` (retrieved in
log-space, ``M = 10**X``, see
:func:`VISSSlib.level3.combined_riming.reflec_logM`), the forward operator
runs the observed VISSS PSD plus a riming-dependent mass-size relation and
SSRGA (Self-Similar Rayleigh-Gans Approximation) scattering through PAMTRA
to predict radar reflectivity, and the retrieval searches for the ``M``
that best reproduces the *observed* radar ``Ze``.
:func:`VISSSlib.level3.combined_riming.ssrga_parameter` has an explicit
fallback for 50°-elevation radars (a polynomial fit in ``M``) alongside the
zenith-pointing (90°) case that PAMTRA's own descriptor file provides
directly — anything in between (or outside ~39–61°) is unsupported.

:func:`VISSSlib.level3.combined_riming.retrieveCombinedRiming` is the
per-day entry point. Its DAG parent is ``level2track`` (merged with the
radar/meteo data from ``aux.py``), and — since running PAMTRA/OE per time
step is comparatively expensive — it gates on four quality conditions
before doing any retrieval work at all: cold enough
(``air_temperature < config.level3.combinedRiming.maxTemp``, default
+2°C), actually precipitating (configured reflectivity variable above
``minZe``), VISSS ``qualityFlags == 0``, and enough particles observed
(``nParticles >= minNParticles``). If *no* time step in the whole day
passes all four, it writes the ``.nodata`` sentinel immediately rather than
attempting (and failing) the retrieval.

``VISSSlib.level3`` API
---------------------------

.. automodule:: VISSSlib.level3
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
