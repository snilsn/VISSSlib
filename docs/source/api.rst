API
===

The VISSSlib processing library provides tools for analyzing Video In Situ
Snowfall Sensor (VISSS) data. The library is organized into several
modules, each serving a specific purpose in the data processing pipeline.
Every module has its own page with a prose overview followed by its full
API reference (rather than duplicating that reference here too) — see:

- :doc:`tools`, :doc:`files` — shared infrastructure
- :doc:`metadata` — metaFrames/metaEvents
- :doc:`av` — video reading
- :doc:`detection` — single-camera particle detection (level1detect)
- :doc:`matching`, :doc:`metaRotation` — stereo particle matching and camera rotation retrieval
- :doc:`tracking` — frame-to-frame particle tracking (level1track)
- :doc:`distributions` — calibration and Level 2 distributions
- :doc:`level3` — derived microphysical products (e.g. riming)
- :doc:`products` — the processing DAG
- :doc:`quicklooks` — plotting
- :doc:`analysis` — interactive stereo-view inspection
- :doc:`fixes` — campaign-specific data patches
