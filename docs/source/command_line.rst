Command-line Options
====================

``python -m VISSSlib`` runs ``VISSSlib/__main__.py``, which parses
arguments with :func:`VISSSlib.tools._create_parser` (below) and then
dispatches on the ``command`` string via a flat ``if/elif`` chain straight
into the corresponding module function (e.g. ``matching.matchParticles``
calls :func:`VISSSlib.matching.matchParticles`). Every subcommand added to
the parser needs a matching branch added there by hand — the two are kept
in sync manually, not generated from each other. This is also the
mechanism :class:`VISSSlib.products.DataProduct` relies on (see
:doc:`products`): the shell commands it generates are just
``python -m VISSSlib <command> ...`` invocations of this same CLI.

.. argparse::
   :module: VISSSlib.tools
   :func: _create_parser
   :prog: python -m VISSSlib

