import os
import sys

import pytest

# Make sure `import VISSSlib` resolves to *this* checkout's src/, not
# wherever `pip install -e .` happened to be run from. The editable install
# is pinned to whatever directory it was installed from and does not follow
# git worktrees, so without this a worktree's `pytest` run can silently
# exercise a different checkout's code (e.g. the main clone's) with no
# warning - green tests then say nothing about the code actually on disk
# here. Must run before any test module imports VISSSlib, hence conftest.py.
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
