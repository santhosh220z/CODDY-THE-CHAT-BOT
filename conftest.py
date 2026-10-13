"""Make the repository root importable however pytest is invoked.

Running `pytest backend/tests` from the repo root works once `backend/` and
friends carry `__init__.py`, but `pytest` run from elsewhere (or via an IDE
with a different cwd) still fails to resolve `import backend...`. Prepending
the root keeps the documented command working regardless.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)