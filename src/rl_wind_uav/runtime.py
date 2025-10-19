"""Runtime helpers for entry-point scripts."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ensure_windows_dll_path() -> None:
    """Ensure Conda DLL search paths are available on Windows.

    When PyCharm or other IDEs launch a Conda interpreter without fully
    activating the environment, Windows may be unable to locate MKL/BLAS DLLs
    required by NumPy, SciPy, and PyTorch. This helper adds the expected
    ``Library/bin`` directory to both ``PATH`` and the DLL search path so the
    native extensions can load reliably.
    """

    if os.name != "nt":  # pragma: no cover - Windows specific behaviour
        return

    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if not prefix:
        return

    candidate = Path(prefix) / "Library" / "bin"
    if not candidate.is_dir():
        return

    candidate_str = str(candidate)

    # Update PATH so child processes inherit the setting as well.
    current_path = os.environ.get("PATH", "")
    if candidate_str not in current_path.split(os.pathsep):
        os.environ["PATH"] = candidate_str + os.pathsep + current_path

    # Windows >= 3.8 requires explicit registration for DLL lookup.
    try:  # pragma: no cover - only executed on Windows
        os.add_dll_directory(candidate_str)
    except (AttributeError, FileNotFoundError):
        # ``add_dll_directory`` is unavailable on older Python versions or
        # when the directory cannot be accessed. In those cases the PATH
        # update above still improves the chance of success.
        pass
