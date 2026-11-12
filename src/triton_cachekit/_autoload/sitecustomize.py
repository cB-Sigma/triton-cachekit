"""Start-up hook used by ``triton-cachekit record -- <command>``.

The ``record`` command puts this directory at the front of ``PYTHONPATH`` so that
every Python process started by the command installs the recorder as soon as it
imports Triton. Processes that never import Triton are not affected.
"""

import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_previous_sitecustomize() -> None:
    """Run the sitecustomize module this one shadows, if there is one."""
    paths = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", paths)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


_chain_previous_sitecustomize()

if os.environ.get("TRITON_CACHEKIT_RECORD"):
    from triton_cachekit.recorder import install_when_triton_is_imported

    install_when_triton_is_imported()
