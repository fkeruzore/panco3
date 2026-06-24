"""Shared test helpers.

panco2 is used as a reference oracle to validate the panco3 port. We load
individual panco2 submodules **by file path** (via importlib) rather than
``import panco2``, because ``panco2/__init__.py`` pulls in the full
inference stack (emcee, chainconsumer, ...) which is not a panco3
dependency. The submodules we need for validation (``cluster``, ``abell``)
have no intra-package relative imports, so they load cleanly in isolation.
"""

import importlib
import importlib.util
import os
import sys
import types

_PANCO2_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "panco2", "panco2")
)


def load_panco2_module(name: str):
    """Load ``panco2/panco2/<name>.py`` as a standalone module."""
    path = os.path.join(_PANCO2_PKG, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_panco2_ref_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_panco2_package_module(name: str):
    """Import ``panco2.<name>`` with intra-package relative imports working.

    Installs a stub ``panco2`` package (pointing at the source dir) into
    ``sys.modules`` so submodules like ``model``/``filtering`` -- which use
    ``from . import ...`` -- import cleanly *without* running the real
    ``panco2/__init__.py`` (which pulls in emcee, chainconsumer, ...).
    """
    pkg = sys.modules.get("panco2")
    if pkg is None or not getattr(pkg, "_panco3_stub", False):
        pkg = types.ModuleType("panco2")
        pkg.__path__ = [_PANCO2_PKG]
        pkg._panco3_stub = True
        sys.modules["panco2"] = pkg
    return importlib.import_module(f"panco2.{name}")
