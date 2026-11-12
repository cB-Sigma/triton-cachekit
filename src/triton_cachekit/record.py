"""Importing this module enables the cache-key recorder in the current process.

    import triton_cachekit.record  # noqa: F401

The output file comes from ``TRITON_CACHEKIT_RECORD`` (default:
``./triton-cachekit-record.jsonl``; ``{pid}`` is substituted, a directory gets
one file per process). Set ``TRITON_CACHEKIT_RECORD_SOURCE=0`` to keep kernel
source text out of the recording.

Raises :class:`triton_cachekit.recorder.UnsupportedTritonError` when the installed
Triton cannot be hooked.
"""

from __future__ import annotations

from .recorder import UnsupportedTritonError, active, install, uninstall

recorder = install()

__all__ = ["UnsupportedTritonError", "active", "install", "recorder", "uninstall"]
