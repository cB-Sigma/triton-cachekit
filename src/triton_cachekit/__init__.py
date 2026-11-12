"""triton-cachekit: inspect, explain, prewarm and ship Triton kernel caches.

Importing this package never imports Triton. The pieces that need a working
Triton install (``record``, ``prewarm``, ``verify``) import it lazily.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
