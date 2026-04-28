from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional

from .adapters import (
    AdapterNotFoundError,
    UnsafeDatasetAdapter,
    available_adapters,
    get_adapter,
)

# Import adapter modules so that they register themselves on import.
from .adapters import beavertails as _beavertails
from .adapters import harmbench as _harmbench
from .adapters import harmbench_jsonl as _harmbench_jsonl
from .adapters import realtoxicity as _realtoxicity
from .adapters import toxigen as _toxigen

__all__ = [
    "AdapterNotFoundError",
    "UnsafeDatasetAdapter",
    "available_adapters",
    "get_adapter",
]
