from __future__ import annotations

from typing import Callable, Dict, Iterator, List

from ..schemas import RawUnsafeRecord


class AdapterNotFoundError(KeyError):
  """Raised when an adapter name is not registered."""


class UnsafeDatasetAdapter:
  """Interface for dataset adapters."""

  name: str

  def iter_unsafe_answers(self) -> Iterator[RawUnsafeRecord]:
    raise NotImplementedError


AdapterFactory = Callable[..., UnsafeDatasetAdapter]


_REGISTRY: Dict[str, AdapterFactory] = {}


def register_adapter(name: str, factory: AdapterFactory) -> None:
  if name in _REGISTRY:
    raise ValueError(f"Adapter '{name}' is already registered.")
  _REGISTRY[name] = factory


def get_adapter(name: str, **kwargs) -> UnsafeDatasetAdapter:
  try:
    factory = _REGISTRY[name]
  except KeyError as exc:
    raise AdapterNotFoundError(name) from exc
  return factory(**kwargs)


def available_adapters() -> List[str]:
  return sorted(_REGISTRY.keys())


__all__ = [
  "AdapterNotFoundError",
  "UnsafeDatasetAdapter",
  "register_adapter",
  "get_adapter",
  "available_adapters",
  "RawUnsafeRecord",
]
