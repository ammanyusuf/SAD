from __future__ import annotations

from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass


@dataclass
class RawUnsafeRecord:
  """Schema for unsafe answer text before tokenization."""

  source: str
  category: str
  answer_text: str
  toxicity_score: Optional[float] = None
  meta: Dict[str, Any] = None

  def __post_init__(self) -> None:
    if self.meta is None:
      self.meta = {}


@dataclass
class TokenizedUnsafeRecord(RawUnsafeRecord):
  """Schema for unsafe answers after tokenization."""

  input_ids: Any = None  # torch.LongTensor
  length: int = 0
  mask_index: Optional[int] = None


class BeaverTailsRecord(BaseModel):
  prompt: str
  response: str
  category: Dict[str, bool]
  is_safe: bool # True = safe, False = unsafe


class RealToxicityPromptRecord(BaseModel):
  filename: Optional[str] = None
  begin: Optional[int] = None
  end: Optional[int] = None
  challenging: Optional[bool] = None
  prompt: Dict[str, Any]
  continuation: Dict[str, Any] # unsafe = complement of safe, ie; any of the following is greater than 0.5 -> safe = continuation[profanity, sexually_explicit , identity_attack, flirtation, threat, insult, severe_toxicity, toxicity] < 0.5 (ie; any one of those is less than 0.5)


class ToxigenTrainRecord(BaseModel):
  prompt: str
  generation: str
  generation_method: str
  group: Optional[str]
  prompt_label: Optional[int] # 0 = safe, 1 = unsafe
  roberta_prediction: Optional[float]
  toxicity_ai: Optional[float] = None
  toxicity_human: Optional[float] = None


class ToxigenAnnotatedRecord(BaseModel):
  text: str
  target_group: Optional[str]
  toxicity_ai: Optional[float]
  toxicity_human: Optional[float]
  predicted_author: Optional[str]
  actual_method: Optional[str]
  group: Optional[str] = None
