"""Centralized constants for dataset field names and enumerations."""

from __future__ import annotations

BEAVERTAILS_PROMPT_FIELD = "prompt"
BEAVERTAILS_RESPONSE_FIELD = "response"
BEAVERTAILS_CATEGORY_FIELD = "category"
BEAVERTAILS_IS_SAFE_FIELD = "is_safe"
BEAVERTAILS_CATEGORY_KEYS = (
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
)

RTP_FILENAME_FIELD = "filename"
RTP_PROMPT_FIELD = "prompt"
RTP_CONTINUATION_FIELD = "continuation"
RTP_TEXT_FIELD = "text"
RTP_SCORE_FIELDS = (
    "toxicity",
    "severe_toxicity",
    "threat",
    "identity_attack",
    "insult",
    "profanity",
    "sexually_explicit",
    "flirtation",
)

TOXIGEN_PROMPT_FIELD = "prompt"
TOXIGEN_GENERATION_FIELD = "generation"
TOXIGEN_METHOD_FIELD = "generation_method"
TOXIGEN_GROUP_FIELD = "group"
TOXIGEN_PROMPT_LABEL_FIELD = "prompt_label"
TOXIGEN_ROBERTA_FIELD = "roberta_prediction"
TOXIGEN_TEXT_FIELD = "text"
TOXIGEN_TARGET_GROUP_FIELD = "target_group"
TOXIGEN_TOXICITY_AI_FIELD = "toxicity_ai"
TOXIGEN_TOXICITY_HUMAN_FIELD = "toxicity_human"
TOXIGEN_GENERATION_METHODS = ("alice", "topk")
TOXIGEN_GROUPS = (
    "asian",
    "black",
    "chinese",
    "jewish",
    "latino",
    "lgbtq",
    "mental_dis",
    "middle_east",
    "mexican",
    "muslim",
    "native_american",
    "physical_dis",
    "women",
)
