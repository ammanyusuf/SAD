import types

from unsafe_prep.adapters.realtoxicity import RealToxicityAdapter
from unsafe_prep.constants import RTP_TEXT_FIELD


def make_row(text: str, scores: dict) -> dict:
    continuation = {RTP_TEXT_FIELD: text}
    continuation.update(scores)
    prompt = {RTP_TEXT_FIELD: "prompt text", "id": "prompt-1"}
    return {"prompt": prompt, "continuation": continuation}


def test_realtoxicity_selects_highest_score(monkeypatch):
    rows = [
        make_row("low toxicity", {"toxicity": 0.2, "insult": 0.1}),
        make_row("high insult", {"toxicity": 0.3, "insult": 0.8}),
    ]
    monkeypatch.setattr(
        "unsafe_prep.adapters.realtoxicity.load_dataset",
        lambda *args, **kwargs: rows,
    )
    adapter = RealToxicityAdapter(toxicity_threshold=0.5, streaming=False)
    records = list(adapter.iter_unsafe_answers())
    assert len(records) == 1
    assert records[0].answer_text == "high insult"
    assert records[0].toxicity_score == 0.8
    assert records[0].meta["toxicity_scores"]["insult"] == 0.8


def test_realtoxicity_applies_filters(monkeypatch):
    rows = [
        make_row("threat", {"threat": 0.9, "toxicity": 0.6}),
        make_row("non threat", {"threat": 0.5, "toxicity": 0.7}),
    ]
    monkeypatch.setattr(
        "unsafe_prep.adapters.realtoxicity.load_dataset",
        lambda *args, **kwargs: rows,
    )
    adapter = RealToxicityAdapter(
        toxicity_threshold=0.5,
        toxicity_filters={"threat": 0.8},
        streaming=False,
    )
    records = list(adapter.iter_unsafe_answers())
    assert len(records) == 1
    assert records[0].answer_text == "threat"
    assert records[0].meta["toxicity_filters"] == {"threat": 0.8}


def test_realtoxicity_falls_back_to_prompt_scores(monkeypatch):
    row = {
        "prompt": {RTP_TEXT_FIELD: "prompt", "toxicity": 0.9},
        "continuation": {RTP_TEXT_FIELD: "use prompt score"},
    }
    monkeypatch.setattr(
        "unsafe_prep.adapters.realtoxicity.load_dataset",
        lambda *args, **kwargs: [row],
    )
    adapter = RealToxicityAdapter(toxicity_threshold=0.5, streaming=False)
    records = list(adapter.iter_unsafe_answers())
    assert len(records) == 1
    assert records[0].answer_text == "use prompt score"
