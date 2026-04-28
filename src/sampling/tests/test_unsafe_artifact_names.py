from sampling.sample_text import _infer_tokenizer_family, _parse_unsafe_artifact_name


def test_parse_unsafe_artifact_name_basic():
    parsed = _parse_unsafe_artifact_name("beavertails-0100")
    assert parsed is not None
    assert parsed["dataset"] == "beavertails"
    assert parsed["sample_size"] == 100
    assert parsed["take_all"] is False
    assert parsed["suffix"] is None


def test_parse_unsafe_artifact_name_with_suffix():
    parsed = _parse_unsafe_artifact_name("real-toxicity-prompts-0100-llada")
    assert parsed is not None
    assert parsed["dataset"] == "real-toxicity-prompts"
    assert parsed["sample_size"] == 100
    assert parsed["take_all"] is False
    assert parsed["suffix"] == "llada"


def test_parse_unsafe_artifact_name_large_sample():
    parsed = _parse_unsafe_artifact_name("toxigen-10000-dream")
    assert parsed is not None
    assert parsed["dataset"] == "toxigen"
    assert parsed["sample_size"] == 10000
    assert parsed["take_all"] is False
    assert parsed["suffix"] == "dream"


def test_parse_unsafe_artifact_name_all():
    parsed = _parse_unsafe_artifact_name("beavertails-all")
    assert parsed is not None
    assert parsed["dataset"] == "beavertails"
    assert parsed["sample_size"] is None
    assert parsed["take_all"] is True
    assert parsed["suffix"] is None


def test_parse_unsafe_artifact_name_invalid():
    assert _parse_unsafe_artifact_name("beavertails") is None
    assert _parse_unsafe_artifact_name("beavertails-xyz") is None


def test_infer_tokenizer_family():
    assert _infer_tokenizer_family("LLaDA-8B-Base") == "llada"
    assert _infer_tokenizer_family("/models/Dream-v0-Instruct-7B") == "dream"
    assert _infer_tokenizer_family("MMaDA-8B-MixCoT") == "mmada"
    assert _infer_tokenizer_family("gpt2-large") == "unknown"
