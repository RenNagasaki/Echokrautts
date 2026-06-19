import sys
import types

import pytest

from src import models
from src.config import Config


@pytest.fixture
def fake_hf(monkeypatch):
    """Install a fake huggingface_hub so resolve_model never hits the network."""
    calls = []
    mod = types.ModuleType("huggingface_hub")

    def hf_hub_download(repo_id, filename, cache_dir=None):
        calls.append((repo_id, filename))
        return f"/cache/{repo_id}/{filename}"

    mod.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return calls


def test_resolve_base_does_not_download(fake_hf):
    rm = models.resolve_model(Config(), "en")
    assert rm.arch == "F5TTS_v1_Base"
    assert rm.ckpt_file == "" and rm.vocab_file == ""
    assert fake_hf == []  # base is auto-downloaded by F5-TTS, not here


def test_resolve_finetune_downloads_ckpt_and_vocab(fake_hf):
    rm = models.resolve_model(Config(), "de")
    assert rm.arch == "F5TTS_Base"
    assert rm.ckpt_file.endswith("model_f5tts_german.safetensors")
    assert rm.vocab_file.endswith("vocab.txt")
    assert ("hvoss-techfak/F5-TTS-German", "model_f5tts_german.safetensors") in fake_hf


def test_resolve_uses_active_language_by_default(fake_hf):
    rm = models.resolve_model(Config(language="fr"))
    assert rm.arch == "F5TTS_Base"
    assert any(repo == "RASPIAUDIO/F5-French-MixedSpeakers-reduced" for repo, _ in fake_hf)


def test_unknown_language_raises():
    with pytest.raises(ValueError):
        models.resolve_model(Config(), "xx")


def test_download_all_covers_every_language(monkeypatch, fake_hf, tmp_path):
    # Fake the base-model class so the "en" branch doesn't import real f5_tts.
    f5mod = types.ModuleType("f5_tts")
    apimod = types.ModuleType("f5_tts.api")
    built = []

    class FakeF5TTS:
        def __init__(self, model=None, hf_cache_dir=None):
            built.append(model)

    apimod.F5TTS = FakeF5TTS
    monkeypatch.setitem(sys.modules, "f5_tts", f5mod)
    monkeypatch.setitem(sys.modules, "f5_tts.api", apimod)

    models.download_all(Config(models_dir=str(tmp_path / "m")))

    assert "F5TTS_v1_Base" in built  # base instantiated
    downloaded_repos = {repo for repo, _ in fake_hf}
    assert "hvoss-techfak/F5-TTS-German" in downloaded_repos
    assert "RASPIAUDIO/F5-French-MixedSpeakers-reduced" in downloaded_repos
    assert "Jmica/F5TTS" in downloaded_repos
