import json
from pathlib import Path

from src.config import Config, load_config


def _write_cfg(tmp_path: Path, **over) -> Path:
    data = {"host": "127.0.0.1", "port": 8765, **over}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(argv=[], config_path=tmp_path / "missing.json", env={})
    assert cfg.port == 8765
    assert cfg.language == "de"
    assert cfg.languages["en"]["arch"] == "F5TTS_v1_Base"
    assert "de" in cfg.languages and "fr" in cfg.languages and "ja" in cfg.languages
    assert cfg.allowed_sample_ext == [".wav", ".flac", ".mp3"]


def test_language_override_via_cli(tmp_path):
    cfg = load_config(argv=["--language", "fr"], config_path=tmp_path / "x.json", env={})
    assert cfg.language == "fr"


def test_stream_chunk_size_default_and_coercion(tmp_path):
    cfg = load_config(argv=[], config_path=tmp_path / "missing.json", env={})
    assert cfg.stream_chunk_size == 20
    over = load_config(argv=[], config_path=tmp_path / "m.json", env={"F5W_STREAM_CHUNK_SIZE": "40"})
    assert over.stream_chunk_size == 40 and isinstance(over.stream_chunk_size, int)


def test_xtts_fp16_default_and_coercion(tmp_path):
    cfg = load_config(argv=[], config_path=tmp_path / "missing.json", env={})
    assert cfg.xtts_fp16 is False
    # ENV string coerces to bool.
    env_on = load_config(argv=[], config_path=tmp_path / "m.json", env={"F5W_XTTS_FP16": "true"})
    assert env_on.xtts_fp16 is True
    # CLI flag overrides.
    cli_on = load_config(argv=["--xtts-fp16", "1"], config_path=tmp_path / "m.json", env={})
    assert cli_on.xtts_fp16 is True


def test_transformers_constraint_default_excludes_5x(tmp_path):
    # XTTS needs isin_mps_friendly (gone in transformers 5.x); the default pin
    # must keep the resolve on the 4.x line.
    cfg = load_config(argv=[], config_path=tmp_path / "missing.json", env={})
    assert cfg.transformers_constraint == "transformers>=4.57,<5"


def test_tts_backend_default_and_overrides(tmp_path):
    cfg = load_config(argv=[], config_path=tmp_path / "missing.json", env={})
    assert cfg.tts_backend == "f5"  # F5 stays the default
    cli = load_config(argv=["--tts-backend", "xtts"], config_path=tmp_path / "m.json", env={})
    assert cli.tts_backend == "xtts"
    envd = load_config(argv=[], config_path=tmp_path / "m.json", env={"F5W_TTS_BACKEND": "xtts"})
    assert envd.tts_backend == "xtts"


def test_languages_map_from_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"languages": {"en": {"arch": "Custom"}}}', encoding="utf-8"
    )
    cfg = load_config(argv=[], config_path=p, env={})
    assert cfg.languages == {"en": {"arch": "Custom"}}


def test_json_is_base(tmp_path):
    path = _write_cfg(tmp_path, port=9000, api_key="abc")
    cfg = load_config(argv=[], config_path=path, env={})
    assert cfg.port == 9000
    assert cfg.api_key == "abc"


def test_env_overrides_json(tmp_path):
    path = _write_cfg(tmp_path, port=9000)
    cfg = load_config(argv=[], config_path=path, env={"F5W_PORT": "5555"})
    assert cfg.port == 5555


def test_cli_overrides_env_and_json(tmp_path):
    path = _write_cfg(tmp_path, port=9000)
    cfg = load_config(
        argv=["--port", "7777"], config_path=path, env={"F5W_PORT": "5555"}
    )
    assert cfg.port == 7777


def test_bool_and_list_coercion(tmp_path):
    path = _write_cfg(tmp_path)
    cfg = load_config(
        argv=[],
        config_path=path,
        env={
            "F5W_ASR_FOR_MISSING_REF_TEXT": "false",
            "F5W_ALLOWED_SAMPLE_EXT": ".wav,.ogg",
            "F5W_MAX_WORKERS": "3",
        },
    )
    assert cfg.asr_for_missing_ref_text is False
    assert cfg.allowed_sample_ext == [".wav", ".ogg"]
    assert cfg.max_workers == 3


def test_nullable_coercion(tmp_path):
    path = _write_cfg(tmp_path, api_key="x")
    cfg = load_config(argv=[], config_path=path, env={"F5W_API_KEY": "null"})
    assert cfg.api_key is None


def test_path_resolution_relative_and_absolute(tmp_path):
    cfg = Config(samples_dir="samples", models_dir=str(tmp_path / "abs"))
    assert cfg.samples_path.is_absolute()
    assert cfg.models_path == tmp_path / "abs"


def test_normalized_exts():
    cfg = Config(allowed_sample_ext=["WAV", ".Flac"])
    assert cfg.normalized_exts == {".wav", ".flac"}


def test_custom_model_path_under_models(tmp_path):
    cfg = Config(models_dir=str(tmp_path / "abs"))
    assert cfg.custom_model_path == tmp_path / "abs" / "echokraut_custom"
