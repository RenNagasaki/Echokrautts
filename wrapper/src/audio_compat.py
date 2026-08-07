"""Keep ``torchaudio.load`` decoding natively, whatever torchaudio version runs.

Why this exists
---------------
The wrapper pins torch/torchaudio to 2.7.x precisely because ``torchaudio.load``
still decodes through the bundled **soundfile** backend there — no system FFmpeg,
no external binaries (see the README). From **torchaudio 2.9** onward, ``load``
is an alias for ``load_with_torchcodec``: decoding moves to TorchCodec, which
needs FFmpeg shared libraries on the machine.

One backend cannot honour the pin: AMD's native-Windows ROCm build ships torch
and torchaudio **2.9.1** and nothing older. Requiring every AMD user to install
FFmpeg (and keeping torchcodec's version glued to torch's) would trade a
self-contained install for a fragile one.

So instead of changing what the engines call, this module changes what that call
does: ``torchaudio.load`` is replaced with a soundfile-based implementation that
returns exactly what the 2.7 API returned — ``(waveform[channels, frames],
sample_rate)``, float32, normalised. ``soundfile`` is already a hard dependency
of the wrapper, so nothing new is installed.

The engines are untouched on purpose. Both call ``torchaudio.load`` from inside
third-party code (``f5_tts.utils_infer``, coqui's XTTS), which we do not fork —
patching the attribute they look up at call time reaches all of them at once.
"""

from __future__ import annotations

from typing import Optional

from . import ndjson

# Marks a patched function so a second call is a no-op and tests can assert it.
_PATCH_FLAG = "_echokrautts_soundfile_shim"


def _needs_shim(torchaudio_version: str, torchcodec_available: bool) -> bool:
    """Decide whether ``torchaudio.load`` must be replaced.

    Pure and version-string driven so it is testable without torch installed.
    The shim is for the case where torchaudio delegates to TorchCodec but
    TorchCodec is not there — with torchcodec present, the stock path works and
    is left alone.
    """
    if torchcodec_available:
        return False
    try:
        major, minor = (int(p) for p in torchaudio_version.split(".")[:2])
    except (ValueError, TypeError):
        # Unparseable version: do not touch a library we do not understand.
        return False
    return (major, minor) >= (2, 9)


def _soundfile_load(uri, *args, **kwargs):
    """Drop-in for ``torchaudio.load`` backed by soundfile.

    Accepts and ignores the 2.7-era keyword arguments that torchaudio 2.9 also
    ignores (``normalize``, ``buffer_size``, ``backend``, ``format``) so callers
    written against either version keep working. ``frame_offset`` and
    ``num_frames`` are honoured — f5-tts passes neither today, but silently
    returning the whole file for a caller that asked for a slice would be a
    nasty bug to chase.
    """
    import soundfile as sf
    import torch

    frame_offset = int(kwargs.get("frame_offset", args[0] if args else 0) or 0)
    num_frames = int(kwargs.get("num_frames", args[1] if len(args) > 1 else -1) or -1)
    channels_first = kwargs.get("channels_first", True)

    with sf.SoundFile(str(uri)) as fh:
        if frame_offset:
            fh.seek(frame_offset)
        data = fh.read(
            frames=-1 if num_frames in (-1, 0) else num_frames,
            dtype="float32",
            always_2d=True,
        )
        sample_rate = fh.samplerate

    tensor = torch.from_numpy(data)  # (frames, channels)
    if channels_first:
        tensor = tensor.transpose(0, 1).contiguous()
    return tensor, sample_rate


def ensure_native_audio_loading(torchaudio_module=None) -> Optional[str]:
    """Patch ``torchaudio.load`` if this torchaudio would need TorchCodec.

    Returns a short reason string when a patch was applied, else ``None``.
    Safe to call repeatedly and safe to call when torchaudio is absent — the
    unit suite mocks torch away entirely.
    """
    if torchaudio_module is None:
        try:
            import torchaudio  # noqa: PLC0415 — optional, absent in the test venv
        except Exception:  # noqa: BLE001 — any import failure means "nothing to patch"
            return None
        torchaudio_module = torchaudio

    if getattr(getattr(torchaudio_module, "load", None), _PATCH_FLAG, False):
        return None

    try:
        import importlib.util

        torchcodec_available = importlib.util.find_spec("torchcodec") is not None
    except Exception:  # noqa: BLE001
        torchcodec_available = False

    version = str(getattr(torchaudio_module, "__version__", "")).split("+")[0]
    if not _needs_shim(version, torchcodec_available):
        return None

    setattr(_soundfile_load, _PATCH_FLAG, True)
    torchaudio_module.load = _soundfile_load
    reason = (
        f"torchaudio {version} routes load() through torchcodec (absent) — "
        "using the bundled soundfile decoder instead"
    )
    ndjson.log_once(reason)
    return reason
