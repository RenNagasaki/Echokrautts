"""The MOSS audio contract, in one place for both MOSS runtimes.

MOSS emits 48 kHz stereo; this wrapper promises 24 kHz mono float32. Two
backends now need that conversion (the PyTorch runtime and the ONNX one), and
the conversion is not trivial enough to write twice — it carries a fix that
cost a user-reported bug to find.
"""

from __future__ import annotations

import numpy as np


class ContractResampler:
    """Per-chunk 48 kHz stereo -> 24 kHz mono, without the seams.

    The conversion has to happen per chunk because a streaming path has no
    "end" to do it at. A resampling filter that starts cold on every chunk is
    audible, though: **measured**, resampling each chunk on its own put the
    largest sample-to-sample jump at the seams at 1.5x the signal's own typical
    step, with ~60 seams per sentence — the crackle a user reported. Feeding the
    tail of the previous chunk back in drops that to 0.7x, i.e. the seams stop
    standing out from the audio around them. Overlap sizes 64/128/256/512/1024
    all measured identical, so 64 (about 1.3 ms at 48 kHz) is what it uses.

    The overlap samples are resampled again and then thrown away. That is the
    price of not keeping a stateful resampler alive across chunks: a few dozen
    samples of arithmetic per chunk, against a filter restart you can hear.

    One instance belongs to one request. :meth:`reset` starts a new signal —
    without it the end of one sentence is spliced onto the start of the next,
    which only shows up in production, never in a single-clip test.
    """

    OVERLAP = 64

    def __init__(self) -> None:
        self.tail = None

    def reset(self) -> None:
        self.tail = None

    def to_contract(self, waveform, source_rate: int, target_rate: int) -> np.ndarray:
        """One chunk of whatever MOSS produced -> float32 mono at the target rate.

        Accepts a torch tensor or anything numpy can adopt, channels-first
        ``(C, N)`` or flat — the ONNX runtime hands out channels-LAST ``(N, C)``,
        so callers there transpose before calling. A single channel passes
        through untouched, and a matching sample rate skips the filter entirely.

        ``target_rate`` is passed per call rather than held here on purpose: the
        worker's ``sample_rate`` is the wrapper's contract, and caching a copy
        of it would make two places disagree the moment one of them changed.
        """
        import torch
        import torchaudio

        tensor = waveform if hasattr(waveform, "dim") else torch.as_tensor(np.asarray(waveform))
        tensor = tensor.detach().to("cpu", dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] > 1:  # stereo (or more) -> mono
            tensor = tensor.mean(dim=0, keepdim=True)
        if not source_rate or source_rate == target_rate:
            return tensor.squeeze(0).numpy().astype(np.float32, copy=False)

        mono = tensor.squeeze(0)
        lead = self.tail
        if lead is not None and lead.numel():
            mono = torch.cat([lead, mono])
            skip = int(round(lead.numel() * target_rate / source_rate))
        else:
            skip = 0
        self.tail = mono[-self.OVERLAP:].clone()
        out = torchaudio.functional.resample(mono.unsqueeze(0), source_rate, target_rate)
        return out.squeeze(0)[skip:].numpy().astype(np.float32, copy=False)
