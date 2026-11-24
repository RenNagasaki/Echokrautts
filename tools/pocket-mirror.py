"""Create our own HuggingFace mirror of the Pocket TTS voice-cloning files.

Copies exactly the files the wrapper loads (per language and model size: model.safetensors +
tokenizer.model) from the pinned source mirror into a repo of ours, with the
attribution CC-BY-4.0 requires and Kyutai's usage policy, then prints the two
config lines to switch to it (`pocket_repo`, `pocket_revision`).

    pip install huggingface_hub
    hf auth login                      # a token with WRITE access to the target
    python tools/pocket-mirror.py <owner>/pocket-tts-mirror

The files are copied byte for byte (no changes), so the existing configs keep
working; only the source of the download moves.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

SOURCE_REPO = "lunahr/pocket-tts-ungated"
SOURCE_REVISION = "d03cd73415a8d46d8eb115c7b524aebb0a729f4a"
# 24-layer models (default) and 6-layer models (option `pocket_layers: 6`); the
# source has no English 24-layer and no French 6-layer model.
FOLDERS = ["german_24l", "french_24l", "italian_24l", "spanish_24l", "portuguese_24l",
           "german", "italian", "spanish", "portuguese", "english"]
FILES = ["model.safetensors", "tokenizer.model"]

README = f"""---
license: cc-by-4.0
---

# Pocket TTS voice-cloning models (mirror)

Unmodified copies of Kyutai's **Pocket TTS** voice-cloning models
(<https://huggingface.co/kyutai/pocket-tts>, code: <https://github.com/kyutai-labs/pocket-tts>),
as used by the EchokrauTTS wrapper. Copied from `{SOURCE_REPO}` at commit `{SOURCE_REVISION}`;
languages: {", ".join(FOLDERS)}.

**Authors / copyright: Kyutai.** Licensed under
[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
No changes were made to the files.

## Usage policy (from Kyutai's model page)

Use of the model must comply with all applicable laws and regulations and must not result in, involve,
or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses
include, without limitation, voice impersonation or cloning without explicit and lawful consent;
misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting
generated content as genuine recordings of real people or events); and the generation of unlawful,
harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content.
"""


def main(target: str) -> int:
    from huggingface_hub import HfApi, snapshot_download

    patterns = [f"languages/{folder}/{name}" for folder in FOLDERS for name in FILES]
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "mirror"
        print(f"Lade {len(patterns)} Dateien aus {SOURCE_REPO}@{SOURCE_REVISION[:7]} …")
        snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION, allow_patterns=patterns, local_dir=str(local))
        missing = [p for p in patterns if not (local / p).is_file()]
        if missing:
            print("Fehlt in der Quelle:", ", ".join(missing))
            return 1
        (local / "README.md").write_text(README, encoding="utf-8")
        api = HfApi()
        api.create_repo(target, repo_type="model", exist_ok=True)
        print(f"Lade nach {target} hoch …")
        commit = api.upload_folder(repo_id=target, folder_path=str(local),
                                   commit_message=f"Mirror of {SOURCE_REPO}@{SOURCE_REVISION[:7]} (CC-BY-4.0, Kyutai)")
    revision = commit.oid
    print("\nFertig. In wrapper/config.json (und src/config.py als Default) eintragen:")
    print(f'  "pocket_repo": "{target}",')
    print(f'  "pocket_revision": "{revision}",')
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
