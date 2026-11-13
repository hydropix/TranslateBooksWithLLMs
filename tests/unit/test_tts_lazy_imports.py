"""
Importing the TTS package must not load the Chatterbox provider or torch.

`import torch` costs seconds and happens at every CLI and server start, even
with TTS disabled, if the package imports the Chatterbox provider eagerly.
Each check runs in a fresh interpreter with an import hook that records any
attempt to import torch, torchaudio or chatterbox (installed or not).
"""
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = textwrap.dedent(
    """
    import sys

    HEAVY = ("torch", "torchaudio", "chatterbox")
    attempts = []

    class Recorder:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in HEAVY:
                attempts.append(name)
            return None

    sys.meta_path.insert(0, Recorder())

    {body}

    loaded = sorted(m for m in sys.modules if m.split(".")[0] in HEAVY)
    provider_loaded = "src.tts.providers.chatterbox_tts" in sys.modules
    print(repr((attempts, loaded, provider_loaded)))
    """
)


def _run(body: str):
    script = _PROBE.format(body=textwrap.indent(textwrap.dedent(body), "").strip())
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return eval(result.stdout.strip().splitlines()[-1])


def test_importing_tts_package_does_not_touch_chatterbox_or_torch():
    attempts, loaded, provider_loaded = _run(
        """
        import src.tts
        from src.tts import AudioProcessor, TTSConfig, create_tts_provider, generate_tts_for_text
        from src.tts.tts_config import TTSConfig as Config
        import src.tts.providers
        """
    )
    assert attempts == []
    assert loaded == []
    assert provider_loaded is False


def test_chatterbox_names_stay_importable_without_importing_torch():
    attempts, loaded, provider_loaded = _run(
        """
        from src.tts.providers import (
            CHATTERBOX_LANGUAGES,
            CHATTERBOX_MAX_TEXT_LENGTH,
            ChatterboxProvider,
            create_chatterbox_provider,
            get_gpu_status,
            is_chatterbox_available,
            sanitize_text_for_tts,
        )
        assert CHATTERBOX_LANGUAGES and CHATTERBOX_MAX_TEXT_LENGTH > 0
        assert callable(is_chatterbox_available)
        """
    )
    assert provider_loaded is True
    assert attempts == []
    assert loaded == []


def test_availability_check_probes_the_backend_on_demand():
    attempts, _loaded, _provider_loaded = _run(
        """
        from src.tts.providers import is_chatterbox_available
        is_chatterbox_available()
        """
    )
    assert "torch" in attempts


def test_audio_processor_uses_chatterbox_chunk_limit():
    from src.tts.audio_processor import AudioProcessor
    from src.tts.providers.chatterbox_tts import MAX_TEXT_LENGTH
    from src.tts.tts_config import TTSConfig

    class FakeChatterbox:
        name = "chatterbox"

    processor = AudioProcessor(TTSConfig(chunk_size=5000), provider=FakeChatterbox())
    assert processor._effective_chunk_size == MAX_TEXT_LENGTH - 50
