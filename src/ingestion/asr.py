"""Local speech-to-text using OpenAI Whisper."""

from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger("edgememory.asr")


class LocalASR:
    """Local speech-to-text using OpenAI Whisper (standard version)."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
    ):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._available = False

    def _load_model(self):
        """Lazy-load Whisper model."""
        if self._model is not None:
            return
        try:
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)
            self._available = True
            logger.info(f"Loaded Whisper model: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        if self._model is None:
            self._load_model()
        return self._available

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe audio file to text."""
        self._load_model()
        if not self._available:
            return {"text": "", "segments": [], "language": "unknown", "error": "Whisper not available"}

        if not Path(audio_path).exists():
            return {"text": "", "segments": [], "language": "unknown", "error": f"File not found: {audio_path}"}

        try:
            result = self._model.transcribe(
                audio_path,
                language=language,
                fp16=False,
            )
            segments = [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
                for seg in result.get("segments", [])
            ]
            return {
                "text": result.get("text", "").strip(),
                "segments": segments,
                "language": result.get("language", "unknown"),
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"text": "", "segments": [], "language": "unknown", "error": str(e)}
