"""Speech-to-text using faster-whisper."""

import asyncio
import os
from dataclasses import dataclass
from functools import partial

from faster_whisper import WhisperModel

MODEL_SIZE = os.environ.get("KIAN_STT_MODEL", "small")
MODEL_DEVICE = os.environ.get("KIAN_STT_DEVICE", "auto")
MODEL_COMPUTE_TYPE = os.environ.get("KIAN_STT_COMPUTE_TYPE", "int8_float16")


@dataclass(slots=True)
class Transcription:
    text: str
    language: str
    language_probability: float


class STT:
    def __init__(
        self,
        model_size: str = MODEL_SIZE,
        device: str = MODEL_DEVICE,
        compute_type: str = MODEL_COMPUTE_TYPE,
    ):
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    async def transcribe(self, audio_f32_16k) -> Transcription:
        """Transcribe a float32 16kHz numpy array and return text plus detected language."""
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            partial(
                self._model.transcribe,
                audio_f32_16k,
                beam_size=1,
                task="transcribe",
                vad_filter=True,
                condition_on_previous_text=False,
            ),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcription(
            text=text,
            language=(info.language or "en").lower(),
            language_probability=float(info.language_probability or 0.0),
        )
