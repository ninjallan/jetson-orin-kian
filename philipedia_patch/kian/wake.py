"""Wake word handling using openWakeWord."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from openwakeword.model import Model

from kian.llm import wake_phrase
from kian.mic import AudioInput, DEVICE_RATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAKE_MODEL = str(PROJECT_ROOT / "models" / "sir_philip.onnx")
DEFAULT_THRESHOLD = float(os.environ.get("KIAN_WAKE_THRESHOLD", "0.45"))
DEFAULT_COOLDOWN_S = float(os.environ.get("KIAN_WAKE_COOLDOWN", "1.5"))
TARGET_RATE = 16000
DOWNSAMPLE_RATIO = DEVICE_RATE // TARGET_RATE


class WakeWordDetector:
    """Continuously watches microphone audio for a custom wake word."""

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
    ):
        self.model_path = model_path or os.environ.get("KIAN_WAKE_MODEL", DEFAULT_WAKE_MODEL)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Wake model not found: {self.model_path}. Provide models/sir_philip.onnx or set KIAN_WAKE_MODEL."
            )
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        model_name = Path(self.model_path).stem
        self._model = Model(wakeword_models=[self.model_path], inference_framework="onnx")
        self._model_name = model_name
        self._last_trigger = 0.0
        self._queue: list[np.ndarray] = []
        self._armed = True

    def _on_audio(self, mono_48k: np.ndarray):
        chunk_16k = mono_48k[::DOWNSAMPLE_RATIO]
        audio_i16 = np.clip(chunk_16k * 32767.0, -32768, 32767).astype(np.int16)
        self._queue.append(audio_i16)

    def wait_for_wake(self) -> None:
        """Block until the wake word is detected."""
        mic = AudioInput.get()
        mic.add_callback(self._on_audio)
        mic.start()
        print(f"[WAKE] listening for '{wake_phrase}' using {Path(self.model_path).name}")
        try:
            while True:
                if not self._queue:
                    time.sleep(0.01)
                    continue
                chunk = self._queue.pop(0)
                scores = self._model.predict(chunk)
                score = float(scores.get(self._model_name, 0.0))
                now = time.monotonic()
                if score >= self.threshold and (now - self._last_trigger) >= self.cooldown_s:
                    self._last_trigger = now
                    print(f"[WAKE] detected '{wake_phrase}' ({score:.3f})")
                    return
        finally:
            mic.remove_callback(self._on_audio)
            self._queue.clear()
