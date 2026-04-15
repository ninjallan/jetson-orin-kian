"""Text-to-speech using Piper with language-aware voice selection."""

import asyncio
import queue
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pulsectl
import sounddevice as sd
from piper import PiperVoice
from piper.config import SynthesisConfig

import kian.llm as llm_mod
from kian.latex_to_speech import latex_to_speech

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TTS_SAMPLE_RATE = 22050

VOLUME_WHISPER = 0.30
VOLUME_INSIDE = 0.50
VOLUME_LOUD = 0.80
VOLUME_SHOUT = 1.0


class TTSPlayer:
    """Synthesizes text and plays audio via a background thread."""

    def __init__(self, speed: float = 1.0, pitch_shift: float = 1.0):
        self._voice_cache: dict[str, PiperVoice] = {}
        self._voice_paths = {
            "en": self._resolve_voice(getattr(llm_mod, "voice_en", "en_GB-alan-medium")),
            "da": self._resolve_voice(getattr(llm_mod, "voice_da", "da_DK-talesyntese-medium")),
        }
        print(f"[TTS] English voice: {self._voice_paths['en'].stem}")
        print(f"[TTS] Danish voice: {self._voice_paths['da'].stem}")
        self._syn_config = SynthesisConfig(length_scale=1.0 / speed)
        self._playback_rate = int(TTS_SAMPLE_RATE * pitch_shift)
        self._beep_audio = self._load_beep()
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    @staticmethod
    def _resolve_voice(stem: str) -> Path:
        path = MODELS_DIR / f"{stem}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"Missing Piper voice model: {path}")
        return path

    def _load_voice(self, language: str) -> PiperVoice:
        key = "da" if language.lower().startswith("da") else "en"
        if key not in self._voice_cache:
            self._voice_cache[key] = PiperVoice.load(str(self._voice_paths[key]))
        return self._voice_cache[key]

    @staticmethod
    def _load_beep() -> np.ndarray:
        beep_path = PROJECT_ROOT / "beepboop.wav"
        with wave.open(str(beep_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def _playback_worker(self):
        self._play_buf: list[np.ndarray] = []
        self._play_lock = threading.Lock()

        def _callback(outdata, frames, time_info, status):
            written = 0
            with self._play_lock:
                while written < frames and self._play_buf:
                    chunk = self._play_buf[0]
                    n = min(frames - written, len(chunk))
                    outdata[written:written + n, 0] = chunk[:n]
                    if n < len(chunk):
                        self._play_buf[0] = chunk[n:]
                    else:
                        self._play_buf.pop(0)
                    written += n
            if written < frames:
                outdata[written:, 0] = 0.0

        with sd.OutputStream(
            samplerate=self._playback_rate,
            channels=1,
            dtype="float32",
            callback=_callback,
            latency="high",
        ):
            time.sleep(0.2)
            while True:
                chunk = self._audio_queue.get()
                if chunk is None:
                    break
                with self._play_lock:
                    self._play_buf.append(chunk.copy())
                while True:
                    with self._play_lock:
                        if not self._play_buf:
                            break
                    time.sleep(0.005)
                self._audio_queue.task_done()

    def _synthesize(self, text: str, language: str) -> np.ndarray:
        voice = self._load_voice(language)
        chunks = []
        for audio_chunk in voice.synthesize(text, self._syn_config):
            chunks.append(audio_chunk.audio_float_array)
        return np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)

    PRONUNCIATION = {
        "Philipedia": "Philip eedia",
        "Sir Philip": "Sir Philip",
        "1920s": "nineteen twenties",
        "1930s": "nineteen thirties",
        "1940s": "nineteen forties",
        "1950s": "nineteen fifties",
        "1960s": "nineteen sixties",
        "1970s": "nineteen seventies",
        "1980s": "nineteen eighties",
        "1990s": "nineteen nineties",
    }
    WORD_FILTER = {"ass": "donkey", "butt": "behind"}
    _WORD_FILTER_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in WORD_FILTER) + r")\b", re.IGNORECASE)
    _DIMENSIONS_RE = re.compile(r"\b(\d+)\s*x\s*(\d+)\b", re.IGNORECASE)
    _YEAR_19XX_RE = re.compile(r"\b19(\d\d)\b")

    def _fix_pronunciation(self, text: str) -> str:
        text = latex_to_speech(text)
        text = text.replace("*", "")
        text = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\ufe0f]', '', text)
        text = self._DIMENSIONS_RE.sub(r"\1 by \2", text)
        text = self._WORD_FILTER_RE.sub(lambda m: self.WORD_FILTER[m.group(0).lower()], text)
        for word, replacement in self.PRONUNCIATION.items():
            text = text.replace(word, replacement)
        text = self._YEAR_19XX_RE.sub(r"nineteen \1", text)
        return text

    async def speak(self, text: str, language: str = "en", tail_silence: float = 0.0):
        text = self._fix_pronunciation(text)
        if not text.strip():
            return
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, self._synthesize, text, language)
        if tail_silence > 0:
            silence = np.zeros(int(self._playback_rate * tail_silence), dtype=np.float32)
            audio = np.concatenate([audio, silence])
        self._audio_queue.put(audio)

    def beep(self):
        self._audio_queue.put(self._beep_audio)

    def set_volume(self, level: float, persist: bool = True):
        with pulsectl.Pulse("kian") as pulse:
            sink = pulse.get_sink_by_name(pulse.server_info().default_sink_name)
            pulse.volume_set_all_chans(sink, level)
        print(f"[TTS] volume: {level:.0%}")
        if persist:
            llm_mod.volume = level
            llm_mod.save_settings()

    def set_voice(self, language: str, stem: str):
        key = "da" if language.lower().startswith("da") else "en"
        path = self._resolve_voice(stem)
        self._voice_paths[key] = path
        if key in self._voice_cache:
            del self._voice_cache[key]
        if key == "da":
            llm_mod.voice_da = stem
        else:
            llm_mod.voice_en = stem
        llm_mod.save_settings()
        print(f"[TTS] {key} voice: {stem}")

    def flush(self):
        with self._play_lock:
            self._play_buf.clear()
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except queue.Empty:
                break

    def drain(self):
        self._audio_queue.join()

    def reset_synth(self):
        self._voice_cache.clear()

    def stop(self):
        self._audio_queue.put(None)
        self._playback_thread.join(timeout=3)
