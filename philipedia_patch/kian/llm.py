"""LLM backend selection and common interface."""

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

_PROMPT_BASE = (
    "You are {name}, a dignified and discreet household companion. "
    "You were awakened by the phrase '{wake_phrase}'. "
    "Your replies will be spoken aloud, so never use markdown, bullet points, emojis, "
    "or stage directions. Speak naturally and clearly. "
    "Reply in Danish when the user speaks Danish, and in English when the user speaks English. "
    "Maintain a refined but warm tone, like a trusted butler or cultured aide. "
    "Never sound childish, cartoonish, overly excited, or like a generic chatbot. "
    "{length_hint}"
)

_HINT_SHORT = "Prefer one to three concise sentences unless the user clearly asks for more detail."

_HINT_EXPLAIN = (
    "Give a well-structured spoken answer of roughly 40 to 140 words. "
    "Be clear, accurate, calm, and conversational."
)

_HINT_STORY = (
    "Answer with richer detail and atmosphere, but still keep the tone composed and elegant. "
    "Avoid slang and avoid sounding theatrical."
)

_STORY_RE = re.compile(
    r"tell me a story|make up a story|imagine|paint a picture|scene|dialogue|describe a|describe the",
    re.IGNORECASE,
)

_EXPLAIN_RE = re.compile(
    r"explain|teach me|tell me about|how does|how do|why does|why do|what is|who is|where is|can you clarify",
    re.IGNORECASE,
)

assistant_name = "Philipedia"
wake_phrase = "Sir Philip"
voice_en = "en_GB-alan-medium"
voice_da = "da_DK-talesyntese-medium"
volume = None


def _pick_length_hint(user_text: str, has_wiki: bool) -> tuple[str, str]:
    """Choose response length hint based on user input. Returns (hint, mode_name)."""
    if _STORY_RE.search(user_text):
        return _HINT_STORY, "story"
    if has_wiki or _EXPLAIN_RE.search(user_text):
        return _HINT_EXPLAIN, "explain"
    return _HINT_SHORT, "short"


def load_settings() -> None:
    global assistant_name, wake_phrase, voice_en, voice_da, volume
    if SETTINGS_PATH.exists():
        data = json.loads(SETTINGS_PATH.read_text())
        assistant_name = data.get("assistant_name", assistant_name)
        wake_phrase = data.get("wake_phrase", wake_phrase)
        voice_en = data.get("voice_en", voice_en)
        voice_da = data.get("voice_da", voice_da)
        volume = data.get("volume", volume)


def save_settings() -> None:
    SETTINGS_PATH.write_text(
        json.dumps(
            {
                "assistant_name": assistant_name,
                "wake_phrase": wake_phrase,
                "voice_en": voice_en,
                "voice_da": voice_da,
                "volume": volume,
            },
            indent=2,
        )
        + "\n"
    )


load_settings()


def system_prompt(length_hint: str = _HINT_SHORT) -> str:
    return _PROMPT_BASE.format(
        name=assistant_name,
        wake_phrase=wake_phrase,
        length_hint=length_hint,
    )


MIN_PHRASE_WORDS = 4


def _longest_common_phrase(a: str, b: str) -> str | None:
    """Find the longest common multi-word phrase between two strings."""
    words_a = a.lower().split()
    words_b = b.lower().split()
    if len(words_a) < MIN_PHRASE_WORDS or len(words_b) < MIN_PHRASE_WORDS:
        return None

    b_ngrams: set[tuple[str, ...]] = set()
    for n in range(MIN_PHRASE_WORDS, len(words_b) + 1):
        for i in range(len(words_b) - n + 1):
            b_ngrams.add(tuple(words_b[i : i + n]))

    for n in range(len(words_a), MIN_PHRASE_WORDS - 1, -1):
        for i in range(len(words_a) - n + 1):
            ngram = tuple(words_a[i : i + n])
            if ngram in b_ngrams:
                return " ".join(ngram)
    return None


def update_system_prompt(history: list[dict], user_text: str = "", has_wiki: bool = False) -> None:
    """Rewrite the system prompt with the appropriate length hint and repetition avoidance."""
    hint, mode = _pick_length_hint(user_text, has_wiki)
    print(f"[MODE] {mode}")
    assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
    base = system_prompt(length_hint=hint)

    if len(assistant_msgs) >= 2:
        phrase = _longest_common_phrase(assistant_msgs[-1], assistant_msgs[-2])
        if phrase:
            base += f" Try to avoid repeating the phrase: '{phrase}'."

    history[0] = {"role": "system", "content": base}


class LLMBackend(Protocol):
    async def chat_stream(self, user_text: str) -> AsyncIterator[str]: ...
    def reset(self) -> None: ...


def create_llm(backend: str = "llamacpp", model: str | None = None) -> LLMBackend:
    """Factory: instantiate the requested LLM backend."""
    if backend == "llamacpp":
        from kian.llm_llamacpp import LlamaLLM

        return LlamaLLM(model_path=model)
    if backend == "ollama":
        from kian.llm_ollama import OllamaLLM

        return OllamaLLM(model=model)
    raise ValueError(f"Unknown backend: {backend!r}  (choose 'llamacpp' or 'ollama')")
