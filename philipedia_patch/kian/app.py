"""Philipedia voice assistant - wake word pipeline."""

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

from kian import leds
from kian import llm as llm_mod
from kian.llm import create_llm
from kian.mic import mic_rms
from kian.naughty import NaughtyDetector
from kian.stt import STT
from kian.tts import TTSPlayer, VOLUME_INSIDE, VOLUME_LOUD, VOLUME_SHOUT, VOLUME_WHISPER
from kian.vad import VADStream
from kian.wake import WakeWordDetector
from kian.wiki import WikiLookup
from kian import safety

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BARGEIN_THRESHOLD = 0.05
SPLIT_RE = re.compile(r"([.!?])\s+")
MIN_CHUNK_LEN = 25
MAX_CHUNK_LEN = 400
PAUSE_SILENCE_S = 0.050
BARGEIN_LISTEN_S = 0.250
BARGEIN_COOLDOWN_S = 0.80
SILENCE_RESET_S = 8 * 60

CONTROL_PHRASES = {
    "Whisper": {"whisper", "be quiet", "whisper please", "shh"},
    "Inside voice": {"inside voice", "normal voice", "normal volume"},
    "Loud": {"loud", "be loud", "speak up", "louder"},
    "Shout": {"shout", "yell", "maximum volume", "max volume"},
    "Reset context": {"lets start fresh", "lets start over", "reset"},
}
_STRIP_PUNCT = re.compile(r"[^\w\s]")
_ACTION_MAP = {
    "Whisper": "whisper",
    "Inside voice": "inside_voice",
    "Loud": "loud",
    "Shout": "shout",
    "Reset context": "reset",
}


def _match_control(text: str) -> str | None:
    normalized = _STRIP_PUNCT.sub("", text).lower().strip()
    for label, phrases in CONTROL_PHRASES.items():
        if normalized in phrases:
            return _ACTION_MAP[label]
    return None


def _language_key(language: str) -> str:
    return "da" if language.lower().startswith("da") else "en"


def _is_repetitive(text: str, phrase_len: int = 3, max_repeats: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < phrase_len * max_repeats:
        return False
    window = words[-(phrase_len * max_repeats * 3):]
    counts: dict[tuple, int] = {}
    for i in range(len(window) - phrase_len + 1):
        ngram = tuple(window[i : i + phrase_len])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] >= max_repeats:
            return True
    return False


async def _stream_response(llm, tts, user_text, language, shutdown):
    bargein = asyncio.Event()
    sentence_q: asyncio.Queue[str | None] = asyncio.Queue()
    naughty_hit = False
    safety_hit = False
    llm_loop = False
    full_response: list[str] = []

    async def producer():
        nonlocal naughty_hit, safety_hit, llm_loop
        naughty = NaughtyDetector()
        loop = asyncio.get_event_loop()
        buf = ""
        first_token = True
        t_llm = time.monotonic()

        async def _check_and_enqueue(text: str):
            nonlocal safety_hit
            while sentence_q.qsize() >= 2 and not bargein.is_set():
                await asyncio.sleep(0.1)
            if bargein.is_set():
                return True
            safe = await loop.run_in_executor(None, safety.classify, text)
            if not safe:
                print(f"[SAFETY] unsafe -- {text}")
                safety_hit = True
                return False
            await sentence_q.put(text)
            return True

        async for token in llm.chat_stream(user_text):
            if shutdown.is_set() or bargein.is_set():
                break
            if first_token:
                print(f"[LLM {time.monotonic() - t_llm:.1f}s] first token")
                first_token = False
            if naughty.check(token):
                naughty_hit = True
                break
            full_response.append(token)
            buf += token
            if _is_repetitive(buf):
                llm_loop = True
                break
            m = None
            for candidate in SPLIT_RE.finditer(buf):
                if candidate.end() >= MIN_CHUNK_LEN:
                    m = candidate
                    break
            if m:
                fragment = buf[: m.end()].strip()
                buf = buf[m.end() :]
                if fragment and not await _check_and_enqueue(fragment):
                    break
            elif len(buf) >= MAX_CHUNK_LEN:
                if not await _check_and_enqueue(buf.strip()):
                    break
                buf = ""

        if buf.strip() and not bargein.is_set() and not shutdown.is_set() and not naughty_hit and not safety_hit and not llm_loop:
            await _check_and_enqueue(buf.strip())
        if not bargein.is_set():
            await sentence_q.put(None)
        print(f"[LLM {time.monotonic() - t_llm:.1f}s] done ({len(full_response)} tokens)")

    async def consumer():
        loop = asyncio.get_event_loop()
        while True:
            sentence = await sentence_q.get()
            if sentence is None:
                break
            print(f"[TTS/{language}] {sentence}")
            await tts.speak(sentence, language=language, tail_silence=PAUSE_SILENCE_S)
            tts.drain()
            rms = await loop.run_in_executor(None, mic_rms, BARGEIN_LISTEN_S, 0.05)
            if rms >= BARGEIN_THRESHOLD:
                print("[INTERRUPTED]")
                tts.flush()
                bargein.set()
                break

    await asyncio.gather(asyncio.create_task(producer()), asyncio.create_task(consumer()))
    return bargein.is_set(), naughty_hit or safety_hit, llm_loop, full_response


async def watch_quit(shutdown: asyncio.Event):
    loop = asyncio.get_event_loop()

    def _read_stdin():
        for line in sys.stdin:
            if line.strip().lower() == "q":
                return

    await loop.run_in_executor(None, _read_stdin)
    print("\nShutting down...")
    shutdown.set()


async def wait_for_command_audio(wake, vad, shutdown):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, wake.wait_for_wake)
    if shutdown.is_set():
        return None
    t0 = time.monotonic()
    async for speech_audio in vad.stream_speech():
        return speech_audio
    print(f"[WAKE] no command captured after {time.monotonic() - t0:.1f}s")
    return None


async def pipeline(backend: str = "llamacpp", model: str | None = None):
    shutdown = asyncio.Event()
    print("Loading models...")
    safety.load()
    vad = VADStream()
    wake = WakeWordDetector()
    stt = STT()
    llm = create_llm(backend=backend, model=model)
    tts = TTSPlayer(pitch_shift=1.05)
    wiki = WikiLookup()
    from kian.llm import volume as saved_volume
    tts.set_volume(saved_volume if saved_volume is not None else VOLUME_INSIDE, persist=False)
    leds.idle()
    print(f"Ready. Wake word: {llm_mod.wake_phrase}. Persona: {llm_mod.assistant_name}.")
    quit_task = None
    if sys.stdin.isatty():
        print("(press Q + Enter to quit)")
        quit_task = asyncio.create_task(watch_quit(shutdown))
    last_speech = time.monotonic()

    while not shutdown.is_set():
        leds.idle()
        speech_audio = await wait_for_command_audio(wake, vad, shutdown)
        if shutdown.is_set():
            break
        if speech_audio is None:
            await asyncio.sleep(0.1)
            continue
        now = time.monotonic()
        if now - last_speech > SILENCE_RESET_S:
            llm.reset()
            wiki.reset()
            print("[RESET] conversation context cleared (8 min silence)")
        last_speech = now
        leds.busy()
        tts.beep()
        audio_sec = len(speech_audio) / 16000
        print(f"\n[VAD {audio_sec:.1f}s]")
        t0 = time.monotonic()
        tx = await stt.transcribe(speech_audio)
        stt_s = time.monotonic() - t0
        if not tx.text.strip():
            print(f"[STT {stt_s:.1f}s] (empty)")
            continue
        language = _language_key(tx.language)
        print(f"[STT {stt_s:.1f}s] ({tx.language} {tx.language_probability:.2f}) {tx.text}")
        rms = await asyncio.get_event_loop().run_in_executor(None, mic_rms, BARGEIN_LISTEN_S, 0.05)
        if rms >= BARGEIN_THRESHOLD:
            print(f"[STT barge-in RMS {rms:.3f} — still speaking, re-listening]")
            continue
        input_naughty = NaughtyDetector()
        if any(input_naughty.check(w) for w in tx.text.split()):
            await tts.speak("I am not quite sure what I am hearing." if language == "en" else "Jeg er ikke helt sikker på, hvad jeg hører.", language=language)
            tts.drain()
            continue
        ctrl = _match_control(tx.text)
        if ctrl:
            print(f"[CTRL] {ctrl}")
            if ctrl == "reset":
                llm.reset(); wiki.reset()
                response = "Starting fresh." if language == "en" else "Vi starter forfra."
            elif ctrl == "whisper":
                tts.set_volume(VOLUME_WHISPER)
                response = "Very well. I shall whisper." if language == "en" else "Naturligvis. Jeg taler lavt."
            elif ctrl == "inside_voice":
                tts.set_volume(VOLUME_INSIDE)
                response = "Of course." if language == "en" else "Naturligvis."
            elif ctrl == "loud":
                tts.set_volume(VOLUME_LOUD)
                response = "Certainly. Louder now." if language == "en" else "Selvfølgelig. Højere nu."
            else:
                tts.set_volume(VOLUME_SHOUT)
                response = "As you wish." if language == "en" else "Som De ønsker."
            await tts.speak(response, language=language)
            tts.drain()
            continue
        wiki_ctx = wiki.search(tx.text, debug=True) if wiki.available else None
        if wiki_ctx:
            title_line = wiki_ctx.splitlines()[0]
            wiki_title = title_line.removeprefix("[Reference: ").removesuffix("]")
            print(f"[WIKI] {wiki_title}")
            llm.set_wiki_context(wiki_ctx, title=wiki_title)
        else:
            llm.set_wiki_context(None)
        interrupted, naughty_hit, llm_loop, full_response = await _stream_response(llm, tts, tx.text, language, shutdown)
        if naughty_hit:
            llm.reset(); wiki.reset()
            await tts.speak("I do not think I should answer that." if language == "en" else "Det mener jeg ikke, at jeg bør svare på.", language=language)
            tts.drain()
        if llm_loop:
            llm.reset(); wiki.reset()
            await tts.speak("I lost my train of thought. Would you ask again?" if language == "en" else "Jeg mistede tråden et øjeblik. Vil De spørge igen?", language=language)
            tts.drain()
        if interrupted:
            await asyncio.sleep(BARGEIN_COOLDOWN_S)

    if quit_task is not None:
        quit_task.cancel()
    tts.stop()
    leds.off()
    print("Goodbye.")


def parse_args():
    p = argparse.ArgumentParser(description="Philipedia voice assistant")
    p.add_argument("--backend", choices=["llamacpp", "ollama"], default="llamacpp")
    p.add_argument("--model", help="Model path or identifier")
    p.add_argument("--speaker", help="Substring to match PulseAudio sink name")
    p.add_argument("--mic", help="Substring to match PulseAudio source name")
    return p.parse_args()


def setup_audio_devices(speaker_match: str | None, mic_match: str | None):
    if not speaker_match and not mic_match:
        return
    import os
    import pulsectl
    for attempt in range(15):
        with pulsectl.Pulse("kian-setup") as pulse:
            sink_name = None
            source_name = None
            if speaker_match:
                for s in pulse.sink_list():
                    if speaker_match.lower() in s.name.lower():
                        sink_name = s.name
                        break
            if mic_match:
                for s in pulse.source_list():
                    if ".monitor" in s.name:
                        continue
                    if mic_match.lower() in s.name.lower():
                        source_name = s.name
                        break
            if (sink_name or not speaker_match) and (source_name or not mic_match):
                if sink_name:
                    os.environ["PULSE_SINK"] = sink_name
                    print(f"[AUDIO] speaker: {sink_name}")
                if source_name:
                    os.environ["PULSE_SOURCE"] = source_name
                    print(f"[AUDIO] mic: {source_name}")
                return
        print(f"[AUDIO] waiting for devices... (attempt {attempt + 1}/15)")
        time.sleep(2)
    print("[AUDIO] WARNING: devices not found after 30s")


def main():
    args = parse_args()
    setup_audio_devices(args.speaker, args.mic)
    try:
        asyncio.run(pipeline(backend=args.backend, model=args.model))
    except KeyboardInterrupt:
        pass
    finally:
        leds.off()
        print("\nGoodbye.")
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
