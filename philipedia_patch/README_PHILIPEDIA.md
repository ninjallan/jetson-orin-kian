# Philipedia patch for Jetson Orin Kian

This patch turns the original Kian demo into a bilingual Danish/English assistant with:

- wake word: `Sir Philip`
- spoken persona: `Philipedia`
- multilingual speech recognition
- separate Danish and English Piper voices
- openWakeWord integration for a custom wake model

## Important limitation

The GitHub connector used for this patch can create new files on the branch, but it cannot overwrite the existing tracked files in-place through this interface. Because of that, the replacement files are stored under `philipedia_patch/`.

Copy them over the originals on your local checkout before testing.

## Files in this patch

- `philipedia_patch/kian/llm.py`
- `philipedia_patch/kian/stt.py`
- `philipedia_patch/kian/tts.py`
- `philipedia_patch/kian/wake.py`
- `philipedia_patch/kian/app.py`
- `philipedia_patch/pyproject.toml`
- `philipedia_patch/scripts/download-models.sh`
- `settings.example.json`

## Install on the Orin

Clone your branch or pull the latest changes:

```bash
git clone -b dev/philipedia https://github.com/ninjallan/jetson-orin-kian.git
cd jetson-orin-kian
```

If you already have the repo locally:

```bash
git fetch origin
git checkout dev/philipedia
git pull
```

Copy the patch files into place:

```bash
cp philipedia_patch/kian/llm.py kian/llm.py
cp philipedia_patch/kian/stt.py kian/stt.py
cp philipedia_patch/kian/tts.py kian/tts.py
cp philipedia_patch/kian/wake.py kian/wake.py
cp philipedia_patch/kian/app.py kian/app.py
cp philipedia_patch/pyproject.toml pyproject.toml
cp philipedia_patch/scripts/download-models.sh scripts/download-models.sh
cp settings.example.json settings.json
```

Install system packages:

```bash
sudo apt update
sudo apt install -y libportaudio2 libsndfile1 pulseaudio-utils
```

Install Python dependencies:

```bash
uv sync
```

Rebuild llama-cpp-python with CUDA:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --force-reinstall --no-binary llama-cpp-python --no-cache llama-cpp-python
```

Download models:

```bash
bash scripts/download-models.sh
```

## Wake word model

You must provide a trained openWakeWord model for `Sir Philip`.

Place it here:

```bash
models/sir_philip.onnx
```

If you want a different path, export:

```bash
export KIAN_WAKE_MODEL=/full/path/to/your/sir_philip.onnx
```

## Optional STT tuning

Default STT settings are multilingual Whisper `small`.

You can override them:

```bash
export KIAN_STT_MODEL=small
export KIAN_STT_DEVICE=auto
export KIAN_STT_COMPUTE_TYPE=int8_float16
```

## Run

```bash
uv run kian
```

## Expected behavior

- the device idles silently while listening for `Sir Philip`
- after wake word detection, it plays the existing acknowledgement beep
- then it listens for the command
- it replies as `Philipedia`
- it responds in Danish when addressed in Danish
- it responds in English when addressed in English

## Recommended first test phrases

Danish:

```text
Sir Philip
Hvad er klokken?
```

English:

```text
Sir Philip
What is the weather like today?
```

## Recommended voices

In `settings.json`:

```json
{
  "assistant_name": "Philipedia",
  "wake_phrase": "Sir Philip",
  "voice_en": "en_GB-alan-medium",
  "voice_da": "da_DK-talesyntese-medium",
  "volume": 0.5
}
```

## Notes

- The English voice should feel more butler-like than the Danish one.
- The Danish voice is mainly there for clarity and bilingual operation.
- If the wake word is too sensitive, raise the threshold in `kian/wake.py`.
- If Danish STT is weak, try `KIAN_STT_MODEL=medium` and retest latency.
