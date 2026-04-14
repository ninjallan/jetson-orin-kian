#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/../../models" && pwd)"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "Downloading Philipedia models to $MODELS_DIR ..."

if [ ! -f silero_vad.onnx ]; then
    echo "Downloading Silero VAD model..."
    wget -q --show-progress https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
else
    echo "Silero VAD already downloaded, skipping."
fi

if [ ! -f en_GB-alan-medium.onnx ]; then
    echo "Downloading English Piper voice..."
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json
else
    echo "English Piper voice already downloaded, skipping."
fi

if [ ! -f da_DK-talesyntese-medium.onnx ]; then
    echo "Downloading Danish Piper voice..."
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/main/da/da_DK/talesyntese/medium/da_DK-talesyntese-medium.onnx
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/main/da/da_DK/talesyntese/medium/da_DK-talesyntese-medium.onnx.json
else
    echo "Danish Piper voice already downloaded, skipping."
fi

if [ ! -f qwen3-4b-instruct-2507-q4_k_m.gguf ]; then
    echo "Downloading Qwen3-4B Instruct 2507 Q4_K_M..."
    wget -q --show-progress https://huggingface.co/enacimie/Qwen3-4B-Instruct-2507-Q4_K_M-GGUF/resolve/main/qwen3-4b-instruct-2507-q4_k_m.gguf
else
    echo "Qwen3-4B Instruct 2507 already downloaded, skipping."
fi

mkdir -p granite-guardian-hap
if [ ! -f granite-guardian-hap/guardian_model_quantized.onnx ]; then
    echo "Downloading Granite Guardian HAP..."
    wget -q --show-progress -O granite-guardian-hap/guardian_model_quantized.onnx \
        https://huggingface.co/KantiArumilli/granite-guardian-hap-38m-onnx/resolve/main/guardian_model_quantized.onnx
    wget -q --show-progress -O granite-guardian-hap/tokenizer.json \
        https://huggingface.co/KantiArumilli/granite-guardian-hap-38m-onnx/resolve/main/tokenizer/tokenizer.json
else
    echo "Granite Guardian HAP already downloaded, skipping."
fi

echo ""
echo "Whisper multilingual model is downloaded automatically on first run by faster-whisper."
echo ""
echo "IMPORTANT: place your trained openWakeWord model here:"
echo "  $MODELS_DIR/sir_philip.onnx"
echo ""
echo "Done."
ls -lh "$MODELS_DIR"
