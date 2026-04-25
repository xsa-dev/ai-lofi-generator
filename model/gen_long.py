#!/usr/bin/env python3
"""Generate a longer lofi track (~90s) by setting higher max_new_tokens."""

import torch
import os
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

print("Loading model...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
model = model.to(device)
print("Ready ✓\n")

prompt = "chill lofi hip hop beats with vinyl crackle, soft piano, dusty drums, relaxing study music, warm and cozy atmosphere"
print(f"Prompt: {prompt}\n")

inputs = processor(text=[prompt], padding=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Try 3x the normal tokens (~90 seconds)
MAX_TOKENS = 1536 * 3
print(f"Generating with max_new_tokens={MAX_TOKENS} (~90 sec)...")

with torch.no_grad():
    audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=MAX_TOKENS)

audio_np = audio[0, 0].cpu().numpy()
path = "/Users/alxy/Desktop/1PROJ/MiniMaxSearch/lofi_tracks/lofi_long_test.wav"
wavfile.write(path, 32000, audio_np)
dur = len(audio_np) / 32000
mb = os.path.getsize(path) / 1024 / 1024
print(f"✓ Saved: {path}")
print(f"  Duration: {dur:.1f}s, Size: {mb:.1f} MB")
