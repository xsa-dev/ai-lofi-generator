#!/usr/bin/env python3
"""Test MusicGen on MPS — generate one 30s lofi track."""

import torch
import os
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

print("MPS available:", torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

print("Loading model (1.5B)... (~3GB download if not cached)")
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
print("Model checkpoint loaded, moving to device...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
model = model.to(device)
print("Model ready on MPS ✓")

prompt = "chill lofi hip hop beat with vinyl crackle, soft piano, relaxing study music"
print(f"\nGenerating: {prompt}")

inputs = processor(text=[prompt], padding=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Synthesizing... (this takes 1-2 min on MPS)")
with torch.no_grad():
    audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=768)

print("Encoding audio...")
audio_np = audio[0, 0].cpu().numpy()
out = os.path.join(os.path.dirname(__file__), "lofi_track_test.wav")
wavfile.write(out, 32000, audio_np)
print(f"✓ Saved: {out}")
print(f"  Duration: {len(audio_np)/32000:.1f}s, Size: {os.path.getsize(out)/1024/1024:.1f} MB")
