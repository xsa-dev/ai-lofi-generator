#!/usr/bin/env python3
"""Generate 5 lofi tracks using MusicGen-medium on Apple MPS."""

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

SAMPLE_RATE = 32000

prompts = [
    "chill lofi hip hop beats with vinyl crackle and rain sounds, relaxing study music",
    "slow mellow lofi with soft piano chords, dusty drums and jazz guitar samples",
    "dreamy lofi sunset beats with muted saxophone and warm bass, chill vibes",
    "lofi chillhop with relaxing bell melody, lo-fi drums and soft texture",
    "focused lofi study beats with gentle guitar chords and calm drum pattern",
]

out_dir = "/Users/alxy/Desktop/1PROJ/MiniMaxSearch/lofi_tracks"
os.makedirs(out_dir, exist_ok=True)

for i, prompt in enumerate(prompts, 1):
    print(f"[{i}/5] {prompt[:55]}...")
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=1536)
    audio_np = audio[0, 0].cpu().numpy()
    path = os.path.join(out_dir, f"lofi_{i:02d}.wav")
    wavfile.write(path, SAMPLE_RATE, audio_np)
    dur = len(audio_np) / SAMPLE_RATE
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✓ {dur:.1f}s, {mb:.1f} MB")

print(f"\nAll done → {out_dir}/")
