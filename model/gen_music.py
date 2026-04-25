#!/usr/bin/env python3
"""Generate lofi music using MusicGen-medium via HuggingFace Transformers on Apple MPS."""

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import os
import numpy as np

# ── Device setup ──────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading MusicGen-medium (1.5B)...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")

# MPS needs float32 (or float16 with care)
model = model.to(dtype=torch.float32, device=device)
model.eval()
print("Model loaded ✓")

# ── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 32000
DURATION = 30  # seconds per track (MusicGen max per call)

prompts = [
    "chill lofi hip hop beats with vinyl crackle and rain sounds, relaxing study music",
    "slow mellow lofi with soft piano chords, dusty drums and jazz guitar samples",
    "dreamy lofi sunset beats with muted saxophone and warm bass, chill vibes",
    "lofi chillhop with relaxing bell melody, lo-fi drums and soft texture",
    "focused lofi study beats with gentle guitar chords and calm drum pattern",
]

output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# ── Generate ──────────────────────────────────────────────────────────────────
for i, prompt in enumerate(prompts, 1):
    print(f"\n[{i}/5] \"{prompt[:55]}...\"")

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=torch.float32) for k, v in inputs.items()}

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=768,  # ~30 sec at 32kHz/hop32
        )

    audio_np = audio_values[0, 0].cpu().float().numpy()
    out_path = os.path.join(output_dir, f"lofi_track_{i:02d}.wav")
    wavfile.write(out_path, SAMPLE_RATE, audio_np)

    dur = len(audio_np) / SAMPLE_RATE
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  ✓ {out_path}  ({dur:.1f}s, {mb:.1f} MB)")

print(f"\nAll 5 tracks done → {output_dir}/")
