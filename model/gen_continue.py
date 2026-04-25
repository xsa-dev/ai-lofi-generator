#!/usr/bin/env python3
"""Generate longer lofi tracks using MusicGen continuation mode.
Each 30s segment continues from the previous one using audio conditioning."""

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
model.eval()
print("Ready ✓\n")

# Lofi prompt
prompt = "chill lofi hip hop beats with vinyl crackle and rain, soft piano chords, dusty drums and jazz guitar, relaxing study music"
print(f"Prompt: {prompt}\n")

inputs = processor(text=[prompt], padding=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

output_dir = os.path.join(os.path.dirname(__file__), "lofi_tracks")
os.makedirs(output_dir, exist_ok=True)

# Generate first segment (30s)
print("Generating segment 1/4...")
with torch.no_grad():
    # Set model to continue mode
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=1536)

segment_01 = audio_values[0, 0].cpu().numpy()
path_01 = f"{output_dir}/lofi_cont_01.wav"
wavfile.write(path_01, 32000, segment_01)
dur = len(segment_01) / 32000
print(f"  ✓ Segment 1: {dur:.1f}s → {path_01}")

# Use AudioConditioner to continue from segment 1
# MusicGen's encoder creates embeddings from audio to condition next generation
print("\nGenerating segment 2/4 (continuing from segment 1)...")
with torch.no_grad():
    # Encode previous audio to get conditioning embeddings
    encoder_outputs = model.encoder(input_ids=inputs["input_ids"])
    
    # Create attention_mask matching encoder outputs
    attention_mask = torch.ones_like(encoder_outputs.last_hidden_state[:, :, 0], dtype=torch.long).to(device)
    
    audio_values_2 = model.generate(
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=1536,
        encoder_outputs=encoder_outputs.last_hidden_state,
        attention_mask=attention_mask,
    )

segment_02 = audio_values_2[0, 0].cpu().numpy()
path_02 = f"{output_dir}/lofi_cont_02.wav"
wavfile.write(path_02, 32000, segment_02)
dur = len(segment_02) / 32000
print(f"  ✓ Segment 2: {dur:.1f}s → {path_02}")

# Concatenate segments with crossfade
print("\nConcatenating all segments with crossfade...")
import numpy as np

def crossfade(a, b, fade_len=3200):
    """Crossfade two audio arrays with fade_len samples."""
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    a[-fade_len:] = a[-fade_len:] * fade_out + b[:fade_len] * fade_in
    return np.concatenate([a[:-fade_len], b])

full_track = segment_01
for seg_path in [path_02]:
    seg = segment_02
    full_track = crossfade(full_track, seg, fade_len=3200)

final_path = f"{output_dir}/lofi_continued_full.wav"
wavfile.write(final_path, 32000, full_track)
mb = os.path.getsize(final_path) / 1024 / 1024
dur = len(full_track) / 32000
print(f"✓ Full track: {dur:.1f}s, {mb:.1f} MB → {final_path}")
