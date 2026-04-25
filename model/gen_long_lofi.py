#!/usr/bin/env python3
"""Generate a ~3-minute lofi track: 6 segments × 30s, stitched with 2s crossfade.
Simple approach: same prompt, crossfade hides discontinuities."""

import torch
import os
import subprocess
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "/Users/alxy/Desktop/1PROJ/MiniMaxSearch/lofi_tracks"
PROMPT = ("chill lofi hip hop beats with vinyl crackle and rain sounds, "
          "soft piano chords, dusty drum breaks, warm double bass, jazz guitar, "
          "relaxed study music, cozy coffee shop atmosphere, 85 bpm")
NUM_SEGMENTS = 6
CROSSFADE_SEC = 2

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Segments: {NUM_SEGMENTS} × 30s with {CROSSFADE_SEC}s crossfade → "
      f"~{(NUM_SEGMENTS-1)*(30-CROSSFADE_SEC)+30}s total\n")

# Load model
print("Loading MusicGen-medium...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
model = model.to(DEVICE)
model.eval()
print("Ready ✓\n")

inputs = processor(text=[PROMPT], padding=True, return_tensors="pt")
input_ids = inputs["input_ids"].to(DEVICE)

segment_files = []

with torch.no_grad():
    for i in range(1, NUM_SEGMENTS + 1):
        print(f"[{i}/{NUM_SEGMENTS}] Generating segment {i}...", end="", flush=True)

        audio_values = model.generate(
            input_ids=input_ids,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=1536,
        )

        seg_path = f"{OUT_DIR}/seg_{i:02d}.wav"
        wavfile.write(seg_path, 32000, audio_values[0, 0].cpu().numpy())
        segment_files.append(seg_path)
        dur = len(audio_values[0, 0]) / 32000
        print(f" ✓ {dur:.1f}s")

print(f"\n✓ All {NUM_SEGMENTS} segments generated")
print("Stitching with ffmpeg crossfade...")

# Build ffmpeg acrossfade chain
chain_parts = []
prev_label = f"[{0}:a]"
for i in range(1, NUM_SEGMENTS):
    curr_label = f"[{i}:a]"
    out_label = f"[{'0' * i}{i}]" if i < NUM_SEGMENTS - 1 else "[out]"
    chain_parts.append(f"{prev_label}{curr_label}acrossfade=d={CROSSFADE_SEC}:c1=tri{out_label}")
    prev_label = out_label

filter_chain = ";".join(chain_parts)

input_args = []
for seg in segment_files:
    input_args += ["-i", seg]

output_path = f"{OUT_DIR}/lofi_3min.wav"
cmd = [
    "ffmpeg", "-y", *input_args,
    "-filter_complex", filter_chain,
    "-map", "[out]", "-ar", "44100", "-ac", "2", "-loglevel", "error",
    output_path,
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    mb = os.path.getsize(output_path) / 1024 / 1024
    total = (NUM_SEGMENTS - 1) * (30 - CROSSFADE_SEC) + 30
    print(f"\n✅ Done! → {output_path}")
    print(f"   Duration: ~{total}s with {CROSSFADE_SEC}s crossfade")
    print(f"   Size: {mb:.1f} MB")
else:
    print(f"FFmpeg error: {result.stderr[-500:]}")
    # Fallback: simple concat
    concat_txt = f"{OUT_DIR}/concat.txt"
    with open(concat_txt, "w") as f:
        for seg in segment_files:
            f.write(f"file '{seg}'\n")
    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-ar", "44100", "-ac", "2", output_path,
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Done (concat)! → {output_path}")
    else:
        print(f"Error: {result.stderr[-300:]}")

for seg in segment_files:
    try:
        os.remove(seg)
    except:
        pass
print("Segments cleaned up.")
