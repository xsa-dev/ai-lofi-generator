# AI Lofi Generator

Generate lofi music tracks using Meta MusicGen on Apple Silicon (MPS).

## Scripts

| File | Description |
|------|-------------|
| `model/gen_test.py` | Generate single 30s lofi track (testing) |
| `model/gen_5.py` | Generate 5 lofi tracks (30s each) |
| `model/gen_long_lofi.py` | Generate 3-min track (6×30s segments + ffmpeg crossfade) |
| `model/gen_music.py` | Custom prompt generation |
| `model/gen_continue.py` | Attempt audio continuation (limited by HF transformers API) |
| `model/mubert_gen.py` | Mubert B2B API generator (needs valid license) |

## Audio Samples

- `model/lofi_tracks/lofi_3min.wav` — 3-minute generated lofi track

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch transformers scipy soundfile sounddevice

# For Mubert API (optional)
uv pip install httpx sentence-transformers
```

## Hardware

- Apple M1 Max 32GB
- Uses MPS (Metal Performance Shaders) for GPU acceleration
- ~5 min per 30s segment on MusicGen-medium (1.5B params)

## Lofi Prompts Used

- "lofi hip hop vinyl crackle rain outside window soft piano dusty drums"
- "lofi chillout soft piano with dusty drums and vinyl crackle"
- "sunset sax gentle lofi"
- "lofi bell melody rain ambient"
- "lofi hip hop gentle guitar rain"

## Notes

- MusicGen generates max 30s per call via HuggingFace transformers
- Longer tracks made by generating segments + crossfading with ffmpeg
- Original Meta audiocraft doesn't build on M1 Max (PyAV compilation fails)
