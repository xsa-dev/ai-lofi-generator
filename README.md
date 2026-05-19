# AI Music Generator (Multi-Genre)

Generate instrumental music tracks via MiniMax Music 2.6 API. Supports multiple built-in genres (lofi, hiphop, rock, metal) plus unlimited custom JSON presets.

## Active script

| File | Description |
|------|-------------|
| `model/gen_minimax_lofi.py` | Random generation with built-in skills + custom JSON presets |

## Usage

```bash
export MINIMAX_API_KEY='...'

# 5 tracks (default: lofi)
python model/gen_minimax_lofi.py

# 10 tracks from a specific genre
python model/gen_minimax_lofi.py 10 --skill rock
python model/gen_minimax_lofi.py 10 --skill metal
python model/gen_minimax_lofi.py 10 --skill hiphop

# custom JSON preset
python model/gen_minimax_lofi.py 5 --skill my_preset
```

## Built-in genres

| Skill | BPM ranges | Moods | Instruments | Drums |
|-------|-----------|-------|-------------|-------|
| `lofi` | 8 ranges (60-115) | 16 | 18 | 10 |
| `hiphop` | 6 ranges (78-115) | 8 | 10 | 6 |
| `rock` | 8 ranges (100-190) | 12 | 10 | 7 |
| `metal` | 6 ranges (100-240) | 12 | 8 | 6 |

All built-in skills: `lofi`, `hiphop`, `rock`, `metal` + any custom JSON in `model/skills/`.

## Custom skills feature

Add your own genre presets as JSON files in `model/skills/`.

Required keys (9 total):
- `bpm_ranges` — list of `[min, max]` integer pairs
- `moods`, `genres`, `instruments`, `drums`, `textures`, `ambiences`, `arrangements`, `extras` — lists of strings
- Optional `name` key overrides filename as skill name

Example: `model/skills/hiphop_dark.json`

## Output files

- `model/lofi_tracks/*.mp3`
- `model/lofi_tracks/*.meta.json` — prompt saved immediately per track
- `model/lofi_tracks/prompts.log`

## API details

- Endpoint: `https://api.minimax.io/v1/music_generation`
- Model: `music-2.6`, `is_instrumental = true`
- Audio: 44.1kHz, 256kbps, MP3
- Timeout: 300s
- Retry: 1 attempt after 15s on failure

## Rate limits

~6 requests/minute. Cap at ~6-10 tracks per 10-minute window.