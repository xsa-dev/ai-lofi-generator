# AI Music Generator (Multi-Genre)

Generate instrumental music tracks via MiniMax Music 2.6 API. Supports multiple built-in genres (lofi, hiphop, rock, metal) plus unlimited custom JSON presets.

## Active script

| File | Description |
|------|-------------|
| `model/gen_minimax_lofi.py` | Random generation with built-in skills: `lofi`, `hiphop` + custom JSON skills |

## Usage

```bash
export MINIMAX_API_KEY='...'

# 5 tracks (default: lofi)
python model/gen_minimax_lofi.py

# 10 hiphop tracks (backward-compatible flag)
python model/gen_minimax_lofi.py 10 --style hiphop

# same through the new feature flag
python model/gen_minimax_lofi.py 10 --skill hiphop

# custom skill from model/skills/*.json
python model/gen_minimax_lofi.py 6 --skill hiphop_dark
```

## Custom skills feature

You can add your own generation skill preset as JSON files in `model/skills/`.

- Each file should contain: `bpm_ranges`, `moods`, `genres`, `instruments`, `drums`, `textures`, `ambiences`, `arrangements`, `extras`
- Optional `name` overrides filename as skill name
- Example included: `model/skills/hiphop_dark.json`

Output files are saved to:
- `model/lofi_tracks/*.mp3`
- `model/lofi_tracks/*.meta.json` — prompt saved immediately per track
- `model/lofi_tracks/prompts.log`

## Notes

~6 requests/minute. Cap at ~6-10 tracks per 10-minute window.