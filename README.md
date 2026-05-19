# AI Lofi / HipHop Generator

Generate instrumental music tracks via MiniMax Music 2.6 API.

## Active script

| File | Description |
|------|-------------|
| `model/gen_minimax_lofi.py` | Random generation with presets: `lofi` and `hiphop` |

## Usage

```bash
export MINIMAX_API_KEY='...'

# default: 5 lofi tracks
python model/gen_minimax_lofi.py

# 10 hiphop tracks
python model/gen_minimax_lofi.py 10 --style hiphop

# explicit lofi
python model/gen_minimax_lofi.py 8 --style lofi
```

Output files are saved to:
- `model/lofi_tracks/*.mp3`
- `model/lofi_tracks/*.meta.json` (prompt saved immediately for every track)
- `model/lofi_tracks/prompts.log`

## Deprecated scripts (planned for deletion)

These scripts are deprecated and no longer part of the current flow. They will be deleted:

- `model/gen_test.py`
- `model/gen_5.py`
- `model/gen_long_lofi.py`
- `model/gen_music.py`
- `model/gen_continue.py`

## Notes

- Uses MiniMax endpoint: `https://api.minimax.io/v1/music_generation`
- Model: `music-2.6`
- `is_instrumental = true`
- MP3 settings: 44.1kHz, 256kbps
