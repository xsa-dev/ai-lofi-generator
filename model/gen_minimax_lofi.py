#!/usr/bin/env python3
"""Generate random instrumental tracks with MiniMax Music 2.6 (lofi/hiphop presets)."""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL = "https://api.minimax.io/v1/music_generation"

OUT_DIR = Path(__file__).parent / "lofi_tracks"
OUT_DIR.mkdir(exist_ok=True)

STYLE_POOLS = {
    "lofi": {
        "bpm_ranges": [
            (60, 70), (68, 78), (72, 82), (75, 85), (80, 90), (85, 95), (90, 105), (100, 115)
        ],
        "moods": [
            "chill and relaxed", "mellow and nostalgic", "dreamy and introspective", "cozy and warm",
            "melancholic and reflective", "peaceful and calm", "laid-back and soulful", "lo-fi and dusty",
            "jazzy and smooth", "rainy day mood", "late night vibes", "morning coffee atmosphere",
            "study focus mood", "chillhop relaxation", "cinematic and emotional", "lo-fi bedroom aesthetic",
        ],
        "genres": [
            "lo-fi hip hop instrumental", "lo-fi chillhop beats", "lo-fi study music",
            "mellow lo-fi rap instrumental", "lo-fi jazz hop", "chill lo-fi beats",
            "relaxing lo-fi instrumental", "lo-fi boom bap", "dusty lo-fi hip hop", "analog lo-fi beats",
            "vinyl lo-fi sample-based instrumental", "lo-fi house instrumental", "ambient lo-fi",
            "lo-fi beats with jazz influences", "lo-fi soul beats", "experimental lo-fi",
        ],
        "instruments": [
            "warm sustain piano chords", "Rhodes electric piano with tube breakup",
            "fingerpicked acoustic guitar chord progression", "nylon guitar fingerstyle",
            "jazz guitar sample with vinyl processing", "electric jazz guitar improvisation",
            "solo grand piano with pedaled sustain", "analog synth pad chords",
            "muted trumpet counter-melody", "tenor saxophone solo melody", "vibraphone and marimba melody",
            "bell and glockenspiel melody", "acoustic piano with jazz voicings",
            "organ chords with percussive stabs", "lo-fi processed Rhodes loop", "warm analog bass line",
            "sub bass groove", "walking bass line", "jazzy chord sample in minor key",
        ],
        "drums": [
            "lazy boom bap drum pattern with dusty kicks and crisp hi-hats",
            "jazz-inspired drum pattern with dusty kicks and brushed snare",
            "slow half-time drums with deep kick and soft snare",
            "lo-fi drum kit with compressed kicks and filtered hi-hats with swing",
            "simple kick on beats 1 and 3, brushed snare on 2 and 4",
            "four-on-the-floor kick pattern with house hi-hats and rimshots",
            "lo-fi breakbeat pattern with chopped hats",
            "relaxed drum groove with lazy snare and soft kicks",
            "dusty breakbeat with filtered hi-hats", "jazz brush strokes on snare, light kick",
        ],
        "textures": [
            "vintage vinyl crackle and pop noise", "warm analog tape saturation and hiss",
            "lo-fi tape distortion and noise floor", "vinyl loop sample with authentic wow and flutter",
            "dusty record crackle throughout", "tape hiss and warm saturation", "lo-fi bit-crushed texture",
            "warm tube amplifier saturation", "cassette tape warmth and noise", "vinyl noise and tape echo",
        ],
        "ambiences": [
            "rain tap sounds and distant thunder", "gentle rain sound effect layered throughout",
            "rainy window ambience in background", "coffee shop background chatter and espresso machine",
            "cozy coffee shop atmosphere", "city traffic ambience distant at night", "ocean wave ambience gentle",
            "wind whispers and outdoor ambience", "图书馆安静的氛围", "深夜城市的遥远声音", "炉火的噼啪声", "树叶沙沙作响的背景",
        ],
        "arrangements": [
            "seamless loop structure", "floating arrangement with layered textures",
            "structured arrangement with intro and outro", "continuous seamless loop",
            "layered harmonic richness throughout", "minimal arrangement, intimate and quiet",
            "rich layered texture building throughout", "simple minimal composition",
            "spacious arrangement with room ambience", "atmospheric pad swells and fades",
        ],
        "extras": [
            "nostalgic 90s sample aesthetic", "concentration enhancing study music",
            "relaxed Sunday morning vibes", "late night focus and productivity", "chill evening unwinding music",
            "deep focus and concentration aid", "relaxing background music for work", "intimate close-mic sound",
            "warm quiet intimate atmosphere", "smooth mellow vibe throughout",
        ],
    },
    "hiphop": {
        "bpm_ranges": [
            (78, 88), (82, 92), (86, 96), (90, 102), (95, 108), (100, 115)
        ],
        "moods": [
            "confident and gritty", "dark and cinematic", "street and raw", "energetic and hard-hitting",
            "bounce-heavy and catchy", "moody and aggressive", "minimal and heavy", "late-night urban vibe",
        ],
        "genres": [
            "hip hop instrumental", "boom bap instrumental", "trap-inspired hip hop beat",
            "old school rap beat", "modern drill-influenced hip hop instrumental",
            "hard-hitting rap production", "sample-based east coast hip hop beat", "west coast bounce hip hop",
        ],
        "instruments": [
            "deep 808 bass line", "detuned piano riff", "dark synth lead",
            "orchestral stab accents", "chopped soul sample", "vinyl sample chops",
            "brass hits and stabs", "minimal guitar pluck motif", "sub bass and sine glide",
        ],
        "drums": [
            "hard kick and snare with swung hi-hats", "trap hats with fast rolls and punchy 808",
            "boom bap drums with crunchy snare", "tight clap on 2 and 4 with syncopated percussion",
            "heavy drums with ghost snare notes", "aggressive drum groove with open hat accents",
        ],
        "textures": [
            "clean but punchy modern mix", "light tape saturation", "vinyl dust layer",
            "bit-crushed percussive texture", "gritty analog saturation",
        ],
        "ambiences": [
            "night city ambience", "subway station reverb tail", "crowd murmur in distance",
            "empty warehouse room tone", "dark alley atmosphere",
        ],
        "arrangements": [
            "intro, verse, hook, verse, outro", "8-bar loop with variation every 4 bars",
            "drop after short filtered intro", "minimal verse sections with hook lift",
        ],
        "extras": [
            "radio-ready low end", "head-nod groove", "club-ready punch", "underground mixtape energy",
            "clean instrumental for rap vocals", "strong transient drums",
        ],
    },
}


def generate_random_prompt(style: str) -> str:
    """Generate a random prompt from selected style pool."""
    pools = STYLE_POOLS[style]

    bpm = random.randint(*random.choice(pools["bpm_ranges"]))
    mood = random.choice(pools["moods"])
    genre = random.choice(pools["genres"])
    inst = random.choice(pools["instruments"])
    drums = random.choice(pools["drums"])
    texture = random.choice(pools["textures"])
    ambience = random.choice(pools["ambiences"])
    arrangement = random.choice(pools["arrangements"])
    extra = random.choice(pools["extras"])

    return (
        f"{genre}, {bpm} BPM, {mood}, {inst}, {drums}, {texture}, "
        f"{ambience}, {arrangement}, {extra}"
    )


def generate_track(prompt: str, track_num: int, style: str) -> Optional[str]:
    """Generate a single track. Returns path on success, None on failure."""
    print(f"\n[{track_num}] Generating ({style})...")
    print(f"  Prompt: {prompt[:120]}...")

    payload = json.dumps({
        "model": "music-2.6",
        "prompt": prompt,
        "is_instrumental": True,
        "audio_setting": {
            "sample_rate": 44100,
            "bitrate": 256000,
            "format": "mp3"
        }
    }).encode()

    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
            msg = body.get("base_resp", {}).get("status_msg", raw[:200])
        except Exception:
            msg = raw[:200]
        print(f"  HTTP {e.code}: {msg}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

    base_resp = result.get("base_resp", {})
    if base_resp.get("status_code") != 0:
        print(f"  API error: {base_resp.get('status_msg')}")
        return None

    audio_hex = result.get("data", {}).get("audio")
    if not audio_hex:
        print("  No audio in response")
        return None

    audio_bytes = bytes.fromhex(audio_hex)
    short_id = str(uuid.uuid4())[:8]
    output_path = OUT_DIR / f"{style}_{short_id}.mp3"

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    extra = result.get("extra_info", {})
    duration_s = (extra.get("music_duration", 0) or 0) / 1000
    size_mb = len(audio_bytes) / 1024 / 1024

    print(f"  ✓ {duration_s:.1f}s, {size_mb:.1f} MB → {output_path.name}")

    # Save prompt immediately alongside track
    meta_path = OUT_DIR / f"{short_id}.meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "id": short_id,
            "style": style,
            "prompt": prompt,
            "duration_s": duration_s,
            "size_mb": size_mb,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random tracks via MiniMax Music 2.6")
    parser.add_argument("count", nargs="?", type=int, default=5, help="Number of tracks to generate")
    parser.add_argument(
        "--style",
        choices=sorted(STYLE_POOLS.keys()),
        default="lofi",
        help="Prompt style preset (default: lofi)",
    )
    return parser.parse_args()


def main():
    if not API_KEY:
        print("Error: MINIMAX_API_KEY not set")
        print("  export MINIMAX_API_KEY='...'")
        sys.exit(1)

    args = parse_args()
    count = args.count
    style = args.style

    print(f"Generating {count} random '{style}' tracks via MiniMax Music 2.6 API")
    print(f"Output: {OUT_DIR}\n")

    generated = 0
    failed = 0
    used_prompts = []

    for i in range(1, count + 1):
        prompt = generate_random_prompt(style)
        used_prompts.append(prompt)

        path = generate_track(prompt, i, style)
        if path:
            generated += 1
        else:
            failed += 1
            print("  ✗ Failed, retrying in 15s...")
            time.sleep(15)
            path = generate_track(prompt, i, style)
            if path:
                generated += 1
                failed -= 1

        if i < count:
            time.sleep(5)

    print(f"\n{'=' * 50}")
    print(f"Done: {generated} generated, {failed} failed")
    print(f"Output: {OUT_DIR}/")

    prompts_log = OUT_DIR / "prompts.log"
    with open(prompts_log, "a") as f:
        f.write(f"\n--- Run at {time.strftime('%Y-%m-%d %H:%M:%S')} [{style}] ---\n")
        for j, p in enumerate(used_prompts, 1):
            f.write(f"Track {j}: {p}\n")

    print(f"Prompts saved: {prompts_log}")


if __name__ == "__main__":
    main()
