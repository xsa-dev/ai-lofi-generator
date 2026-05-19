#!/usr/bin/env python3
"""Generate random instrumental tracks with MiniMax Music 2.6 (multi-genre presets: lofi, hiphop, rock, metal, or custom)."""

import argparse
import json
import os
import random
import socket
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
SKILLS_DIR = Path(__file__).parent / "skills"

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
    "rock": {
        "bpm_ranges": [
            (100, 120), (110, 130), (120, 140), (130, 150), (140, 160), (150, 170), (160, 180), (170, 190),
        ],
        "moods": [
            "energetic and powerful", "raw and gritty", "melodic and anthemic",
            "dark and intense", "uplifting and triumphant", "nostalgic and rebellious",
            "brooding and heavy", "passionate and emotional", "carefree and defiant",
            "epic and cinematic", "delicate and introspective", "fierce and aggressive",
        ],
        "genres": [
            "rock instrumental", "guitar-driven rock instrumental", "alternative rock instrumental",
            "indie rock instrumental", "hard rock instrumental", "classic rock instrumental",
            "psychedelic rock instrumental", "post-rock instrumental", "instrumental rock anthem",
            "melodic rock instrumental", "blues rock instrumental", "punk rock instrumental",
        ],
        "instruments": [
            "distorted electric guitar riff", "dual harmonized guitar leads",
            "clean fingerpicked guitar arpeggios", "power chords with palm muting",
            "bluesy guitar solo", "sliding guitar bends", "acoustic guitar strumming",
            "layered electric guitar chords", "tremolo picked guitar", "wah pedal guitar",
        ],
        "drums": [
            "driving rock drum beat with crashes on the backbeat",
            "hard-hitting double bass kick pattern",
            "steady four-on-the-floor with fills on snare",
            "aggressive punk rock drumming with fast hi-hats",
            "groovy rock beat with syncopated snare hits",
            "double-time feel with tom fills",
            "crash-heavy rock beat for maximum impact",
        ],
        "textures": [
            "warm tube amplifier saturation", "raw and unpolished guitar tone",
            "vintage analog warmth", "thick wall of guitars",
            "clean production with natural room reverb", "lo-fi garage rock aesthetic",
            " polished studio sheen", "gritty overdriven distortion",
        ],
        "ambiences": [
            "stadium crowd ambience in the distance", "empty concert hall reverb",
            "desert highway open air atmosphere", "rain on a tin roof",
            "crackling fireplace ambience", "storm approaching",
            "city lights at night", "mountain wind howling",
        ],
        "arrangements": [
            "verse, chorus, bridge, solo, chorus, outro",
            "extended guitar intro building to a climax",
            "minimal verse with explosive chorus",
            "epic build from quiet to loud dynamics",
            "tension and release throughout",
            "free-form structure with improvisational feel",
        ],
        "extras": [
            "headbang-ready groove", "radio rock polish", "mixtape energy",
            "shout-along anthem feel", "cinematic soundtrack quality",
            "nostalgic 70s rock revival", "modern alternative edge",
        ],
    },
    "metal": {
        "bpm_ranges": [
            (100, 130), (120, 150), (140, 170), (160, 190), (180, 210), (200, 240),
        ],
        "moods": [
            "savage and brutal", "dark and apocalyptic", "technical and precise",
            "epic and majestic", "menacing and ominous", "fierce and relentless",
            "cold and mechanical", "haughty and aggressive", "storming and relentless",
            "brooding and crushing", "blazing and fast", "cathartic and intense",
        ],
        "genres": [
            "metal instrumental", "heavy metal instrumental", "death metal instrumental",
            "doom metal instrumental", "black metal instrumental", "thrash metal instrumental",
            "progressive metal instrumental", "metalcore instrumental", "instrumental metal anthem",
            "instrumental metal riffage", "djent instrumental", "symphonic metal instrumental",
        ],
        "instruments": [
            "downtuned heavy guitar riffs", "sweeping guitar arpeggios",
            "blast beat drumming", "guttural bass rumble",
            "pinch harmonic guitar leads", "tremolo picking patterns",
            "polyrhythmic guitar work", "low-tuned six-string breakdowns",
        ],
        "drums": [
            "non-stop blast beats with thunderous kick drum",
            "technical blast beat with precise fills",
            "slow crushing doom beats with cymbal swells",
            "double bass pedal relentless assault",
            "crust punk d-beat pattern",
            "syncopated djent-style drumming",
        ],
        "textures": [
            "compressed wall of guitars", "raw recording with natural distortion",
            "cold digital precision", "grimy analog saturation",
            "dark reverb-drenched guitars", "thin jagged tone",
            "thick layering of multiple guitar tracks", "clean production cutting through",
        ],
        "ambiences": [
            "empty dark hall ambience", "distant thunder and rain",
            "wind howling through ruins", "fire crackling",
            "underground bunker atmosphere", "void-like emptiness",
            "howling wolves in the distance", "church bells in decay",
        ],
        "arrangements": [
            "riff, breakdown, solo, riff",
            "slow build into crushing climax",
            "technical precision with dynamic shifts",
            "relentless attack throughout",
            "ambient intro into full assault",
            "multiple tempo changes with progressive structure",
        ],
        "extras": [
            "maximum headbang factor", "mosh-pit ready", "extended guitar solo showcase",
            "negative nostalgia", "soundtrack-quality composition",
            "brutal breakdown showcase", "technical wankery",
        ],
    },
}


def load_custom_skills() -> dict:
    """Load optional skill presets from model/skills/*.json."""
    custom = {}
    if not SKILLS_DIR.exists():
        return custom

    required = {
        "bpm_ranges", "moods", "genres", "instruments", "drums",
        "textures", "ambiences", "arrangements", "extras",
    }

    for path in sorted(SKILLS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            skill_name = data.get("name") or path.stem
            missing = sorted(required - set(data.keys()))
            if missing:
                print(f"[warn] Skip {path.name}: missing keys {', '.join(missing)}")
                continue
            custom[skill_name] = {k: data[k] for k in required}
        except Exception as exc:
            print(f"[warn] Skip {path.name}: {exc}")

    return custom


def generate_random_prompt(style: str, pools: dict) -> str:
    """Generate a random prompt from selected style pool."""
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
        "--skill",
        default="lofi",
        help="Skill preset name (built-in: lofi, hiphop; custom: model/skills/*.json)",
    )
    parser.add_argument(
        "--style",
        dest="skill",
        help="Backward-compatible alias for --skill",
    )
    return parser.parse_args()


def check_connectivity(timeout: int = 5) -> bool:
    """Check if api.minimax.io is reachable (fails fast under VPN)."""
    try:
        socket.create_connection(("api.minimax.io", 443), timeout=timeout)
        return True
    except OSError:
        return False


def main():
    if not API_KEY:
        print("Error: MINIMAX_API_KEY not set")
        print("  export MINIMAX_API_KEY='...'")
        sys.exit(1)

    all_skills = dict(STYLE_POOLS)
    all_skills.update(load_custom_skills())

    args = parse_args()
    count = args.count
    style = args.skill

    if style not in all_skills:
        print(f"Error: unknown skill '{style}'")
        print(f"Available skills: {', '.join(sorted(all_skills.keys()))}")
        sys.exit(2)

    print(f"Generating {count} random '{style}' tracks via MiniMax Music 2.6 API")
    print(f"Output: {OUT_DIR}\n")

    generated = 0
    failed = 0
    used_prompts = []

    for i in range(1, count + 1):
        prompt = generate_random_prompt(style, all_skills[style])
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
