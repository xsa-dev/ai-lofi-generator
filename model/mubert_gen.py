#!/usr/bin/env python3
"""Generate lofi tracks via Mubert API."""

import httpx
import json
import time
import os
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./lofi_tracks")
OUTPUT_DIR.mkdir(exist_ok=True)

# Demo credentials from Mubert colab (works for quick test)
LICENSE = "ttmmubertlicense#f0acYBenRcfeFpNT4wpYGaTQIyDI4mJGv5fIhBFz97NXDwDNFHmMRsBSzmGsJwbTpP1A6i07AXcIeAHo5"
API_TOKEN = "4951f6428e83172a4f39de05d5b3ab10d58560b8"

# Lofi-specific tags from Mubert's tag list
LOFI_PROMPTS = [
    "chill lofi hip hop beats perfect for studying and relaxing",
    "lofi beat with vinyl crackle jazz chords and dusty drums",
    "relaxing lofi study music with soft piano and mellow bass",
    "chill lofi vibes with warm vinyl texture and calm drums",
    "dreamy lofi sunset beats with muted sax and gentle groove",
]

DURATION = 180  # 3 minutes


# ── Get PAT ──────────────────────────────────────────────────────────────────
def get_pat(email: str = "demo@example.com") -> str:
    r = httpx.post(
        "https://api-b2b.mubert.com/v2/GetServiceAccess",
        json={
            "method": "GetServiceAccess",
            "params": {
                "email": email,
                "license": LICENSE,
                "token": API_TOKEN,
                "mode": "track",
            },
        },
        timeout=30,
    )
    data = r.json()
    if data.get("status") != 1:
        raise Exception(f"PAT error: {data}")
    return data["data"]["pat"]


# ── Generate track ────────────────────────────────────────────────────────────
def generate_track(pat: str, tags: list, duration: int, output_path: Path) -> bool:
    # Start generation
    r = httpx.post(
        "https://api-b2b.mubert.com/v2/RecordTrackTTM",
        json={
            "method": "RecordTrackTTM",
            "params": {
                "pat": pat,
                "duration": duration,
                "tags": tags,
                "mode": "track",
            },
        },
        timeout=30,
    )
    data = r.json()
    if data.get("status") != 1:
        print(f"  Error: {data}")
        return False

    track_url = data["data"]["tasks"][0]["download_link"]
    print(f"  Generating... ", end="", flush=True)

    # Poll until ready (max 60s)
    for _ in range(60):
        r = httpx.get(track_url, timeout=30)
        if r.status_code == 200:
            output_path.write_bytes(r.content)
            return True
        time.sleep(1)
        print(".", end="", flush=True)

    print(" (timeout)")
    return False


# ── Map prompt → Mubert tags via sentence similarity ────────────────────────
def get_tags_for_prompts(prompts: list) -> list:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        minilm = SentenceTransformer("all-MiniLM-L6-v2")

        mubert_tags_str = (
            "tribal,action,kids,neo-classic,run 130,pumped,jazz / funk,ethnic,"
            "dubtechno,reggae,acid jazz,liquidfunk,funk,witch house,tech house,"
            "underground,artists,mystical,disco,sensorium,r&b,agender,psychedelic,"
            "peaceful,run 140,piano,run 160,setting,meditation,christmas,ambient,"
            "horror,cinematic,electro house,idm,bass,minimal,underscore,drums,"
            "glitchy,beautiful,technology,tribal house,country pop,jazz & funk,"
            "documentary,space,classical,valentines,chillstep,experimental,trap,"
            "new jack swing,drama,post-rock,tense,corporate,neutral,happy,analog,"
            "funky,spiritual,chill hop,dramatic,catchy,holidays,fitness 90,"
            "optimistic,orchestra,acid techno,energizing,romantic,minimal house,"
            "breaks,hyper pop,warm up,dreamy,dark,urban,microfunk,dub,nu disco,"
            "vogue,keys,hardcore,aggressive,indie,electro funk,beauty,relaxing,"
            "trance,pop,hiphop,soft,acoustic,chillrave / ethno-house,deep techno,"
            "angry,dance,fun,dubstep,tropical,latin pop,heroic,world music,"
            "inspirational,uplifting,atmosphere,art,epic,advertising,chillout,"
            "scary,spooky,slow ballad,saxophone,summer,erotic,jazzy,energy 100,"
            "kara mar,xmas,atmospheric,indie pop,hip-hop,yoga,reggaeton,lounge,"
            "travel,running,folk,chillrave & ethno-house,detective,darkambient,"
            "chill,fantasy,minimal techno,special,night,tropical house,downtempo,"
            "lullaby,meditative,upbeat,glitch hop,fitness,neurofunk,sexual,"
            "indie rock,future pop,jazz,cyberpunk,melancholic,happy hardcore,"
            "family / kids,synths,electric guitar,comedy,psychedelic rock,calm,"
            "zen,bells,podcast,melodic house,ethnic percussion,nature,heavy,"
            "bassline,indie dance,techno,drumnbass,synth pop,vaporwave,sad,"
            "8-bit,chillgressive,deep,orchestral,futuristic,hardtechno,"
            "nostalgic,big room,sci-fi,tutorial,joyful,pads,minimal 170,drill,"
            "ethnic 108,amusing,sleepy ambient,psychill,italo disco,lofi,house,"
            "acoustic guitar,bassline house,rock,k-pop,synthwave,deep house,"
            "electronica,gabber,nightlife,sport & fitness,road trip,celebration,"
            "electronic,hardstyle,garage,uk garage,trumpet,blues,rock & roll,"
            "samba,latin,afrobeat,funky house,groove,world,electro,disco house,"
            "hands up,core,raw,hard dance,hardcore,光头,忧伤,流行,民谣,古风,复古,"
            "红色,摇滚,重金属,朋克,英伦,电子,爵士,古典,拉丁,乡村,蓝调,说唱,嘻哈,"
            " R and b"
        )
        mubert_tags = np.array(mubert_tags_str.split(","))
        tag_embeddings = minilm.encode(mubert_tags)
        prompt_embeddings = minilm.encode(prompts)

        results = []
        for pe in prompt_embeddings:
            scores = 1 - (pe @ tag_embeddings.T) / (
                np.linalg.norm(pe) * np.linalg.norm(tag_embeddings, axis=1)
            )
            idxs = np.argsort(scores)[:3]
            results.append(list(mubert_tags[idxs]))
        return results
    except ImportError:
        # Fallback: use lofi tag directly
        return [["lofi"]] * len(prompts)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Getting PAT...")
    pat = get_pat()
    print(f"PAT received: {pat[:20]}...\n")

    print("Computing tags for lofi prompts...")
    tag_lists = get_tags_for_prompts(LOFI_PROMPTS)
    for i, (prompt, tags) in enumerate(zip(LOFI_PROMPTS, tag_lists)):
        print(f"  [{i+1}] {prompt[:60]}...")
        print(f"      → {tags}")

    print(f"\nGenerating {len(LOFI_PROMPTS)} lofi tracks ({DURATION}s each)...\n")

    for i, (prompt, tags) in enumerate(zip(LOFI_PROMPTS, tag_lists), 1):
        fname = OUTPUT_DIR / f"lofi_track_{i:02d}.mp3"
        print(f"[{i}/5] {prompt[:50]}...")
        ok = generate_track(pat, tags, DURATION, fname)
        if ok:
            size = fname.stat().st_size
            print(f"  ✓ Saved: {fname} ({size // 1024} KB)")
        else:
            print(f"  ✗ Failed")

    print(f"\nDone. Tracks in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
