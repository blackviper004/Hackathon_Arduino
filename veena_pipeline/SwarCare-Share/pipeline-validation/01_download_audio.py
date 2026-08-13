"""
SwarCare Pipeline Validation - Step 1: Download Veena Audio
=============================================================
PURPOSE:
    Download Saraswati Veena audio recordings from Freesound.org
    These will serve as our "healthy baseline" training data for
    validating the Edge Impulse → Brick pipeline.

SOURCES:
    1. Veena-Murthy-19-4-2011.wav (3:04) — Saraswati Veena by M.V.N. Murthy
       - Carnatic music, clean recording
       - License: CC0 (Creative Commons Zero)
       - URL: https://freesound.org/people/xserra/sounds/125655/

    2. veena_classical.wav (4s) — Classical veena excerpt
       - Short clip, clean
       - License: CC0
       - URL: https://freesound.org/people/anjds/sounds/380750/

    3. Veena_Recording.wav (4:18) — Live veena via contact pickup
       - Captured through guitar pickup (similar to our piezo approach!)
       - License: Attribution (credit: iskweldog)
       - URL: https://freesound.org/people/iskweldog/sounds/216060/

NOTE:
    Freesound requires OAuth2 authentication for direct downloads.
    This script provides TWO options:
      Option A: Manual download (recommended for first time)
      Option B: Freesound API download (requires API key)

WHAT THIS GIVES US:
    ~7+ minutes of real Saraswati Veena audio in WAV format,
    ready for segmentation in the next step.
"""

import os
import sys

# === CONFIGURATION ===
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw_audio")

# Freesound sources we want
SOURCES = [
    {
        "name": "Veena-Murthy-19-4-2011.wav",
        "id": 125655,
        "duration": "3:04",
        "description": "Saraswati Veena by M.V.N. Murthy - Carnatic music",
        "url": "https://freesound.org/people/xserra/sounds/125655/",
        "license": "CC0",
    },
    {
        "name": "veena_classical.wav",
        "id": 380750,
        "duration": "0:04",
        "description": "Classical veena excerpt - clean",
        "url": "https://freesound.org/people/anjds/sounds/380750/",
        "license": "CC0",
    },
    {
        "name": "Veena_Recording.wav",
        "id": 216060,
        "duration": "4:18",
        "description": "Live veena via contact pickup (like our piezo!)",
        "url": "https://freesound.org/people/iskweldog/sounds/216060/",
        "license": "Attribution - iskweldog",
    },
]


def manual_download_instructions():
    """Print instructions for manual download from Freesound."""
    print("=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Freesound requires a free account to download. Steps:")
    print()
    print("1. Create a free account at: https://freesound.org/home/register/")
    print("2. Download each file below and save to:")
    print(f"   {OUTPUT_DIR}")
    print()

    for i, src in enumerate(SOURCES, 1):
        print(f"   File {i}: {src['name']}")
        print(f"   URL:  {src['url']}")
        print(f"   Duration: {src['duration']}")
        print(f"   License: {src['license']}")
        print(f"   Info: {src['description']}")
        print()

    print("-" * 70)
    print("After downloading, run: python 02_segment_audio.py")
    print("=" * 70)


def api_download(api_key: str):
    """
    Download via Freesound API (requires API key).
    Get your API key at: https://freesound.org/apiv2/apply/
    """
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for src in SOURCES:
        sound_id = src["id"]
        filename = src["name"]
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            print(f"[SKIP] {filename} already exists")
            continue

        print(f"[DOWNLOADING] {filename} ({src['duration']})...")

        # Get sound metadata
        meta_url = f"https://freesound.org/apiv2/sounds/{sound_id}/"
        headers = {"Authorization": f"Token {api_key}"}
        resp = requests.get(meta_url, headers=headers)

        if resp.status_code != 200:
            print(f"  [ERROR] Failed to get metadata: {resp.status_code}")
            continue

        data = resp.json()
        download_url = data.get("download")

        if not download_url:
            print(f"  [ERROR] No download URL in response")
            continue

        # Download the actual file
        dl_resp = requests.get(download_url, headers=headers, stream=True)
        if dl_resp.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  [OK] Saved to {filepath}")
        else:
            print(f"  [ERROR] Download failed: {dl_resp.status_code}")

    print("\nDone! Run: python 02_segment_audio.py")


if __name__ == "__main__":
    print("SwarCare Pipeline Validation - Audio Download")
    print("=" * 50)
    print()

    # Check if API key is provided
    api_key = os.environ.get("FREESOUND_API_KEY")

    if api_key:
        print("Using Freesound API (key found in environment)")
        api_download(api_key)
    else:
        print("No FREESOUND_API_KEY found in environment.")
        print("Showing manual download instructions instead.")
        print()
        manual_download_instructions()
        print()
        print("TIP: To use API download, get a key from:")
        print("  https://freesound.org/apiv2/apply/")
        print("Then run: $env:FREESOUND_API_KEY='your_key'; python 01_download_audio.py")
