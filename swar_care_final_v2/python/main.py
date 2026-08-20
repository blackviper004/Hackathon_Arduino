# server.py
#
# SwarCare web backend — FastAPI replacement for the old Streamlit app.
#
# This file is the ONLY thing that talks to SwarCareEngine. It never touches
# engine.py or model.py — it only calls the public methods engine.py already
# exposed (start_recording, pause_recording, stop_recording,
# get_audio_buffer_snapshot, get_terminal_lines_snapshot, analyze_veena_ai,
# and the plain attributes such as .state,
# .piezo_samples_recorded, etc.) exactly the way main.py used to.
#
# Run with:  uvicorn server:app --host 0.0.0.0 --port 8000

import asyncio
import hashlib
import io
import json
import os
import time
import zipfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import SwarCareEngine
from model import IST

# ---------------------------------------------------------------------------
# Engine singleton — identical instance the old main.py used
# ---------------------------------------------------------------------------
backend = SwarCareEngine.get_instance()
DATA_DIR = backend.recordings_dir

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="SwarCare Hub API")


# ---------------------------------------------------------------------------
# Helpers (ported 1:1 from main.py's caching logic)
# ---------------------------------------------------------------------------

def _get_recordings_signature() -> str:
    """Lightweight signature of the recordings folder — changes only when
    files actually change on disk. Same approach main.py used to invalidate
    its ZIP cache."""
    if not os.path.exists(DATA_DIR):
        return "empty"
    try:
        files = sorted(os.listdir(DATA_DIR))
        meta = []
        for f in files:
            if f.endswith((".csv", ".wav")):
                p = os.path.join(DATA_DIR, f)
                meta.append(f"{f}:{os.path.getmtime(p)}:{os.path.getsize(p)}")
        return hashlib.md5("".join(meta).encode("utf-8")).hexdigest()
    except Exception:
        return str(time.time())


@lru_cache(maxsize=4)
def _generate_recordings_zip_cached(dir_signature: str) -> bytes:
    """Cached ZIP generator — only recompresses when the signature changes."""
    zip_buffer = io.BytesIO()
    if os.path.exists(DATA_DIR):
        raw_files = sorted(os.listdir(DATA_DIR))
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in raw_files:
                if fname.endswith(".csv") or fname.endswith(".wav"):
                    fpath = os.path.join(DATA_DIR, fname)
                    if os.path.exists(fpath):
                        zf.write(fpath, arcname=fname)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def _safe_filename(name: str) -> str:
    """Reject path traversal — filenames must stay inside DATA_DIR."""
    name = os.path.basename((name or "").strip())
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


def _status_payload() -> dict:
    if backend.state == "RECORDING":
        elapsed = backend.get_synced_time() - backend.start_system_time
    elif backend.state == "PAUSED":
        elapsed = backend.pause_start_time - backend.start_system_time
    elif backend.state == "STOPPING":
        elapsed = backend.stop_system_time - backend.start_system_time
    else:
        elapsed = 0.0
    return {
        "state": backend.state,
        "elapsed_s": round(max(0.0, elapsed), 3),
        "server_now_ms": int(backend.get_synced_time() * 1000),
        "piezo_samples": backend.piezo_samples_recorded,
        "audio_samples": backend.audio_samples_recorded,
        "piezo_sample_rate_hz": backend.PIEZO_SAMPLE_RATE_HZ,
        "audio_sample_rate_hz": backend.AUDIO_SAMPLE_RATE_HZ,
        "time_synced": backend.time_synced,
    }


def _audio_payload() -> dict:
    snap = backend.get_audio_buffer_snapshot()
    return {
        "samples": snap,
        "elapsed_s": round(
            backend.audio_samples_recorded / float(backend.AUDIO_SAMPLE_RATE_HZ), 4
        ),
        "state": backend.state,
    }


def _piezo_payload() -> dict:
    snap = backend.get_terminal_lines_snapshot()
    return {
        "lines": [{"text": t, "active": a} for t, a in snap],
        "state": backend.state,
    }


def _latest_prefix() -> Optional[str]:
    if backend.state in ("RECORDING", "PAUSED", "STOPPING") and backend.current_prefix:
        return backend.current_prefix
    if os.path.exists(DATA_DIR):
        vfiles = [
            f for f in os.listdir(DATA_DIR)
            if f.endswith("_piezo.csv") or f.endswith("_audio.wav") or f.endswith("_audio.tmp")
        ]
        all_rec = sorted(
            {
                f.replace("_piezo.csv", "").replace("_audio.wav", "").replace("_audio.tmp", "")
                for f in vfiles
            },
            reverse=True,
        )
        return all_rec[0] if all_rec else None
    return None


# ---------------------------------------------------------------------------
# Time sync — device sends its real clock, server corrects its epoch
# ---------------------------------------------------------------------------

class TimeSyncRequest(BaseModel):
    epoch_ms: int  # Device's Date.now() — Unix epoch in milliseconds


@app.post("/api/sync_time")
def sync_time(req: TimeSyncRequest):
    device_epoch_s = req.epoch_ms / 1000.0
    backend.set_time_offset(device_epoch_s)
    corrected_now = datetime.fromtimestamp(backend.get_synced_time(), tz=IST)
    return {
        "synced": True,
        "offset_ms": round(backend.time_offset * 1000),
        "server_time_ist": corrected_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


# ---------------------------------------------------------------------------
# Live telemetry / status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    return _status_payload()


@app.get("/api/audio_data")
def get_audio_data():
    return _audio_payload()


@app.get("/api/piezo_lines")
def get_piezo_lines():
    return _piezo_payload()


@app.get("/api/storage_metrics")
def get_storage_metrics():
    p_samples = backend.piezo_samples_recorded
    a_samples = backend.audio_samples_recorded
    return {
        "piezo_samples": p_samples,
        "piezo_lines": p_samples + 1 if p_samples > 0 else 0,
        "piezo_duration_s": round(p_samples / float(backend.PIEZO_SAMPLE_RATE_HZ), 2),
        "audio_samples": a_samples,
        "audio_duration_s": round(a_samples / float(backend.AUDIO_SAMPLE_RATE_HZ), 2),
    }


# ---------------------------------------------------------------------------
# Recording control (was: st.button Start/Pause/Stop calling backend.*)
# ---------------------------------------------------------------------------

@app.post("/api/record/start")
def record_start():
    backend.start_recording()
    return _status_payload()


@app.post("/api/record/pause")
def record_pause():
    backend.pause_recording()
    return _status_payload()


@app.post("/api/record/stop")
def record_stop():
    backend.stop_recording()
    return _status_payload()


# ---------------------------------------------------------------------------
# Veena AI diagnostics (was: backend.analyze_veena_ai(...) in the fragment)
# ---------------------------------------------------------------------------

@app.get("/api/veena_analysis")
def veena_analysis(
    tonic_hz: float = 130.81,
    cents_threshold: float = 15.0,
    string_label: Optional[str] = None,
):
    prefix = _latest_prefix()
    if not prefix:
        return {"available": False, "error": "No recording session found yet.", "prefix": None}

    # "auto-detect" was represented client-side as an empty/omitted value
    if string_label in ("", "auto", "null", "None"):
        string_label = None

    result = backend.analyze_veena_ai(
        prefix,
        tonic_hz=tonic_hz,
        cents_threshold=cents_threshold,
        string_label=string_label,
    )
    result["prefix"] = prefix
    return result




# ---------------------------------------------------------------------------
# Saved Records Explorer
# ---------------------------------------------------------------------------

class RenameRequest(BaseModel):
    old_name: str
    new_name: str


@app.get("/api/recordings")
def list_recordings(limit: int = 3):
    if os.path.exists(DATA_DIR):
        raw_files = sorted(os.listdir(DATA_DIR), reverse=True)
        csv_files = [f for f in raw_files if f.endswith(".csv")]
        wav_files = [f for f in raw_files if f.endswith(".wav")]
    else:
        csv_files, wav_files = [], []

    def meta(files):
        out = []
        for f in files:
            p = os.path.join(DATA_DIR, f)
            if os.path.exists(p):
                out.append({
                    "name": f,
                    "size_bytes": os.path.getsize(p),
                    "mtime": os.path.getmtime(p),
                })
        return out

    return {
        "csv_files": meta(csv_files[:limit]),
        "wav_files": meta(wav_files[:limit]),
        "csv_total": len(csv_files),
        "wav_total": len(wav_files),
        "limit": limit,
    }


@app.get("/api/recordings/download/{filename}")
def download_recording(filename: str, as_name: Optional[str] = None):
    filename = _safe_filename(filename)
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    download_name = filename
    if as_name:
        as_name = as_name.strip()
        if as_name:
            ext = ".csv" if filename.endswith(".csv") else ".wav"
            download_name = as_name if as_name.endswith(ext) else as_name + ext

    media_type = "text/csv" if filename.endswith(".csv") else "audio/wav"
    return FileResponse(file_path, media_type=media_type, filename=download_name)


@app.get("/api/recordings/audio/{filename}")
def stream_audio(filename: str):
    """Used by the <audio> replay control — same file, inline (not a download)."""
    filename = _safe_filename(filename)
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path) or not filename.endswith(".wav"):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")


@app.post("/api/recordings/rename")
def rename_recording(req: RenameRequest):
    old_name = _safe_filename(req.old_name)
    new_name = _safe_filename(req.new_name)

    old_path = os.path.join(DATA_DIR, old_name)
    if not os.path.exists(old_path):
        raise HTTPException(status_code=404, detail="File not found")

    ext = ".csv" if old_name.endswith(".csv") else ".wav"
    if not new_name.endswith(ext):
        new_name += ext

    new_path = os.path.join(DATA_DIR, new_name)
    try:
        os.rename(old_path, new_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "new_name": new_name}


@app.delete("/api/recordings/{filename}")
def delete_recording(filename: str):
    filename = _safe_filename(filename)
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


@app.get("/api/recordings/zip")
def download_all_zip():
    dir_sig = _get_recordings_signature()
    zip_bytes = _generate_recordings_zip_cached(dir_sig)
    if not zip_bytes:
        raise HTTPException(status_code=404, detail="No recordings available")
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"SwarCare_All_Recordings_{ts_now}.zip"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@app.post("/api/recordings/clear_all")
def clear_all_recordings():
    if os.path.exists(DATA_DIR):
        for file_to_wipe in os.listdir(DATA_DIR):
            try:
                os.remove(os.path.join(DATA_DIR, file_to_wipe))
            except Exception:
                pass
    _generate_recordings_zip_cached.cache_clear()
    return {"ok": True}


# ---------------------------------------------------------------------------
# WebSocket: single live-telemetry channel
# (replaces the localStorage bridge + three separate polling loops)
# ---------------------------------------------------------------------------

TELEMETRY_PUSH_INTERVAL_S = 0.12  # ~8Hz, matches the old 100-180ms polling


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = {
                "status": _status_payload(),
                "audio": _audio_payload(),
                "piezo": _piezo_payload(),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(TELEMETRY_PUSH_INTERVAL_S)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Captive-portal handlers
# When the Arduino UNO Q acts as a WiFi access point, the OS-level captive
# portal detector on the connecting device hits well-known URLs.  Without
# explicit handling these fall through to the StaticFiles mount and return a
# JSON 404.  Serving the dashboard HTML directly ensures the SwarCare page
# opens automatically on laptops (Windows, macOS) and any other client.
# ---------------------------------------------------------------------------

_INDEX_HTML = STATIC_DIR / "index.html"

def _serve_index():
    """Return the dashboard index.html as a direct HTML response."""
    if _INDEX_HTML.exists():
        return Response(content=_INDEX_HTML.read_bytes(), media_type="text/html")
    return Response(content="<h1>SwarCare</h1><p>index.html not found</p>", media_type="text/html")

# Windows (msftconnecttest.com/redirect, msftconnecttest.com/connecttest.txt)
@app.get("/redirect")
@app.get("/connecttest.txt")
def _captive_windows():
    return _serve_index()

# Android (connectivitycheck.gstatic.com/generate_204)
@app.get("/generate_204")
def _captive_android():
    return _serve_index()

# Apple (captive.apple.com/hotspot-detect.html)
@app.get("/hotspot-detect.html")
def _captive_apple():
    return _serve_index()


# ---------------------------------------------------------------------------
# Static frontend (the Arduino App Lab "WebUI brick" folder)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")