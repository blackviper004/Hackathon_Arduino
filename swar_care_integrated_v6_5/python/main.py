import io
import json
import os
import time
import urllib.parse
import zipfile
import hashlib
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from engine import SwarCareEngine

# Initialize SwarCare Core Engine Singleton
backend = SwarCareEngine.get_instance()
DATA_DIR = backend.recordings_dir

st.set_page_config(page_title="SwarCare Hub", page_icon="📡", layout="centered")

# Sync client time context with backend engine
components.html(
    """
<script>
fetch('/settime', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({epoch_ms: Date.now()})
}).catch(() => {});
</script>
""",
    height=0,
)

# Inject Modern Viewport & Responsive CSS Rules
st.markdown(
    """
<style>
    /* Centralized Color Tokens & Design System */
    :root {
        --bg-primary: #0E1117;
        --bg-surface: #161B22;
        --bg-surface-elevated: #21262D;
        --border-subtle: #30363D;
        --border-accent: #00BCD4;
        --text-primary: #F0F6FC;
        --text-secondary: #8B949E;
        --text-muted: #6E7681;
        --accent-cyan: #00BCD4;
        --accent-green: #00E676;
        --accent-amber: #FFA000;
        --accent-red: #FF5252;
        --btn-bg: #21262D;
        --btn-text: #F0F6FC;
        --btn-border: #30363D;
        --btn-hover-bg: #30363D;
        --btn-disabled-bg: #161B22;
        --btn-disabled-text: #6E7681;
        --btn-disabled-border: #21262D;
    }

    /* Global Clean Dark Typography & Theme */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Mobile-First Layout Adjustments */
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }
        h1 { font-size: 1.4rem !important; color: #FFFFFF !important; }
        h2 { font-size: 1.15rem !important; color: var(--text-primary) !important; }
        h3 { font-size: 0.95rem !important; color: var(--text-primary) !important; }

        /* Metric Scaling for Mobile Screens */
        [data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
            color: #FFFFFF !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            color: var(--text-secondary) !important;
        }
        .stButton button {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            font-size: 0.82rem !important;
        }
    }

    /* Clean Seamless Iframes with Zero Outer Overflow */
    iframe {
        width: 100% !important;
        border: none !important;
        overflow: hidden !important;
        display: block !important;
    }

    /* Polished Card Container Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border-subtle) !important;
        background-color: var(--bg-surface);
        border-radius: 8px;
        transition: border-color 0.2s ease-in-out;
    }

    /* Centralized Action Button Styling with Maximum Readability */
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        background-color: var(--btn-bg) !important;
        color: var(--btn-text) !important;
        border: 1px solid var(--btn-border) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease-in-out !important;
    }

    div[data-testid="stButton"] button:hover:not(:disabled),
    div[data-testid="stDownloadButton"] button:hover:not(:disabled) {
        background-color: var(--btn-hover-bg) !important;
        border-color: var(--border-accent) !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0, 188, 212, 0.25) !important;
    }

    div[data-testid="stButton"] button:disabled,
    div[data-testid="stDownloadButton"] button:disabled {
        background-color: var(--btn-disabled-bg) !important;
        color: var(--btn-disabled-text) !important;
        border-color: var(--btn-disabled-border) !important;
        opacity: 0.65 !important;
        cursor: not-allowed !important;
    }

    /* Hardware Control Deck Dynamic Button Palette */
    div[data-testid="column"]:nth-child(1) div[data-testid="stButton"] button:not(:disabled) {
        border-color: rgba(0, 230, 118, 0.4) !important;
        color: var(--accent-green) !important;
    }
    div[data-testid="column"]:nth-child(1) div[data-testid="stButton"] button:hover:not(:disabled) {
        background-color: rgba(0, 230, 118, 0.15) !important;
        border-color: var(--accent-green) !important;
        box-shadow: 0 2px 10px rgba(0, 230, 118, 0.25) !important;
    }

    div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] button:not(:disabled) {
        border-color: rgba(255, 160, 0, 0.4) !important;
        color: var(--accent-amber) !important;
    }
    div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] button:hover:not(:disabled) {
        background-color: rgba(255, 160, 0, 0.15) !important;
        border-color: var(--accent-amber) !important;
        box-shadow: 0 2px 10px rgba(255, 160, 0, 0.25) !important;
    }

    div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] button:not(:disabled) {
        border-color: rgba(255, 82, 82, 0.4) !important;
        color: var(--accent-red) !important;
    }
    div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] button:hover:not(:disabled) {
        background-color: rgba(255, 82, 82, 0.15) !important;
        border-color: var(--accent-red) !important;
        box-shadow: 0 2px 10px rgba(255, 82, 82, 0.25) !important;
    }

    /* Tabs Styling */
    div[data-testid="stTabs"] button {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom-color: var(--accent-cyan) !important;
    }

    /* Inputs, Selectboxes & Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-surface) !important;
        border-color: var(--border-subtle) !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="input"] input {
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state initialization
if "visible_records_limit" not in st.session_state:
    st.session_state.visible_records_limit = 3

if "prev_engine_state" not in st.session_state:
    st.session_state.prev_engine_state = backend.state

st.title("🎵 SwarCare Anomaly Detection Hub")
st.write("Real-Time Vibration & Audio Stream Intelligence")
st.markdown("---")


# ===========================================================================
# 🚀 PERFORMANCE LAYER: CACHED FILE I/O & OPTIMIZED DATA PIPELINE
# ===========================================================================

def _get_recordings_signature() -> str:
    """Generate a lightweight signature of the recordings folder to invalidate cache only when files change."""
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


@st.cache_data(show_spinner=False)
def generate_recordings_zip(dir_signature: str) -> bytes:
    """Cached generator for the recordings ZIP archive.
    Recompresses ONLY when directory contents actually change on disk."""
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


@st.cache_data(show_spinner=False, max_entries=30)
def get_cached_file_bytes(file_path: str, mtime: float) -> bytes:
    """Lazy cached binary reader for download buttons."""
    if not os.path.exists(file_path):
        return b""
    with open(file_path, "rb") as f:
        return f.read()


# ==========================================
# 📑 CONSOLIDATED 2-TAB INTERFACE
# ==========================================
tab_live_ai, tab_records = st.tabs(
    ["📡 Live & Veena Diagnostics", "📁 Saved Records Explorer"]
)

LIVE_SYNC_SEC = 1.0 / 2.0  # Telemetry sync bridge frequency (2 Hz)
AI_REFRESH_SEC = 2.5  # Periodic refresh rate for AI diagnostic display

# ==========================================
# TAB 1: LIVE CONTROL, AI & TELEMETRY
# ==========================================
with tab_live_ai:
    # --- 1. HARDWARE CONTROL DECK ---
    st.subheader("⚙️ Hardware Control Deck")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_label = (
            "▶ RESUME" if backend.state == "PAUSED" else "▶ START RECORDING"
        )
        if st.button(
            start_label,
            use_container_width=True,
            disabled=(backend.state in ["RECORDING", "STOPPING"]),
            help="Start capturing telemetry",
        ):
            backend.start_recording()
            st.session_state.prev_engine_state = backend.state
            st.rerun()
    with col2:
        if st.button(
            "⏸ PAUSE",
            use_container_width=True,
            disabled=(backend.state != "RECORDING"),
            help="Pause live telemetry capture",
        ):
            backend.pause_recording()
            st.session_state.prev_engine_state = backend.state
            st.rerun()
    with col3:
        if st.button(
            "⏹ STOP & SAVE",
            use_container_width=True,
            disabled=(backend.state not in ["RECORDING", "PAUSED"]),
            help="Finalize and save streams",
        ):
            backend.stop_recording()
            st.session_state.prev_engine_state = backend.state
            st.rerun()

    is_live = backend.state in ("RECORDING", "PAUSED", "STOPPING")

    # --- INVISIBLE DATA BUS BRIDGE (LocalStorage Sync & State Transition Watcher) ---
    @st.fragment(run_every=LIVE_SYNC_SEC if is_live else None)
    def sync_telemetry_to_local_storage():
        current_state = backend.state
        if (
            st.session_state.prev_engine_state == "STOPPING"
            and current_state == "STOPPED"
        ):
            st.session_state.prev_engine_state = "STOPPED"
            st.toast("💾 Recording saved and verified successfully!", icon="🎉")
            time.sleep(0.1)
            st.rerun()
        else:
            st.session_state.prev_engine_state = current_state

        if backend.state == "RECORDING":
            server_elapsed = time.time() - backend.start_system_time
        elif backend.state == "PAUSED":
            server_elapsed = (
                backend.pause_start_time - backend.start_system_time
            )
        elif backend.state == "STOPPING":
            server_elapsed = (
                backend.stop_system_time - backend.start_system_time
            )
        else:
            server_elapsed = 0.0
        server_elapsed = max(0.0, server_elapsed)

        status_payload = {
            "state": backend.state,
            "elapsed_s": round(server_elapsed, 3),
            "server_now_ms": int(time.time() * 1000),
            "piezo_samples": backend.piezo_samples_recorded,
            "audio_samples": backend.audio_samples_recorded,
        }

        audio_snap = backend.get_audio_buffer_snapshot()
        audio_elapsed = backend.audio_samples_recorded / float(
            backend.AUDIO_SAMPLE_RATE_HZ
        )
        audio_payload = {
            "samples": audio_snap,
            "elapsed_s": round(audio_elapsed, 4),
            "state": backend.state,
        }

        terminal_snap = backend.get_terminal_lines_snapshot()
        piezo_payload = {
            "lines": [{"text": t, "active": a} for t, a in terminal_snap],
            "state": backend.state,
        }

        enc_status = urllib.parse.quote(json.dumps(status_payload))
        enc_audio = urllib.parse.quote(json.dumps(audio_payload))
        enc_piezo = urllib.parse.quote(json.dumps(piezo_payload))

        bridge_js = f"""
        <script>
        (function() {{
            try {{
                localStorage.setItem('swarcare_status', decodeURIComponent('{enc_status}'));
                localStorage.setItem('swarcare_audio', decodeURIComponent('{enc_audio}'));
                localStorage.setItem('swarcare_piezo', decodeURIComponent('{enc_piezo}'));
            }} catch(e) {{}}
        }})();
        </script>
        """
        components.html(bridge_js, height=0)

    sync_telemetry_to_local_storage()

    # --- 2. RECORDING TIMER (Strictly Monotonic Local Clock Engine) ---
    timer_static_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * { box-sizing: border-box; }
            html, body { margin:0; padding:0; background:transparent; overflow:hidden; font-family:Consolas, 'Courier New', monospace; width:100%; height:100%; }
            .box { 
                background-color:#161B22; 
                border:1px solid #30363D; 
                border-radius:6px; 
                padding:8px 12px; 
                text-align:center; 
                width:100%; 
                height:68px; 
                display:flex; 
                flex-direction:column; 
                justify-content:center; 
                align-items:center;
                transition: border-color 0.25s ease, box-shadow 0.25s ease; 
            }
            .lab { color:#8B949E; font-weight:bold; font-size:11px; letter-spacing:1px; transition: color 0.25s ease; }
            .val { color:#C9D1D9; font-size:26px; font-weight:bold; letter-spacing:2px; margin-top:2px; font-variant-numeric: tabular-nums; }
        </style>
    </head>
    <body>
        <div class="box" id="mainbox">
            <div class="lab" id="mainlab">⏱️ RECORDING TIME</div>
            <div class="val" id="tval">00:00:00</div>
        </div>
        <script>
            let state = "STOPPED";
            let serverElapsed = 0.0;
            let anchorStartMs = null;
            let maxDisplaySec = 0.0;
            let tryHttp = true;
            let fetchInFlight = false;

            function fmt(sec) {
                sec = Math.max(0, Math.floor(sec));
                const hh = String(Math.floor(sec / 3600)).padStart(2, '0');
                const mm = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
                const ss = String(sec % 60).padStart(2, '0');
                return hh + ':' + mm + ':' + ss;
            }

            function updateUI() {
                const box = document.getElementById('mainbox');
                const lab = document.getElementById('mainlab');
                const val = document.getElementById('tval');
                if (!box || !lab || !val) return;

                if (state === "PAUSED") {
                    anchorStartMs = null;
                    box.style.borderColor = "#FFA000";
                    lab.style.color = "#FFA000";
                    lab.textContent = "⏱️ PAUSED AT";
                    val.textContent = fmt(serverElapsed);
                } else if (state === "STOPPING") {
                    anchorStartMs = null;
                    box.style.borderColor = "#00BCD4";
                    lab.style.color = "#00BCD4";
                    lab.textContent = "⏱️ SAVING RECORDING...";
                    val.textContent = fmt(serverElapsed);
                } else if (state === "RECORDING") {
                    box.style.borderColor = "#00E676";
                    lab.style.color = "#00E676";
                    lab.textContent = "⏱️ RECORDING TIME";

                    const now = performance.now();
                    if (anchorStartMs === null) {
                        anchorStartMs = now - (serverElapsed * 1000);
                        maxDisplaySec = serverElapsed;
                    }

                    let calculatedSec = (now - anchorStartMs) / 1000.0;

                    if (serverElapsed > calculatedSec + 1.5) {
                        anchorStartMs = now - (serverElapsed * 1000);
                        calculatedSec = serverElapsed;
                    }

                    if (calculatedSec > maxDisplaySec) {
                        maxDisplaySec = calculatedSec;
                    }

                    val.textContent = fmt(maxDisplaySec);
                } else {
                    anchorStartMs = null;
                    maxDisplaySec = 0.0;
                    box.style.borderColor = "#30363D";
                    lab.style.color = "#8B949E";
                    lab.textContent = "⏱️ RECORDING TIME";
                    val.textContent = fmt(0);
                }
            }

            function syncData() {
                // 1. Read directly from localStorage (instant, synchronous)
                try {
                    const raw = localStorage.getItem('swarcare_status');
                    if (raw) {
                        const data = JSON.parse(raw);
                        state = data.state;
                        serverElapsed = data.elapsed_s;
                    }
                } catch(e) {}

                // 2. Non-blocking sidecar HTTP probe
                if (tryHttp && !fetchInFlight) {
                    fetchInFlight = true;
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 250);
                    let host = 'localhost';
                    try {
                        if (window.parent && window.parent.location && window.parent.location.hostname) host = window.parent.location.hostname;
                        else if (window.location && window.location.hostname) host = window.location.hostname;
                    } catch(e){}
                    if (!host) host = 'localhost';

                    fetch('http://' + host + ':7654/status', {cache: 'no-store', signal: controller.signal})
                        .then(res => res.ok ? res.json() : null)
                        .then(data => {
                            if (data) {
                                state = data.state;
                                serverElapsed = data.elapsed_s;
                            }
                        })
                        .catch(() => {})
                        .finally(() => {
                            clearTimeout(timeoutId);
                            fetchInFlight = false;
                        });
                }
            }

            setInterval(syncData, 100);

            function animLoop() {
                updateUI();
                requestAnimationFrame(animLoop);
            }
            requestAnimationFrame(animLoop);
            syncData();
        </script>
    </body>
    </html>
    """
    st.iframe(timer_static_html, height=72)

    st.markdown("---")

    # --- 3. STATUS BANNER ---
    @st.fragment(run_every=1.0 if is_live else None)
    def render_status_banner():
        if backend.state == "STOPPED":
            st.error("System Status: **STOPPED / IDLE** ⏹️")
        elif backend.state == "RECORDING":
            st.success("System Status: **RECORDING SENSORS LIVE** ▶️")
        elif backend.state == "PAUSED":
            st.warning("System Status: **RECORDING PAUSED** ⏸️")
        elif backend.state == "STOPPING":
            st.info(
                "System Status: **WRITING DATA STREAMS GRACEFULLY... PLEASE"
                " WAIT** ⏳"
            )

    render_status_banner()
    st.write("")

    # --- 4. UNIFIED VEENA ANOMALY & DIAGNOSTIC MONITOR ---
    st.subheader("🎯 Single Unified Anomaly & Diagnostic Monitor")
    st.caption("Parallel Hybrid Architecture: Physics Pitch Engine (Tuning) + ML Structural Fault Classifier running simultaneously.")

    _TONIC_OPTIONS_UI = {
        "A1 (55 Hz)":   55.00,  "A#1 (58 Hz)":  58.27,  "B1 (62 Hz)":   61.74,
        "C2 (65 Hz)":   65.41,  "C#2 (69 Hz)":  69.30,  "D2 (73 Hz)":   73.42,
        "D#2 (78 Hz)":  77.78,  "E2 (82 Hz)":   82.41,  "F2 (87 Hz)":   87.31,
        "F#2 (93 Hz)":  92.50,  "G2 (98 Hz)":   98.00,  "G#2 (104 Hz)": 103.83,
        "A2 (110 Hz)":  110.00, "A#2 (117 Hz)": 116.54, "B2 (123 Hz)":  123.47,
        "C3 (131 Hz)":  130.81, "C#3 (139 Hz)": 138.59, "D3 (147 Hz)":  146.83,
        "D#3 (156 Hz)": 155.56, "E3 (165 Hz)":  164.81, "F3 (175 Hz)":  174.61,
        "F#3 (185 Hz)": 185.00,
    }
    _TONIC_KEYS = list(_TONIC_OPTIONS_UI.keys())
    _STRING_OPTIONS = {
        "S1 — Sarani (Tara Sa, 2× Sa)": "S1",
        "S2 — Panchama (Pa, 1.5× Sa)": "S2",
        "S3 — Mandra Sa (tonic)": "S3",
        "S4 — Anumandra (lower Pa, 0.75× Sa)": "S4",
        "T1 — Chikari 1 (Sa, 4×)": "T1",
        "T2 — Chikari 2 (Pa, 6×)": "T2",
        "T3 — Chikari 3 (Sa, 8×)": "T3",
        "🔄 Auto-detect from pitch": None,
    }
    _STRING_KEYS = list(_STRING_OPTIONS.keys())

    _STRUCTURAL_DEPTH_DESCRIPTIONS = {
        -2: {"title": "Silence / System Idle", "summary": "No audio excitation detected. Waiting for musician to pluck a Veena string."},
        -1: {"title": "Non-Veena Sound / Human Voice", "summary": "Acoustic signal analysis identified non-instrument sound (human speech, vocalization, or ambient noise). System requires genuine Saraswati Veena string resonance for diagnostic evaluation."},
        0: {"title": "Structurally Sound & Resonant", "summary": "Instrument exhibits optimal acoustic resonance across the resonator (Kudam), bridge (Kudirai), and neck (Dandi). No structural damping detected."},
        1: {"title": "Structurally Sound & Resonant", "summary": "Instrument exhibits optimal acoustic resonance across the resonator (Kudam), bridge (Kudirai), and neck (Dandi). No structural damping detected."},
        2: {"title": "Fret Wear / Misalignment", "summary": "Bronze fret surface wear or loose fret fixing along the wax ledge (Melakku), causing non-linear buzz and fret contact impedance."},
        3: {"title": "String Corrosion / Oxidation", "summary": "Surface oxidation and metal fatigue on steel/bronze core strings, altering linear mass density and harmonic purity."},
        4: {"title": "Bridge Tilt / Kudirai Asymmetry", "summary": "Angular misalignment or uneven base contact of the main bridge (Kudirai) on the resonator soundboard plate."},
        5: {"title": "Kudam Crack / Resonator Shell Fracture", "summary": "Structural hairline crack or wood joint separation in the Jackwood resonator shell (Kudam), causing internal cavity acoustic leakage."},
        6: {"title": "Loose Peg / Birudai Slippage", "summary": "Taper pin friction failure in the tuning peg box (Birudai), causing continuous mechanical tension slippage under string load."},
        7: {"title": "String Buzz / Jiva Thread Mismatch", "summary": "Improper contact angle between string and bridge curvature or damaged cotton/silk buzzing thread (Jiva/Javali)."},
        8: {"title": "Sympathetic Resonance Dampening", "summary": "Acoustic damping in auxiliary drone resonators or wax wall structure, reducing sustain of unplucked resonant strings."},
        9: {"title": "Finish Degradation / Shellac Flaking", "summary": "Degraded French polish / shellac coating on the wood body, affecting micro-porosity and ambient moisture protection."},
        10: {"title": "Detached Bridge / Base Separation", "summary": "Partial adhesive separation between the bone/wood Kudirai base and the soundboard wood plate."},
        11: {"title": "Nut Groove Wear / Meru Slit Wear", "summary": "Deepened or widened string guide grooves at the upper bridge (Meru), causing string rattling and open-string buzzing."},
    }

    vcfg_col1, vcfg_col2 = st.columns(2)
    with vcfg_col1:
        _tonic_key = st.selectbox(
            "🎵 Sa / Tonic Frequency",
            _TONIC_KEYS,
            index=_TONIC_KEYS.index("C3 (131 Hz)"),
            key="veena_tonic_select",
            help="Sa anchor. All 7 string targets compute from this.",
        )
        veena_tonic_hz = _TONIC_OPTIONS_UI[_tonic_key]
    with vcfg_col2:
        _str_key = st.selectbox(
            "🎻 String Being Plucked",
            _STRING_KEYS,
            index=0,
            key="veena_string_select",
            help="Which Veena string you are currently plucking.",
        )
        veena_string_label = _STRING_OPTIONS[_str_key]

    @st.fragment(
        run_every=AI_REFRESH_SEC if backend.state == "RECORDING" else None
    )
    def render_unified_anomaly_monitor():
        # Identify the active or latest session
        if backend.state in ("RECORDING", "PAUSED", "STOPPING") and backend.current_prefix:
            _latest = backend.current_prefix
        elif os.path.exists(DATA_DIR):
            _vfiles = [
                f for f in os.listdir(DATA_DIR)
                if f.endswith("_piezo.csv") or f.endswith("_audio.wav") or f.endswith("_audio.tmp")
            ]
            _all_rec = sorted(
                {f.replace("_piezo.csv", "").replace("_audio.wav", "").replace("_audio.tmp", "") for f in _vfiles},
                reverse=True,
            )
            _latest = _all_rec[0] if _all_rec else None
        else:
            _latest = None

        if not _latest:
            with st.container(border=True):
                st.info(
                    "💡 **Anomaly Monitor Standing By:** Start recording and pluck a string to trigger real-time physics tuning and structural diagnostics."
                )
            return

        _vkey = f"veena_res_{_latest}_{veena_tonic_hz:.2f}_{veena_string_label}"
        _now = time.time()

        if (
            _vkey not in st.session_state
            or (_now - st.session_state.get(f"{_vkey}_ts", 0)) > 2.0
        ):
            st.session_state[_vkey] = backend.analyze_veena_ai(
                _latest,
                tonic_hz=veena_tonic_hz,
                cents_threshold=15.0,
                string_label=veena_string_label,
            )
            st.session_state[f"{_vkey}_ts"] = _now

        vres = st.session_state.get(_vkey, {})

        if not vres.get("available", False):
            with st.container(border=True):
                st.warning(f"⏳ {vres.get('error', 'Diagnostic engine buffering audio…')}")
            return

        tuning = vres.get("tuning", {})
        quality = vres.get("quality", {})
        is_healthy = vres.get("is_healthy", False)
        is_veena = vres.get("is_veena", True)
        sound_type = vres.get("sound_type", "Veena String Resonance")
        master_status = vres.get("status", "Unknown")
        
        _tstat = tuning.get("status", "NO_PITCH")
        _cdev = tuning.get("cents_dev", 0.0)
        _f0 = tuning.get("f0_hz", 0.0)
        _tgt = tuning.get("target_hz", 0.0)
        _sname = tuning.get("string_name", "—")
        _tmsg = tuning.get("message", "")
        _tconf = tuning.get("confidence", 0.0)

        _qok = quality.get("is_healthy", False)
        _qlabel = quality.get("label", "Healthy")
        _qconf = quality.get("confidence", 0.0)
        _qcls = quality.get("fault_class", 0)

        # 1. Master Verdict Classification Banner
        if master_status == "Silence" or _tstat == "SILENCE" or _qcls == -2:
            st.info("⚪ **SYSTEM IDLE / SILENCE** — Waiting for audio input. Pluck a Saraswati Veena string to begin diagnostics.")
        elif not is_veena or master_status == "Non-Veena Sound Detected" or _tstat == "NON_VEENA" or _qcls == -1:
            st.error(f"🚨 **ANOMALY DETECTED — {_qlabel.upper()}** (Acoustic signature does not match Saraswati Veena string).")
        elif is_healthy and _tstat == "IN_TUNE":
            st.success("🟢 **HEALTHY & IN TUNE** — Instrument is structurally sound and tuned accurately within ±15 cents.")
        elif is_healthy and _tstat in ["FLAT", "SHARP"]:
            st.warning(f"🟡 **HEALTHY (TUNING WATCH: {_tstat})** — Instrument structure is sound, but string is {_tstat} by {abs(_cdev):.1f} cents.")
        elif not is_healthy and _qcls > 1:
            st.error(f"🚨 **ANOMALY DETECTED — {_qlabel.upper()}** (Structural defect identified by ML Classifier).")
        elif not is_healthy and _tstat in ["FLAT", "SHARP"]:
            st.warning(f"🟡 **TUNING MISALIGNMENT: {_tstat}** — String pitch is {_tstat} by {abs(_cdev):.1f} cents.")
        elif not is_healthy and _tstat == "NO_PITCH":
            st.warning("⏳ **ACOUSTIC DAMPENING / NO PITCH** — Unclear fundamental string pitch detected.")
        else:
            st.info("ℹ️ **DIAGNOSTIC ACTIVE** — Analyzing resonance and tuning...")

        # 2. Key Metrics Row
        with st.container(border=True):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                if master_status == "Silence" or _qcls == -2:
                    _badge = "🔇 Silence / Idle"
                    _delta = "No Input"
                    _dcolor = "off"
                elif not is_veena or _qcls == -1:
                    _badge = f"🚨 {_qlabel}"
                    _delta = "Non-Veena Anomaly"
                    _dcolor = "inverse"
                else:
                    _badge = "✅ Healthy" if is_healthy else f"🚨 {_qlabel}"
                    _delta = f"{_qconf:.1f}% confidence" if _qconf > 0 else "Active"
                    _dcolor = "normal" if is_healthy else "inverse"

                st.metric(
                    label="Overall Health Status",
                    value=_badge,
                    delta=_delta,
                    delta_color=_dcolor,
                )
            with mc2:
                _STATUS_EMOJI = {
                    "IN_TUNE": "✅", "FLAT": "⬇️", "SHARP": "⬆️",
                    "NO_PITCH": "🔇", "SILENCE": "🟤", "NON_VEENA": "🚫",
                }
                if not is_veena or _tstat == "NON_VEENA":
                    _t_label = "🚫 Non-Veena"
                    _t_delta = "Non-instrument sound"
                    _t_dcolor = "inverse"
                elif _tstat == "SILENCE":
                    _t_label = "🔇 Silence"
                    _t_delta = "No pitch"
                    _t_dcolor = "off"
                else:
                    _t_label = f"{_STATUS_EMOJI.get(_tstat, '?')} {_tstat}"
                    _t_delta = f"{_cdev:+.1f} cents" if _f0 > 0 else "No pitch"
                    _t_dcolor = "normal" if _tstat == "IN_TUNE" else "inverse"

                st.metric(
                    label=f"Tuning ({_sname})",
                    value=_t_label,
                    delta=_t_delta,
                    delta_color=_t_dcolor,
                )
            with mc3:
                _depth_entry = _STRUCTURAL_DEPTH_DESCRIPTIONS.get(_qcls, {"title": _qlabel, "summary": ""})
                st.metric(
                    label="Identified Anomaly / Defect",
                    value=_depth_entry["title"],
                    delta=f"Class {_qcls}" if _qcls >= 0 else "Sound Validation",
                    delta_color="normal" if is_healthy else ("off" if _qcls == -2 else "inverse"),
                )

            # High-Precision Responsive Tuning Gauge
            if is_veena and _f0 > 0 and _tgt > 0:
                clamped_progress = max(0.0, min(1.0, (_cdev + 50.0) / 100.0))
                st.progress(
                    clamped_progress,
                    text=f"Detected: {_f0:.1f} Hz │ Target: {_tgt:.1f} Hz │ Dev: {_cdev:+.1f} cents (Target: ±15 cents)",
                )
            elif not is_veena:
                st.progress(
                    0.0,
                    text=f"Non-Veena Acoustic Event ({sound_type}) │ Requires genuine Veena string excitation",
                )
            if _tmsg:
                st.caption(f"📌 **Tuning Guide:** {_tmsg}")

        # 3. Structural Depth (What Could Be The Anomaly)
        _depth_info = _STRUCTURAL_DEPTH_DESCRIPTIONS.get(_qcls, {"title": _qlabel, "summary": "Structural analysis performed via YAMNet embeddings."})
        with st.container(border=True):
            st.markdown(f"**🔬 Structural Issues Depth:** `{_depth_info['title']}`")
            st.write(_depth_info["summary"])
            st.caption(f"Evaluated Target Session: `📄 {_latest}` │ Sound Type: **{sound_type}** │ Feature Vector: **527-D Parallel Hybrid**")

        # 4. Expandable Reference & Diagnostic Report
        with st.expander("🎷 String Reference Hz — all 7 strings at current tonic"):
            _rz = {
                "S1 Sarani":    round(veena_tonic_hz * 2.0,  2),
                "S2 Panchama":  round(veena_tonic_hz * 1.5,  2),
                "S3 Mandra Sa": round(veena_tonic_hz * 1.0,  2),
                "S4 Anumandra": round(veena_tonic_hz * 0.75, 2),
                "T1 Chikari":   round(veena_tonic_hz * 4.0,  2),
                "T2 Chikari":   round(veena_tonic_hz * 6.0,  2),
                "T3 Chikari":   round(veena_tonic_hz * 8.0,  2),
            }
            _rc1, _rc2, _rc3, _rc4 = st.columns(4)
            items = list(_rz.items())
            for i, col in enumerate([_rc1, _rc2, _rc3, _rc4]):
                with col:
                    if i < len(items):
                        col.metric(items[i][0], f"{items[i][1]} Hz")
                    if i + 4 < len(items):
                        col.metric(items[i + 4][0], f"{items[i + 4][1]} Hz")

        with st.expander("🔍 Full Diagnostic Telemetry Report (Raw JSON)"):
            st.json(vres)

    render_unified_anomaly_monitor()
    st.markdown("---")

    # --- 5. DUAL TELEMETRY MONITOR STATIONS ---

    # A. VIBRATION SERIAL MONITOR (Piezo Sensor)
    st.subheader("Vibration Serial Monitor (Piezo Sensor)")

    terminal_static_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; }
            html, body { margin:0; padding:0; background-color:#0E1117; overflow:hidden; font-family:Consolas, 'Courier New', monospace; width:100%; height:100%; }
            .terminal-box {
                background-color:#161B22;
                border:1px solid #30363D;
                border-radius:6px;
                height:100%;
                width:100%;
                padding:8px;
                overflow-y:auto;
                overflow-x:hidden;
                font-size: clamp(9.5px, 2.5vw, 11.5px);
                letter-spacing: -0.3px;
                font-variant-numeric: tabular-nums;
                white-space: pre-wrap;
                word-break: break-all;
                transition: border-color 0.25s ease;
            }
            .idle-wrap {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
                color: #8B949E;
                font-size: 12px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div id="term-box" class="terminal-box">
            <div class="idle-wrap">--- SERIAL MONITOR PIPELINE IDLE ---</div>
        </div>
        <script>
            let lastSecBlock = null;
            let baseTimeSec = null;
            let pRate = 2000;
            let tryHttp = true;
            let fetchInFlight = false;
            let lastRenderedSignature = "";

            function parseSec(txt, idx) {
                const hhmmss = txt.match(/(?:TIME|T)\\s*:\\s*(\\d{1,2}):(\\d{2}):([\\d.]+)/i);
                if (hhmmss) return parseFloat(hhmmss[1]) * 3600 + parseFloat(hhmmss[2]) * 60 + parseFloat(hhmmss[3]);
                const mmss = txt.match(/(?:TIME|T)\\s*:\\s*(\\d{1,2}):([\\d.]+)/i);
                if (mmss) return parseFloat(mmss[1]) * 60 + parseFloat(mmss[2]);
                const relT = txt.match(/(?:TIME|T|SEC|SECS)\\s*:\\s*([\\d.]+)/i);
                if (relT) return parseFloat(relT[1]);
                return idx / pRate;
            }

            function renderLines(linesData, state) {
                const box = document.getElementById('term-box');
                if (!box) return;

                if (state === "RECORDING") box.style.borderColor = "#00E676";
                else if (state === "PAUSED") box.style.borderColor = "#FFA000";
                else if (state === "STOPPING") box.style.borderColor = "#00BCD4";
                else box.style.borderColor = "#30363D";

                if (!linesData || linesData.length === 0) {
                    if (lastRenderedSignature !== "IDLE") {
                        box.innerHTML = '<div class="idle-wrap">--- SERIAL MONITOR PIPELINE IDLE ---</div>';
                        lastRenderedSignature = "IDLE";
                    }
                    return;
                }

                // Fast signature check to avoid DOM thrashing if lines have not changed
                const currentSig = linesData.map(l => (l.text || '') + (l.active ? '1' : '0')).join('|');
                if (currentSig === lastRenderedSignature) return;
                lastRenderedSignature = currentSig;

                let html = "";
                lastSecBlock = null;
                baseTimeSec = null;

                linesData.forEach((item, idx) => {
                    let txt = item.text || "";
                    let isAct = item.active || false;

                    let compact = txt.replace(/\\s*:\\s*/g, ':').replace(/\\s*\\|\\s*/g, '|');
                    let secVal = parseSec(txt, idx);
                    if (baseTimeSec === null) baseTimeSec = secVal;

                    let secBlock = Math.floor(secVal / 2) * 2;
                    if (lastSecBlock !== null && secBlock !== lastSecBlock) {
                        let relSec = Math.floor((secVal - baseTimeSec) / 2) * 2;
                        html += `<div style="border-top: 1px dotted #00BCD4; background: rgba(0, 188, 212, 0.08); color: #00BCD4; text-align: center; font-size: 9.5px; margin: 4px 0; padding: 2px 0; font-weight: bold; letter-spacing: 0.5px;">⏱️ ┈┈ 2s MARKER (+${relSec}s) ┈┈</div>`;
                    }
                    lastSecBlock = secBlock;

                    let col = isAct ? "#00E676" : "#C9D1D9";
                    html += `<div style="color:${col}; font-weight:bold; line-height:1.35;">${compact}</div>`;
                });

                box.innerHTML = html;
                box.scrollTop = box.scrollHeight;
            }

            function syncTerminal() {
                let linesData = null;
                let state = "STOPPED";

                // 1. Read from localStorage directly (fast & synchronous)
                try {
                    const raw = localStorage.getItem('swarcare_piezo');
                    if (raw) {
                        const data = JSON.parse(raw);
                        linesData = data.lines;
                        state = data.state;
                    }
                } catch(e) {}

                // 2. Non-blocking sidecar HTTP probe
                if (tryHttp && !fetchInFlight) {
                    fetchInFlight = true;
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 250);
                    let host = 'localhost';
                    try {
                        if (window.parent && window.parent.location && window.parent.location.hostname) host = window.parent.location.hostname;
                        else if (window.location && window.location.hostname) host = window.location.hostname;
                    } catch(e){}
                    if (!host) host = 'localhost';

                    fetch('http://' + host + ':7654/piezo_lines', {cache: 'no-store', signal: controller.signal})
                        .then(res => res.ok ? res.json() : null)
                        .then(data => {
                            if (data && data.lines) {
                                renderLines(data.lines, data.state);
                            }
                        })
                        .catch(() => {})
                        .finally(() => {
                            clearTimeout(timeoutId);
                            fetchInFlight = false;
                        });
                }

                if (linesData) {
                    renderLines(linesData, state);
                }
            }

            setInterval(syncTerminal, 180);
            syncTerminal();
        </script>
    </body>
    </html>
    """
    st.iframe(terminal_static_html, height=195)
    st.write("")

    # B. AUDIO LIVE MONITOR (USB Microphone - 24 FPS Oscilloscope)
    st.subheader("Audio Live Monitor (USB Microphone)")

    audio_static_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; }
            html, body { margin:0; padding:0; background-color:#0E1117; overflow:hidden; width:100%; height:100%; }
            #aC { display:block; background:#161B22; border-radius:6px; border:1px solid #30363D; width:100%; height:100%; }
        </style>
    </head>
    <body>
        <canvas id="aC"></canvas>
        <script>
            (function() {
                const canvas = document.getElementById('aC');
                const ctx = canvas.getContext('2d');

                function updateCanvasDimensions() {
                    const dpr = window.devicePixelRatio || 1;
                    const containerWidth = window.innerWidth || 360;
                    const containerHeight = window.innerHeight || 200;
                    canvas.width = containerWidth * dpr;
                    canvas.height = containerHeight * dpr;
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                    return { w: containerWidth, h: containerHeight };
                }

                let dims = updateCanvasDimensions();
                window.addEventListener('resize', () => { dims = updateCanvasDimensions(); });

                const visibleWindowSec = 4.0;
                let history = [];
                let serverTimeSec = 0.0;
                let lastFetchMs = performance.now();
                let lastFrameMs = performance.now();
                let displayTimeSec = 0.0;
                let state = "STOPPED";
                let tryHttp = true;
                let fetchInFlight = false;

                const targetFPS = 24;
                const frameIntervalMs = 1000 / targetFPS;
                let lastRenderMs = performance.now();

                function fetchAudioData() {
                    // 1. Read from localStorage directly (fast & synchronous)
                    try {
                        const raw = localStorage.getItem('swarcare_audio');
                        if (raw) {
                            const data = JSON.parse(raw);
                            if (data.samples && data.samples.length > 0) {
                                history = data.samples;
                            }
                            serverTimeSec = data.elapsed_s || 0.0;
                            state = data.state || "STOPPED";
                            lastFetchMs = performance.now();
                        }
                    } catch(e) {}

                    // 2. Non-blocking sidecar HTTP probe
                    if (tryHttp && !fetchInFlight) {
                        fetchInFlight = true;
                        const controller = new AbortController();
                        const timeoutId = setTimeout(() => controller.abort(), 250);
                        let host = 'localhost';
                        try {
                            if (window.parent && window.parent.location && window.parent.location.hostname) host = window.parent.location.hostname;
                            else if (window.location && window.location.hostname) host = window.location.hostname;
                        } catch(e){}
                        if (!host) host = 'localhost';

                        fetch('http://' + host + ':7654/audio_data', {cache: 'no-store', signal: controller.signal})
                            .then(res => res.ok ? res.json() : null)
                            .then(data => {
                                if (data && data.samples && data.samples.length > 0) {
                                    history = data.samples;
                                    serverTimeSec = data.elapsed_s || 0.0;
                                    state = data.state || "STOPPED";
                                    lastFetchMs = performance.now();
                                }
                            })
                            .catch(() => {})
                            .finally(() => {
                                clearTimeout(timeoutId);
                                fetchInFlight = false;
                            });
                    }
                }

                function draw() {
                    const dpr = window.devicePixelRatio || 1;
                    const w = canvas.width / dpr;
                    const h = canvas.height / dpr;

                    const nowMs = performance.now();
                    lastFrameMs = nowMs;

                    if (state === "RECORDING") {
                        let targetTimeSec = serverTimeSec + ((nowMs - lastFetchMs) / 1000.0);
                        if (displayTimeSec === 0 || Math.abs(displayTimeSec - targetTimeSec) > 1.0) {
                            displayTimeSec = targetTimeSec;
                        } else {
                            displayTimeSec += (targetTimeSec - displayTimeSec) * 0.25;
                        }
                    } else {
                        displayTimeSec = serverTimeSec;
                    }

                    const isMobile = w < 480;
                    const padLeft = isMobile ? 56 : 68;
                    const padBottom = isMobile ? 28 : 32;
                    const padTop = 18;
                    const padRight = isMobile ? 12 : 20;

                    const plotW = Math.max(10, w - padLeft - padRight);
                    const plotH = Math.max(10, h - padTop - padBottom);
                    const midY = padTop + (plotH / 2);

                    ctx.clearRect(0, 0, w, h);

                    // Grid lines
                    ctx.strokeStyle = '#21262D';
                    ctx.lineWidth = 1;
                    for(let i = 0; i <= 4; i++) {
                        let y = padTop + (plotH / 4) * i;
                        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(w - padRight, y); ctx.stroke();
                    }

                    ctx.strokeStyle = '#30363D';
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(padLeft, padTop, plotW, plotH);

                    ctx.setLineDash([3, 3]);
                    ctx.strokeStyle = '#8B949E';
                    ctx.beginPath(); ctx.moveTo(padLeft, midY); ctx.lineTo(w - padRight, midY); ctx.stroke();
                    ctx.setLineDash([]);

                    ctx.fillStyle = '#8B949E';
                    ctx.font = isMobile ? '8.5px monospace' : '10px monospace';
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('+1.0', padLeft - 6, padTop);
                    ctx.fillText('0.0', padLeft - 6, midY);
                    ctx.fillText('-1.0', padLeft - 6, padTop + plotH);

                    const N = history.length;
                    if(N > 0 && state !== "STOPPED") {
                        ctx.fillStyle = state === "RECORDING" ? '#00E676' : '#FFA000';
                        const barWidth = isMobile ? 1.5 : 2.0;

                        for (let col = 0; col < plotW; col += barWidth) {
                            let x = padLeft + col;
                            let sampleIdx = Math.floor((col / plotW) * N);
                            if (sampleIdx >= N) sampleIdx = N - 1;

                            let amplitude = Math.abs(history[sampleIdx]);
                            let barHeight = Math.max(2, amplitude * (plotH * 0.88));
                            ctx.fillRect(x, midY - (barHeight / 2), barWidth, barHeight);
                        }
                    } else if (N > 0 && state === "STOPPED") {
                        ctx.fillStyle = '#8B949E';
                        const barWidth = isMobile ? 1.5 : 2.0;
                        for (let col = 0; col < plotW; col += barWidth) {
                            let x = padLeft + col;
                            let sampleIdx = Math.floor((col / plotW) * N);
                            if (sampleIdx >= N) sampleIdx = N - 1;
                            let amplitude = Math.abs(history[sampleIdx]);
                            let barHeight = Math.max(2, amplitude * (plotH * 0.88));
                            ctx.fillRect(x, midY - (barHeight / 2), barWidth, barHeight);
                        }
                    } else {
                        ctx.fillStyle = '#8B949E';
                        ctx.font = isMobile ? '10px monospace' : '11.5px monospace';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText('--- AUDIO MONITOR IDLE ---', padLeft + plotW / 2, midY);
                    }

                    // Clean 2-second time markers
                    const pixelsPerSec = plotW / visibleWindowSec;
                    const start2sMarker = Math.floor(Math.max(0, displayTimeSec - visibleWindowSec) / 2.0) * 2.0;
                    const end2sMarker = displayTimeSec + 2.0;

                    for(let tMark = start2sMarker; tMark <= end2sMarker; tMark += 2.0) {
                        let markOffset = displayTimeSec - tMark;
                        let markX = (w - padRight) - (markOffset * pixelsPerSec);

                        if (markX >= padLeft && markX <= (w - padRight)) {
                            ctx.save();
                            ctx.setLineDash([2, 2]);
                            ctx.strokeStyle = '#00BCD4';
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(markX, padTop);
                            ctx.lineTo(markX, padTop + plotH);
                            ctx.stroke();
                            ctx.restore();

                            ctx.fillStyle = '#00BCD4';
                            ctx.font = isMobile ? 'bold 8px monospace' : 'bold 9.5px monospace';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText(tMark.toFixed(1) + 's', markX, padTop - 2);

                            ctx.fillStyle = '#C9D1D9';
                            ctx.textBaseline = 'top';
                            ctx.fillText(tMark.toFixed(1) + 's', markX, padTop + plotH + 3);
                        }
                    }

                    ctx.fillStyle = '#8B949E';
                    ctx.font = isMobile ? 'bold 9.5px monospace' : 'bold 11px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillText('Time', padLeft + plotW / 2, h - 12);

                    ctx.save();
                    ctx.translate(isMobile ? 12 : 14, padTop + plotH / 2);
                    ctx.rotate(-Math.PI / 2);
                    ctx.fillStyle = '#C9D1D9';
                    ctx.font = isMobile ? 'bold 9px monospace' : 'bold 11px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('Normalised Amplitude', 0, 0);
                    ctx.restore();
                }

                setInterval(fetchAudioData, 180);

                function anim(nowMs) {
                    requestAnimationFrame(anim);
                    const elapsed = nowMs - lastRenderMs;
                    if (elapsed >= frameIntervalMs) {
                        lastRenderMs = nowMs - (elapsed % frameIntervalMs);
                        draw();
                    }
                }
                requestAnimationFrame(anim);
            })();
        </script>
    </body>
    </html>
    """
    st.iframe(audio_static_html, height=205)
    st.markdown("---")

    # C. SYNCHRONIZED STORAGE METRICS PANEL
    @st.fragment(run_every=1.0 if is_live else None)
    def render_storage_metrics():
        p_samples = backend.piezo_samples_recorded
        p_lines = p_samples + 1 if p_samples > 0 else 0
        p_time = p_samples / float(backend.PIEZO_SAMPLE_RATE_HZ)

        a_samples = backend.audio_samples_recorded
        a_time = a_samples / float(backend.AUDIO_SAMPLE_RATE_HZ)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Vibration Samples", f"{p_samples:,}")
            st.caption(f"📝 Piezo Lines: **{p_lines:,}**")
            st.caption(f"⏱️ Piezo Duration: **{p_time:.2f} s**")
        with c2:
            st.metric("Audio Samples", f"{a_samples:,}")
            st.caption(f"🔊 Audio WAV: **{a_samples:,}**")
            st.caption(f"⏱️ Audio Duration: **{a_time:.2f} s**")

    render_storage_metrics()

# ==========================================
# TAB 2: SAVED RECORDS EXPLORER
# ==========================================
with tab_records:
    st.subheader("📁 Saved Records Explorer")

    if os.path.exists(DATA_DIR):
        raw_files = sorted(os.listdir(DATA_DIR), reverse=True)
        csv_files = [f for f in raw_files if f.endswith(".csv")]
        wav_files = [f for f in raw_files if f.endswith(".wav")]
    else:
        csv_files, wav_files = [], []

    # --- FEATURE 2: DOWNLOAD ALL AS ZIP ARCHIVE (HIGH PERFORMANCE CACHED) ---
    if csv_files or wav_files:
        dir_sig = _get_recordings_signature()
        zip_bytes = generate_recordings_zip(dir_sig)
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"SwarCare_All_Recordings_{ts_now}.zip"

        st.download_button(
            label="📦 Download All Recordings (.zip)",
            data=zip_bytes,
            file_name=zip_filename,
            mime="application/zip",
            use_container_width=True,
            help=(
                "Download all stored CSV vibration data and WAV audio"
                " recordings in a single ordered ZIP archive."
            ),
        )
        st.markdown("---")

    limit = st.session_state.visible_records_limit

    col_csv, col_wav = st.columns(2)

    # --- LEFT COLUMN: .csv Files ---
    with col_csv:
        st.markdown("### 📝 `.csv` (Vibration)")
        displayed_csv = csv_files[:limit]

        if not displayed_csv:
            st.caption("No CSV records available.")
        else:
            for file in displayed_csv:
                file_path = os.path.join(DATA_DIR, file)
                if not os.path.exists(file_path):
                    continue

                with st.container(border=True):
                    st.write(f"📄 **{file}**")

                    custom_name = st.text_input(
                        "Rename download:",
                        value=file,
                        key=f"rename_{file}",
                        help="Enter custom filename for download",
                    )

                    target_name = (
                        custom_name.strip() if custom_name.strip() else file
                    )
                    if not target_name.endswith(".csv"):
                        target_name += ".csv"

                    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
                    with btn_col1:
                        file_mtime = os.path.getmtime(file_path)
                        file_data = get_cached_file_bytes(file_path, file_mtime)
                        st.download_button(
                            label="📥 CSV",
                            data=file_data,
                            file_name=target_name,
                            mime="text/csv",
                            key=f"dl_{file}_{target_name}",
                            use_container_width=True,
                        )
                    with btn_col2:
                        if st.button(
                            "✏️",
                            key=f"rn_disk_{file}",
                            use_container_width=True,
                            help="Rename file on disk",
                        ):
                            if target_name != file:
                                new_path = os.path.join(DATA_DIR, target_name)
                                try:
                                    os.rename(file_path, new_path)
                                    st.toast(
                                        f"Renamed to {target_name}!", icon="✏️"
                                    )
                                    time.sleep(0.1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    with btn_col3:
                        if st.button(
                            "🗑️",
                            key=f"del_{file}",
                            use_container_width=True,
                            help="Delete file",
                        ):
                            try:
                                os.remove(file_path)
                                st.toast(f"Removed {file}!", icon="🗑️")
                                time.sleep(0.1)
                                st.rerun()
                            except Exception as e:
                                f"Error: {e}"

    # --- RIGHT COLUMN: .wav Files ---
    with col_wav:
        st.markdown("### 🔊 `.wav` (Audio)")
        displayed_wav = wav_files[:limit]

        if not displayed_wav:
            st.caption("No WAV records available.")
        else:
            for file in displayed_wav:
                file_path = os.path.join(DATA_DIR, file)
                if not os.path.exists(file_path):
                    continue

                with st.container(border=True):
                    st.write(f"🎧 **{file}**")

                    custom_wav_name = st.text_input(
                        "Rename download:",
                        value=file,
                        key=f"rename_wav_{file}",
                        help="Enter custom filename for download",
                    )

                    target_wav_name = (
                        custom_wav_name.strip()
                        if custom_wav_name.strip()
                        else file
                    )
                    if not target_wav_name.endswith(".wav"):
                        target_wav_name += ".wav"

                    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
                    with btn_col1:
                        wav_mtime = os.path.getmtime(file_path)
                        wav_data = get_cached_file_bytes(file_path, wav_mtime)
                        st.download_button(
                            label="📥 WAV",
                            data=wav_data,
                            file_name=target_wav_name,
                            mime="audio/wav",
                            key=f"dl_wav_{file}_{target_wav_name}",
                            use_container_width=True,
                        )
                    with btn_col2:
                        if st.button(
                            "✏️",
                            key=f"rn_disk_wav_{file}",
                            use_container_width=True,
                            help="Rename file on disk",
                        ):
                            if target_wav_name != file:
                                new_path = os.path.join(DATA_DIR, target_wav_name)
                                try:
                                    os.rename(file_path, new_path)
                                    st.toast(
                                        f"Renamed to {target_wav_name}!",
                                        icon="✏️",
                                    )
                                    time.sleep(0.1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    with btn_col3:
                        if st.button(
                            "🗑️",
                            key=f"del_wav_{file}",
                            use_container_width=True,
                            help="Delete file",
                        ):
                            try:
                                os.remove(file_path)
                                st.toast(f"Removed {file}!", icon="🗑️")
                                time.sleep(0.1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    with st.expander("▶️ Replay"):
                        st.audio(file_path, format="audio/wav")

    # --- WORKSPACE FOLDER DECK CONTROLS ---
    st.write("")
    total_max_files = max(len(csv_files), len(wav_files))
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

    with ctrl_col1:
        if limit < total_max_files:
            if st.button("🔽 Show More", use_container_width=True):
                st.session_state.visible_records_limit += 5
                st.rerun()
        else:
            st.button(
                "✨ All Displayed", disabled=True, use_container_width=True
            )

    with ctrl_col2:
        if limit > 3:
            if st.button("🔄 Reset View", use_container_width=True):
                st.session_state.visible_records_limit = 3
                st.rerun()
        else:
            st.button("🔄 Reset View", disabled=True, use_container_width=True)

    with ctrl_col3:
        if st.button(
            "💥 Clear All",
            use_container_width=True,
            help="Wipe all recordings",
        ):
            if os.path.exists(DATA_DIR):
                for file_to_wipe in os.listdir(DATA_DIR):
                    try:
                        os.remove(os.path.join(DATA_DIR, file_to_wipe))
                    except Exception:
                        pass
            st.session_state.visible_records_limit = 3
            st.toast("Storage directory wiped clean!", icon="💥")
            time.sleep(0.2)
            st.rerun()
