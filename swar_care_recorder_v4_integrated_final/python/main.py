import io
import json
import os
import time
import urllib.parse
import zipfile
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

# Inject Mobile Viewport & Touch Responsive CSS Rules
st.markdown(
    """
<style>
    /* Mobile-First Layout Adjustments */
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }
        h1 { font-size: 1.45rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.0rem !important; }

        /* Metric Scaling for Mobile Screens */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }
        /* Mobile Touch Target Padding */
        .stButton button {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            font-size: 0.82rem !important;
        }
    }

    /* Clean Seamless Iframes */
    iframe {
        width: 100% !important;
        border: none !important;
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


# Helper function to generate ZIP archive of all stored recordings in order
def generate_recordings_zip():
    zip_buffer = io.BytesIO()
    if os.path.exists(DATA_DIR):
        raw_files = sorted(os.listdir(DATA_DIR))
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in raw_files:
                if fname.endswith(".csv") or fname.endswith(".wav"):
                    fpath = os.path.join(DATA_DIR, fname)
                    zf.write(fpath, arcname=fname)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ==========================================
# 📑 CONSOLIDATED 2-TAB INTERFACE
# ==========================================
tab_live_ai, tab_records = st.tabs(
    ["📡 Live Telemetry & AI Diagnostics", "📁 Saved Records Explorer"]
)

LIVE_SYNC_SEC = 1.0 / 24.0  # Exactly 24 FPS telemetry sync bridge (~0.0417s)
AI_REFRESH_SEC = 4.0  # Optimized to avoid UI stutter during live inference

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

    # --- INVISIBLE DATA BUS BRIDGE (LocalStorage Sync) ---
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
        <style>
            body { margin:0; background:transparent; font-family:Consolas, 'Courier New', monospace; }
            .box { background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 12px; text-align:center; box-sizing:border-box; transition: border-color 0.3s; }
            .lab { color:#8B949E; font-weight:bold; font-size:11px; letter-spacing:1px; transition: color 0.3s; }
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
                    box.style.borderColor = "#CCA000";
                    lab.style.color = "#CCA000";
                    lab.textContent = "⏱️ PAUSED AT";
                    val.textContent = fmt(serverElapsed);
                } else if (state === "STOPPING") {
                    anchorStartMs = null;
                    box.style.borderColor = "#00BCD4";
                    lab.style.color = "#00BCD4";
                    lab.textContent = "⏱️ SAVING RECORDING...";
                    val.textContent = fmt(serverElapsed);
                } else if (state === "RECORDING") {
                    box.style.borderColor = "#00C853";
                    lab.style.color = "#00C853";
                    lab.textContent = "⏱️ RECORDING TIME";

                    const now = Date.now();
                    if (anchorStartMs === null) {
                        anchorStartMs = now - (serverElapsed * 1000);
                        maxDisplaySec = serverElapsed;
                    }

                    let calculatedSec = (now - anchorStartMs) / 1000.0;

                    if (serverElapsed > calculatedSec + 2.0) {
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

            async function syncData() {
                let synced = false;
                if (tryHttp) {
                    try {
                        const res = await fetch('http://localhost:7654/status', {cache: 'no-store'});
                        if (res.ok) {
                            const data = await res.json();
                            state = data.state;
                            serverElapsed = data.elapsed_s;
                            synced = true;
                        } else {
                            tryHttp = false;
                        }
                    } catch(e) {
                        tryHttp = false;
                    }
                }

                if (!synced) {
                    try {
                        const raw = localStorage.getItem('swarcare_status');
                        if (raw) {
                            const data = JSON.parse(raw);
                            state = data.state;
                            serverElapsed = data.elapsed_s;
                        }
                    } catch(e) {}
                }
                updateUI();
            }

            setInterval(syncData, 41);

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
    st.iframe(timer_static_html, height=76)

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

    # --- 4. AI ANOMALY MONITOR DECK (Non-Blocking Cached Fragment) ---
    st.subheader("🧠 Real-Time AI Diagnostic & Anomaly Center")

    @st.fragment(
        run_every=AI_REFRESH_SEC if backend.state == "RECORDING" else None
    )
    def render_ai_diagnostics():
        if os.path.exists(DATA_DIR):
            valid_files = [
                f
                for f in os.listdir(DATA_DIR)
                if f.endswith("_piezo.csv") or f.endswith("_audio.wav")
            ]
            all_recordings = sorted(
                [
                    f.replace("_piezo.csv", "").replace("_audio.wav", "")
                    for f in valid_files
                ],
                reverse=True,
            )
            unique_prefixes = sorted(list(set(all_recordings)), reverse=True)
        else:
            unique_prefixes = []

        if unique_prefixes:
            latest_prefix = unique_prefixes[0]

            # Non-blocking state cache for AI inference
            cache_key = f"ai_res_{latest_prefix}"
            now_time = time.time()

            if (
                cache_key not in st.session_state
                or (now_time - st.session_state.get(f"{cache_key}_ts", 0)) > 3.5
            ):
                st.session_state[cache_key] = backend.analyze_recording_ai(
                    latest_prefix
                )
                st.session_state[f"{cache_key}_ts"] = now_time

            ai_results = st.session_state[cache_key]

            with st.container(border=True):
                col_ai1, col_ai2, col_ai3 = st.columns(3)

                status_color_text = ai_results.get("status", "UNKNOWN")
                with col_ai1:
                    st.markdown(
                        f"**Diagnostic Status**\n### {status_color_text}"
                    )
                with col_ai2:
                    score = ai_results.get("score", 0.0)
                    st.metric("Anomaly Score (0-1)", f"{score:.3f}")
                    st.progress(min(1.0, max(0.0, score)))
                with col_ai3:
                    conf = ai_results.get("confidence", 0)
                    st.metric("Confidence Level", f"{conf}%")
                    st.progress(min(1.0, max(0.0, conf / 100.0)))

                st.caption(f"Target Session Evaluated: `📄 {latest_prefix}`")

                with st.expander(
                    "🔍 Expand Full AI Diagnostic Report & Telemetry Metrics"
                ):
                    st.json(ai_results)
        else:
            with st.container(border=True):
                st.info(
                    "💡 **AI Engine Standing By:** Record a session using the"
                    " control deck above to trigger background feature"
                    " extraction and live classification."
                )

    render_ai_diagnostics()

    st.markdown("---")

    # --- 5. DUAL TELEMETRY MONITOR STATIONS ---

    # A. VIBRATION SERIAL MONITOR (Piezo Sensor)
    st.subheader("Vibration Serial Monitor (Piezo Sensor)")

    terminal_static_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { margin:0; background-color:#0E1117; overflow:hidden; font-family:Consolas, 'Courier New', monospace; }
            .terminal-box {
                background-color:#161B22;
                border:1px solid #30363D;
                border-radius:6px;
                height:190px;
                padding:6px 6px;
                box-sizing:border-box;
                overflow-y:auto;
                overflow-x:auto;
                font-size: clamp(9.5px, 2.6vw, 11.8px);
                letter-spacing: -0.35px;
                font-variant-numeric: tabular-nums;
                white-space: nowrap;
                transition: border-color 0.3s;
            }
        </style>
    </head>
    <body>
        <div id="term-box" class="terminal-box">
            <div style="color:#8B949E; padding-top:80px; text-align:center; font-size:12px;">--- SERIAL MONITOR PIPELINE IDLE ---</div>
        </div>
        <script>
            let lastSecBlock = null;
            let baseTimeSec = null;
            let pRate = 2000;
            let tryHttp = true;

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

                if (state === "RECORDING") box.style.borderColor = "#00C853";
                else if (state === "PAUSED") box.style.borderColor = "#CCA000";
                else if (state === "STOPPING") box.style.borderColor = "#00BCD4";
                else box.style.borderColor = "#30363D";

                if (!linesData || linesData.length === 0) {
                    box.innerHTML = '<div style="color:#8B949E; padding-top:80px; text-align:center; font-size:12px;">--- SERIAL MONITOR PIPELINE IDLE ---</div>';
                    return;
                }

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
                        html += `<div style="border-top: 2px dotted #00BCD4; background: rgba(0, 188, 212, 0.12); color: #00BCD4; text-align: center; font-size: 10px; margin: 5px 0; padding: 2px 0; font-weight: bold; letter-spacing: 0.5px;">⏱️ ┈┈┈┈┈ 2s MARKER (+${relSec}s) ┈┈┈┈┈</div>`;
                    }
                    lastSecBlock = secBlock;

                    let col = isAct ? "#00E676" : "#C9D1D9";
                    html += `<div style="color:${col}; font-weight:bold; line-height:1.35;">${compact}</div>`;
                });

                box.innerHTML = html;
                box.scrollTop = box.scrollHeight;
            }

            async function syncTerminal() {
                let linesData = null;
                let state = "STOPPED";

                if (tryHttp) {
                    try {
                        const res = await fetch('http://localhost:7654/piezo_lines', {cache: 'no-store'});
                        if (res.ok) {
                            const data = await res.json();
                            linesData = data.lines;
                            state = data.state;
                        } else {
                            tryHttp = false;
                        }
                    } catch(e) {
                        tryHttp = false;
                    }
                }

                if (!linesData) {
                    try {
                        const raw = localStorage.getItem('swarcare_piezo');
                        if (raw) {
                            const data = JSON.parse(raw);
                            linesData = data.lines;
                            state = data.state;
                        }
                    } catch(e) {}
                }
                renderLines(linesData, state);
            }

            setInterval(syncTerminal, 41);
            syncTerminal();
        </script>
    </body>
    </html>
    """
    st.iframe(terminal_static_html, height=200)
    st.write("")

    # B. AUDIO LIVE MONITOR (USB Microphone - 24 FPS Oscilloscope)
    st.subheader("Audio Live Monitor (USB Microphone)")

    audio_static_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { margin:0; background-color:#0E1117; overflow:hidden; }
            #aC { display:block; background:#161B22; border-radius:6px; border:1px solid #30363D; width:100%; height:200px; }
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
                    canvas.width = containerWidth * dpr;
                    canvas.height = 200 * dpr;
                    ctx.scale(dpr, dpr);
                    return { w: containerWidth, h: 200 };
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

                const targetFPS = 24;
                const frameIntervalMs = 1000 / targetFPS;
                let lastRenderMs = performance.now();

                async function fetchAudioData() {
                    let synced = false;
                    if (tryHttp) {
                        try {
                            const res = await fetch('http://localhost:7654/audio_data', {cache: 'no-store'});
                            if (res.ok) {
                                const data = await res.json();
                                history = data.samples || [];
                                serverTimeSec = data.elapsed_s || 0.0;
                                state = data.state || "STOPPED";
                                lastFetchMs = performance.now();
                                synced = true;
                            } else {
                                tryHttp = false;
                            }
                        } catch(e) {
                            tryHttp = false;
                        }
                    }

                    if (!synced) {
                        try {
                            const raw = localStorage.getItem('swarcare_audio');
                            if (raw) {
                                const data = JSON.parse(raw);
                                history = data.samples || [];
                                serverTimeSec = data.elapsed_s || 0.0;
                                state = data.state || "STOPPED";
                                lastFetchMs = performance.now();
                            }
                        } catch(e) {}
                    }
                }

                function draw() {
                    const dpr = window.devicePixelRatio || 1;
                    const w = canvas.width / dpr;
                    const h = canvas.height / dpr;

                    const nowMs = performance.now();
                    const dt = Math.min(0.1, (nowMs - lastFrameMs) / 1000.0);
                    lastFrameMs = nowMs;

                    if (state === "RECORDING") {
                        let targetTimeSec = serverTimeSec + ((nowMs - lastFetchMs) / 1000.0);
                        if (displayTimeSec === 0 || Math.abs(displayTimeSec - targetTimeSec) > 1.0) {
                            displayTimeSec = targetTimeSec;
                        } else {
                            displayTimeSec += (targetTimeSec - displayTimeSec) * 0.2;
                        }
                    } else {
                        displayTimeSec = serverTimeSec;
                    }

                    const isMobile = w < 480;
                    const padLeft = isMobile ? 48 : 65;
                    const padBottom = isMobile ? 32 : 35;
                    const padTop = 20;
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

                    ctx.setLineDash([4, 4]);
                    ctx.strokeStyle = '#8B949E';
                    ctx.beginPath(); ctx.moveTo(padLeft, midY); ctx.lineTo(w - padRight, midY); ctx.stroke();
                    ctx.setLineDash([]);

                    ctx.fillStyle = '#8B949E';
                    ctx.font = isMobile ? '8.5px monospace' : '10px monospace';
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('+1.0', padLeft - 4, padTop);
                    ctx.fillText('0.0', padLeft - 4, midY);
                    ctx.fillText('-1.0', padLeft - 4, h - padBottom);

                    const N = history.length;
                    if(N > 0) {
                        ctx.fillStyle = '#00E676';
                        const barWidth = isMobile ? 1.5 : 2.0;

                        for (let col = 0; col < plotW; col += barWidth) {
                            let x = padLeft + col;
                            let sampleIdx = Math.floor((col / plotW) * N);
                            if (sampleIdx >= N) sampleIdx = N - 1;

                            let amplitude = Math.abs(history[sampleIdx]);
                            let barHeight = Math.max(2, amplitude * (plotH * 0.85));
                            ctx.fillRect(x, midY - (barHeight / 2), barWidth, barHeight);
                        }
                    } else {
                        ctx.fillStyle = '#8B949E';
                        ctx.font = isMobile ? '11px monospace' : '12px monospace';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText('--- AUDIO MONITOR IDLE ---', padLeft + plotW / 2, midY);
                    }

                    // Smooth 2-second time markers
                    const pixelsPerSec = plotW / visibleWindowSec;
                    const start2sMarker = Math.floor(Math.max(0, displayTimeSec - visibleWindowSec) / 2.0) * 2.0;
                    const end2sMarker = displayTimeSec + 2.0;

                    for(let tMark = start2sMarker; tMark <= end2sMarker; tMark += 2.0) {
                        let markOffset = displayTimeSec - tMark;
                        let markX = (w - padRight) - (markOffset * pixelsPerSec);

                        if (markX >= padLeft && markX <= (w - padRight)) {
                            ctx.save();
                            ctx.setLineDash([3, 3]);
                            ctx.strokeStyle = '#00BCD4';
                            ctx.lineWidth = 1.2;
                            ctx.beginPath();
                            ctx.moveTo(markX, padTop);
                            ctx.lineTo(markX, h - padBottom);
                            ctx.stroke();
                            ctx.restore();

                            ctx.fillStyle = '#00BCD4';
                            ctx.font = isMobile ? 'bold 8.5px monospace' : 'bold 10px monospace';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText(tMark.toFixed(1) + 's', markX, padTop - 2);

                            ctx.fillStyle = '#C9D1D9';
                            ctx.textBaseline = 'top';
                            ctx.fillText(tMark.toFixed(1) + 's', markX, h - padBottom + 4);
                        }
                    }

                    ctx.fillStyle = '#8B949E';
                    ctx.font = isMobile ? 'bold 10px monospace' : 'bold 11px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillText('Time', padLeft + plotW / 2, h - 14);

                    ctx.save();
                    ctx.translate(isMobile ? 12 : 16, padTop + plotH / 2);
                    ctx.rotate(-Math.PI / 2);
                    ctx.fillStyle = '#C9D1D9';
                    ctx.font = isMobile ? 'bold 9.5px monospace' : 'bold 12px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('Normalised Amplitude', 0, 0);
                    ctx.restore();
                }

                setInterval(fetchAudioData, 41);

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
    st.iframe(audio_static_html, height=210)
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

    # --- FEATURE 2: DOWNLOAD ALL AS ZIP ARCHIVE ---
    if csv_files or wav_files:
        zip_bytes = generate_recordings_zip()
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
                        with open(file_path, "rb") as f:
                            file_data = f.read()
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
                                    time.sleep(0.2)
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
                                time.sleep(0.2)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

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
                        with open(file_path, "rb") as f:
                            wav_data = f.read()
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
                                new_path = os.path.join
                                try:
                                    os.rename(
                                        file_path,
                                        os.path.join(DATA_DIR, target_wav_name),
                                    )
                                    st.toast(
                                        f"Renamed to {target_wav_name}!",
                                        icon="✏️",
                                    )
                                    time.sleep(0.2)
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
                                time.sleep(0.2)
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
            time.sleep(0.3)
            st.rerun()