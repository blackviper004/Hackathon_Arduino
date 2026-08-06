import os
import re
import time  
import streamlit as st
from engine import SwarCareEngine
import streamlit.components.v1 as components
from streamlit.errors import StreamlitAPIException

# Sync time context with backend engine
components.html("""
<script>
fetch('/settime', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({epoch_ms: Date.now()})
}).catch(() => {});
</script>
""", height=0)

# Initialize SwarCare Core Engine
backend = SwarCareEngine.get_instance()
DATA_DIR = backend.recordings_dir

st.set_page_config(page_title="SwarCare Hub", page_icon="📡", layout="centered")

# Inject Mobile Viewport & Touch Responsive CSS Rules
st.markdown("""
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
""", unsafe_allow_html=True)

# Session state initialization
if "visible_records_limit" not in st.session_state:
    st.session_state.visible_records_limit = 3

st.title("🎵 SwarCare Anomaly Detection Hub")
st.write("Real-Time Vibration & Audio Stream Intelligence")
st.markdown("---")

# Render Session Banners
if backend.state == "STOPPED":
    st.error("System Status: **STOPPED / IDLE** ⏹️")
elif backend.state == "RECORDING":
    st.success("System Status: **RECORDING SENSORS LIVE** ▶️")
elif backend.state == "PAUSED":
    st.warning("System Status: **RECORDING PAUSED** ⏸️")
elif backend.state == "STOPPING":
    st.info("System Status: **WRITING DATA STREAMS GRACEFULLY... PLEASE WAIT** ⏳")

st.write("")  

# ==========================================
# 📑 CONSOLIDATED 2-TAB INTERFACE
# ==========================================
tab_live_ai, tab_records = st.tabs([
    "📡 Live Telemetry & AI Diagnostics", 
    "📁 Saved Records Explorer"
])

# ==========================================
# TAB 1: LIVE CONTROL, AI & TELEMETRY
# ==========================================
with tab_live_ai:
    # --- 1. HARDWARE CONTROL DECK ---
    st.subheader("⚙️ Hardware Control Deck")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_label = "▶ RESUME" if backend.state == "PAUSED" else "▶ START RECORDING"
        if st.button(start_label, use_container_width=True, disabled=(backend.state in ["RECORDING", "STOPPING"]), help="Start capturing telemetry"):
            backend.start_recording()
            st.rerun()
    with col2:
        if st.button("⏸ PAUSE", use_container_width=True, disabled=(backend.state != "RECORDING"), help="Pause live telemetry capture"):
            backend.pause_recording()
            st.rerun()
    with col3:
        if st.button("⏹ STOP & SAVE", use_container_width=True, disabled=(backend.state not in ["RECORDING", "PAUSED"]), help="Finalize and save streams"):
            backend.stop_recording()
            st.rerun()

    st.markdown("---")

    # --- 2. AI ANOMALY MONITOR DECK ---
    st.subheader("🧠 Real-Time AI Diagnostic & Anomaly Center")

    if os.path.exists(DATA_DIR):
        valid_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_piezo.csv") or f.endswith("_audio.wav")]
        all_recordings = sorted([f.replace("_piezo.csv", "").replace("_audio.wav", "") for f in valid_files], reverse=True)
        unique_prefixes = sorted(list(set(all_recordings)), reverse=True)
    else:
        unique_prefixes = []

    if unique_prefixes:
        latest_prefix = unique_prefixes[0]
        ai_results = backend.analyze_recording_ai(latest_prefix)
        
        with st.container(border=True):
            col_ai1, col_ai2, col_ai3 = st.columns(3)
            
            status_color_text = ai_results.get("status", "UNKNOWN")
            with col_ai1:
                st.markdown(f"**Diagnostic Status**\n### {status_color_text}")
            with col_ai2:
                score = ai_results.get('score', 0.0)
                st.metric("Anomaly Score (0-1)", f"{score:.3f}")
                st.progress(min(1.0, max(0.0, score)))
            with col_ai3:
                conf = ai_results.get('confidence', 0)
                st.metric("Confidence Level", f"{conf}%")
                st.progress(min(1.0, max(0.0, conf / 100.0)))
                
            st.caption(f"Target Session Evaluated: `📄 {latest_prefix}`")
            
            with st.expander("🔍 Expand Full AI Diagnostic Report & Telemetry Metrics"):
                st.json(ai_results)
    else:
        with st.container(border=True):
            st.info("💡 **AI Engine Standing By:** Record a session using the control deck above to trigger background feature extraction and live classification.")

    st.markdown("---")

    # --- 3. DUAL TELEMETRY MONITOR FRAGMENT ---
    @st.fragment()
    def render_hybrid_telemetry_station():
        if backend.state == "STOPPING":
            with st.spinner("Executing thread cleanup and matching wave structures precisely..."):
                while backend.state == "STOPPING":
                    time.sleep(0.02)
            st.rerun()

        # Active Status Indicator
        p_rate = getattr(backend, 'PIEZO_SAMPLE_RATE_HZ', 100) or 100
        p_samples_live = backend.piezo_samples_recorded
        current_duration = p_samples_live / float(p_rate)
        current_2s_block = int(current_duration // 2) * 2

        if backend.state == "RECORDING":
            st.markdown(f"""
            <div style="background-color: #161B22; border: 1px solid #00BCD4; border-radius: 6px; padding: 8px 10px; text-align: center; margin-bottom: 12px;">
                <span style="color: #00BCD4; font-weight: bold; font-family: monospace; font-size: 13px; letter-spacing: 0.5px;">
                    ⏱️ ACTIVE CAPTURE: ~{current_2s_block}s to {current_2s_block + 2}s REACHED
                </span>
            </div>
            """, unsafe_allow_html=True)

        # --- A. VIBRATION SERIAL MONITOR ---
        st.subheader("Vibration Serial Monitor (Piezo Sensor)")
        terminal_snapshots = backend.get_terminal_lines_snapshot()
        border_color = "#00C853" if backend.state == "RECORDING" else ("#CCA000" if backend.state == "PAUSED" else "#30363D")

        terminal_html_lines = ""
        if len(terminal_snapshots) == 0:
            terminal_html_lines = '<div style="color:#8B949E; padding-top:80px; text-align:center; font-size:12px;">--- SERIAL MONITOR PIPELINE IDLE ---</div>'
        else:
            last_sec_block = None
            base_time_sec = None

            for idx, (txt_line, is_active) in enumerate(terminal_snapshots):
                # 1. Compact spacing around colons and pipes
                compact_line = re.sub(r'\s*:\s*', ':', txt_line)
                compact_line = re.sub(r'\s*\|\s*', '|', compact_line)

                # 2. Extract seconds accurately (handles HH:MM:SS.mmm, MM:SS.mmm, and SS.mmm)
                sec_val = None
                
                # Match HH:MM:SS.mmm (e.g. TIME:22:40:00.889)
                hhmmss = re.search(r'(?:TIME|T)\s*:\s*(\d{1,2}):(\d{2}):([\d.]+)', txt_line, re.IGNORECASE)
                if hhmmss:
                    sec_val = float(hhmmss.group(1)) * 3600 + float(hhmmss.group(2)) * 60 + float(hhmmss.group(3))
                else:
                    # Match MM:SS.mmm
                    mmss = re.search(r'(?:TIME|T)\s*:\s*(\d{1,2}):([\d.]+)', txt_line, re.IGNORECASE)
                    if mmss:
                        sec_val = float(mmss.group(1)) * 60 + float(mmss.group(2))
                    else:
                        # Match relative numeric TIME: 1.25 or 1.25s
                        rel_t = re.search(r'(?:TIME|T|SEC|SECS)\s*:\s*([\d.]+)', txt_line, re.IGNORECASE)
                        if rel_t:
                            try:
                                sec_val = float(rel_t.group(1))
                            except ValueError:
                                sec_val = None

                # Fallback: estimate time from sample index if line has no time header
                if sec_val is None:
                    sec_val = idx / float(max(1, p_rate))

                if base_time_sec is None:
                    base_time_sec = sec_val

                # 2-second block calculation
                sec_block = int(sec_val // 2) * 2

                # 3. Inject Cyan Dotted 2s Marker Line directly into the stream
                if last_sec_block is not None and sec_block != last_sec_block:
                    rel_sec = int((sec_val - base_time_sec) // 2) * 2
                    terminal_html_lines += (
                        f'<div style="border-top: 2px dotted #00BCD4; background: rgba(0, 188, 212, 0.12); '
                        f'color: #00BCD4; text-align: center; font-size: 10px; margin: 5px 0; padding: 2px 0; '
                        f'font-weight: bold; letter-spacing: 0.5px;">'
                        f'⏱️ ┈┈┈┈┈ 2s MARKER (+{rel_sec}s) ┈┈┈┈┈</div>'
                    )

                last_sec_block = sec_block

                color = "#00E676" if is_active else "#C9D1D9"
                terminal_html_lines += f'<div style="color:{color}; font-weight:bold; line-height:1.35;">{compact_line}</div>'

        terminal_window_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ margin:0; background-color:#0E1117; overflow:hidden; font-family:Consolas, 'Courier New', monospace; }}
                .terminal-box {{ 
                    background-color:#161B22; 
                    border:1px solid {border_color}; 
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
                }}
            </style>
        </head>
        <body>
            <div id="term-box" class="terminal-box">{terminal_html_lines}</div>
            <script>
                const box = document.getElementById('term-box');
                if (box) {{ box.scrollTop = box.scrollHeight; }}
            </script>
        </body>
        </html>
        """
        st.iframe(terminal_window_code, height=200)
        st.write("")

        # --- B. AUDIO LIVE MONITOR ---
        st.subheader("Audio Live Monitor (USB Microphone)")
        
        a_snap = backend.get_audio_buffer_snapshot()
        a_csv = ",".join([f"{x:.4f}" for x in a_snap])
        is_recording_flag = "true" if backend.state == "RECORDING" else "false"

        a_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ margin:0; background-color:#0E1117; overflow:hidden; }}
                #aC {{ display:block; background:#161B22; border-radius:6px; border:1px solid #30363D; width:100%; height:200px; }}
            </style>
        </head>
        <body>
            <canvas id="aC"></canvas>
            <script>
                (function() {{
                    const canvas = document.getElementById('aC');
                    const ctx = canvas.getContext('2d');
                    
                    function updateCanvasDimensions() {{
                        const dpr = window.devicePixelRatio || 1;
                        const containerWidth = window.innerWidth || 360;
                        canvas.width = containerWidth * dpr;
                        canvas.height = 200 * dpr;
                        ctx.scale(dpr, dpr);
                        return {{ w: containerWidth, h: 200 }};
                    }}

                    let dims = updateCanvasDimensions();
                    window.addEventListener('resize', () => {{ dims = updateCanvasDimensions(); }});

                    const isRecording = {is_recording_flag};

                    let history = JSON.parse(localStorage.getItem('sc_wave_history') || '[]');
                    let totalCaptureCount = parseInt(localStorage.getItem('sc_total_points') || '0');
                    
                    if(!isRecording && history.length > 0) {{
                        history = []; 
                        totalCaptureCount = 0;
                        localStorage.setItem('sc_total_points', '0');
                    }}

                    const inbound = [{a_csv}];
                    if (isRecording && inbound.length > 0) {{
                        for(let i = 0; i < inbound.length; i++) {{
                            history.push(inbound[i]);
                            totalCaptureCount++;
                        }}
                        localStorage.setItem('sc_total_points', totalCaptureCount);
                    }}

                    const maxHistorySpan = 3000; 
                    if (history.length > maxHistorySpan) {{
                        history.splice(0, history.length - maxHistorySpan);
                    }}
                    localStorage.setItem('sc_wave_history', JSON.stringify(history));

                    const sampleRate = 500.0;
                    
                    function draw() {{
                        const dpr = window.devicePixelRatio || 1;
                        const w = canvas.width / dpr;
                        const h = canvas.height / dpr;

                        const isMobile = w < 480;
                        const padLeft = isMobile ? 48 : 65;    
                        const padBottom = isMobile ? 32 : 35;  
                        const padTop = 20;
                        const padRight = isMobile ? 12 : 20;
                        
                        const plotW = Math.max(10, w - padLeft - padRight);
                        const plotH = Math.max(10, h - padTop - padBottom);
                        const midY = padTop + (plotH / 2);

                        const currentTimeSec = totalCaptureCount / sampleRate;
                        const visibleTimeWindowSec = 4.0;
                        const pixelsPerSec = plotW / visibleTimeWindowSec;

                        ctx.clearRect(0, 0, w, h);
                        
                        ctx.strokeStyle = '#21262D';
                        ctx.lineWidth = 1;
                        for(let i = 0; i <= 4; i++) {{
                            let y = padTop + (plotH / 4) * i;
                            ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(w - padRight, y); ctx.stroke();
                        }}

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

                        if(history.length > 0) {{
                            ctx.fillStyle = '#00E676';
                            const barWidth = isMobile ? 1.5 : 2;
                            const rightmostX = w - padRight;

                            for(let i = history.length - 1; i >= 0; i--) {{
                                let sampleTime = (totalCaptureCount - (history.length - 1 - i)) / sampleRate;
                                let timeOffset = currentTimeSec - sampleTime;
                                let x = rightmostX - (timeOffset * pixelsPerSec);

                                if (x < padLeft) break; 

                                let amplitude = Math.abs(history[i]);
                                let barHeight = Math.max(2, amplitude * (plotH * 0.85));
                                ctx.fillRect(x - (barWidth / 2), midY - (barHeight / 2), barWidth, barHeight);
                            }}
                        }}

                        const start2sMarker = Math.floor(Math.max(0, currentTimeSec - visibleTimeWindowSec) / 2.0) * 2.0;
                        const end2sMarker = currentTimeSec + 2.0;

                        for(let tMark = start2sMarker; tMark <= end2sMarker; tMark += 2.0) {{
                            let markOffset = currentTimeSec - tMark;
                            let markX = (w - padRight) - (markOffset * pixelsPerSec);

                            if (markX >= padLeft && markX <= (w - padRight)) {{
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
                            }}
                        }}

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

                        requestAnimationFrame(draw);
                    }}
                    draw();
                }})();
            </script>
        </body>
        </html>
        """
        st.iframe(a_html, height=210)
        st.markdown("---")

        # --- C. SYNCHRONIZED STORAGE METRICS PANEL ---
        a_rate = getattr(backend, 'AUDIO_SAMPLE_RATE_HZ', 44100) or 44100
        p_samples = backend.piezo_samples_recorded
        p_lines = p_samples + 1 if p_samples > 0 else 0
        p_time = p_samples / float(p_rate)

        a_samples = backend.audio_samples_recorded
        a_time = a_samples / float(a_rate)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Vibration Samples", f"{p_samples:,}")
            st.caption(f"📝 Piezo Lines: **{p_lines:,}**")
            st.caption(f"⏱️ Piezo Duration: **{p_time:.2f} s**")
        with c2:
            st.metric("Audio Samples", f"{a_samples:,}")
            st.caption(f"🔊 Audio WAV: **{a_samples:,}**")
            st.caption(f"⏱️ Audio Duration: **{a_time:.2f} s**")

        if backend.state == "RECORDING":
            time.sleep(0.04)  
            try:
                st.rerun(scope="fragment")
            except Exception:
                st.rerun()

    render_hybrid_telemetry_station()

# ==========================================
# TAB 2: SAVED RECORDS EXPLORER
# ==========================================
with tab_records:
    st.subheader("📁 Saved Records Explorer")

    if os.path.exists(DATA_DIR):
        raw_files = sorted(os.listdir(DATA_DIR), reverse=True)
        csv_files = [f for f in raw_files if f.endswith('.csv')]
        wav_files = [f for f in raw_files if f.endswith('.wav')]
    else:
        csv_files, wav_files = [], []

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
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="📥 CSV",
                                data=f,
                                file_name=file,
                                mime="text/csv",
                                key=f"dl_{file}",
                                use_container_width=True
                            )
                    with btn_col2:
                        if st.button("🗑️", key=f"del_{file}", use_container_width=True, help="Delete file"):
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
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="📥 WAV",
                                data=f,
                                file_name=file,
                                mime="audio/wav",
                                key=f"dl_wav_{file}",
                                use_container_width=True
                            )
                    with btn_col2:
                        if st.button("🗑️", key=f"del_wav_{file}", use_container_width=True, help="Delete file"):
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
            st.button("✨ All Displayed", disabled=True, use_container_width=True)

    with ctrl_col2:
        if limit > 3:
            if st.button("🔄 Reset View", use_container_width=True):
                st.session_state.visible_records_limit = 3
                st.rerun()
        else:
            st.button("🔄 Reset View", disabled=True, use_container_width=True)

    with ctrl_col3:
        if st.button("💥 Clear All", use_container_width=True, help="Wipe all recordings"):
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