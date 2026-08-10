import os
import time  
import streamlit as st
from engine import SwarCareEngine

import streamlit.components.v1 as components

components.html("""
<script>
fetch('/settime', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({epoch_ms: Date.now()})
}).catch(() => {});
</script>
""", height=0)

# Initialize the SwarCare Core Backend Engine
backend = SwarCareEngine.get_instance()
DATA_DIR = backend.recordings_dir

st.set_page_config(page_title="SwarCare Hub", page_icon="📡", layout="centered")

# Initialize UI view state management registers
if "visible_records_limit" not in st.session_state:
    st.session_state.visible_records_limit = 3

st.title("🎵 SwarCare Anomaly Detection Hub")
st.write("Anomaly Detection Using Vibration & Audio Streams")
st.markdown("---")

# Render Dynamic Operational Session Banners
if backend.state == "STOPPED":
    st.error("System Status: **STOPPED / IDLE** ⏹️")
elif backend.state == "RECORDING":
    st.success("System Status: **RECORDING SENSORS LIVE** ▶️")
elif backend.state == "PAUSED":
    st.warning("System Status: **RECORDING PAUSED** ⏸️")
elif backend.state == "STOPPING":
    st.info("System Status: **WRITING DATA STREAMS GRACEFULLY... PLEASE WAIT** ⏳")

st.write("")  

# Action Control Deck
col1, col2, col3 = st.columns(3)
with col1:
    start_label = "▶ RESUME" if backend.state == "PAUSED" else "▶ START RECORDING"
    if st.button(start_label, use_container_width=True, disabled=(backend.state in ["RECORDING", "STOPPING"])):
        backend.start_recording()
        st.rerun()
with col2:
    if st.button("⏸ PAUSE", use_container_width=True, disabled=(backend.state != "RECORDING")):
        backend.pause_recording()
        st.rerun()
with col3:
    if st.button("⏹ STOP & SAVE", use_container_width=True, disabled=(backend.state not in ["RECORDING", "PAUSED"])):
        backend.stop_recording()
        st.rerun()

st.divider()

# --- DUAL TELEMETRY MONITOR FRAGMENT ---
@st.fragment()
def render_hybrid_telemetry_station():
    if backend.state == "STOPPING":
        with st.spinner("Executing thread cleanup and matching wave structures precisely..."):
            while backend.state == "STOPPING":
                time.sleep(0.02)
        st.rerun()

    # --- 1. ARDUINO IDE MONOSPACE SCROLLING TERMINAL MONITOR ---
    st.subheader("Vibration Serial Monitor (Piezo Sensor)")
    terminal_snapshots = backend.get_terminal_lines_snapshot()
    border_color = "#00C853" if backend.state == "RECORDING" else ("#CCA000" if backend.state == "PAUSED" else "#30363D")

    terminal_html_lines = ""
    if len(terminal_snapshots) == 0:
        terminal_html_lines = '<div style="color:#8B949E; padding-top:80px; text-align:center;">--- SERIAL MONITOR PIPELINE IDLE ---</div>'
    else:
        for txt_line, is_active in terminal_snapshots:
            color = "#00E676" if is_active else "#C9D1D9"
            terminal_html_lines += f'<div style="color:{color}; font-weight:bold; line-height:1.5;">{txt_line}</div>'

    terminal_window_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin:0; background-color:#0E1117; overflow:hidden; font-family:Consolas, Monaco, 'Courier New', monospace; }}
            .terminal-box {{ background-color:#161B22; border:1px solid {border_color}; border-radius:6px; height:200px; padding:12px; box-sizing:border-box; overflow-y:auto; font-size:16px; }}
        </style>
    </head>
    <body>
        <div class="terminal-box">{terminal_html_lines}</div>
    </body>
    </html>
    """
    st.iframe(terminal_window_code, height=210)
    st.write("")

    # --- 2. VOICE RECORDER PHONE-STYLE ROLLING VISUALIZER ---
    st.subheader("Audio Live Monitor (USB Microphone) ")
    
    a_snap = backend.get_audio_buffer_snapshot()
    a_csv = ",".join([f"{x:.4f}" for x in a_snap])
    is_recording_flag = "true" if backend.state == "RECORDING" else "false"

    a_html = f"""
    <!DOCTYPE html>
    <html>
    <head><style>body {{ margin:0; background-color:#0E1117; overflow:hidden; }}</style></head>
    <body>
        <canvas id="aC" style="display:block; background:#161B22; border-radius:6px; border:1px solid #30363D; width:100%; height:200px;"></canvas>
        <script>
            (function() {{
                const canvas = document.getElementById('aC');
                const ctx = canvas.getContext('2d');
                
                const dpr = window.devicePixelRatio || 1;
                canvas.width = (window.innerWidth || 700) * dpr;
                canvas.height = 200 * dpr;
                ctx.scale(dpr, dpr);
                
                const w = canvas.width / dpr;
                const h = canvas.height / dpr;
                
                const padLeft = 55;    
                const padBottom = 35;  
                const padTop = 15;
                const padRight = 15;
                
                const plotW = w - padLeft - padRight;
                const plotH = h - padTop - padBottom;
                const midY = padTop + (plotH / 2);

                const isRecording = {is_recording_flag};

                let history = JSON.parse(localStorage.getItem('sc_wave_history') || '[]');
                if(!isRecording && history.length > 0) {{
                    history = []; 
                }}

                const inbound = [{a_csv}];
                if (isRecording && inbound.length > 0) {{
                    for(let i = 0; i < inbound.length; i++) {{
                        history.push(inbound[i]);
                    }}
                }}

                const maxBars = Math.floor(plotW / 4); 
                if (history.length > maxBars) {{
                    history.splice(0, history.length - maxBars);
                }}
                localStorage.setItem('sc_wave_history', JSON.stringify(history));

                function draw() {{
                    ctx.clearRect(0, 0, w, h);
                    
                    ctx.strokeStyle = '#21262D';
                    ctx.lineWidth = 1;
                    for(let i = 0; i <= 4; i++) {{
                        let y = padTop + (plotH / 4) * i;
                        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(w - padRight, y); ctx.stroke();
                    }}
                    for(let i = 0; i <= 5; i++) {{
                        let x = padLeft + (plotW / 5) * i;
                        ctx.beginPath(); ctx.moveTo(x, padTop); ctx.lineTo(x, h - padBottom); ctx.stroke();
                    }}

                    ctx.strokeStyle = '#30363D';
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(padLeft, padTop, plotW, plotH);
                    
                    ctx.setLineDash([4, 4]);
                    ctx.strokeStyle = '#8B949E';
                    ctx.beginPath(); ctx.moveTo(padLeft, midY); ctx.lineTo(w - padRight, midY); ctx.stroke();
                    ctx.setLineDash([]); 

                    ctx.fillStyle = '#8B949E';
                    ctx.font = '10px monospace';
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('+1.0', padLeft - 6, padTop);
                    ctx.fillText('0.0', padLeft - 6, midY);
                    ctx.fillText('-1.0', padLeft - 6, h - padBottom);

                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    for(let i = 0; i <= 5; i++) {{
                        let x = padLeft + (plotW / 5) * i;
                        let elapsedSec = 0.2 * i;
                        ctx.fillText(elapsedSec.toFixed(1) + 's', x, h - padBottom + 6);
                    }}

                    // X-AXIS TITLE
                    ctx.fillStyle = '#C9D1D9';
                    ctx.font = 'bold 11px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillText('Time (s)', padLeft + plotW / 2, h - 12);

                    // Y-AXIS TITLE (rotated)
                    ctx.save();
                    ctx.translate(14, padTop + plotH / 2);
                    ctx.rotate(-Math.PI / 2);
                    ctx.fillStyle = '#C9D1D9';
                    ctx.font = 'bold 13px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('Normalised Amplitude', 0, 0);
                    ctx.restore();

                    if(history.length > 0) {{
                        ctx.fillStyle = '#00E676';
                        const barWidth = 2;
                        const gap = 2;
                        let startX = w - padRight - barWidth; 
                        
                        for(let i = history.length - 1; i >= 0; i--) {{
                            let amplitude = Math.abs(history[i]);
                            let barHeight = Math.max(2, amplitude * (plotH * 0.9));
                            let x = startX - (history.length - 1 - i) * (barWidth + gap);
                            
                            if (x < padLeft) break; 
                            
                            ctx.fillRect(x, midY - (barHeight / 2), barWidth, barHeight);
                        }}
                    }}
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

    # --- SYNCHRONIZED STORAGE METRICS PANEL ---
    p_samples = backend.piezo_samples_recorded
    p_lines = p_samples + 1 if p_samples > 0 else 0
    p_time = p_samples / backend.PIEZO_SAMPLE_RATE_HZ

    a_samples = backend.audio_samples_recorded
    a_time = a_samples / backend.AUDIO_SAMPLE_RATE_HZ

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Vibration Samples Logged", f"{p_samples:,}")
        st.caption(f"📝 Piezo CSV Lines: **{p_lines:,}**")
        st.caption(f"⏱️ Piezo Duration: **{p_time:.2f} s**")
    with c2:
        st.metric("Audio Samples Logged", f"{a_samples:,}")
        st.caption(f"🔊 Audio WAV Samples: **{a_samples:,}**")
        st.caption(f"⏱️ Audio Duration: **{a_time:.2f} s**")

    if backend.state == "RECORDING":
        time.sleep(0.04)  
        st.rerun()

render_hybrid_telemetry_station()

st.markdown("---")

# --- 📁 DYNAMIC SAVED RECORDS PANEL (SIDE-BY-SIDE SPLIT) ---
st.subheader("📁 Saved Records")

if os.path.exists(DATA_DIR):
    raw_files = sorted(os.listdir(DATA_DIR), reverse=True)
    csv_files = [f for f in raw_files if f.endswith('.csv')]
    wav_files = [f for f in raw_files if f.endswith('.wav')]
else:
    csv_files, wav_files = [], []

limit = st.session_state.visible_records_limit

# Side-by-Side Headings Layout
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
                    if st.button("🗑️", key=f"del_{file}", use_container_width=True):
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
                            key=f"dl_{file}",
                            use_container_width=True
                        )
                with btn_col2:
                    if st.button("🗑️", key=f"del_{file}", use_container_width=True):
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
    if st.button("💥 Clear All", use_container_width=True):
        if os.path.exists(DATA_DIR):
            for file_to_wipe in os.listdir(DATA_DIR):
                try:
                    os.remove(os.path.join(DATA_DIR, file_to_wipe))
                except Exception:
                    pass
        st.session_state.visible_records_limit = 3
        st.toast("Storage directory wiped clean! Counter reset to 001.", icon="💥")
        time.sleep(0.3)
        st.rerun()