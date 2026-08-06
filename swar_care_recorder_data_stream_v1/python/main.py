import time  
import streamlit as st
                    
from engine import SwarCareEngine

backend = SwarCareEngine.get_instance()

st.set_page_config(page_title="SwarCare Hub", page_icon="📡", layout="centered")

st.title("📡 SwarCare Hybrid Station")
st.write("Unified Synchronized Workspace for Concurrent Vibration & Audio Streams")
st.markdown("---")

# Intercept interface compilation passes while background disk operations run
is_writing = backend.disk_write_lock.locked()

# Render Dynamic Operational Session Banners
if is_writing:
    st.info("System Status: **FLUSHING COMPACTED VOLATILE MEMORY BLOCKS SAFELY TO STORAGE...** ⏳")
elif backend.state == "STOPPED":
    st.error("System Status: **STOPPED / IDLE** ⏹️")
elif backend.state == "RECORDING":
    st.success("System Status: **RECORDING SENSORS LIVE** ▶️")
elif backend.state == "PAUSED":
    st.warning("System Status: **RECORDING PAUSED** ⏸️")
elif backend.state == "STOPPING":
    st.info("System Status: **SEALING STORAGE PLATES...** ⏳")

st.write("")  

# Action Control Deck
col1, col2, col3 = st.columns(3)
with col1:
    start_label = "▶ RESUME" if backend.state == "PAUSED" else "▶ START RECORDING"
    if st.button(start_label, use_container_width=True, disabled=(backend.state in ["RECORDING", "STOPPING"] or is_writing)):
        backend.start_recording()
        st.rerun()
with col2:
    if st.button("⏸ PAUSE", use_container_width=True, disabled=(backend.state != "RECORDING" or is_writing)):
        backend.pause_recording()
        st.rerun()
with col3:
    if st.button("⏹ STOP & SAVE", use_container_width=True, disabled=(backend.state not in ["RECORDING", "PAUSED"] or is_writing)):
        backend.stop_recording()
        st.rerun()

st.divider()

# --- DUAL TELEMETRY MONITOR FRAGMENT ---
@st.fragment()
def render_hybrid_telemetry_station():
    
    # --- 1. ARDUINO IDE MONOSPACE SCROLLING TERMINAL MONITOR ---
    st.subheader("Vibration Serial Monitor (Piezo Ingestion)")
    
    terminal_snapshots = backend.get_terminal_lines_snapshot()
    
    if backend.state == "RECORDING":
        border_color = "#00C853"
    elif backend.state == "PAUSED":
        border_color = "#CCA000"
    else:
        border_color = "#30363D"

    terminal_html_lines = ""
    if len(terminal_snapshots) == 0:
        terminal_html_lines = '<div style="color:#8B949E; padding-top:80px; text-align:center;">--- SERIAL MONITOR PIPELINE IDLE ---</div>'
    else:
        for txt_line, is_active in terminal_snapshots:
            if is_active:
                terminal_html_lines += f'<div style="color:#00E676; font-weight:bold; line-height:1.4;">{txt_line}</div>'
            else:
                terminal_html_lines += f'<div style="color:#C9D1D9; line-height:1.4;">{txt_line}</div>'

    terminal_window_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin:0; background-color:#0E1117; overflow:hidden; font-family:Consolas, Monaco, 'Courier New', monospace; }}
            .terminal-box {{
                background-color:#161B22; 
                border:1px solid {border_color}; 
                border-radius:6px; 
                height:200px; 
                padding:10px; 
                box-sizing:border-box; 
                overflow-y:hidden; 
                font-size:12.5px;
            }}
        </style>
    </head>
    <body>
        <div class="terminal-box">
            {terminal_html_lines}
        </div>
    </body>
    </html>
    """
    st.iframe(terminal_window_code, height=210)
    st.write("")

    # --- 2. MICROPHONE CANVAS TELEMETRY MONITOR ---
    st.subheader("Audio (USB Microphone) Monitor")
    a_snap = backend.get_audio_buffer_snapshot()
    a_csv = ",".join([f"{x:.4f}" for x in a_snap])

    a_html = f"""
    <!DOCTYPE html>
    <html>
    <head><style>body {{ margin:0; background-color:#0E1117; overflow:hidden; }}</style></head>
    <body>
        <canvas id="aC" style="display:block; background:#161B22; border-radius:6px; border:1px solid #30363D; width:100%; height:200px;"></canvas>
        <script>
            (function() {{
                const c = document.getElementById('aC'); const ctx = c.getContext('2d');
                c.width = window.innerWidth || 700; c.height = 200;
                const d = [{a_csv}];
                ctx.strokeStyle = '#21262D'; ctx.lineWidth = 1;
                for(let i=1; i<5; i++) {{ let y=(c.height/5)*i; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(c.width,y); ctx.stroke(); }}
                if(d.length > 0) {{
                    ctx.strokeStyle = '#00E676'; ctx.lineWidth = 2; ctx.beginPath();
                    for(let i=0; i<d.length; i++) {{
                        let x = (c.width / (d.length - 1)) * i; let y = (c.height / 2) - (d[i] * (c.height / 2.4));
                        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
                    }}
                    ctx.stroke();
                }}
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
    a_lines = a_samples + 1 if a_samples > 0 else 0
    a_time = a_samples / backend.AUDIO_SAMPLE_RATE_HZ

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Vibration Samples Logged", f"{p_samples:,}")
        st.caption(f"📝 Piezo Lines: **{p_lines:,}**")
        st.caption(f"⏱️ Piezo Duration: **{p_time:.2f} s**")
    with c2:
        st.metric("Audio Samples Logged", f"{a_samples:,}")
        st.caption(f"📝 Audio Lines: **{a_lines:,}**")
        st.caption(f"⏱️ Audio Duration: **{a_time:.2f} s**")

    # Force immediate layout refreshes if an active recording session or a background write is active
    if backend.state == "RECORDING" or backend.disk_write_lock.locked():
        time.sleep(0.08) 
        st.rerun()

render_hybrid_telemetry_station()