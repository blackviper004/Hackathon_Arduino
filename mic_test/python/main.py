import time  
import streamlit as st
from engine import SwarCareEngine

# Access the global data acquisition core singleton instance
backend = SwarCareEngine.get_instance()

# --- STREAMLIT UI PAGE STRUCTURE INITIALIZATION ---
st.set_page_config(page_title="SwarCare Hub", page_icon="📡", layout="centered")

st.title("📡 SwarCare Hybrid Station")
st.write("Clean Audio Recording Station with Non-Blocking Instant Data Architecture")
st.markdown("---")

# Render Dynamic Hardware State Banners
if backend.state == "STOPPED":
    st.error("System Status: **STOPPED** ⏹️")
elif backend.state == "RECORDING":
    st.success("System Status: **RECORDING RUN LIVE** ▶️")
elif backend.state == "PAUSED":
    st.warning("System Status: **STREAM BUFFER PAUSED** ⚖️")

st.write("")  

# Action Button Layout Interface Rows
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶ START / RESUME", use_container_width=True, disabled=(backend.state == "RECORDING")):
        backend.start_recording()
        st.rerun()
with col2:
    if st.button("⏸ PAUSE", use_container_width=True, disabled=(backend.state != "RECORDING")):
        backend.state = "PAUSED"
        st.rerun()
with col3:
    if st.button("⏹ STOP & SAVE", use_container_width=True, disabled=(backend.state == "STOPPED")):
        backend.stop_recording()
        st.rerun()

st.divider()

# --- TELEMETRY GRAPHING PLATFORM ---
st.subheader("Live Telemetry & Performance Tracking")

@st.fragment()
def render_live_telemetry_station():
    buffer_snapshot = backend.get_live_buffer_snapshot()
    live_samples_csv = ",".join([f"{x:.4f}" for x in buffer_snapshot])

    # Clean UI component graphics string 
    canvas_html_brick = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; background-color: #0E1117; overflow: hidden; font-family: sans-serif; }}
            canvas {{ display: block; background: #161B22; border-radius: 6px; border: 1px solid #30363D; width: 100%; height: 250px; }}
        </style>
    </head>
    <body>
        <canvas id="liveWaveformCanvas"></canvas>
        <script>
            (function() {{
                const canvas = document.getElementById('liveWaveformCanvas');
                const ctx = canvas.getContext('2d');
                
                canvas.width = window.innerWidth || 700;
                canvas.height = 250;
                
                const rawData = [{live_samples_csv}];
                
                // Draw background grid lines
                ctx.strokeStyle = '#21262D';
                ctx.lineWidth = 1;
                for(let i = 1; i < 5; i++) {{
                    let y = (canvas.height / 5) * i;
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
                }}
                
                // Draw clean microphone live trace mapping line using green (#00E676)
                if (rawData.length > 0) {{
                    ctx.strokeStyle = '#00E676';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    for(let i = 0; i < rawData.length; i++) {{
                        let x = (canvas.width / (rawData.length - 1)) * i;
                        let y = (canvas.height / 2) - (rawData[i] * (canvas.height / 2.4));
                        
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }}
                    ctx.stroke();
                }}
            }})();
        </script>
    </body>
    </html>
    """
    
    # Modern native st.iframe implementation embeds raw HTML string directly
    st.iframe(canvas_html_brick, height=260)

    # --- TEXT METRICS DATA LAYOUT PANEL ---
    c1, c2 = st.columns(2)
    total_samples = backend.samples_recorded
    csv_file_lines = total_samples + 1 if total_samples > 0 else 0
    elapsed_seconds = total_samples / backend.SAMPLE_RATE_HZ

    c1.metric("Raw Samples Tracked", f"{total_samples:,}")
    c1.caption(f"📝 Total lines written to disk CSV file: **{csv_file_lines:,}**")
    c2.metric("Continuous Timeline Tracked", f"{elapsed_seconds:.2f} s")

    # Refresh pacing matching embedded framework performance requirements
    if backend.state == "RECORDING":
        time.sleep(0.08)
        st.rerun()

render_live_telemetry_station()