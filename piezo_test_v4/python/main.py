# ==============================================================================
# SWARCARE HUB: STREAMLIT APP ENGINE & LIFECYCLE ROUTER (main.py)
# ==============================================================================
import signal
signal.signal = lambda signum, handler: None 

import subprocess
import atexit
import time  
import streamlit as st
import streamlit.components.v1 as components

from engine import SwarCareEngine, ON_DEVICE

# ==============================================================================
# 1. CRASH-PROOF AUTOMATED HOTSPOT ENGINE
# ==============================================================================
def find_wifi_interface():
    cmd = ["nmcli", "-t", "-f", "DEVICE,TYPE", "device"]
    try:
        try:
            res = subprocess.check_output(["sudo"] + cmd, text=True)
        except FileNotFoundError:
            res = subprocess.check_output(cmd, text=True)
            
        for line in res.strip().split("\n"):
            parts = line.split(":")
            if len(parts) == 2 and parts[1] == "wifi":
                return parts[0]
    except (FileNotFoundError, Exception):
        pass
    return "wlan0"

def setup_runtime_hotspot():
    if not ON_DEVICE:
        return
        
    print("\n[SwarCare Hotspot Initialization Sequence Started]")
    try: 
        try:
            subprocess.run(["sudo", "rfkill", "unblock", "wifi"], check=False)
        except FileNotFoundError:
            subprocess.run(["rfkill", "unblock", "wifi"], check=False)
    except FileNotFoundError:
        pass
        
    try: 
        try:
            subprocess.run(["sudo", "nmcli", "radio", "wifi", "on"], check=False)
        except FileNotFoundError:
            subprocess.run(["nmcli", "radio", "wifi", "on"], check=False)
    except FileNotFoundError:
        return  
        
    try:
        try:
            res = subprocess.run(["sudo", "nmcli", "connection", "up", "Swar_Care"], capture_output=True, text=True)
            is_fallback = (res.returncode != 0)
        except FileNotFoundError:
            res = subprocess.run(["nmcli", "connection", "up", "Swar_Care"], capture_output=True, text=True)
            is_fallback = (res.returncode != 0)
        
        if is_fallback:
            wifi_iface = find_wifi_interface()
            spawn_cmd = [
                "nmcli", "device", "wifi", "hotspot", 
                "ifname", wifi_iface, 
                "ssid", "Swar_Care",
                "password", "12345678"
            ]
            try:
                subprocess.run(["sudo"] + spawn_cmd, check=False)
            except FileNotFoundError:
                subprocess.run(spawn_cmd, check=False)
    except Exception:
        pass 

def shutdown_runtime_hotspot():
    if not ON_DEVICE:
        return
    wifi_iface = find_wifi_interface()
    try:
        try:
            subprocess.run(["sudo", "nmcli", "connection", "down", "Swar_Care"], check=False)
            subprocess.run(["sudo", "nmcli", "device", "disconnect", wifi_iface], check=False)
        except FileNotFoundError:
            subprocess.run(["nmcli", "connection", "down", "Swar_Care"], check=False)
            subprocess.run(["nmcli", "device", "disconnect", wifi_iface], check=False)
    except Exception:
        pass

atexit.register(shutdown_runtime_hotspot)


# ==============================================================================
# 2. RUNTIME INITIALIZATION GATEWAY
# ==============================================================================
if "hotspot_configured" not in st.session_state:
    setup_runtime_hotspot()
    st.session_state["hotspot_configured"] = True

backend = SwarCareEngine.get_instance()


# ==============================================================================
# 3. STREAMLIT USER INTERFACE LAYOUT
# ==============================================================================
st.set_page_config(page_title="SwarCare Hub", page_icon="📡", layout="centered")

st.title("📡 SwarCare Hybrid Station")
st.write("SwarCare Anomaly Detection Hub with Client-Side Canvas Injection")
st.markdown("---")

if backend.state == "STOPPED":
    st.error("System Status: **STOPPED** ⏹️")
elif backend.state == "RECORDING":
    st.success("System Status: **RECORDING RUN LIVE** ▶️")
elif backend.state == "PAUSED":
    st.warning("System Status: **STREAM BUFFER PAUSED** ⚖️")

st.write("")  

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


# ==============================================================================
# 4 & 5. COMBINED HIGH-SPEED RUNTIME MONITOR (HANDLES CANVAS & METRICS)
# ==============================================================================
st.subheader("Live Telemetry & Performance Tracking")

@st.fragment()
def render_live_telemetry_station():
    buffer_snapshot = backend.get_live_buffer_snapshot()
    live_samples_csv = ",".join([f"{x:.4f}" for x in buffer_snapshot[::3]])

    # Self-contained layout with zero external file dependencies
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
                
                // Draw background hardware grids
                ctx.strokeStyle = '#21262D';
                ctx.lineWidth = 1;
                for(let i = 1; i < 5; i++) {{
                    let y = (canvas.height / 5) * i;
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(canvas.width, y);
                    ctx.stroke();
                }}
                
                // Draw live metrics traces
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
    # Using components.html here creates a sandbox that updates with the fragment loop
    components.html(canvas_html_brick, height=260, scrolling=False)

    # --- TEXT METRICS ROWPASS ---
    c1, c2 = st.columns(2)
    metric1_placeholder = c1.empty()
    metric2_placeholder = c2.empty()

    total_samples = backend.samples_recorded
    csv_file_lines = total_samples + 1 if total_samples > 0 else 0
    elapsed_seconds = total_samples / backend.SAMPLE_RATE_HZ

    with metric1_placeholder.container():
        st.metric("Raw Samples Tracked", f"{total_samples:,}")
        st.caption(f"📝 Total lines written to disk CSV file: **{csv_file_lines:,}**")
    metric2_placeholder.metric("Continuous Timeline Tracked", f"{elapsed_seconds:.2f} s")

    # Force immediate fragment rerun if system is actively logging
    if backend.state == "RECORDING":
        time.sleep(0.08)
        st.rerun()

# Instantiate layout component window
render_live_telemetry_station()