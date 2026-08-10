    import os
    import wave
    import queue
    import time
    import struct
    import subprocess
    import threading
    from datetime import datetime, timezone, timedelta
    from collections import deque
    
    # --- TARGET SPECIFICATION: INDIAN STANDARD TIME (UTC +5:30) ---
    IST = timezone(timedelta(hours=5, minutes=30))
    PACKET_FORMAT = "<IQ40H" 
    RAW_PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
    
    # --- ADC / VOLTAGE CONVERSION CONSTANTS ---
    ADC_MAX_VALUE = 4095.0        # 12-bit ADC (0-4095)
    ADC_REFERENCE_VOLTAGE = 3.3   # Reference voltage of the ADC (adjust to match your board, e.g. 5.0V)
    
    try:
        from arduino.app_utils import App, Bridge
        ON_DEVICE = True
    except ImportError:
        ON_DEVICE = False
        class App:
            @staticmethod
            def run():
                while True:
                    time.sleep(1)
        class Bridge:
            @staticmethod
            def provide(n, c):
                pass
    
    
    class SwarCareEngine:
        _instance = None
        _singleton_lock = threading.Lock()
    
        @classmethod
        def get_instance(cls):
            """Thread-safe Singleton accessor ensuring exactly one global engine exists."""
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance
    
        def __init__(self):
            if hasattr(self, "_initialized") and self._initialized:
                return
                
            self.state = "STOPPED"  # STOPPED, RECORDING, PAUSED, STOPPING
            self.state_lock = threading.Lock() 
            
            # --- THREAD-ISOLATED ACCELERATED QUEUES ---
            self.piezo_queue = queue.Queue(maxsize=5000)
            self.piezo_file_lock = threading.Lock()
            self.audio_file_lock = threading.Lock()
            self.audio_buffer_lock = threading.Lock()
            self.terminal_lock = threading.Lock()
            
            self.PIEZO_SAMPLE_RATE_HZ = 2000
            self.AUDIO_SAMPLE_RATE_HZ = 16000
            
            # Optimized Storage Targets
            self.piezo_csv = None
            self.audio_raw_file = None
            self.audio_tmp_path = ""
            
            self.piezo_samples_recorded = 0
            self.audio_samples_recorded = 0
            
            # Absolute Phase Synchronization Epoch Anchors
            self.start_system_time = 0.0
            self.start_piezo_us = None
            self.pause_start_time = 0.0
            self.current_prefix = ""
            
            # High-Speed Telemetry Deque Rings for Fluid Visualizer Updates
            self.piezo_terminal_lines = deque(maxlen=14)
            self.audio_live_buffer = deque([0.0] * 200, maxlen=200)
            
            self.recordings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
            self.DATA_DIR = self.recordings_dir
            os.makedirs(self.recordings_dir, exist_ok=True)
            
            self.audio_process = None
            
            # Start persistent hardware serial processor thread immediately
            threading.Thread(target=self._process_piezo_stream, daemon=True).start()
            
            Bridge.provide("piezo_batch", self.handle_piezo_packet)
            threading.Thread(target=App.run, daemon=True).start()
            
            self._initialized = True
    
        def handle_piezo_packet(self, payload):
            try:
                self.piezo_queue.put_nowait(bytes(payload))
            except queue.Full:
                pass 
    
        def _determine_audio_device(self):
            try:
                result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True)
                for line in result.stdout.splitlines():
                    if "card" in line and "USB" in line:
                        card_num = line.split(":")[0].split("card")[1].strip()
                        return f"plughw:{card_num},0"
            except Exception:
                pass
            return "plug:default"
    
        def get_audio_buffer_snapshot(self):
            with self.audio_buffer_lock:
                snapshot = list(self.audio_live_buffer)
                self.audio_live_buffer.clear() 
                return snapshot
    
        def get_terminal_lines_snapshot(self):
            with self.terminal_lock:
                return list(self.piezo_terminal_lines)
    
        def start_recording(self):
            with self.state_lock:
                if self.state == "PAUSED":
                    paused_duration = time.time() - self.pause_start_time
                    self.start_system_time += paused_duration
                    self.state = "RECORDING"
                    return
                if self.state != "STOPPED":
                    return
    
                # HARDWARE RESET UTILITY: Kill stray drivers to unlock the sound card channel resource
                try:
                    subprocess.run(["pkill", "-9", "arecord"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(0.05)
                except Exception:
                    pass
    
                try:
                    existing_files = os.listdir(self.recordings_dir)
                    run_numbers = [
                        int(f.split("_")[1]) for f in existing_files 
                        if f.startswith("Rec_") and "_" in f and f.split("_")[1].isdigit()
                    ]
                    next_run_id = max(run_numbers) + 1 if run_numbers else 1
                except Exception:
                    next_run_id = 1
    
                date_stamp = datetime.now(tz=IST).strftime("%Y%m%d_%H%M%S")
                self.current_prefix = f"Rec_{next_run_id:03d}_{date_stamp}"
    
                self.piezo_samples_recorded = 0
                self.audio_samples_recorded = 0
                self.start_piezo_us = None
    
                with self.piezo_file_lock:
                    self.piezo_csv = open(os.path.join(self.recordings_dir, f"{self.current_prefix}_piezo.csv"), "w", newline="", encoding="utf-8", buffering=65536)
                    self.piezo_csv.write("sample_index,real_time_s,raw_adc,amplitude,voltage\n")
    
                with self.audio_file_lock:
                    self.audio_tmp_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_audio.tmp")
                    self.audio_raw_file = open(self.audio_tmp_path, "wb", buffering=65536)
    
                with self.terminal_lock:
                    self.piezo_terminal_lines.clear()
                with self.audio_buffer_lock:
                    self.audio_live_buffer.clear()
    
                while not self.piezo_queue.empty():
                    try:
                        self.piezo_queue.get_nowait()
                    except queue.Empty:
                        break
    
                target_hw = self._determine_audio_device()
                audio_cmd = [
                    "arecord", "-D", target_hw, "-f", "S16_LE", 
                    "-r", str(self.AUDIO_SAMPLE_RATE_HZ), "-c", "1", "-t", "raw",
                    "--buffer-time=30000"
                ]
                self.audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                
                self.start_system_time = time.time()
                self.state = "RECORDING"
    
                threading.Thread(target=self._audio_capture_worker, args=(self.audio_process,), daemon=True).start()
            return
    
        def pause_recording(self):
            with self.state_lock:
                if self.state == "RECORDING":
                    self.state = "PAUSED"
                    self.pause_start_time = time.time()
    
        def stop_recording(self):
            with self.state_lock:
                if self.state in ["RECORDING", "PAUSED"]:
                    self.state = "STOPPING"
                    
                    if self.audio_process:
                        try:
                            self.audio_process.terminate()
                            self.audio_process.wait(timeout=1.0)
                        except Exception:
                            pass
                        self.audio_process = None
                    
                    threading.Thread(target=self._safe_single_threaded_cleanup, daemon=True).start()
    
        def _process_piezo_stream(self):
            last_sec = -1
            base_time_str = ""
            last_flush = time.time()
    
            while True:
                try:
                    payload = self.piezo_queue.get(timeout=0.01)
                    payload = bytes(payload)
                    
                    if len(payload) > RAW_PACKET_SIZE:
                        payload = payload[-RAW_PACKET_SIZE:]
                    if len(payload) != RAW_PACKET_SIZE:
                        continue
    
                    unpacked = struct.unpack(PACKET_FORMAT, payload)
                    batch_start_us = unpacked[1]
                    raw_values = unpacked[2:]
    
                    if self.state != "RECORDING":
                        continue
    
                    if self.start_piezo_us is None:
                        self.start_piezo_us = batch_start_us
    
                    buffer_list = []
                    us_offset = (batch_start_us - self.start_piezo_us) / 1000000.0
    
                    for idx, val in enumerate(raw_values):
                        amp = (val / ADC_MAX_VALUE) * 2.0 - 1.0
                        voltage = (val / ADC_MAX_VALUE) * ADC_REFERENCE_VOLTAGE
                        sample_offset = idx / self.PIEZO_SAMPLE_RATE_HZ
                        sample_time = self.start_system_time + us_offset + sample_offset
                        
                        sec_int = int(sample_time)
                        
                        if sec_int != last_sec:
                            dt_obj = datetime.fromtimestamp(sec_int, tz=IST)
                            base_time_str = dt_obj.strftime("%H:%M:%S")
                            last_sec = sec_int
    
                        ms = int((sample_time - sec_int) * 1000)
                        t_str = f"{base_time_str}.{ms:03d}"
                        
                        buffer_list.append(f"{self.piezo_samples_recorded},{t_str},{val},{amp:.4f},{voltage:.4f}\n")
                        
                        if self.piezo_samples_recorded % 160 == 0:
                            line_entry = f"TIME: {t_str} | ADC: {val:04d} | AMP: {amp:+.4f} | V: {voltage:.3f}"
                            with self.terminal_lock:
                                self.piezo_terminal_lines.append((line_entry, abs(amp) > 0.02))
                                
                        self.piezo_samples_recorded += 1
    
                    with self.piezo_file_lock:
                        if self.piezo_csv:
                            self.piezo_csv.write("".join(buffer_list))
                            if time.time() - last_flush >= 0.25:
                                self.piezo_csv.flush()
                                last_flush = time.time()
                except queue.Empty:
                    continue
                except Exception:
                    continue
    
        def _audio_capture_worker(self, proc):
            while self.state in ["RECORDING", "PAUSED"] and proc.poll() is None:
                try:
                    raw_bytes = proc.stdout.read(512)
                    if not raw_bytes:
                        break
    
                    if self.state != "RECORDING":
                        continue
    
                    total_samples = len(raw_bytes) // 2
                    audio_ints = struct.unpack(f"<{total_samples}h", raw_bytes)
    
                    max_val = max(abs(val) for val in audio_ints) if audio_ints else 0
                    visual_multiplier = 1.0
                    if 0 < max_val < 6000:
                        visual_multiplier = 6.0  
                    elif 6000 <= max_val < 15000:
                        visual_multiplier = 2.5
    
                    with self.audio_buffer_lock:
                        for val in audio_ints[::32]:  
                            normalized_float = (val / 32768.0) * visual_multiplier
                            clamped_float = max(-1.0, min(1.0, normalized_float))
                            self.audio_live_buffer.append(clamped_float)
    
                    self.audio_samples_recorded += total_samples
    
                    with self.audio_file_lock:
                        if self.audio_raw_file:
                            self.audio_raw_file.write(raw_bytes)
                except Exception:
                    break
    
        def _safe_single_threaded_cleanup(self):
            time.sleep(0.15)
            
            with self.piezo_file_lock, self.audio_file_lock:
                p_dur = self.piezo_samples_recorded / self.PIEZO_SAMPLE_RATE_HZ
                a_dur = self.audio_samples_recorded / self.AUDIO_SAMPLE_RATE_HZ
                
                # CRITICAL PROTECTION CHECK: Prevent empty data streams from truncating valid metrics to 0
                if a_dur == 0 and self.audio_samples_recorded > 0:
                    a_dur = self.audio_samples_recorded / self.AUDIO_SAMPLE_RATE_HZ
                
                synchronized_duration = min(p_dur, a_dur) if (p_dur > 0 and a_dur > 0) else max(p_dur, a_dur)
                
                final_piezo_count = int(synchronized_duration * self.PIEZO_SAMPLE_RATE_HZ)
                final_audio_count = int(synchronized_duration * self.AUDIO_SAMPLE_RATE_HZ)
                
                if self.piezo_csv:
                    self.piezo_csv.flush()
                    os.fsync(self.piezo_csv.fileno())
                    self.piezo_csv.close()
                    self.piezo_csv = None
                    
                try:
                    p_csv_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_piezo.csv")
                    with open(p_csv_path, "r", encoding="utf-8") as f:
                        p_lines = f.readlines()
                    slice_limit = min(final_piezo_count + 1, len(p_lines))
                    with open(p_csv_path, "w", encoding="utf-8") as f:
                        f.writelines(p_lines[:slice_limit])
                except Exception:
                    pass
    
                if self.audio_raw_file:
                    self.audio_raw_file.flush()
                    os.fsync(self.audio_raw_file.fileno())
                    self.audio_raw_file.close()
                    self.audio_raw_file = None
                    
                try:
                    final_wav_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_audio.wav")
                    if os.path.exists(self.audio_tmp_path) and final_audio_count > 0:
                        wav_file = wave.open(final_wav_path, "wb")
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.AUDIO_SAMPLE_RATE_HZ)
                        
                        with open(self.audio_tmp_path, "rb") as f_raw:
                            exact_pcm_payload = f_raw.read(final_audio_count * 2)
                            wav_file.writeframes(exact_pcm_payload)
                        wav_file.close()
                    
                    if os.path.exists(self.audio_tmp_path):
                        os.remove(self.audio_tmp_path)
                except Exception:
                    pass
    
                self.piezo_samples_recorded = final_piezo_count
                self.audio_samples_recorded = final_audio_count
    
            with self.state_lock:
                self.state = "STOPPED"