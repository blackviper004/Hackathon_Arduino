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

try:
    from arduino.app_utils import App, Bridge
    ON_DEVICE = True
except ImportError:
    ON_DEVICE = False
    class App:
        @staticmethod
        def run():
            while True: time.sleep(1)
    class Bridge:
        @staticmethod
        def provide(n, c): pass


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
        self.piezo_queue = queue.Queue()
        self.piezo_file_lock = threading.Lock()
        self.audio_file_lock = threading.Lock()
        self.audio_buffer_lock = threading.Lock()
        self.terminal_lock = threading.Lock()
        
        self.PIEZO_SAMPLE_RATE_HZ = 2000
        self.AUDIO_SAMPLE_RATE_HZ = 16000
        
        # Live Active File Descriptors for Real-Time Serialization
        self.piezo_wav = None
        self.piezo_csv = None
        self.audio_wav = None
        self.audio_csv = None
        
        self.piezo_samples_recorded = 0
        self.audio_samples_recorded = 0
        
        # AI-Grade Microsecond Synchronization Time Anchors
        self.start_system_time = 0.0
        self.start_piezo_us = None
        self.pause_start_time = 0.0
        self.current_prefix = ""
        
        # Telemetry Cache Registers for Web Viewports
        self.piezo_terminal_lines = deque(maxlen=14)
        self.latest_piezo_amplitude = 0.0
        self.audio_live_buffer = deque([0.0] * 300, maxlen=300)
        
        self.recordings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        self.audio_process = None
        
        # Boot persistent background daemon execution loops immediately
        threading.Thread(target=self._process_piezo_stream, daemon=True).start()
        threading.Thread(target=self._audio_daemon_loop, daemon=True).start()
        
        Bridge.provide("piezo_batch", self.handle_piezo_packet)
        threading.Thread(target=App.run, daemon=True).start()
        
        self._initialized = True

    def handle_piezo_packet(self, payload):
        self.piezo_queue.put(payload)

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
            return list(self.audio_live_buffer)

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
            self.latest_piezo_amplitude = 0.0
            self.start_piezo_us = None

            # Open real-time streaming file handles and write headings immediately
            with self.piezo_file_lock:
                self.piezo_wav = wave.open(os.path.join(self.recordings_dir, f"{self.current_prefix}_piezo.wav"), "wb")
                self.piezo_wav.setnchannels(1)
                self.piezo_wav.setsampwidth(2)
                self.piezo_wav.setframerate(self.PIEZO_SAMPLE_RATE_HZ)
                
                self.piezo_csv = open(os.path.join(self.recordings_dir, f"{self.current_prefix}_piezo.csv"), "w", newline="", encoding="utf-8")
                self.piezo_csv.write("sample_index,real_time_s,raw_adc,amplitude\n")

            with self.audio_file_lock:
                self.audio_wav = wave.open(os.path.join(self.recordings_dir, f"{self.current_prefix}_audio.wav"), "wb")
                self.audio_wav.setnchannels(1)
                self.audio_wav.setsampwidth(2)
                self.audio_wav.setframerate(self.AUDIO_SAMPLE_RATE_HZ)
                
                self.audio_csv = open(os.path.join(self.recordings_dir, f"{self.current_prefix}_audio.csv"), "w", newline="", encoding="utf-8")
                self.audio_csv.write("sample_index,real_time_s,raw_adc,amplitude\n")

            with self.terminal_lock:
                self.piezo_terminal_lines.clear()
            with self.audio_buffer_lock:
                self.audio_live_buffer.clear()
                self.audio_live_buffer.extend([0.0] * 300)

            # Purge the audio device's input pipe allocation buffers
            if self.audio_process and self.audio_process.stdout:
                fd = self.audio_process.stdout.fileno()
                os.set_blocking(fd, False)
                try:
                    while True:
                        if not self.audio_process.stdout.read(32768): break
                except Exception: pass
                os.set_blocking(fd, True)

            while not self.piezo_queue.empty():
                try: self.piezo_queue.get_nowait()
                except queue.Empty: break

            self.start_system_time = time.time()
            self.state = "RECORDING"

    def pause_recording(self):
        with self.state_lock:
            if self.state == "RECORDING":
                self.state = "PAUSED"
                self.pause_start_time = time.time()

    def stop_recording(self):
        with self.state_lock:
            if self.state in ["RECORDING", "PAUSED"]:
                self.state = "STOPPING"
                # Spawns exactly one worker thread to manage cleanup safely
                threading.Thread(target=self._safe_single_threaded_cleanup, daemon=True).start()

    def _process_piezo_stream(self):
        """Thread-isolated loop streaming vibration data directly to disk files live."""
        last_sec = -1
        base_time_str = ""
        last_flush = time.time()

        while True:
            try:
                try:
                    payload = self.piezo_queue.get(timeout=0.02)
                except queue.Empty:
                    continue

                if len(payload) > RAW_PACKET_SIZE: payload = payload[-RAW_PACKET_SIZE:]
                if not isinstance(payload, bytes) or len(payload) != RAW_PACKET_SIZE: continue

                unpacked = struct.unpack(PACKET_FORMAT, payload)
                batch_start_idx = unpacked[0]
                batch_start_us = unpacked[1]
                raw_values = unpacked[2:]

                if len(raw_values) > 0:
                    self.latest_piezo_amplitude = (raw_values[-1] / 4095.0) * 2.0 - 1.0

                # FIXED: Strictly isolate writing to active recording states to prevent counter inflation
                if self.state != "RECORDING":
                    continue

                if self.state == "RECORDING":
                    if self.start_piezo_us is None:
                        self.start_piezo_us = batch_start_us

                    pcm_buffer = bytearray()
                    buffer_list = []

                    for idx, val in enumerate(raw_values):
                        amp = (val / 4095.0) * 2.0 - 1.0
                        
                        # High-Precision Hardware Time Alignment Mapping
                        us_offset = (batch_start_us - self.start_piezo_us) / 1000000.0
                        sample_offset = idx / self.PIEZO_SAMPLE_RATE_HZ
                        sample_time = self.start_system_time + us_offset + sample_offset
                        
                        sec_int = int(sample_time)
                        ms = int((sample_time - sec_int) * 1000)

                        if sec_int != last_sec:
                            dt_obj = datetime.fromtimestamp(sec_int, tz=IST)
                            base_time_str = dt_obj.strftime("%H:%M:%S")
                            last_sec = sec_int

                        t_str = f"{base_time_str}.{ms:03d}"
                        int_pcm = max(-32768, min(32767, int(amp * 32767)))
                        pcm_buffer.extend(int_pcm.to_bytes(2, byteorder="little", signed=True))
                        
                        buffer_list.append(f"{self.piezo_samples_recorded},{t_str},{val},{amp:.4f}\n")
                        
                        if self.piezo_samples_recorded % 133 == 0:
                            line_entry = f"TIME: {t_str} | ADC: {val:04d} | AMP: {amp:+.4f}"
                            with self.terminal_lock:
                                self.piezo_terminal_lines.append((line_entry, abs(amp) > 0.02))
                                
                        self.piezo_samples_recorded += 1

                    with self.piezo_file_lock:
                        if self.piezo_csv and self.piezo_wav:
                            self.piezo_wav.writeframes(pcm_buffer)
                            self.piezo_csv.write("".join(buffer_list))
                            if time.time() - last_flush >= 0.5:
                                self.piezo_csv.flush()
                                last_flush = time.time()
            except Exception:
                continue

    def _audio_daemon_loop(self):
        """Continuous background hardware listener streaming audio data frames directly to files."""
        target_hw = self._determine_audio_device()
        audio_cmd = [
            "arecord", "-D", target_hw, "-f", "S16_LE", 
            "-r", str(self.AUDIO_SAMPLE_RATE_HZ), "-c", "1", "-t", "raw",
            "--buffer-time=500000"
        ]
        self.audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        last_sec = -1
        base_time_str = ""
        last_flush = time.time()

        while True:
            try:
                raw_bytes = self.audio_process.stdout.read(2048)
                if not raw_bytes:
                    time.sleep(0.001)
                    continue

                # FIXED: Freeze streaming and buffer modifications instantly when stopping
                if self.state != "RECORDING":
                    with self.audio_buffer_lock:
                        if len(self.audio_live_buffer) > 0 and (self.audio_live_buffer[0] != 0.0 or self.audio_live_buffer[-1] != 0.0):
                            self.audio_live_buffer.clear()
                            self.audio_live_buffer.extend([0.0] * 300)
                    continue

                if self.state == "RECORDING":
                    total_samples = len(raw_bytes) // 2
                    audio_ints = struct.unpack(f"<{total_samples}h", raw_bytes)

                    with self.audio_buffer_lock:
                        for val in audio_ints[::12]:
                            self.audio_live_buffer.append(val / 32768.0)

                    buffer_list = []
                    for val in audio_ints:
                        amp = val / 32768.0
                        sample_time = self.start_system_time + (self.audio_samples_recorded / self.AUDIO_SAMPLE_RATE_HZ)
                        
                        sec_int = int(sample_time)
                        ms = int((sample_time - sec_int) * 1000)

                        if sec_int != last_sec:
                            dt_obj = datetime.fromtimestamp(sec_int, tz=IST)
                            base_time_str = dt_obj.strftime("%H:%M:%S")
                            last_sec = sec_int

                        t_str = f"{base_time_str}.{ms:03d}"
                        emulated_adc = int((amp + 1.0) * 2047.5)
                        buffer_list.append(f"{self.audio_samples_recorded},{t_str},{emulated_adc},{amp:.4f}\n")
                        self.audio_samples_recorded += 1

                    with self.audio_file_lock:
                        if self.audio_csv and self.audio_wav:
                            self.audio_wav.writeframes(raw_bytes)
                            self.audio_csv.write("".join(buffer_list))
                            if time.time() - last_flush >= 0.5:
                                self.audio_csv.flush()
                                last_flush = time.time()
            except Exception:
                time.sleep(0.001)

    def _safe_single_threaded_cleanup(self):
        """Safely flushes data, closes handles, and runs timeline truncation on an isolated thread."""
        time.sleep(0.15)  # Wait briefly for background loops to exit their active blocks
        
        with self.piezo_file_lock, self.audio_file_lock:
            # Calculate absolute matching limits
            p_dur = self.piezo_samples_recorded / self.PIEZO_SAMPLE_RATE_HZ
            a_dur = self.audio_samples_recorded / self.AUDIO_SAMPLE_RATE_HZ
            synchronized_duration = min(p_dur, a_dur)
            
            final_piezo_count = int(synchronized_duration * self.PIEZO_SAMPLE_RATE_HZ)
            final_audio_count = int(synchronized_duration * self.AUDIO_SAMPLE_RATE_HZ)
            
            # --- 1. CLOSE PIEZO HANDLES ---
            if self.piezo_wav:
                self.piezo_wav.close()
                self.piezo_wav = None
            if self.piezo_csv:
                self.piezo_csv.flush()
                os.fsync(self.piezo_csv.fileno())
                self.piezo_csv.close()
                self.piezo_csv = None
                
            try:
                p_csv_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_piezo.csv")
                with open(p_csv_path, "r", encoding="utf-8") as f:
                    p_lines = f.readlines()[:final_piezo_count + 1]
                with open(p_csv_path, "w", encoding="utf-8") as f:
                    f.writelines(p_lines)
            except Exception: pass

            # --- 2. CLOSE AUDIO HANDLES ---
            if self.audio_wav:
                self.audio_wav.close()
                self.audio_wav = None
            if self.audio_csv:
                self.audio_csv.flush()
                os.fsync(self.audio_csv.fileno())
                self.audio_csv.close()
                self.audio_csv = None
                
            try:
                a_csv_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_audio.csv")
                with open(a_csv_path, "r", encoding="utf-8") as f:
                    a_lines = f.readlines()[:final_audio_count + 1]
                with open(a_csv_path, "w", encoding="utf-8") as f:
                    f.writelines(a_lines)
            except Exception: pass

            # Sync engine counters with the exact truncation metrics
            self.piezo_samples_recorded = final_piezo_count
            self.audio_samples_recorded = final_audio_count

        with self.state_lock:
            self.state = "STOPPED"