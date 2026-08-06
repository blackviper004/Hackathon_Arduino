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
            
        self.state = "STOPPED"  # STOPPED, RECORDING, PAUSED, STOPPING, PROCESSING
        self.state_lock = threading.Lock() 
        
        # --- RESILIENT MEMORY STORAGE REGISTERS ---
        self.piezo_queue = queue.Queue()
        self.piezo_raw_storage = []
        self.audio_raw_storage = []
        
        self.piezo_file_lock = threading.Lock()
        self.audio_file_lock = threading.Lock()
        self.audio_buffer_lock = threading.Lock()
        self.terminal_lock = threading.Lock()
        self.disk_write_lock = threading.Lock()
        
        self.PIEZO_SAMPLE_RATE_HZ = 2000
        self.AUDIO_SAMPLE_RATE_HZ = 16000
        
        self.piezo_samples_recorded = 0
        self.audio_samples_recorded = 0
        
        # Microsecond Synchronization Time Anchors
        self.start_system_time = 0.0
        self.start_piezo_us = None
        self.pause_start_time = 0.0
        self.current_prefix = ""
        
        # Continuous Scrolling Monospace Serial Monitor Deque Cache
        self.piezo_terminal_lines = deque(maxlen=14)
        self.latest_piezo_amplitude = 0.0
        self.audio_live_buffer = deque([0.0] * 300, maxlen=300)
        
        self.recordings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        self.audio_process = None
        
        # Initialize background processing daemons immediately on startup
        threading.Thread(target=self._process_piezo_queue, daemon=True).start()
        threading.Thread(target=self._audio_daemon_loop, daemon=True).start()
        
        Bridge.provide("piezo_batch", self.handle_piezo_packet)
        threading.Thread(target=App.run, daemon=True).start()
        
        self._initialized = True

    def handle_piezo_packet(self, payload):
        """Buffers raw incoming hardware telemetry packets instantly into RAM queues."""
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
                self.state = "RECORDING"
                return
            if self.state != "STOPPED" or self.disk_write_lock.locked():
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

            self.piezo_raw_storage.clear()
            self.audio_raw_storage.clear()
            self.piezo_samples_recorded = 0
            self.audio_samples_recorded = 0
            self.latest_piezo_amplitude = 0.0
            self.start_piezo_us = None

            with self.terminal_lock:
                self.piezo_terminal_lines.clear()
            with self.audio_buffer_lock:
                self.audio_live_buffer.clear()
                self.audio_live_buffer.extend([0.0] * 300)

            # Purge any stale idle bytes from the OS sound pipe buffer
            if self.audio_process and self.audio_process.stdout:
                fd = self.audio_process.stdout.fileno()
                os.set_blocking(fd, False)
                try:
                    while True:
                        stale_bytes = self.audio_process.stdout.read(32768)
                        if not stale_bytes: break
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

    def _process_piezo_queue(self):
        """High-speed worker thread parsing incoming vibration arrays to RAM frames."""
        last_sec = -1
        base_time_str = ""

        while True:
            try:
                try:
                    payload = self.piezo_queue.get(timeout=0.02)
                except queue.Empty:
                    if self.state == "STOPPING":
                        self._check_and_trigger_deferred_compilation()
                    continue

                if len(payload) > RAW_PACKET_SIZE: payload = payload[-RAW_PACKET_SIZE:]
                if not isinstance(payload, bytes) or len(payload) != RAW_PACKET_SIZE: continue

                unpacked = struct.unpack(PACKET_FORMAT, payload)
                batch_start_idx = unpacked[0]
                batch_start_us = unpacked[1]
                raw_values = unpacked[2:]

                if len(raw_values) > 0:
                    last_raw_val = raw_values[-1]
                    self.latest_piezo_amplitude = (last_raw_val / 4095.0) * 2.0 - 1.0

                if self.state in ["STOPPED", "PAUSED", "STOPPING", "PROCESSING"]:
                    continue

                if self.state == "RECORDING":
                    if self.start_piezo_us is None:
                        self.start_piezo_us = batch_start_us

                    self.piezo_raw_storage.append((batch_start_idx, batch_start_us, raw_values))
                    
                    # --- SCROLLING MONOSPACE MONITOR STRIPPED CORE ENGINE ---
                    for i, v in enumerate(raw_values):
                        if (self.piezo_samples_recorded + i) % 133 == 0:
                            s_time = self.start_system_time + ((batch_start_us - self.start_piezo_us) / 1000000.0) + (i / self.PIEZO_SAMPLE_RATE_HZ)
                            s_sec = int(s_time)
                            s_ms = int((s_time - s_sec) * 1000)
                            
                            if s_sec != last_sec:
                                base_time_str = datetime.fromtimestamp(s_sec, tz=IST).strftime("%H:%M:%S")
                                last_sec = s_sec
                                
                            val_amp = (v / 4095.0) * 2.0 - 1.0
                            # FIXED: Completely stripped RAW_INDEX mapping text block to clean monitor space
                            line_entry = f"TIME: {base_time_str}.{s_ms:03d} | ADC: {v:04d} | AMP: {val_amp:+.4f}"
                            
                            with self.terminal_lock:
                                self.piezo_terminal_lines.append((line_entry, abs(val_amp) > 0.02))

                    self.piezo_samples_recorded += len(raw_values)
            except Exception:
                continue

    def _audio_daemon_loop(self):
        """Continuous background hardware listener streaming audio data frames directly into RAM caches."""
        target_hw = self._determine_audio_device()
        audio_cmd = [
            "arecord", "-D", target_hw, "-f", "S16_LE", 
            "-r", str(self.AUDIO_SAMPLE_RATE_HZ), "-c", "1", "-t", "raw",
            "--buffer-time=500000"
        ]
        self.audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        while True:
            try:
                raw_bytes = self.audio_process.stdout.read(2048)
                if not raw_bytes:
                    time.sleep(0.001)
                    continue

                if self.state in ["STOPPED", "PAUSED", "STOPPING", "PROCESSING"]:
                    with self.audio_buffer_lock:
                        if self.audio_live_buffer[0] != 0.0 or self.audio_live_buffer[-1] != 0.0:
                            self.audio_live_buffer.clear()
                            self.audio_live_buffer.extend([0.0] * 300)
                    continue

                if self.state == "RECORDING":
                    total_samples = len(raw_bytes) // 2
                    audio_ints = struct.unpack(f"<{total_samples}h", raw_bytes)

                    with self.audio_buffer_lock:
                        for val in audio_ints[::12]:
                            self.audio_live_buffer.append(val / 32768.0)

                    self.audio_raw_storage.append(raw_bytes)
                    self.audio_samples_recorded += total_samples
            except Exception:
                time.sleep(0.001)

    def _check_and_trigger_deferred_compilation(self):
        with self.state_lock:
            if self.state == "STOPPING":
                self.state = "STOPPED"  # Instantly release UI control back to idle status
                threading.Thread(target=self._compile_ai_datasets, daemon=True).start()

    def _compile_ai_datasets(self):
        """Asynchronously flushes cached RAM blocks to disk with exact timeline truncation."""
        with self.disk_write_lock:
            last_sec = -1
            base_time_str = ""

            # Unpack audio data into a continuous array
            audio_samples = []
            for chunk in self.audio_raw_storage:
                total_samples = len(chunk) // 2
                audio_samples.extend(struct.unpack(f"<{total_samples}h", chunk))

            # Unpack piezo data into a continuous list of samples
            piezo_samples = []
            for batch_idx, batch_us, raw_vals in self.piezo_raw_storage:
                for idx, val in enumerate(raw_vals):
                    piezo_samples.append((batch_idx + idx, batch_us, val, idx))

            # Calculate the maximum synchronous duration threshold
            total_piezo_available = len(piezo_samples)
            total_audio_available = len(audio_samples)
            
            max_piezo_duration = total_piezo_available / self.PIEZO_SAMPLE_RATE_HZ
            max_audio_duration = total_audio_available / self.AUDIO_SAMPLE_RATE_HZ
            
            synchronized_duration = min(max_piezo_duration, max_audio_duration)
            
            # Truncate both datasets to match the synchronization threshold perfectly
            final_piezo_count = int(synchronized_duration * self.PIEZO_SAMPLE_RATE_HZ)
            final_audio_count = int(synchronized_duration * self.AUDIO_SAMPLE_RATE_HZ)
            
            piezo_samples = piezo_samples[:final_piezo_count]
            audio_samples = audio_samples[:final_audio_count]

            # Update metrics values to reflect the truncated values
            self.piezo_samples_recorded = len(piezo_samples)
            self.audio_samples_recorded = len(audio_samples)

            # --- 1. COMPILE SYNCHRONIZED PIEZO TELEMETRY DATASETS ---
            p_csv_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_piezo.csv")
            p_wav_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_piezo.wav")
            
            p_wav = wave.open(p_wav_path, "wb")
            p_wav.setnchannels(1)
            p_wav.setsampwidth(2)
            p_wav.setframerate(self.PIEZO_SAMPLE_RATE_HZ)
            
            p_pcm = bytearray()
            lines_buffer = ["sample_index,real_time_s,raw_adc,amplitude\n"]
            
            for p_idx, (b_idx, b_us, val, idx) in enumerate(piezo_samples):
                amp = (val / 4095.0) * 2.0 - 1.0
                sample_time = self.start_system_time + ((b_us - self.start_piezo_us) / 1000000.0) + (idx / self.PIEZO_SAMPLE_RATE_HZ)
                
                sec_int = int(sample_time)
                ms = int((sample_time - sec_int) * 1000)
                
                if sec_int != last_sec:
                    base_time_str = datetime.fromtimestamp(sec_int, tz=IST).strftime("%H:%M:%S")
                    last_sec = sec_int
                    
                t_str = f"{base_time_str}.{ms:03d}"
                int_pcm = max(-32768, min(32767, int(amp * 32767)))
                p_pcm.extend(int_pcm.to_bytes(2, byteorder="little", signed=True))
                lines_buffer.append(f"{b_idx},{t_str},{val},{amp:.4f}\n")
                
            with open(p_csv_path, "w", encoding="utf-8") as f:
                f.write("".join(lines_buffer))
            p_wav.writeframes(p_pcm)
            p_wav.close()

            # --- 2. COMPILE SYNCHRONIZED ACOUSTIC AUDIO DATASETS ---
            a_csv_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_audio.csv")
            a_wav_path = os.path.join(self.recordings_dir, f"{self.current_prefix}_audio.wav")
            
            a_wav = wave.open(a_wav_path, "wb")
            a_wav.setnchannels(1)
            a_wav.setsampwidth(2)
            a_wav.setframerate(self.AUDIO_SAMPLE_RATE_HZ)
            
            a_pcm = bytearray()
            lines_buffer = ["sample_index,real_time_s,raw_adc,amplitude\n"]
            last_sec = -1
            
            for a_idx, val in enumerate(audio_samples):
                amp = val / 32768.0
                sample_time = self.start_system_time + (a_idx / self.AUDIO_SAMPLE_RATE_HZ)
                
                sec_int = int(sample_time)
                ms = int((sample_time - sec_int) * 1000)
                
                if sec_int != last_sec:
                    base_time_str = datetime.fromtimestamp(sec_int, tz=IST).strftime("%H:%M:%S")
                    last_sec = sec_int
                    
                t_str = f"{base_time_str}.{ms:03d}"
                emulated_adc = int((amp + 1.0) * 2047.5)
                
                int_pcm = max(-32768, min(32767, val))
                a_pcm.extend(int_pcm.to_bytes(2, byteorder="little", signed=True))
                lines_buffer.append(f"{a_idx},{t_str},{emulated_adc},{amp:.4f}\n")
                
            with open(a_csv_path, "w", encoding="utf-8") as f:
                f.write("".join(lines_buffer))
            a_wav.writeframes(a_pcm)
            a_wav.close()
            
            # Flush core data arrays to save RAM space
            self.piezo_raw_storage.clear()
            self.audio_raw_storage.clear()