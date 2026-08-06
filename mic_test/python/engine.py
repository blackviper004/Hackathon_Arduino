import os
import wave
import struct
import time
import subprocess
import threading
from datetime import datetime, timezone, timedelta
from collections import deque

# --- TARGET SPECIFICATION: INDIAN STANDARD TIME (UTC +5:30) ---
IST = timezone(timedelta(hours=5, minutes=30))

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
            
        self.state = "STOPPED"
        self.file_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        
        self.wav_file = None
        self.csv_file = None
        self.samples_recorded = 0
        self.SAMPLE_RATE_HZ = 16000  
        
        self.live_buffer = deque([0.0] * 300, maxlen=300) 
        
        # RESTORED: Exact matching path tracking configuration from your original code
        self.recordings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        self.process = None
        self.reader_thread = None
        self._initialized = True

    def _determine_audio_device(self):
        """Scans the system sound cards to identify the hardware USB Mic line."""
        try:
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                if "card" in line and "USB" in line:
                    card_num = line.split(":")[0].split("card")[1].strip()
                    return f"plughw:{card_num},0"
        except Exception:
            pass
        return "plug:default"

    def get_live_buffer_snapshot(self):
        with self.buffer_lock:
            return list(self.live_buffer)

    def start_recording(self):
        with self.file_lock:
            if self.state == "PAUSED":
                self.state = "RECORDING"
                return
            if self.state != "STOPPED": 
                return

            try:
                existing_files = os.listdir(self.recordings_dir)
                run_numbers = []
                for f in existing_files:
                    if f.startswith("Rec_") and "_" in f:
                        try:
                            num_token = f.split("_")[1]
                            run_numbers.append(int(num_token))
                        except (ValueError, IndexError):
                            continue
                next_run_id = max(run_numbers) + 1 if run_numbers else 1
            except Exception:
                next_run_id = 1

            date_stamp = datetime.now(tz=IST).strftime("%Y%m%d_%H%M%S")
            file_prefix = f"Rec_{next_run_id:03d}_{date_stamp}"
            
            # Open clean target files directly in the restored recordings directory
            self.wav_file = wave.open(os.path.join(self.recordings_dir, f"{file_prefix}.wav"), "wb")
            self.wav_file.setnchannels(1)
            self.wav_file.setsampwidth(2) 
            self.wav_file.setframerate(self.SAMPLE_RATE_HZ)
            
            self.csv_file = open(os.path.join(self.recordings_dir, f"{file_prefix}.csv"), "w", newline="", encoding="utf-8")
            self.csv_file.write("sample_index,real_time_s,raw_adc,amplitude\n")
            self.csv_file.flush()
            
            self.samples_recorded = 0
            
            with self.buffer_lock:
                self.live_buffer.clear()
                self.live_buffer.extend([0.0] * 300)

            target_hw = self._determine_audio_device()
            
            cmd = [
                "arecord",
                "-D", target_hw,
                "-f", "S16_LE",   
                "-r", str(self.SAMPLE_RATE_HZ),
                "-c", "1",
                "-t", "raw"       
            ]
            
            # Capture stderr to flag immediate hardware system constraints
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
            self.state = "RECORDING"
            
            self.reader_thread = threading.Thread(target=self._stream_reader_loop, daemon=True)
            self.reader_thread.start()

    def _stream_reader_loop(self):
        """Processes binary packets from the Linux hardware sound pipe live to disk."""
        last_sec = -1
        base_time_str = ""
        last_flush_time = time.time()
        
        while self.state in ["RECORDING", "PAUSED"] and self.process.poll() is None:
            raw_bytes = self.process.stdout.read(2048)
            if not raw_bytes:
                break
                
            if self.state == "PAUSED":
                continue 
                
            total_samples = len(raw_bytes) // 2
            audio_ints = struct.unpack(f"<{total_samples}h", raw_bytes)
            
            chunk_arrival_time = time.time()
            buffer_list = []
            
            with self.buffer_lock:
                for idx, val in enumerate(audio_ints):
                    amplitude = val / 32768.0
                    
                    if idx % 12 == 0:
                        self.live_buffer.append(amplitude)
                        
                    sample_offset = (total_samples - idx) / self.SAMPLE_RATE_HZ
                    sample_time = chunk_arrival_time - sample_offset
                    sec_int = int(sample_time)
                    ms = int((sample_time - sec_int) * 1000)
                    
                    if sec_int != last_sec:
                        dt = datetime.fromtimestamp(sec_int, tz=IST)
                        base_time_str = dt.strftime("%H:%M:%S")
                        last_sec = sec_int
                        
                    time_stamp_str = f"{base_time_str}.{ms:03d}"
                    emulated_adc = int((amplitude + 1.0) * 2047.5)
                    
                    buffer_list.append(f"{self.samples_recorded},{time_stamp_str},{emulated_adc},{amplitude:.4f}\n")
                    self.samples_recorded += 1
            
            with self.file_lock:
                if self.csv_file and self.wav_file:
                    self.wav_file.writeframes(raw_bytes)
                    self.csv_file.write("".join(buffer_list))
                    
                    current_time = time.time()
                    if current_time - last_flush_time >= 0.5:
                        self.csv_file.flush()
                        last_flush_time = current_time

        # If the recording process exited with an error, print it to the App Lab console
        if self.process and self.process.poll() is not None and self.process.returncode != 0:
            err_output = self.process.stderr.read().decode('utf-8', errors='ignore')
            print(f"❌ OS Sound Engine Error: {err_output.strip()}")

    def stop_recording(self):
        if self.state in ["RECORDING", "PAUSED"]:
            self.state = "STOPPED"
            
            if self.process:
                self.process.terminate()
                self.process.wait()
                self.process = None
                
            with self.file_lock:
                if self.wav_file: 
                    self.wav_file.close()
                if self.csv_file:
                    self.csv_file.flush()
                    os.fsync(self.csv_file.fileno())
                    self.csv_file.close()
                
                self.wav_file = None
                self.csv_file = None