# ==============================================================================
# SWARCARE HUB: BACKGROUND PROCESSING & CORE DATA STORAGE (engine.py)
# ==============================================================================
import os
import wave
import queue
import time
import threading
import struct
from datetime import datetime, timezone, timedelta
from collections import deque

PACKET_FORMAT = "<IQ40H" 
RAW_PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# --- HARDCODED TARGET SPECIFICATION: INDIAN STANDARD TIME (UTC +5:30) ---
IST = timezone(timedelta(hours=5, minutes=30))

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
            
        self.state = "STOPPED"
        self.data_queue = queue.Queue()
        self.file_lock = threading.Lock()
        self.buffer_lock = threading.Lock() # Dedicated thread lock for live_buffer modifications
        
        self.wav_file = None
        self.csv_file = None
        
        self.mcu_to_system_us_delta = None
        self.samples_recorded = 0
        self.SAMPLE_RATE_HZ = 2000
        
        # Pre-allocated sliding window ring buffer (stores last 600 points)
        self.live_buffer = deque([0.0] * 600, maxlen=600) 
        
        self.recordings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Parallel thread separation architecture
        threading.Thread(target=self._process_queue_stream, daemon=True).start()
        Bridge.provide("piezo_batch", self.handle_packet)
        threading.Thread(target=App.run, daemon=True).start()
        
        self._initialized = True

    def handle_packet(self, payload):
        # Allow queuing only during active capture sessions
        if self.state in ["RECORDING", "PAUSED"]:
            self.data_queue.put(payload)

    def get_live_buffer_snapshot(self):
        """Thread-safe snapshot getter for the Streamlit UI loop to prevent race conditions."""
        with self.buffer_lock:
            return list(self.live_buffer)

    def _process_queue_stream(self):
        """Ultra-low latency parallel processing engine."""
        last_flush_time = time.time()
        
        while True:
            try:
                # If engine is paused, hold items safely ordered in queue and rest
                if self.state == "PAUSED":
                    time.sleep(0.02)
                    continue

                # Pull incoming telemetry packet
                try:
                    payload = self.data_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                payloads_batch = [payload]
                
                # Instant Batch Drainer: Pulls all backlogged items from RAM queue at once
                while True:
                    try:
                        payloads_batch.append(self.data_queue.get_nowait())
                    except queue.Empty:
                        break
                
                pcm_buffer = bytearray()
                buffer_list = []
                local_samples_count = 0
                
                # Fast vector parsing loop execution
                for p in payloads_batch:
                    if len(p) > RAW_PACKET_SIZE:
                        p = p[-RAW_PACKET_SIZE:]
                    if not isinstance(p, bytes) or len(p) != RAW_PACKET_SIZE:
                        continue
                        
                    unpacked_data = struct.unpack(PACKET_FORMAT, p)
                    batch_start_idx = unpacked_data[0]
                    batch_start_us = unpacked_data[1]  
                    raw_values = unpacked_data[2:]     
                    
                    if self.mcu_to_system_us_delta is None:
                        now = time.time()
                        total_system_us = int(now * 1_000_000)
                        self.mcu_to_system_us_delta = total_system_us - batch_start_us

                    now_us_base = self.mcu_to_system_us_delta + batch_start_us
                    
                    # Lock buffer momentarily during modifications to shield from UI read operations
                    with self.buffer_lock:
                        for idx, val in enumerate(raw_values):
                            normalized_amp = (val / 4095.0) * 2.0 - 1.0
                            self.live_buffer.append(normalized_amp)
                            
                            # Cache serialization payload structure
                            sample_system_us = now_us_base + (idx * 500)
                            sample_dt = datetime.fromtimestamp(sample_system_us / 1_000_000, tz=IST)
                            time_stamp_str = sample_dt.strftime("%H:%M:%S.%f")[:-3]
                            
                            int_pcm_s16 = max(-32768, min(32767, int(normalized_amp * 32767)))
                            pcm_buffer.extend(int_pcm_s16.to_bytes(2, byteorder="little", signed=True))
                            buffer_list.append(f"{batch_start_idx + idx},{time_stamp_str},{val},{normalized_amp:.4f}\n")
                    
                    local_samples_count += len(raw_values)
                
                # Bulk thread-safe I/O commit execution (checks for file object validation over state variables)
                if len(buffer_list) > 0:
                    with self.file_lock:
                        if self.csv_file and self.wav_file:
                            self.wav_file.writeframes(pcm_buffer)
                            self.csv_file.write("".join(buffer_list))
                            self.samples_recorded += local_samples_count
                            
                            # Defer flushing to 500ms bounds to resolve physical I/O bottlenecks
                            current_time = time.time()
                            if current_time - last_flush_time >= 0.5:
                                self.csv_file.flush()
                                last_flush_time = current_time
                                
            except Exception:
                continue

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

            date_stamp = datetime.now(tz=IST).strftime("%Y-%m-%d_%H-%M-%S")
            file_prefix = f"Rec_{next_run_id:03d}_{date_stamp}"
            
            self.wav_file = wave.open(os.path.join(self.recordings_dir, f"{file_prefix}.wav"), "wb")
            self.wav_file.setnchannels(1)
            self.wav_file.setsampwidth(2)
            self.wav_file.setframerate(self.SAMPLE_RATE_HZ)
            
            self.csv_file = open(os.path.join(self.recordings_dir, f"{file_prefix}.csv"), "w", newline="", encoding="utf-8")
            self.csv_file.write("sample_index,real_time_s,raw_adc,amplitude\n")
            self.csv_file.flush()
            
            self.mcu_to_system_us_delta = None
            self.samples_recorded = 0
            
            with self.buffer_lock:
                self.live_buffer.clear()
                for _ in range(600):
                    self.live_buffer.append(0.0)
                
            self.state = "RECORDING"

    def stop_recording(self):
        if self.state in ["RECORDING", "PAUSED"]:
            self.state = "STOPPED"
            
            # Allow background thread loop 150ms window to drain final backlogged payload
            time.sleep(0.15)
            
            with self.file_lock:
                if self.wav_file: 
                    self.wav_file.close()
                if self.csv_file:
                    self.csv_file.flush()
                    os.fsync(self.csv_file.fileno())
                    self.csv_file.close()
                
                self.wav_file = None
                self.csv_file = None
            
            # Clean lingering storage elements from queue frame context
            while not self.data_queue.empty():
                try: 
                    self.data_queue.get_nowait()
                except queue.Empty: 
                    break