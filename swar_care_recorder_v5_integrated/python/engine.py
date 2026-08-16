# engine.py

import os
import json as _json
import wave
import queue
import time
import struct
import subprocess
import threading
from datetime import datetime
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler

from model import (
    IST,
    PACKET_FORMAT,
    RAW_PACKET_SIZE,
    ADC_MAX_VALUE,
    ADC_REFERENCE_VOLTAGE,
    App,
    Bridge,
    AnomalyDetectionModel,
    VeenaDiagnosticModel,
)


# ===========================================================================
# HTTP SIDECAR — localhost:7654 JSON API for iframe JS clients
# ===========================================================================

class _SwarCareAPIHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler for the live telemetry sidecar."""
    engine_ref = None  # Injected at startup

    def log_message(self, format, *args):  # noqa: A002
        pass  # Suppress default access-log noise

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        eng = _SwarCareAPIHandler.engine_ref
        if eng is None:
            self.send_response(503)
            self.end_headers()
            return
        try:
            if self.path == "/status":
                if eng.state == "RECORDING":
                    elapsed = time.time() - eng.start_system_time
                elif eng.state == "PAUSED":
                    elapsed = eng.pause_start_time - eng.start_system_time
                elif eng.state == "STOPPING":
                    elapsed = eng.stop_system_time - eng.start_system_time
                else:
                    elapsed = 0.0
                self._json({
                    "state": eng.state,
                    "elapsed_s": round(max(0.0, elapsed), 3),
                    "server_now_ms": int(time.time() * 1000),
                    "piezo_samples": eng.piezo_samples_recorded,
                    "audio_samples": eng.audio_samples_recorded,
                })

            elif self.path == "/audio_data":
                snap = eng.get_audio_buffer_snapshot()
                self._json({
                    "samples": snap,
                    "elapsed_s": round(
                        eng.audio_samples_recorded / float(eng.AUDIO_SAMPLE_RATE_HZ), 4
                    ),
                    "state": eng.state,
                })

            elif self.path == "/piezo_lines":
                snap = eng.get_terminal_lines_snapshot()
                self._json({
                    "lines": [{"text": t, "active": a} for t, a in snap],
                    "state": eng.state,
                })

            else:
                self.send_response(404)
                self._cors()
                self.end_headers()
        except Exception:
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data):
        body = _json.dumps(data, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)


def _start_http_sidecar(engine_instance, port: int = 7654) -> None:
    """Bind the sidecar to 0.0.0.0:<port> in a daemon thread."""
    _SwarCareAPIHandler.engine_ref = engine_instance
    try:
        server = HTTPServer(("0.0.0.0", port), _SwarCareAPIHandler)
        threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="swarcare-http-sidecar"
        ).start()
    except Exception:
        pass


# ===========================================================================
# MAIN ENGINE
# ===========================================================================

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

        # --- THREAD-ISOLATED ACCELERATED QUEUES & LOCKS ---
        self.piezo_queue = queue.Queue(maxsize=5000)
        self.piezo_file_lock = threading.Lock()
        self.audio_file_lock = threading.Lock()
        self.audio_buffer_lock = threading.Lock()
        self.terminal_lock = threading.Lock()

        # NOTE: model.py's VIBRATION_SAMPLE_RATE_HZ (used at analysis time) must
        # match whatever rate the vibration K-means model was actually trained
        # at. If you change PIEZO_SAMPLE_RATE_HZ here, re-check model.py's
        # calibration notes before trusting vibration anomaly verdicts.
        self.PIEZO_SAMPLE_RATE_HZ = 2000
        self.AUDIO_SAMPLE_RATE_HZ = 16000

        # Optimized Storage Targets
        self.piezo_csv = None
        self.audio_raw_file = None
        self.audio_tmp_path = ""

        self.piezo_samples_recorded = 0
        self.audio_samples_recorded = 0
        self.audio_bytes_written = 0
        self._audio_unpack_carry = b""

        # Absolute Phase Synchronization Epoch Anchors
        self.start_system_time = 0.0
        self.start_piezo_us = None
        self.pause_start_time = 0.0
        self.stop_system_time = 0.0
        self.current_prefix = ""

        # High-Speed Telemetry Deque Rings for Fluid Visualizer Updates
        self.piezo_terminal_lines = deque(maxlen=14)
        self.AUDIO_WINDOW_SEC = 4.0
        self.AUDIO_EFFECTIVE_DISPLAY_RATE_HZ = 500
        self.audio_live_buffer = deque(
            maxlen=int(self.AUDIO_WINDOW_SEC * self.AUDIO_EFFECTIVE_DISPLAY_RATE_HZ)
        )

        # --- LIVE AI-ANALYSIS ROLLING WINDOWS ---
        # Separate from the buffers above (those are decimated/rescaled for
        # the waveform *display* only, not suitable for inference). These
        # hold the last few seconds of FULL-RESOLUTION raw samples so
        # analyze_recording_ai() can score a bounded, recent window instead
        # of re-reading the entire (ever-growing) recording on every poll.
        self.LIVE_ANALYSIS_WINDOW_SEC = 3.0
        self.piezo_raw_window = deque(
            maxlen=int(self.LIVE_ANALYSIS_WINDOW_SEC * self.PIEZO_SAMPLE_RATE_HZ)
        )
        self.audio_raw_window = deque(
            maxlen=int(self.LIVE_ANALYSIS_WINDOW_SEC * self.AUDIO_SAMPLE_RATE_HZ)
        )
        self.piezo_raw_window_lock = threading.Lock()
        self.audio_raw_window_lock = threading.Lock()

        self.recordings_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "recordings")
        )
        self.DATA_DIR = self.recordings_dir
        os.makedirs(self.recordings_dir, exist_ok=True)

        self.audio_process = None

        # Start persistent hardware serial processor thread immediately
        threading.Thread(target=self._process_piezo_stream, daemon=True).start()

        Bridge.provide("piezo_batch", self.handle_piezo_packet)
        threading.Thread(target=App.run, daemon=True).start()

        # Start HTTP sidecar
        _start_http_sidecar(self)

        self._initialized = True

    # -----------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -----------------------------------------------------------------------

    def handle_piezo_packet(self, payload):
        try:
            self.piezo_queue.put_nowait(bytes(payload))
        except queue.Full:
            pass

    def _determine_audio_device(self):
        try:
            result = subprocess.run(
                ["arecord", "-l"], capture_output=True, text=True, check=True
            )
            for line in result.stdout.splitlines():
                if "card" in line and "USB" in line:
                    card_num = line.split(":")[0].split("card")[1].strip()
                    return f"plughw:{card_num},0"
        except Exception:
            pass
        return "plug:default"

    def _determine_audio_command(self, target_hw):
        """Constructs ultra-low-latency 20ms period ALSA stream command."""
        base_cmd = [
            "arecord", "-D", target_hw, "-f", "S16_LE",
            "-r", str(self.AUDIO_SAMPLE_RATE_HZ), "-c", "1", "-t", "raw",
            "--period-size=320",   # 320 samples = 20ms low latency hardware period
            "--buffer-size=1280",  # 80ms buffer depth to prevent underruns
            "--avail-min=320"      # Wake reader instantly every 20ms
        ]
        # Bypass Linux OS stdout pipe block buffering if stdbuf is present
        try:
            subprocess.run(
                ["stdbuf", "--version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            return ["stdbuf", "-o0", "-e0"] + base_cmd
        except Exception:
            return base_cmd

    def get_audio_buffer_snapshot(self):
        """Returns the current ~4-second rolling audio window. Non-destructive."""
        with self.audio_buffer_lock:
            return list(self.audio_live_buffer)

    def get_terminal_lines_snapshot(self):
        with self.terminal_lock:
            return list(self.piezo_terminal_lines)

    def get_recent_piezo_raw_window(self):
        """Last ~LIVE_ANALYSIS_WINDOW_SEC seconds of raw (unscaled) ADC ints."""
        with self.piezo_raw_window_lock:
            return list(self.piezo_raw_window)

    def get_recent_audio_raw_window(self):
        """Last ~LIVE_ANALYSIS_WINDOW_SEC seconds of raw int16 PCM audio samples."""
        with self.audio_raw_window_lock:
            return list(self.audio_raw_window)

    # -----------------------------------------------------------------------
    # RECORDING LIFECYCLE
    # -----------------------------------------------------------------------

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
                subprocess.run(
                    ["pkill", "-9", "arecord"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
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
            self.audio_bytes_written = 0
            self._audio_unpack_carry = b""
            self.start_piezo_us = None

            with self.piezo_file_lock:
                self.piezo_csv = open(
                    os.path.join(self.recordings_dir, f"{self.current_prefix}_piezo.csv"),
                    "w", newline="", encoding="utf-8", buffering=65536
                )
                self.piezo_csv.write("sample_index,real_time_s,raw_adc,amplitude,voltage\n")

            with self.audio_file_lock:
                self.audio_tmp_path = os.path.join(
                    self.recordings_dir, f"{self.current_prefix}_audio.tmp"
                )
                self.audio_raw_file = open(self.audio_tmp_path, "wb", buffering=65536)

            with self.terminal_lock:
                self.piezo_terminal_lines.clear()
            with self.audio_buffer_lock:
                self.audio_live_buffer.clear()
            with self.piezo_raw_window_lock:
                self.piezo_raw_window.clear()
            with self.audio_raw_window_lock:
                self.audio_raw_window.clear()

            while not self.piezo_queue.empty():
                try:
                    self.piezo_queue.get_nowait()
                except queue.Empty:
                    break

            target_hw = self._determine_audio_device()
            audio_cmd = self._determine_audio_command(target_hw)
            self.audio_process = subprocess.Popen(
                audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

            self.start_system_time = time.time()
            self.stop_system_time = 0.0
            self.state = "RECORDING"

            threading.Thread(
                target=self._audio_capture_worker,
                args=(self.audio_process,),
                daemon=True
            ).start()

    def pause_recording(self):
        with self.state_lock:
            if self.state == "RECORDING":
                self.state = "PAUSED"
                self.pause_start_time = time.time()

    def stop_recording(self):
        with self.state_lock:
            if self.state in ["RECORDING", "PAUSED"]:
                self.stop_system_time = time.time()
                self.state = "STOPPING"

                if self.audio_process:
                    try:
                        self.audio_process.terminate()
                        try:
                            if self.audio_process.stdout:
                                self.audio_process.stdout.read(65536)
                        except Exception:
                            pass
                        self.audio_process.wait(timeout=1.0)
                    except Exception:
                        pass
                    self.audio_process = None

                threading.Thread(
                    target=self._safe_single_threaded_cleanup, daemon=True
                ).start()

    # -----------------------------------------------------------------------
    # WORKER THREADS
    # -----------------------------------------------------------------------

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
                us_offset = (batch_start_us - self.start_piezo_us) / 1_000_000.0

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

                    ms = min(999, max(0, int((sample_time - sec_int) * 1000)))
                    t_str = f"{base_time_str}.{ms:03d}"

                    buffer_list.append(
                        f"{self.piezo_samples_recorded},{t_str},{val},{amp:.4f},{voltage:.4f}\n"
                    )

                    if self.piezo_samples_recorded % 160 == 0:
                        line_entry = (
                            f"TIME: {t_str} | ADC: {val:04d} | AMP: {amp:+.4f} | V: {voltage:.3f}"
                        )
                        with self.terminal_lock:
                            self.piezo_terminal_lines.append((line_entry, abs(amp) > 0.02))

                    self.piezo_samples_recorded += 1

                with self.piezo_raw_window_lock:
                    self.piezo_raw_window.extend(raw_values)

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
        # Read in 640-byte chunks (20ms at 16kHz int16) for instant, low-latency audio streaming
        CHUNK_SIZE = 640
        while self.state in ["RECORDING", "PAUSED"] and proc.poll() is None:
            try:
                raw_bytes = proc.stdout.read(CHUNK_SIZE)
                if not raw_bytes:
                    break

                if self.state != "RECORDING":
                    continue

                with self.audio_file_lock:
                    if self.audio_raw_file:
                        self.audio_raw_file.write(raw_bytes)
                        self.audio_bytes_written += len(raw_bytes)
                        self.audio_samples_recorded = self.audio_bytes_written // 2

                try:
                    combined = self._audio_unpack_carry + raw_bytes
                    total_samples = len(combined) // 2
                    self._audio_unpack_carry = combined[total_samples * 2:]
                    if total_samples == 0:
                        continue
                    usable_bytes = combined[:total_samples * 2]
                    audio_ints = struct.unpack(f"<{total_samples}h", usable_bytes)

                    with self.audio_raw_window_lock:
                        self.audio_raw_window.extend(audio_ints)

                    max_val = max(abs(v) for v in audio_ints) if audio_ints else 0
                    visual_multiplier = 1.0
                    if 0 < max_val < 6000:
                        visual_multiplier = 6.0
                    elif 6000 <= max_val < 15000:
                        visual_multiplier = 2.5

                    with self.audio_buffer_lock:
                        for val in audio_ints[::32]:
                            normalized = (val / 32768.0) * visual_multiplier
                            self.audio_live_buffer.append(max(-1.0, min(1.0, normalized)))
                except Exception:
                    pass
            except Exception:
                continue

    def _safe_single_threaded_cleanup(self):
        time.sleep(0.3)

        with self.piezo_file_lock, self.audio_file_lock:
            p_dur = self.piezo_samples_recorded / self.PIEZO_SAMPLE_RATE_HZ
            a_dur = self.audio_samples_recorded / self.AUDIO_SAMPLE_RATE_HZ

            final_piezo_count = self.piezo_samples_recorded
            final_audio_count = self.audio_samples_recorded

            if p_dur > 0 and a_dur > 0:
                synchronized_duration = min(p_dur, a_dur)
                final_piezo_count = int(synchronized_duration * self.PIEZO_SAMPLE_RATE_HZ)
                final_audio_count = int(synchronized_duration * self.AUDIO_SAMPLE_RATE_HZ)

            if self.piezo_csv:
                self.piezo_csv.flush()
                os.fsync(self.piezo_csv.fileno())
                self.piezo_csv.close()
                self.piezo_csv = None

                try:
                    p_csv_path = os.path.join(
                        self.recordings_dir, f"{self.current_prefix}_piezo.csv"
                    )
                    with open(p_csv_path, "r", encoding="utf-8") as f:
                        p_lines = f.readlines()
                    if len(p_lines) > final_piezo_count + 1:
                        with open(p_csv_path, "w", encoding="utf-8") as f:
                            f.writelines(p_lines[:final_piezo_count + 1])
                except Exception:
                    pass

            if self.audio_raw_file:
                self.audio_raw_file.flush()
                os.fsync(self.audio_raw_file.fileno())
                self.audio_raw_file.close()
                self.audio_raw_file = None

                try:
                    final_wav_path = os.path.join(
                        self.recordings_dir, f"{self.current_prefix}_audio.wav"
                    )
                    if os.path.exists(self.audio_tmp_path) and final_audio_count > 0:
                        with wave.open(final_wav_path, "wb") as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(self.AUDIO_SAMPLE_RATE_HZ)
                            with open(self.audio_tmp_path, "rb") as f_raw:
                                wav_file.writeframes(f_raw.read(final_audio_count * 2))

                    if os.path.exists(self.audio_tmp_path):
                        os.remove(self.audio_tmp_path)
                except Exception:
                    pass

            self.piezo_samples_recorded = final_piezo_count
            self.audio_samples_recorded = final_audio_count

        with self.state_lock:
            self.state = "STOPPED"

    # -----------------------------------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------------------------------

    def analyze_recording_ai(self, prefix):
        """Delegates evaluation to the dedicated model class in model.py.
        Returns a JSON-serializable dict: {status, score (0-1), confidence,
        std_dev, max_deviation, sample_count, piezo:{...}, audio:{...}}.

        While a recording is in progress, scores a bounded ~few-second
        rolling window from memory (fast, and the only way to see audio at
        all before the WAV is finalized). Once stopped, falls back to the
        file-based analysis of the completed recording."""
        if self.state in ("RECORDING", "PAUSED"):
            piezo_raw = self.get_recent_piezo_raw_window()
            audio_raw = self.get_recent_audio_raw_window()
            return AnomalyDetectionModel.evaluate_live(
                piezo_raw, self.PIEZO_SAMPLE_RATE_HZ,
                audio_raw, self.AUDIO_SAMPLE_RATE_HZ,
            )
        return AnomalyDetectionModel.evaluate(
            self.recordings_dir, prefix, piezo_sr=self.PIEZO_SAMPLE_RATE_HZ
        )

    def analyze_veena_ai(
        self,
        prefix: str,
        tonic_hz: float = 130.81,
        cents_threshold: float = 15.0,
        string_label: str = None,
    ) -> dict:
        """Parallel Veena diagnostic: tuning (Physics) + quality (ML) simultaneously.

        While recording is live, scores the in-memory audio window so results
        are available instantly without waiting for the WAV to be finalized.
        Once stopped, falls back to reading the completed WAV from disk.

        Args:
            prefix:           Recording prefix (e.g. 'Rec_001_20260816_182200').
            tonic_hz:         Sa (tonic) frequency in Hz. Default = C3 (130.81 Hz).
            cents_threshold:  Tuning tolerance in cents. Default ±15 cents.
            string_label:     Which string was plucked, e.g. 'S1'. None = auto-detect.

        Returns:
            dict with keys: status, is_healthy, tuning (physics result),
            quality (ML result), tonic_hz, string_label.
        """
        if self.state in ("RECORDING", "PAUSED"):
            audio_raw = self.get_recent_audio_raw_window()
            return VeenaDiagnosticModel.evaluate_live(
                audio_raw,
                self.AUDIO_SAMPLE_RATE_HZ,
                tonic_hz=tonic_hz,
                cents_threshold=cents_threshold,
                string_label=string_label,
            )
        return VeenaDiagnosticModel.evaluate(
            self.recordings_dir,
            prefix,
            tonic_hz=tonic_hz,
            cents_threshold=cents_threshold,
            string_label=string_label,
            audio_sr=self.AUDIO_SAMPLE_RATE_HZ,
        )