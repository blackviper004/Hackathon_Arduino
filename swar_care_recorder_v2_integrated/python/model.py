import os
import struct
import time
import math
from datetime import timezone, timedelta
from dataclasses import dataclass

# --- TARGET SPECIFICATION: INDIAN STANDARD TIME (UTC +5:30) ---
IST = timezone(timedelta(hours=5, minutes=30))
PACKET_FORMAT = "<IQ40H" 
RAW_PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# --- ADC / VOLTAGE CONVERSION CONSTANTS ---
ADC_MAX_VALUE = 4095.0        # 12-bit ADC (0-4095)
ADC_REFERENCE_VOLTAGE = 3.3   # Reference voltage of the ADC

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


@dataclass
class AnomalyReport:
    status: str
    score: float
    confidence: float
    std_dev: float
    max_deviation: float
    sample_count: int


class AnomalyDetectionModel:
    """Encapsulates the AI/Statistical anomaly detection logic for vibration streams."""
    
    @staticmethod
    def evaluate(recordings_dir: str, prefix: str) -> dict:
        p_csv_path = os.path.join(recordings_dir, f"{prefix}_piezo.csv")
        if not os.path.exists(p_csv_path):
            return {
                "status": "No Data", 
                "score": 0.0, 
                "confidence": 0.0, 
                "details": "CSV data stream missing."
            }
        
        voltages = []
        try:
            with open(p_csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 5:
                        voltages.append(float(parts[4]))
        except Exception:
            pass
        
        if not voltages:
            return {
                "status": "Normal", 
                "score": 0.0, 
                "confidence": 50.0, 
                "details": "No voltage samples parsed."
            }
        
        mean_v = sum(voltages) / len(voltages)
        variance = sum((v - mean_v) ** 2 for v in voltages) / len(voltages)
        std_dev = math.sqrt(variance)
        max_deviation = max(abs(v - mean_v) for v in voltages)
        
        # AI heuristic anomaly classification computation
        anomaly_score = min(1.0, (std_dev * 3.0) + (max_deviation * 0.25))
        is_anomaly = anomaly_score > 0.35
        
        status = "⚠️ ANOMALY DETECTED" if is_anomaly else "✅ NORMAL OPERATION"
        confidence = round(min(99.8, 75.0 + (anomaly_score * 24.0)), 1)
        
        report = AnomalyReport(
            status=status,
            score=round(anomaly_score, 3),
            confidence=confidence,
            std_dev=round(std_dev, 4),
            max_deviation=round(max_deviation, 4),
            sample_count=len(voltages)
        )
        
        return report.__dict__