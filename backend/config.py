"""Application settings. Thresholds are application-level decision thresholds
(NOT scientifically universal) and can be calibrated via env vars."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "backend" / "models"
MODEL_PATH = MODEL_DIR / "phishing_model.joblib"
ONNX_PATH = MODEL_DIR / "phishing_model.onnx"
METADATA_PATH = MODEL_DIR / "metadata.json"
DASHBOARD_DIR = ROOT / "dashboard"
REPORTS_DIR = ROOT / "reports"

SUSPICIOUS_THRESHOLD = float(os.getenv("SNAPSHIELD_SUSPICIOUS_T", "0.35"))
PHISHING_THRESHOLD = float(os.getenv("SNAPSHIELD_PHISHING_T", "0.70"))
USE_ONNX = os.getenv("SNAPSHIELD_USE_ONNX", "0") == "1"
MAX_BODY_BYTES = 4096
