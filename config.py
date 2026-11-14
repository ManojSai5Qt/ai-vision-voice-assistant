"""
Configuration and Constants for AI Object Detection & Voice Assistant
VisionVoice AI - Hackathon Edition
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "test_assets"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# YOLO Detection Settings
DEFAULT_MODEL = "yolov8n.pt"  # Lightweight nano model for real-time edge performance
DEFAULT_CONFIDENCE = 0.50     # TC10: Filter uncertain predictions below 0.50
DEFAULT_IOU_THRESHOLD = 0.45  # Non-maximum suppression threshold
INPUT_FRAME_WIDTH = 640
INPUT_FRAME_HEIGHT = 480

# Common COCO Object Classes of Interest for Visually Impaired Assistance
HIGH_PRIORITY_CLASSES = {
    "person": 1,
    "car": 1,
    "motorcycle": 1,
    "bus": 1,
    "truck": 1,
    "traffic light": 1,
    "stop sign": 1,
    "bicycle": 2,
    "dog": 2,
    "chair": 2,
    "couch": 2,
    "dining table": 2,
    "bed": 2,
    "stairs": 1,
}

MEDIUM_PRIORITY_CLASSES = {
    "bottle": 3,
    "cup": 3,
    "cell phone": 3,
    "laptop": 3,
    "backpack": 3,
    "handbag": 3,
    "suitcase": 3,
    "door": 2,
    "tv": 3,
    "keyboard": 4,
    "mouse": 4,
    "book": 4,
}

# Voice & Announcement Settings
DEFAULT_SPEECH_RATE = 165     # Words per minute for pyttsx3 (clear & intelligible)
DEFAULT_COOLDOWN_SECONDS = 4.0  # TC08: Prevent repeated announcements of the same object
SPEECH_QUEUE_MAX_SIZE = 10

# Proximity & Distance Thresholds (Normalized bbox height/area relative to frame)
# Height ratio = bbox_height / frame_height
PROXIMITY_IMMEDIATE_RATIO = 0.55  # Object takes > 55% of vertical frame height (Very close, < 1 meter)
PROXIMITY_CLOSE_RATIO = 0.30      # 30% - 55% height (Nearby, 1 - 2 meters)
PROXIMITY_MODERATE_RATIO = 0.12   # 12% - 30% height (Moderate, 2 - 4 meters)
# Below 0.12 = Distant (TC04, > 4 meters)

# Low-Light Processing Settings (TC05)
LOW_LIGHT_LUMINANCE_THRESHOLD = 50.0  # Mean grayscale intensity threshold (0-255)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_GRID_SIZE = (8, 8)
GAMMA_CORRECTION_FACTOR = 1.6         # Lightens shadows

# Spatial Partitioning (Horizontal 3-zone split)
LEFT_ZONE_BOUNDARY = 0.35   # 0 to 0.35 = Left
RIGHT_ZONE_BOUNDARY = 0.65  # 0.65 to 1.0 = Right
# 0.35 to 0.65 = Directly ahead / Center
