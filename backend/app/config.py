import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

DATABASE_URL = "sqlite:///./image_quality.db"

MIN_WIDTH = 1000
MIN_HEIGHT = 1000
BLUR_THRESHOLD = 100.0
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 200