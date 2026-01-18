import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

DATABASE_URL = "sqlite:///./image_quality.db"

# Existing thresholds
MIN_WIDTH = 1000
MIN_HEIGHT = 1000
BLUR_THRESHOLD = 100.0
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 200

# New ecommerce thresholds
MIN_SHARPNESS = 50.0  # Laplacian variance for edge detection
IDEAL_ASPECT_RATIOS = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]  # Common product image ratios
ASPECT_RATIO_TOLERANCE = 0.1
MIN_BACKGROUND_SCORE = 0.7  # For white/clean background detection (0-1 scale)
TEXT_CONFIDENCE_THRESHOLD = 0.5  # For watermark/text detection