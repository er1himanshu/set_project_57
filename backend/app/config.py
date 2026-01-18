import os

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Database configuration
DATABASE_URL = "sqlite:///./image_quality.db"

# File upload constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# Quality thresholds
MIN_WIDTH = 1000
MIN_HEIGHT = 1000
BLUR_THRESHOLD = 100.0
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 200
MIN_SHARPNESS = 50.0  # Laplacian variance for edge detection

# Ecommerce standards
IDEAL_ASPECT_RATIOS = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]  # Common product image ratios
ASPECT_RATIO_TOLERANCE = 0.1
MIN_BACKGROUND_SCORE = 0.7  # For white/clean background detection (0-1 scale)
TEXT_CONFIDENCE_THRESHOLD = 0.5  # For watermark/text detection

# Image analysis constants
TEXT_DETECTION_LINE_RATIO = 0.005  # Ratio of organized line pixels indicating text (0.5%)
COLOR_SIMILARITY_THRESHOLD = 100  # Color distance threshold for description matching
MAX_PIXELS_FOR_COLOR_SAMPLING = 10000  # Maximum pixels to sample for color analysis

# CLIP configuration
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", None)  # Optional: path to fine-tuned model
CLIP_SIMILARITY_THRESHOLD = float(os.getenv("CLIP_SIMILARITY_THRESHOLD", "0.25"))  # Threshold for mismatch detection
CLIP_BATCH_SIZE = int(os.getenv("CLIP_BATCH_SIZE", "8"))  # Batch size for inference

# CLIP training configuration
CLIP_TRAIN_BATCH_SIZE = int(os.getenv("CLIP_TRAIN_BATCH_SIZE", "16"))
CLIP_TRAIN_LEARNING_RATE = float(os.getenv("CLIP_TRAIN_LEARNING_RATE", "5e-6"))
CLIP_TRAIN_EPOCHS = int(os.getenv("CLIP_TRAIN_EPOCHS", "3"))
CLIP_CHECKPOINT_DIR = os.getenv("CLIP_CHECKPOINT_DIR", os.path.join(BASE_DIR, "clip_checkpoints"))

# Default product categories for class-based comparison
DEFAULT_CATEGORIES = [
    "shoes", "sneakers", "boots", "sandals",
    "shirt", "t-shirt", "blouse", "dress", "pants", "jeans", "shorts", "skirt",
    "jacket", "coat", "sweater", "hoodie",
    "handbag", "backpack", "purse", "wallet",
    "watch", "sunglasses", "jewelry", "necklace", "bracelet", "earrings",
    "hat", "cap", "scarf", "gloves",
    "electronics", "phone", "laptop", "tablet", "camera",
    "furniture", "chair", "table", "sofa", "bed",
    "toy", "book", "cosmetics", "perfume"
]