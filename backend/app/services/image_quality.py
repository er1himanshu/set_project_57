import cv2
import numpy as np
from skimage import exposure
from ..config import (
    MIN_WIDTH, MIN_HEIGHT, BLUR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    MIN_SHARPNESS, IDEAL_ASPECT_RATIOS, ASPECT_RATIO_TOLERANCE, MIN_BACKGROUND_SCORE
)

def analyze_image(path: str, description: str = None):
    """
    Analyze image quality metrics for ecommerce product listing.
    
    Args:
        path: File system path to the image file
        description: Optional product description for consistency check
        
    Returns:
        dict: Comprehensive analysis results including all ecommerce metrics
        None: If the image file cannot be read or is invalid
    """
    image = cv2.imread(path)
    if image is None:
        return None

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Basic metrics
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_score = float(np.mean(gray))
    contrast_score = float(np.std(gray))
    
    # Ecommerce-specific metrics
    aspect_ratio = width / height if height > 0 else 0
    sharpness_score = calculate_sharpness(gray)
    background_score = assess_background(image)
    has_watermark = detect_text_watermark(gray)
    
    # Description consistency
    description_consistency = check_description_consistency(description, image, path)
    
    # Quality checks
    reasons = []
    suggestions = []
    passed = True

    # Resolution check
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        passed = False
        reasons.append("Resolution too low")
        suggestions.append(f"Increase resolution to at least {MIN_WIDTH}x{MIN_HEIGHT} pixels for better quality")

    # Blur check
    if blur_score < BLUR_THRESHOLD:
        passed = False
        reasons.append("Image too blurry")
        suggestions.append("Use a tripod or better focus to reduce blur")

    # Brightness check
    if brightness_score < MIN_BRIGHTNESS:
        passed = False
        reasons.append("Image too dark")
        suggestions.append("Improve lighting or increase exposure")
    elif brightness_score > MAX_BRIGHTNESS:
        passed = False
        reasons.append("Image too bright")
        suggestions.append("Reduce lighting or decrease exposure to avoid overexposure")

    # Sharpness check
    if sharpness_score < MIN_SHARPNESS:
        passed = False
        reasons.append("Insufficient sharpness")
        suggestions.append("Ensure proper focus and use higher quality camera settings")

    # Aspect ratio check
    if not is_good_aspect_ratio(aspect_ratio):
        reasons.append("Non-standard aspect ratio")
        suggestions.append("Use standard aspect ratios like 1:1, 4:3, or 16:9 for better product display")

    # Background check
    if background_score < MIN_BACKGROUND_SCORE:
        reasons.append("Background not clean/white")
        suggestions.append("Use a plain white or neutral background to highlight the product")

    # Watermark check
    if has_watermark:
        reasons.append("Text or watermark detected")
        suggestions.append("Remove watermarks and text overlays for cleaner product images")

    # Description consistency
    if description and description_consistency != "Consistent":
        reasons.append(f"Description mismatch: {description_consistency}")
        suggestions.append("Ensure product description accurately matches the image content")

    reason = "OK" if passed else ", ".join(reasons)
    improvement_suggestions = "; ".join(suggestions) if suggestions else "Image meets quality standards"

    return {
        "width": width,
        "height": height,
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "passed": passed,
        "reason": reason,
        "aspect_ratio": aspect_ratio,
        "sharpness_score": sharpness_score,
        "background_score": background_score,
        "has_watermark": has_watermark,
        "description_consistency": description_consistency,
        "improvement_suggestions": improvement_suggestions
    }


def calculate_sharpness(gray_image):
    """Calculate image sharpness using Laplacian variance (edge detection)."""
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = laplacian.var()
    return float(sharpness)


def assess_background(image):
    """
    Assess if the image has a clean/white background.
    Returns a score between 0 and 1, where 1 is perfect white background.
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get the border pixels (likely background)
    border_thickness = 20
    h, w = image.shape[:2]
    
    # Extract border regions
    top_border = image[0:border_thickness, :]
    bottom_border = image[h-border_thickness:h, :]
    left_border = image[:, 0:border_thickness]
    right_border = image[:, w-border_thickness:w]
    
    # Combine all borders
    borders = np.concatenate([
        top_border.reshape(-1, 3),
        bottom_border.reshape(-1, 3),
        left_border.reshape(-1, 3),
        right_border.reshape(-1, 3)
    ])
    
    # Calculate mean brightness of borders
    border_brightness = np.mean(borders)
    
    # Calculate standard deviation (uniformity)
    border_std = np.std(borders)
    
    # White background: high brightness, low variation
    # Normalize to 0-1 scale
    brightness_score = min(border_brightness / 255.0, 1.0)
    uniformity_score = max(1.0 - (border_std / 128.0), 0.0)
    
    # Combined score
    background_score = (brightness_score * 0.7 + uniformity_score * 0.3)
    
    return float(background_score)


def detect_text_watermark(gray_image):
    """
    Detect if image contains text or watermarks using edge detection patterns.
    Returns True if text-like patterns are detected.
    """
    # Apply edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Look for horizontal and vertical line patterns typical of text
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    
    # Count significant line pixels
    h_line_pixels = np.sum(horizontal_lines > 0)
    v_line_pixels = np.sum(vertical_lines > 0)
    
    # Heuristic: if we have many organized lines, likely text
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    line_ratio = (h_line_pixels + v_line_pixels) / total_pixels
    
    # If more than 0.5% of pixels are organized lines, consider it text
    return line_ratio > 0.005


def is_good_aspect_ratio(aspect_ratio):
    """Check if aspect ratio is close to standard ecommerce ratios."""
    for ideal_w, ideal_h in IDEAL_ASPECT_RATIOS:
        ideal_ratio = ideal_w / ideal_h
        if abs(aspect_ratio - ideal_ratio) / ideal_ratio <= ASPECT_RATIO_TOLERANCE:
            return True
    return False


def check_description_consistency(description, image, image_path):
    """
    Check consistency between product description and image content.
    Uses a deterministic heuristic approach.
    
    Returns: "Consistent", "Inconsistent", or "No description provided"
    """
    if not description or description.strip() == "":
        return "No description provided"
    
    description_lower = description.lower()
    
    # Analyze image colors
    colors = analyze_dominant_colors(image)
    
    # Simple keyword matching heuristic
    color_keywords = {
        'red': (0, 0, 200),
        'blue': (200, 0, 0),
        'green': (0, 200, 0),
        'yellow': (0, 200, 200),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
        'orange': (0, 165, 255),
        'purple': (128, 0, 128),
        'pink': (203, 192, 255),
        'brown': (42, 42, 165)
    }
    
    # Check if mentioned colors are present in image
    mentioned_colors = [color for color in color_keywords.keys() if color in description_lower]
    
    if mentioned_colors:
        # Check if at least one mentioned color is dominant in image
        color_match_found = False
        for color_name in mentioned_colors:
            target_color = color_keywords[color_name]
            for dominant_color in colors:
                # Calculate color similarity
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(dominant_color, target_color)))
                if distance < 100:  # Threshold for color similarity
                    color_match_found = True
                    break
            if color_match_found:
                break
        
        if not color_match_found:
            return "Color mismatch detected"
    
    # If description is very short (< 10 chars), consider it insufficient
    if len(description.strip()) < 10:
        return "Description too brief"
    
    return "Consistent"


def analyze_dominant_colors(image, k=3):
    """Extract dominant colors from image using k-means clustering."""
    # Reshape image to be a list of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # Sample pixels for performance (use up to 10000 pixels)
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]
    
    # Apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to regular Python lists
    dominant_colors = [tuple(map(int, color)) for color in centers]
    
    return dominant_colors