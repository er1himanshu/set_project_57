import cv2
import numpy as np
from skimage import exposure
from ..config import MIN_WIDTH, MIN_HEIGHT, BLUR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS

def analyze_image(path: str):
    """
    Analyze image quality metrics including resolution, blur, brightness, and contrast.
    
    Args:
        path: File system path to the image file
        
    Returns:
        dict: Analysis results containing width, height, blur_score, brightness_score,
              contrast_score, passed (bool), and reason (str)
        None: If the image file cannot be read or is invalid
    """
    image = cv2.imread(path)
    if image is None:
        return None

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    brightness_score = float(np.mean(gray))
    contrast_score = float(np.std(gray))

    reasons = []
    passed = True

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        passed = False
        reasons.append("Resolution too low")

    if blur_score < BLUR_THRESHOLD:
        passed = False
        reasons.append("Image too blurry")

    if brightness_score < MIN_BRIGHTNESS or brightness_score > MAX_BRIGHTNESS:
        passed = False
        reasons.append("Poor lighting")

    reason = "OK" if passed else ", ".join(reasons)

    return {
        "width": width,
        "height": height,
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "passed": passed,
        "reason": reason
    }