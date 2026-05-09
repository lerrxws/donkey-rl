from src.detection.roi import extract_score_rois
from src.detection.preprocessing import preprocess_score_image, crop_to_content, normalize_digit
from src.detection.digits import predict_score_value
from src.detection.objects import detect_object
from src.detection.score import load_score_templates, read_score_counters
from src.state import build_state


def detect_one(*args, color=None, **kwargs):
    return detect_object(*args, **kwargs)


__all__ = [
    "extract_score_rois",
    "preprocess_score_image",
    "crop_to_content",
    "normalize_digit",
    "predict_score_value",
    "detect_object",
    "detect_one",
    "load_score_templates",
    "read_score_counters",
    "build_state",
]