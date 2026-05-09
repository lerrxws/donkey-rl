from src.detection.roi import extract_score_rois
from src.detection.preprocessing import preprocess_score_image, crop_to_content, normalize_digit
from src.detection.digits import predict_score_value
from src.detection.objects import detect_object
from src.detection.score import load_score_templates, read_score_counters