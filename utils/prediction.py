"""
utils/prediction.py
Model loading (MobileNetV2) and prediction logic.
The model is cached in Streamlit's session state so it loads only once.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model():
    """
    Load MobileNetV2 pretrained on ImageNet.
    Weights are downloaded automatically on first run (~14 MB) and cached.

    Returns:
        Compiled Keras Model ready for inference.
    """
    from tensorflow.keras.applications import MobileNetV2
    model = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, preprocessed_frame: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    Run inference on a single preprocessed frame.

    Args:
        model: Loaded Keras MobileNetV2 model.
        preprocessed_frame: Output of preprocess_frame() — shape (1, 224, 224, 3).
        top_k: Number of top predictions to return.

    Returns:
        List of dicts: [{"label": str, "confidence": float}, ...] sorted by confidence desc.
    """
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

    preds = model.predict(preprocessed_frame, verbose=0)
    decoded = decode_predictions(preds, top=top_k)[0]  # [(class_id, label, prob), ...]

    results = []
    for _, label, prob in decoded:
        results.append({
            "label": label.replace("_", " "),
            "confidence": float(prob),
        })
    return results


def top_prediction(model, preprocessed_frame: np.ndarray) -> Tuple[str, float]:
    """
    Convenience wrapper — returns only the single best (label, confidence) pair.
    """
    preds = predict(model, preprocessed_frame, top_k=1)
    if preds:
        return preds[0]["label"], preds[0]["confidence"]
    return "unknown", 0.0
