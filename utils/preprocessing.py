"""
utils/preprocessing.py — works with PyTorch pipeline
"""
import numpy as np


def preprocess_frame(frame_bgr: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    import cv2
    resized = cv2.resize(frame_bgr, target_size)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr     = rgb.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)


def preprocess_pil(pil_image, target_size: tuple = (224, 224)) -> np.ndarray:
    img = pil_image.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)
