"""
utils/preprocessing.py
Image preprocessing utilities for MobileNetV2 inference.
"""

import numpy as np


def preprocess_frame(frame_bgr: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Prepare a raw BGR OpenCV frame for MobileNetV2.

    Steps:
      1. Resize to (224, 224) — MobileNetV2 input resolution.
      2. Convert BGR → RGB (OpenCV uses BGR by default).
      3. Cast to float32 and apply MobileNetV2 preprocess_input scaling
         (scales pixel values from [0, 255] to [-1, 1]).
      4. Add batch dimension → shape (1, 224, 224, 3).

    Args:
        frame_bgr: Raw frame from cv2.VideoCapture in BGR colour order.
        target_size: (width, height) tuple for resizing.

    Returns:
        Preprocessed numpy array ready for model.predict().
    """
    import cv2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Resize
    resized = cv2.resize(frame_bgr, target_size)

    # BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Float + MobileNetV2 scaling
    arr = preprocess_input(rgb.astype(np.float32))

    # Add batch dim
    return np.expand_dims(arr, axis=0)


def preprocess_pil(pil_image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Prepare a PIL Image for MobileNetV2.
    Used for the 'Upload Image' fallback path.

    Args:
        pil_image: PIL.Image object (any mode; converted to RGB internally).
        target_size: (width, height) tuple for resizing.

    Returns:
        Preprocessed numpy array ready for model.predict().
    """
    import numpy as np
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = pil_image.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)
