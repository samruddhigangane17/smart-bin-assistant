"""
utils/prediction.py
Uses tflite-runtime instead of full TensorFlow for cloud deployment.
"""
from __future__ import annotations
import numpy as np
import urllib.request
import os
import json
from typing import Tuple


MODEL_PATH = "/tmp/mobilenet_v2.tflite"
LABELS_PATH = "/tmp/imagenet_labels.txt"

MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224_quant_and_labels.zip"
TFLITE_URL = "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/2.tflite"
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"


def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(TFLITE_URL, MODEL_PATH)
    if not os.path.exists(LABELS_PATH):
        urllib.request.urlretrieve(LABELS_URL, LABELS_PATH)
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    with open(LABELS_PATH) as f:
        labels = [line.strip() for line in f.readlines()]
    return {"interpreter": interpreter, "labels": labels}


def predict(model, preprocessed_frame: np.ndarray, top_k: int = 5) -> list[dict]:
    interpreter = model["interpreter"]
    labels = model["labels"]
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # tflite quant model expects uint8
    inp = preprocessed_frame.copy()
    if input_details[0]['dtype'] == np.uint8:
        inp = ((inp + 1.0) / 2.0 * 255).astype(np.uint8)
    else:
        inp = inp.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    if output.dtype == np.uint8:
        output = output.astype(np.float32) / 255.0

    top_indices = np.argsort(output)[::-1][:top_k]
    results = []
    for i in top_indices:
        label = labels[i] if i < len(labels) else "unknown"
        results.append({"label": label.replace("_", " ").lower(), "confidence": float(output[i])})
    return results


def top_prediction(model, preprocessed_frame: np.ndarray) -> Tuple[str, float]:
    preds = predict(model, preprocessed_frame, top_k=1)
    if preds:
        return preds[0]["label"], preds[0]["confidence"]
    return "unknown", 0.0
