"""
utils/prediction.py — PyTorch MobileNetV2
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def load_model():
    import torch
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.eval()
    return {"model": model, "weights": weights}


def predict(model_dict, preprocessed_frame: np.ndarray, top_k: int = 5) -> list[dict]:
    import torch
    model   = model_dict["model"]
    weights = model_dict["weights"]

    # preprocessed_frame is (1, 224, 224, 3) numpy float32 scaled to [-1,+1]
    # Convert to torch tensor (1, 3, 224, 224) in [0,1]
    arr = (preprocessed_frame[0] + 1.0) / 2.0          # [-1,1] → [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(tensor)

    probs     = torch.nn.functional.softmax(outputs[0], dim=0)
    top_probs, top_indices = torch.topk(probs, top_k)

    categories = weights.meta["categories"]
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = categories[idx].replace("_", " ").lower()
        results.append({"label": label, "confidence": float(prob)})
    return results


def top_prediction(model_dict, preprocessed_frame: np.ndarray) -> Tuple[str, float]:
    preds = predict(model_dict, preprocessed_frame, top_k=1)
    if preds:
        return preds[0]["label"], preds[0]["confidence"]
    return "unknown", 0.0
