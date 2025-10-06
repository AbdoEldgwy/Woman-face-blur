import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn

"""
Usage:
    clf = GenderClassifierModel("best_model.pth", device="cuda")
    frame = cv2.imread("some.jpg")        # BGR np.ndarray
    pred = clf.predict(frame)             # {'label': 'male', 'confidence': 0.97, 'probs': [..]}
    
    # From a video stream:
    cap = cv2.VideoCapture("video.mp4")
    ok, frame = cap.read()
    if ok:
        pred = clf.predict(frame)
    
    # Batch:
    preds = clf.predict_batch([frame1, frame2, ...])
"""

class GenderClassifierModel:

    def __init__(self, weights_path='outputs/best_mobilenetv3_small_gender.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)

        # Must match your training config
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Build the exact same model head you trained
        self.model = models.mobilenet_v3_small(weights=None)
        in_feats = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_feats, 2)
        
        state = torch.load(weights_path, map_location=self.device)
        # If you saved a pure state_dict:
        if isinstance(state, dict) and "state_dict" not in state:
            self.model.load_state_dict(state)
        else:
            # handle checkpoints like {"state_dict": ...}
            self.model.load_state_dict(state["state_dict"])
            
        self.model.to(self.device)
        self.model.eval()

        # class names
        self.id2label = {0: "female", 1: "male"}

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            raise ValueError("Input is not a valid OpenCV BGR image.")

        # Some detectors may yield grayscale crops; enforce 3ch
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # BGR->RGB then PIL for torchvision
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        x = self.transform(pil_img)  # C,H,W
        return x

    @torch.inference_mode()
    def predict(self, img_bgr: np.ndarray) -> dict:
        x = self._preprocess(img_bgr).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            "label": self.id2label[idx],
            "confidence": float(probs[idx]),
            "probs": probs.tolist(),
        }

    @torch.inference_mode()
    def predict_batch(self, imgs_bgr: list[np.ndarray]) -> list[dict]:
        if len(imgs_bgr) == 0:
            return []
        tensors = torch.stack([self._preprocess_one(im) for im in imgs_bgr]).to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            logits = self.model(tensors)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()

        preds = []
        for p in probs:
            idx = int(np.argmax(p))
            preds.append({
                "label": self.id2label[idx],
                "confidence": float(p[idx]),
                "probs": p.tolist(),
            })
        return preds
