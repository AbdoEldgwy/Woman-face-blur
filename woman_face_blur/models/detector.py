from ultralytics import YOLO
import torch

class YOLOPersonDetector:
    def __init__(self,weights ='yolov8n.pt' , device='cuda', conf=.3):
        self.device = device
        self.model = YOLO(weights).to(self.device)
        self.model.conf = conf
        
    def detect(self,img):
        """
        image: BGR numpy image (H, W, C) or path
        returns: list of dicts: {'box': (x1,y1,x2,y2), 'conf': float}
        Only returns COCO class `person` (class id 0).
        """
        