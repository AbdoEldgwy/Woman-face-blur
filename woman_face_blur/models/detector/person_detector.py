# (COCO Pretrained) -> for person detect 
# if class detected is 0 -> person class ✅

from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, img):
        results = self.model(img, conf=self.conf_threshold)
        person_boxes = []
        
        for result in results: 
            for box in result.boxes:
                if int(box.cls) == 0:
                    person_boxes.append(box.xyxy[0].tolist())
                    
        return person_boxes