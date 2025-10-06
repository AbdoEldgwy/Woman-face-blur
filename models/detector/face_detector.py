# (WIDER FACE Pretrained for this model)

from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path="models/yolov8n-face.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, img):
        results = self.model(img, conf=self.conf_threshold)
        face_boxes = []

        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:
                    face_boxes.append(box.xyxy[0].tolist())

        return face_boxes
    