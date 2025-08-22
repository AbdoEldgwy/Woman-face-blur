'''
1. Detect persons in the image/video (YOLO or any detector).
2. For each person:
    |__ Crop their face region.
    |__ Run the Gender Classifier.
    |__ If woman → blur the whole person bounding box.
    |__ If man → keep as is.
3. Return/save the processed image/video.

'''

import cv2
import torch
from models.detector.person_detector import PersonDetector
from models.detector.face_detector import FaceDetector
from models.classifier import GenderClassifierModel
from utils.blurring import blur_box

class WomanBlurPipeline:
    def __init__(self):
        self.person_detector = PersonDetector("yolov8n.pt")
        self.face_detector = FaceDetector("yolov8n-face.pt")
        self.gender_classifier = GenderClassifierModel()
        

    def process_image(self, img_path):
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {img_path}")

        persons_bboxs = self.person_detector.detect(image)    # return list of person bounding boxes

        for bbox in persons_bboxs:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = image[y1:y2, x1:x2]

            face_box = self.face_detector.detect(person_crop)    # return list of face bounding boxes
            if not face_box:
                continue
            
            # Select the first face from the list
            fx1, fy1, fx2, fy2 = map(int, face_box[0])
            face_crop = person_crop[fy1:fy2, fx1:fx2]

            # Preprocess fro classifier
            face_resized = cv2.resize(face_crop,(224, 224))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255.0

            gender = self.gender_classifier.predict(face_tensor)

            if gender == "female":
                blur_box(image, bbox)

        return image
    
    def process_video(self, input_path, output_path, display=False):
        cap = cv2.VideoCapture(input_path)

        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            processed_frame = self.process_image(frame)

            out.write(processed_frame)

            if display:
                cv2.imshow("Processed", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()