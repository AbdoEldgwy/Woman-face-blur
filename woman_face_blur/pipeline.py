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
from models.genderClassifier import GenderClassifierModel
from utils.blurring import blur_box

class WomanBlurPipeline:
    def __init__(self):
        self.person_detector = PersonDetector()
        self.face_detector = FaceDetector()
        self.gender_classifier = GenderClassifierModel()

    def process_image(self, img_path, output_path, display=False):
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {img_path}")

        output_image = image.copy()
        persons_bboxs = self.person_detector.detect(image)
        print(f"Detected {len(persons_bboxs)} persons.")

        for bbox in persons_bboxs:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = image[y1:y2, x1:x2]
            face_bboxs = self.face_detector.detect(person_crop)
            
            if not face_bboxs:
                continue
            fx1, fy1, fx2, fy2 = map(int, face_bboxs[0])
            face_crop = image[y1+fy1:y1+fy2, x1+fx1:x1+fx2]

            gender_cls = self.gender_classifier.predict(face_crop)
            
            if gender_cls['label'] == "female":
                output_image = blur_box(output_image, bbox)
            print(f"Processed bbox: {bbox}, Gender: {gender_cls}")


        if display:
            cv2.imshow("Processed", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cv2.imwrite(output_path, output_image)
        return output_image

    def process_video(self, video_path, output_path, display=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Process each frame as an image (without saving to disk)
            output_frame = self._process_frame(frame)
            out.write(output_frame)
            if display:
                cv2.imshow("Processed", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def _process_frame(self, image):
        output_image = image.copy()
        persons_bboxs = self.person_detector.detect(image)
        for bbox in persons_bboxs:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = image[y1:y2, x1:x2]
            face_box = self.face_detector.detect(person_crop)
            if not face_box:
                continue
            fx1, fy1, fx2, fy2 = map(int, face_box[0])
            face_crop = person_crop[fy1:fy2, fx1:fx2]
            face_resized = cv2.resize(face_crop, (224, 224))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255.0
            gender = self.gender_classifier.predict(face_tensor)
            if gender == "female":
                output_image = blur_box(output_image, bbox)
        return output_image