import cv2
from models.detector.person_detector import PersonDetector
from models.detector.face_detector import FaceDetector
from models.genderClassifier import GenderClassifierModel
from utils.blurring import blur_box

class WomanBlurPipeline:
    def __init__(self):
        self.person_detector = PersonDetector()
        self.face_detector = FaceDetector()
        self.gender_classifier = GenderClassifierModel()

    def _process_frame(self, frame):
        """
        Process a single BGR frame:
        - detect persons
        - detect face inside person region
        - classify gender from face
        - blur full person box if female
        """
        output = frame.copy()
        persons_bboxs = self.person_detector.detect(frame)

        for bbox in persons_bboxs:
            x1, y1, x2, y2 = map(int, bbox)
            # guard rails for out-of-bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]
            face_bboxs = self.face_detector.detect(person_crop)
            if not face_bboxs:
                continue

            fx1, fy1, fx2, fy2 = map(int, face_bboxs[0])
            # map face box back to full-frame coords
            gx1, gy1 = x1 + fx1, y1 + fy1
            gx2, gy2 = x1 + fx2, y1 + fy2

            gx1, gy1 = max(0, gx1), max(0, gy1)
            gx2, gy2 = min(w, gx2), min(h, gy2)
            if gx2 <= gx1 or gy2 <= gy1:
                continue

            face_crop = frame[gy1:gy2, gx1:gx2]
            gender_cls = self.gender_classifier.predict(face_crop)

            if gender_cls.get('label') == "female":
                output = blur_box(output, (x1, y1, x2, y2))
        return output

    def process_image(self, img_path, output_path, display=False):
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {img_path}")

        output_image = self._process_frame(image)

        if display:
            cv2.imshow("Processed Image", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(output_path, output_image)
        return output_image

    def process_video(self, video_path, output_path, display=False):
        """
        Read video, process frame by frame, save to output_path.
        Keeps input FPS and frame size when possible.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # MP4-friendly fourcc. Change to 'XVID' for .avi if needed.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = self._process_frame(frame)
            out.write(processed)

            if display:
                cv2.imshow("Processed Video", processed)
                # Press q to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

    def process_real_time_camera(self, camera_index=0, display=True):
        """
        Open a webcam stream and process frames in real time.
        Press q to exit.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Failed to open camera index: {camera_index}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = self._process_frame(frame)

            if display:
                cv2.imshow("Processed Camera", processed)
                # 1 ms delay so UI remains responsive. q to quit.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()
            
