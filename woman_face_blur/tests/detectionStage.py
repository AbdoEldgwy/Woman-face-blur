from models.detector.face_detector import FaceDetector
from models.genderClassifier import GenderClassifierModel
import cv2 as cv


def draw_predictions(image, predictions):
    """
    image: numpy array (BGR) from cv.imread
    predictions: list of tuples [(gender, [x1,y1,x2,y2]), ...]
    """
    for gender, bbox in predictions:
        x1, y1, x2, y2 = map(int, bbox)

        color = (255, 0, 0) if gender == "female" else (0, 255, 0)
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            image,gender,(x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

face_detector = FaceDetector()

img = cv.imread(r"data\test_images\woman-test.jpg")
if img is None:
    raise FileNotFoundError("Image not found or could not be loaded.")

persons = face_detector.detect(img)
classifier = GenderClassifierModel(weights_path='outputs/best_mobilenetv3_small_gender.pth')

person_bboxes = face_detector.detect(img)
print(f"Detected {len(person_bboxes)} persons.")

clsList = []
for person_bbox in person_bboxes:
    x1, y1, x2, y2 = map(int, person_bbox)
    person_classifier_result = classifier.predict(img[y1:y2, x1:x2])
    clsList.append((person_classifier_result['label'], person_bbox))

print(clsList)

output_img = draw_predictions(img, clsList)
cv.imshow("Image", img)
cv.waitKey(0)
