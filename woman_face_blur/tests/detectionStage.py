from models.detector.person_detector import PersonDetector
from models.detector.face_detector import FaceDetector
import cv2

person_detector = PersonDetector()
face_detector = FaceDetector()

image = cv2.imread(r"data\celeba\img_align_celeba\000001.jpg")
if image is None:
    raise FileNotFoundError("Image 'test_image.jpg' not found or could not be loaded.")

persons = person_detector.detect(image)
faces = face_detector.detect(image)

print(persons)
# print(faces)

# for each person, crop and detect face
for person in persons:
    x_min, y_min, x_max, y_max = map(int, person)  # Convert floats to ints
    person_image = image[y_min:y_max, x_min:x_max] # Cropping
    person_faces = face_detector.detect(person_image)

    print(person_faces)