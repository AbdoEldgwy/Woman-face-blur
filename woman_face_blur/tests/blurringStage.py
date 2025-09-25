import cv2 as cv
from utils.blurring import blur_box

from models.detector.person_detector import PersonDetector
from models.detector.face_detector import FaceDetector

img = cv.imread(r"data\celeba\img_align_celeba\000001.jpg")
if img is None:
    raise FileNotFoundError("Image not found or path is incorrect.")

person_detector = PersonDetector()
face_detector = FaceDetector()

person_bboxes = person_detector.detect(img)
for person_bbox in person_bboxes:
    x1, y1, x2, y2 = map(int, person_bbox)
    img = blur_box(img, person_bbox, ksize=(99, 99))

# print("Face BBoxes:", face_bboxes)

cv.imshow("Image", img)
cv.waitKey(0)

'''
for fast way:
1) if image contains more than one person, 
2) then detect faces for each person,
3) then classificate women or mens 
4) then blur faces of women
'''