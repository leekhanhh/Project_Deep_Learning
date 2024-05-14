import numpy as np
import math
from PIL import Image
import deepface
from deepface import commons
import numpy as np

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# this function aligns given face in img based on left and right eye coordinates
def alignment_procedure(img, left_eye, right_eye, bbox):

    # Crop Face
    x, y, w, h = bbox
    img_roi = img[y:y+h, x:x+w]

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock

    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = euclidean_distance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0: 

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        # Image ROI
        img_roi = Image.fromarray(img_roi)
        img_roi = np.array(img_roi.rotate(direction * angle))

    # -----------------------

    return img_roi  # return img anyway
