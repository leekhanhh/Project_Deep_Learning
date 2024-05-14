import os
import cv2
import glob
import argparse
from utils import alignment_procedure
from mtcnn import MTCNN


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-o", "--save", type=str, default='Norm',
                help="path to save dir")


args = vars(ap.parse_args())
dir_path = args["dataset"]
path = args['save']

Flage = True
detector = MTCNN()

list_dir = []
list_save = []
list_update = []
if os.path.exists(path):
    list_save = os.listdir(path)
    list_dir = os.listdir(dir_path)
    list_update =  list(set(list_dir)^set(list_save))
else:
    os.makedirs(path)

if len(list_update) == 0:
    if len(list_dir) == 0 and len(list_save) == 0:
        class_list = os.listdir(dir_path)
    else:
        if (set(list_dir) == set(list_save)):
            Flage = False
        else:
            Flage = True
else:
    class_list = list_update


if Flage:
    class_list = sorted(class_list)
    for name in class_list:
        img_list = glob.glob(os.path.join(dir_path, name) + '/*')
        
        # Create Save Folder
        save_folder = os.path.join(path, name)
        os.makedirs(save_folder, exist_ok=True)
        
        for img_path in img_list:
            img = cv2.imread(img_path)

            detections = detector.detect_faces(img)
            
            if len(detections)>0:
                right_eye = detections[0]['keypoints']['right_eye']
                left_eye = detections[0]['keypoints']['left_eye']
                bbox = detections[0]['box']
                norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)

                # Save Norm ROI
                cv2.imwrite(f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
                print(f'[INFO] Successfully normalised {img_path}')

            else:
                print(f'[INFO] Not detected eyes in {img_path}')

        print(f'[INFO] Successfully normalised all images from {len(os.listdir(path))} classes\n')
    print(f"[INFO] All normalised images saved in '{path}'")

else:
    print('[INFO] Already normalized all data..')
