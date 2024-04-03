from ultralytics import YOLO
import cv2
import os
import numpy as np


def yolo_crop(folder_path, crop_output):

    model = YOLO("./yolov8n-face.pt")
    cant = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            img = cv2.imread(image_path)
            
            results = model.predict(img, show=False, batch=8)
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            
            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    crop_obj = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if not os.path.exists(crop_output):
                        os.makedirs(crop_output)
                    save_path = os.path.join(crop_output, file_name)
                    cv2.imwrite(save_path, crop_obj)
            else:
                cant.append(image_path)

    if cant is not None:
        if not os.path.exists("./cantlist"):
            os.makedirs("./cantlist")
        np.savetxt('./cantlist/cant.txt', cant, fmt='%s')
