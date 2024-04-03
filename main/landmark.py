# MTCNN을 이용한 랜드마크

from facenet_pytorch import MTCNN
import face_alignment

from PIL import Image
import numpy as np
import cv2
import os
from skimage import io
from sklearn.metrics.pairwise import euclidean_distances


#MTCNN Landmark

def mtcnn_landmark(input_folder_path, save_folder_path):

    mtcnn = MTCNN(image_size=160, margin= 50)
    cant = []

    for root, _, files in os.walk(input_folder_path):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            img = Image.open(image_path)
            img = img.convert("RGB")

            _, probs, landmarks = mtcnn.detect(img, landmarks=True)

            if probs:
                landmarks_2d = landmarks.reshape(-1, 2)
                distances = euclidean_distances(landmarks_2d)
                embedding = distances[np.triu_indices(len(landmarks_2d), k=1)]
                embedding = np.array(embedding)
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                save_path = os.path.join(save_folder_path, file_name.replace('.png', '.npy'))
                np.save(save_path, embedding)
            else:
                cant.append(image_path)
    
    if cant is not None:
        if not os.path.exists("./cantlist"):
            os.makedirs("./cantlist")
        np.savetxt('./cantlist/cant_mtcnn_list.txt', cant, fmt='%s')


# FAN landmark
        
EYEBROW_L = (17, 21)
EYEBROW_R = (22, 26)
BETWEEN_BROWS = (21, 22)
EYE_L = (36, 39)
EYE_R = (43, 45)
NOSE_W = (31, 25)
NOSE_L = (27, 33)
LIP = (48, 54)
FACE_W = (1, 15)

def fan_lanmark(input_folder_path, save_folder_path):

    model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, face_detector='sfd')

    cant = []

    for root, _, files in os.walk(input_folder_path):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            input = io.imread(image_path)
            resized_image = cv2.resize(input, (160, 160))
            preds = model.get_landmarks(resized_image)

            if preds is not None:  # 랜드마크가 검출되었는지 확인
                distances = []
                # 원하는 랜드마크 쌍의 인덱스 지정
                landmark_indices = [EYEBROW_L, EYEBROW_R, BETWEEN_BROWS, EYE_L, EYE_R, NOSE_L, NOSE_W, LIP, FACE_W]
                for idx1, idx2 in landmark_indices:
                    # 두 지점 간의 거리 계산하여 distances 리스트에 추가
                    distance = euclidean_distances([preds[0][idx1]], [preds[0][idx2]])[0][0]
                    distances.append(distance)
                distances = np.array(distances)
                distances = distances.reshape(1, -1)
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                save_path = os.path.join(save_folder_path, file_name.replace('.png', '.npy'))
                np.save(save_path, distances)
            else:
                cant.append(file_name)

    if cant is not None:
        if not os.path.exists("./cantlist"):
            os.makedirs("./cantlist")
        np.savetxt('./cantlist/cant_fan_list.txt', cant, fmt='%s')

