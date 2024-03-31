# Same_person_detection
2D이미지를 이용하여 동일인 감지 및 3D Reconstruction

## 이미지 Crop
YOLOv8의 yolov8n-face.pt을 사용하여 이미지에서 얼굴을 검출 후 Crop해 저장하였다

## 이미지 Embedding
crop된 이미지를 resnet을 사용하여 vecter화 했다


## 2D Landmark 추출
+ MTCNN 사용시 5개 포인트 추출
![image](https://github.com/cmsk6405/Same_person_detection/assets/97841700/87aed66e-9ab1-4219-a341-9db5a9085a8d)
+ FAN 사용시 68개의 포인트 추출
+ ![image](https://github.com/cmsk6405/Same_person_detection/assets/97841700/68f6a597-4ff0-48b4-ba25-db19e9e8cc00)


## 3D Reconstruction은 아래의 Git주소 활용하여 2D이미지를 3D로 구현 하였다
https://github.com/ascust/3DMM-Fitting-Pytorch?tab=readme-ov-file
