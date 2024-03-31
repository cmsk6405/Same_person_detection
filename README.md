# Same_person_detection
2D이미지를 이용하여 동일인 감지 및 3D Reconstruction  

## 이미지 Crop
YOLOv8의 yolov8n-face.pt을 사용하여 이미지에서 얼굴을 검출 후 Crop해 저장하였다.  

## 이미지 Embedding
crop된 이미지를 resnet을 사용하여 vector화 했다.  

## 2D Landmark 추출
두 모델중 하나를 선택하여 랜드마크들간의 거리를 측정한다.  
나는 MTCNN으로 진행하였다.  
+ MTCNN 사용시 5개 포인트 추출
![image](https://github.com/cmsk6405/Same_person_detection/assets/97841700/87aed66e-9ab1-4219-a341-9db5a9085a8d)  
5개의 포인트뿐이므로 각 포인트간의 유클리드 거리(총 10개)를 측정한다.  

+ FAN 사용시 68개의 포인트 추출
  
![image](https://github.com/cmsk6405/Same_person_detection/assets/97841700/68f6a597-4ff0-48b4-ba25-db19e9e8cc00)  
임의의 포인트를 설정하여 유클리드 거리를 측정하여 저장한다.  
나는 아래의 포인트들을 지정하여 거리를 측정했다.  
  + EYEBROW_L = (17, 21)  
  + EYEBROW_R = (22, 26)  
  + BETWEEN_BROWS = (21, 22)  
  + EYE_L = (36, 39)  
  + EYE_R = (43, 45)  
  + NOSE_W = (31, 25)  
  + NOSE_L = (27, 33)  
  + LIP = (48, 54)  
  + FACE_W = (1, 15)  
  

## verctor merge
Embedding과 Landmark거리를 저장한 .npy파일들을 같은 사람끼리 모아 하나의 .npy파일로 저장한다.  

## 비교
새로운 이미지들도 위 와 같은 과정을 거친뒤 미리 만들어 놓은 .npy파일들과 유클리드 거리를 비교하여 일정 수준 아래면 같은 사람이라고 판단한다.  
일치한다면 아래와 같이 결과가 나온다.  
![image](https://github.com/cmsk6405/Same_person_detection/assets/97841700/dbc55df7-a280-40d8-a73f-d6706a8367e3)

## 3D Reconstruction은 아래의 Git주소 활용하여 2D이미지를 3D로 구현 하였다
https://github.com/ascust/3DMM-Fitting-Pytorch?tab=readme-ov-file
