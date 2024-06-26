# Same_person_detection
실내에서 여러 CCTV를 통한 한 사람의 동선등을 추적하기 위해서 2D이미지를 이용하여 동일인 감지 및 3D Reconstruction을 진행하였다.  

프로젝트의 구성은 아래의 이미지처럼 구성하였다.
![image](https://github.com/cmsk6405/Same_person_detection/assets/97841700/08408265-d8c4-4d54-9104-bd2f174110a7)


## 사용
2d_same_person_detection.ipynb를 참조하거나  
    `cd Same_person_detection/main`  
    `python main.py`  
을 이용하여 사용할수있다.  
내가 실험에 사용한 이미지 몇장을 sample로 올려놓았다.  
main.py를 이용할 경우 config.py를 통해 저장 경로를 설정해 놓았다.

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
내가 사용한 데이터의 경우 파일이름 앞의 숫자 4자리가 ID로써 사람을 구별하였다. 그래서 이걸 이용하여 같은 사람을 판별하여 merge했다.  
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71416  

## 비교
새로운 이미지들도 위 와 같은 과정을 거친뒤 미리 만들어 놓은 .npy파일들과 유클리드 거리를 비교하여 일정 수준 아래면 같은 사람이라고 판단한다.  
일치한다면 아래와 같이 결과가 나온다.  
![image](https://github.com/cmsk6405/Same_person_detection/assets/97841700/43df56a1-506e-42f0-9e3e-fc7d2101bab4)  

## 3D Reconstruction은 아래의 Git주소 활용하여 2D이미지를 3D로 구현 하였다
https://github.com/ascust/3DMM-Fitting-Pytorch?tab=readme-ov-file

내가 가진 이미지로 구현한 것  
![image](https://github.com/cmsk6405/Same_person_detection/assets/97841700/af5a37bb-801c-43fe-9d65-d5dbeb5780c7)  
이렇게 3D로 구현을 하면 mesh와 3d이미지가 생성되는데 이를 이용하여 Embedding을 하거나 face와 vertice의 Landmark를 이용하여 유클리드 거리를 측정하여 더욱 정확하게 동일인을 식별해낼 수 있다

