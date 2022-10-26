# face-mesh

![image](https://user-images.githubusercontent.com/57317290/197903654-d48e03af-859f-4991-9e9f-8652c483f796.png)


## 목적
- 구글에서 제공하는 Face Mesh - mediapipe 활용해보기
- 머신러닝/딥러닝 API 서비스 만들어보기

## 개발 스택
- Windows 10
- VS Code
- Anaconda3-2022.10 (https://www.anaconda.com/products/distribution)
- Python 3.8.13

## 기능
[클라이언트]
- 카메라를 이용해 얼굴 중에서도 왼쪽 눈을 추적합니다.
- 눈을 깜박일 때를 트리거로 잡고 깜박일 때 얼굴 사진을 서버쪽으로 API 전송합니다.

[서버]
- 서버에서는 받은 사진을 성별, 나이 추론하는 기능을 API 로 제공
