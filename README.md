# -
2024데모데이

1. 이 프로젝트는 헬멧 착용 여부를 분류하는 CNN 기반 이미지 분석 애플리케이션입니다.
이미지와 주석 파일(XML)을 불러와 학습하고, 최적화된 모델로 새로운 이미지의 착용 여부를 예측합니다.

2. 
prediction.py 실행

images/ : 이미지 데이터 디렉토리
annotations/ : XML 형식의 주석 파일 디렉토리
best_model.keras : 학습 중 생성된 모델 파일 

3. python==3.12.8

4. tensorflow==2.13.0
keras==2.13.1
scikit-learn==1.2.0
numpy==1.23.5
pandas==1.5.3
matplotlib==3.6.2
seaborn==0.12.2
opencv-python==4.7.0.72
