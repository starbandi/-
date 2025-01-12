import os

# 드라이브에서 파일 고유번호를 통해 불러오고, 압축풀기
image_path = "C:/Users/minyu/Desktop/Demo/images"
annotations_path = "C:/Users/minyu/Desktop/Demo/annotations"

image_files = os.listdir(image_path)
annotation_files = os.listdir(annotations_path)


PATH = '.'
IMG_DIR = os.path.join(PATH, "images")
XML_DIR = os.path.join(PATH, "annotations")

xml_files = [os.path.join(XML_DIR, x) for x in os.listdir(XML_DIR) if x.endswith(".xml")]

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

#xml 파일에서 이미지 위치를 가져오기
def extract_info_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    labels = []
    filename = root.find('filename').text
    for boxes in root.iter('object'):
        label = boxes.find('name').text
        if label == "head" or label == "person":
          return filename, 0

    return filename, 1


## 이미지(images) 및 라벨(annotations) 파일 로드 ##
import cv2
import numpy as np

# 이미지 특징 추출하기
img_data = []
label_data = []

for xml_file in xml_files:
    
    # XML 파일 파싱하기
    filename, label = extract_info_from_xml(xml_file)

    # 이미지 파일 읽기 및 사이즈 조정
    img_file = os.path.join(IMG_DIR, filename)
    img = cv2.imread(img_file)
    label_data.append(label)
    resized_img = cv2.resize(img, (224, 224))
    img_data.append(resized_img)

img_data = np.array(img_data)
label_data = np.array(label_data)

len(img_data), len(label_data)


from collections import Counter

label_counter = Counter(label_data)
label_counter


import seaborn as sns

sns.countplot(x=label_data)
plt.title('Danger / Safe')
plt.show()

# 0 : Danger / 1 : Safe
labels_to_names = {0: 'DANGER', 1: 'SAFE'}
plt.figure(figsize=(16,16))
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.imshow(cv2.cvtColor(img_data[i], cv2.COLOR_BGR2RGB))
  label_name = labels_to_names[label_data[i]]
  plt.title(label_name)
  plt.axis('off')
plt.show()


## CNN 모델 설계 ##
from sklearn.model_selection import train_test_split

# keras를 사용해 CNN 구성
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential

# 레이어 쌓기
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Dropout - 오버피팅 방지 / Dense 1 - 1차원으로 정의
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 모델 가져오기
from tensorflow.keras.metrics import Recall, Precision, AUC

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        Recall(name='recall'),
        Precision(name='precision'),
        AUC(name='auc')])


# ModelCheckpoint 설정
from tensorflow.keras.callbacks import ModelCheckpoint

model_checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1)

# 데이터(train, test) 나누기
x_train, x_test, y_train, y_test = train_test_split(img_data, label_data, test_size=0.1, stratify=label_data, random_state=42)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 모델 학습
base_history = model.fit(x_train, y_train, batch_size=32, validation_split=0.2, epochs=20, callbacks=[model_checkpoint])


# base_history 시각화
plt.plot(base_history.history['loss'])
plt.plot(base_history.history['val_loss'])
plt.title('Loss')
plt.xticks(range(0, 21, 2)) # x축 눈금 지정
plt.ylim([0, 10]) # y축 범위 지정
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.legend(['train', 'val'], loc='upper left')
plt.show()


from tensorflow.keras.models import load_model

# 저장된 모델 불러오기
model = load_model('best_model.keras')

# 불러온 모델로부터 테스트 데이터에 대한 확률값을 예측합니다.
predicted_probabilities = model.predict(x_test)

# 확률값을 0 또는 1로 바꿀 임계값을 정의합니다.
threshold = 0.6

# 확률값을 예측 라벨(0 또는 1)로 바꿉니다.
y_pred = (predicted_probabilities > threshold).astype(int)


import seaborn as sns
from sklearn.metrics import confusion_matrix

# 혼동 행렬을 계산합니다.
cm = confusion_matrix(y_test, y_pred)

# seaborn의 heatmap을 사용하여 혼동 행렬을 시각화합니다.
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16},
            xticklabels=['DANGER', 'SAFE'], yticklabels=['DANGER', 'SAFE'])
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.title('Confusion Matrix')
plt.show()


## 새로운 이미지를 받아 헬멧 착용여부 예측 ##

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# 저장된 모델 불러오기
model = load_model('best_model.keras')

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # 모델의 입력 크기로 조정
    img_array = image.img_to_array(img)  # 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array /= 255.0
    return img_array

# 사용자로부터 이미지 파일 경로 입력 받기
img_path = input("이미지 파일 경로를 입력하세요: ")

# 이미지를 로드하고 전처리
img_array = load_and_preprocess_image(img_path)

# 이미지가 성공적으로 로드되었는지 확인
if img_array is None or img_array.size == 0:
    raise ValueError("이미지를 로드할 수 없습니다. 경로를 확인하세요.")

# 모델을 사용하여 예측 수행
predicted_probabilities = model.predict(img_array)

# 확률값을 0 또는 1로 바꿀 임계값을 정의
threshold = 0.5
y_pred = (predicted_probabilities > threshold).astype(int)

# 결과 출력
labels_to_names = {0: 'DANGER (No Helmet)', 1: 'SAFE (Wearing Helmet)'}
result = labels_to_names[y_pred[0][0]]

print(f"예측 결과: {result}")

# 예측 결과를 시각화
plt.imshow(img_array[0])
plt.title(f'Predicted: {result}')
plt.axis('off')
plt.show()