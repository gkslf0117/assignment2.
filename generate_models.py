import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10

# 폴더 생성
os.makedirs('models', exist_ok=True)

# 데이터 로드 (5,000장 사용)
(x_train, y_train), _ = cifar10.load_data()
x_train = x_train[:5000].astype('float32') / 255.0
y_train = y_train[:5000]

print("=== Model 1 학습 시작 (Adam) ===")
model1 = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(x_train, y_train, epochs=1, batch_size=128)
model1.save('models/resnet50_cifar10_v1.h5')

print("\n=== Model 2 학습 시작 (RMSprop) ===")
model2 = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
model2.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=1, batch_size=128)
model2.save('models/resnet50_cifar10_v2.h5')

print("\n[성공] 모델 2개가 models/ 폴더에 저장됨")
