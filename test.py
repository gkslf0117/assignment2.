import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# 결과 저장 폴더 생성
os.makedirs('results', exist_ok=True)

def load_cifar10_data():
    _, (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    return x_test, y_test

def load_models():
    print("Loading models...")
    # 앞서 만든 모델 경로 연결
    model1 = tf.keras.models.load_model('models/resnet50_cifar10_v1.h5')
    model2 = tf.keras.models.load_model('models/resnet50_cifar10_v2.h5')
    return model1, model2
