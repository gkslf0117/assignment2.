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

def calc_neuron_coverage(model, input_data, threshold=0.5):
    """
    DeepXplore - 뉴런 커버리지 계산
    특정 임계값 이상으로 활성화된 뉴런의 비율을 측정
    """
    # 마지막 레이어를 제외한 모든 중간 레이어의 출력 추출
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'dense' in layer.name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(input_data, verbose=0)
    
    total_neurons = 0
    activated_neurons = 0
    
    for layer_activation in activations:
        # 각 뉴런의 최대 활성화 값이 threshold를 넘었는지 확인
        max_activations = np.max(layer_activation, axis=0)
        activated_neurons += np.sum(max_activations > threshold)
        total_neurons += np.prod(max_activations.shape)
        
    return (activated_neurons / total_neurons) * 100

def visualize_and_save(image, pred1, pred2, index):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.title(f"Model1: {class_names[pred1]} | Model2: {class_names[pred2]}")
    plt.axis('off')
    plt.savefig(f'results/disagreement_{index}.png')
    plt.close()
