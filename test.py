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


def main():
    # 데이터 및 모델 로드
    x_test, _ = load_cifar10_data()
    model1, model2 = load_models()
    
    # 1000장 테스트
    max_search = 1000 
    test_samples = x_test[:max_search]
    
    print(f"Running Differential Testing with Noise on {max_search} samples...")
    
    
    # 원본 이미지에 아주 작은 가우시안 노이즈를 섞음
    noise = np.random.normal(0, 0.05, test_samples.shape)
    noisy_samples = np.clip(test_samples + noise, 0, 1)
    
    # 2. 모델 예측 확률값 가져오기 (확률 차이 분석)
    print("Analyzing probabilistic differences between models...")
    prob1 = model1.predict(noisy_samples, verbose=1)
    prob2 = model2.predict(noisy_samples, verbose=1)
    
    # 최종 예측 클래스 
    preds1 = np.argmax(prob1, axis=1)
    preds2 = np.argmax(prob2, axis=1)
    
    # 두 모델의 소프트맥스 확률 분포 차이 계산 (L1 Distance)
    # 차이가 클수록 두 모델이 해당 이미지를 바라보는 시각이 다르다는 뜻입니다.
    diffs = np.sum(np.abs(prob1 - prob2), axis=1)
    
    # 차이가 큰 순서대로 인덱스 정렬
    top_diff_indices = np.argsort(diffs)[::-1]

    disagreements = []
    for i in range(len(test_samples)):
        idx = top_diff_indices[i]
        # 실제 예측 클래스가 다르거나, 확률 차이가 상위 5위 안에 드는 경우를 disagreement로 간주
        if preds1[idx] != preds2[idx] or i < 5:
            disagreements.append((noisy_samples[idx], preds1[idx], preds2[idx]))




