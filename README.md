# assignment2.

이 과제는 CIFAR-10 데이터셋과 ResNet50 모델을 사용하여 모델이 서로 불일치하게 예측하는 입력을 찾고, Neuron Coverage를 측정하는 것을 목표로 합니다. DeepXplore를 지금 환경에 맞게 수정하여 구현하였습니다.

설정 및 설치 방법 이 과제를 실행하기 위해서는 Python 3.10 이상의 버전이 필요합니다. 아래 명령어를 통해 필요한 라이브러리를 설치할 수 있습니다. pip install -r requirements.txt

실행 순서 1단계. 모델 생성 Differential Testing을 위한 다양성을 위해 서로 다른 옵티마이저(Adam, RMSprop)를 사용한 두 개의 ResNet50 모델을 학습하고 저장합니다. python generate_models.py

2단계. DeepXplore 테스트 실행 test.py를 실행하여 모델들의 불일치한 예측을 찾고 뉴런 커버리지를 계산합니다 python test.py

DeepXplore 수정 사항 원본 DeepXplore 프레임워크를 최신 라이브러리 제가 사용하는 환경에 맞게 다음과 같이 수정했습니다.

1)Keras 1.x 기반의 원본 코드를 TensorFlow 2.x 환경에서 동작하도록 재구현했습니다. 
2)원본의 데이터셋별 스크립트를 CIFAR-10 및 ResNet50에 최적화된 통합 test.py로 대체했습니다. 
3)현대적인 Keras 모델의 컨볼루션 레이어에서 활성화 값을 추출하여 커버리지를 계산하는 calc_neuron_coverage 함수를 사용했습니다. 
4)원본의 그라디언트 기반 최적화 방식 대신, 가우시안 노이즈(std=0.05)와 확률적 차이 분석을 결합하여 모델 간 불일치를 유발하는 입력을 효율적으로 탐색하도록 수정했습니다.

과제 구조 
1)generate_models.py: 두 개의 ResNet50 모델을 학습 
2)test.py: DeepXplore 테스팅을 수행 
3)results/: 불일치가 발견된 이미지들이 저장되는 폴더 
4)requirements.txt: 필요한 파이썬 라이브러리 목록 
5)report.pdf: Connecting Attacks and Testing 관한 에세이

실험 결과 노이즈가 포함된 입력에 대해 모델 1과 모델 2의 예측이 서로 다른 사례를 많이 발견했습니다. 
임계값 0.2를 기준으로 기본 테스트 대비 유의미한 뉴런 커버리지 상승을 확인했습니다.
epoch를 조절해보면서 성능을 향상시킬 수 있었습니다.

#Acknowledge
과제 구현 과정에서 OpenAI의 도움을 받았음 밝힙니다.
