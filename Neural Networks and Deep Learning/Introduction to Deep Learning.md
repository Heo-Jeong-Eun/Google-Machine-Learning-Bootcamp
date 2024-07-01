# Introduction to Deep Learning

## What You’ll Learn

1. Neural Networks and Deep Learning
    - NN과 DL에 대해 배우고, NN을 만드는 방법과 데이터를 학습하는 방법에 대해 배울 것이다. → 고양이 이미지 인식기 제작
2. Improving Deep Neural Networks : Hyperparameter Tuning, Regularization and Optimization
    - DL의 실질적인 부분을 알아가고, 어떻게 잘 동작하게 하는지 배울 것이다.
3. Structuring your Machine Learning Project
    - ML을 어떻게 구조화 할 것인가에 대해 배우고 여러 가지 경험을 공유할 것이다.
4. Convolution Neural Networks
    - 주로 Image 인식 / 처리에 사용되는 CNN에 대해 배울 것이다.
5. Natural Language Processing : Building Sequence Models
    - Sequence Data 처리에 유용하게 사용되는 Recurrent Neural Network(RNN), LSTM Model에 대해 배울 것이다. → 자연어 처리나 음성 인식, 음악 생성에 사용된다.

## What is a Neural Networks

Deep Learning == Neural Network은 인간의 뇌와 신경 체계 수학적 Model을 의미한다. 

<img width = "700" height = "350" alt="스크린샷 2024-07-01 오후 3 33 14" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/e1a86e0f-6769-4674-9e8a-c64d8cfdd4a8">

> 주택 가격 예측 문제 <br>
> 6개의 집에 대한 Data, 집의 크기와 가격을 알고 있을 때, 집의 크기에 대해 집 값을 예측할 수 있는 함수를 도출한다. 

간단한 신경망으로 생각했을 때, Input X, size of house가 Neuron, Node(A Single Neuron)으로 들어가고 Output Y, price를 출력한다. 

<img width = "700" height = "350" alt="스크린샷 2024-07-01 오후 4 59 15" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/ad8f9803-0a2c-4b09-8472-ea4072ea3d9b">

Neuron을 여러 개 쌓아 더 큰 신경망 Model을 구현할 수 있고, 더 다양한 Feature를 사용한 신경망 Model은 위 사진과 같이 나타낼 수 있다. 

즉, **신경망은 Input X와 출력 Y를 Matching 해주는 함수를 찾는 과정이라고 할 수 있다.** 

## Supervised Learning with Neural Networks

ML에서 Supervised Learning은 정답 Label이 주어진 Data를 이용해 학습 시키는 방법이다. 

앞서 살펴본 주택 가격 예측 문제도 신경망을 사용해 지도학습을 구현할 수 있다. 

<img width = "700" height = "350" alt="스크린샷 2024-07-01 오후 3 45 13" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/fd2b6b68-68e8-4bef-bb76-6b949591fa64">

**분야에 따라 적용되는 신경망이 다르고 Data의 형태에 따라 Model 구현 방식이 다르다.** 

<img width = "700" height = "350" alt="스크린샷 2024-07-01 오후 3 46 25" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/ef436510-0721-4c64-aec8-bece58137915">

Image Data의 경우 CNN 합성곱 신경망, 음성 Data(1차원의 시계열 Data로 나타나는 Sequence)는 RNN을 사용한다. 

<img width = "700" height = "350" alt="스크린샷 2024-07-01 오후 3 50 35" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/205a98e7-b90a-42f6-aa0a-bff55d22d703">

신경망 Model은 SW가 비구조적 Data를 잘 학습하도록 발전해왔다. 

- **구조적 Data : 정보의 특성이 잘 정의된 Database로 표현된 Data**
- **비구조적 Data : Image, Audio와 같이 특징적인 값을 추출하기 어려운 형태의 Data**

## **Why Deep Learning Is Rising**

**Data 양의 증가, Computer 성능 향상, Algorithm 개선으로 성능이 향상 되었기 때문에 DL이 강력한 도구로 부상할 수 있었다.** 

<img width = "700" height = "350" alt="스크린샷 2024-07-01 오후 3 52 20" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/381a5daf-3827-4047-ae76-5e99a5cc7368">

> Large NN / Medium NN / Small NN / Traditional Learning Algo

**x축, Amount of Data = m**

- m Training Sample Size가 작을 때, Algorithm의 상대적 순위가 잘 정의되어 있지 않고, 구현 방법에 따라 성능이 결정 된다.
- Training Sample Size가 아주 클 때, 큰 신경망이 일관되게 다른 방법을 압도한다.

**Algorithm 개선, Sigmoid  → ReLU**

- 신경망 함수의 활성화 함수 Sigmoid 함수를 ReLU 함수로 바꾸면서 경사 하강법 Algorithm 작동에 큰 성과가 있었다.

<img width = "700" height = "250" alt="스크린샷_2024-07-01_오후_3 57 58" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/5b2b26e6-933c-4735-99f8-9745b216b428">

**Sigmoid 함수** 

- 왼쪽, 오른쪽 끝 부분으로 가면 미분 값이 0이 된다. 이는 경사 하강법(Gradient Descent)을 사용할 때 사용할 파라미터의 값을 아주 천천히 바뀌게 하고, 학습 속도가 급격히 저하되는 문제가 발생한다. → Gradient Vanising

**ReLU 함수**

- Input 값이 양수인 경우 미분 값이 1로 모두 동일하므로 0에 수렴할 가능성이 훨씬 적다.
- Rectified Linear Unit, Rectify는 0과 결과 중 큰 값을 취하라는 의미이다 → **max(0, x)** / 0과 x중 더 큰 값을 선택한다. 만일 0보다 작은 경우에는 0이 되는 구조

<img width = "700" height = "350" alt="스크린샷 2024-07-01 오후 4 10 47" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/99cb6495-bb3a-41bb-a8d4-1c6e2d37d672">

Sigmoid 함수에서 ReLU 변경하면서 Gradient Descent Algorithm을 더 빠르게 작동시키는 Algorithm을 만들었다. 

빠른 계산 속도가 중요한  이유는 Network를 Training 시키는 과정은 Idea → Code → Experiment의 순서로 반복적으로 이루어 지는데 Training이 길어지게 되면 Cycle이 진행되는 시간이 늘어나며 생산성에서 큰 차이를 가져오기 때문이다.

> The Lecture Notes referred to <a href = 'https://www.deeplearning.ai/'>DeepLearning.AI</a>