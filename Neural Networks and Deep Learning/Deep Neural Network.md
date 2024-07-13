# Deep Neural Network

## Deep L-Layer Neural Network

![스크린샷 2024-07-11 오전 10 40 06](https://github.com/user-attachments/assets/1094a6e3-27bc-4685-b28d-d22d3228a1c3)

**DL에서 Layer의 갯수**를 말할 때에는 보통 **Input Layer를 제외한 Output Layer와 Hidden Layer 갯수만을 말한다.** 

Layer가 많이 쌓였다 해서 무조건 좋은 것은 아니지만 해결하고자 하는 현실의 복잡한 문제들은 **비선형 문제**이기 때문에 보통 **Layer가 깊을 수록 문제를 더 효과적으로 해결할 수 있다.** 

그렇다고 해서 Layer를 과도하게 많이 쌓게 되면 Training 시간이 너무 길어지거나 Overfitting 문제가 생길 수 있기 때문에 상황에 따라 적절한 Layer 갯수, 매개변수 설정이 중요하다. 

<img width="598" alt="스크린샷 2024-07-11 오전 10 44 08" src="https://github.com/user-attachments/assets/02933fc4-e115-4977-9620-6c014d06e2e6">

위 NN은 4개의 Layer로 구성되어 있으며 Input Data는 a[0]이다. 

이때 Input Layer는 따로 계산하지 않으며 오른쪽부터 1-4 Layer로 표현한다. 

Input Data x는 각각의 Perceptron에 입력될 때 각기 다른 가중치 w를 곱하고 각 Perceptron 마다 가지는 고유 값인 bias를 더해주고 이는 z로 표현되며 Activation Function g의 Input이 된다.

그리고 계산되어 나오는 Activation Function의 Ouput인 a[n]이 다음 Layer의 Input이 되는 방식으로 동작한다. 

각 층마다 위와 같은 과정을 거치며 최종적인 Output은 Y-hat이 된다. 

이것이 **NN Training에서 Forward Propagtion** 계산이다. 

## Forward Propagation in a Deep Network

NN Training 과정은 크게 2가지 단계로 나눌 수 있다. 

1. **Forward Propagation**
    - Input Data가 각각의 NN Layer를 지나며 Data에 대한 예상치인 Y-hat을 계산하는 과정이다.
2. **Backward Propagation**
    - 1. Forward Propagation 단계에서 산출된 Y-hat을 통해 나온 Loss 값을 통해 Loss를 최소화 하는 좋은 방법인 Gradient Descent를 사용해 매개변수 w, b를 수정하는 단계이다.

**DL의 Training 목적**은 **적절한 매개변수 w, b의 값을 찾아 수정함으로써 가장 낮은 Loss를 만드는 것이 목적**이다. → 가장 낮은 Loss는 DL Model의 예측이 비슷하다는 의미로 가장 정확하다는 의미이다. 

<img width="598" alt="스크린샷 2024-07-11 오전 11 15 30" src="https://github.com/user-attachments/assets/2030fdb0-4e63-4f85-8254-04a2abcb593d">

Forward Propagation을 수학적으로 표현하면 위와 같다. 

Vector를 이용해 계산하게 되면 NN을 훨씬 편리하게 나타낼 수 있다. 

예를 들어 Layer 1에 있는 5개의 Perceptron들을 Vector, 즉 행렬로 묶어 한번에 연산하게 되면 계산 과정이 간단해지고 실제 Training 시간도 단축된다. 

**Z[1] = W[1] * a[0] + b[1]**

- Z[1]은 Layer 1에서 가중치 w와 Input a의 곱 + bias 값을 나타내는데, 이는 Activation Function의 Input을 의미한다.
- W[1]은 Input Data-Layer 1 사이의 가중치 w를 모아둔 Vector, 즉 행렬이다.
- a[0]은 Input Data를 모아둔 Vector이다.
- b[1]은 Layer 1의 Perceptron들의 bias를 모아둔 Vector이다.
- A[1]은 Layer 1에서 Activation Function의 Output을 모아둔 Vector, 즉 행렬이다.

이 값은 a[1]으로, 다음 Layer인 Layer 2의 Input이 된다. 

이렇게 Layer를 거쳐 마지막에 산출되는 값인 a[4]가 Y-hat이 된다. 

## Getting your Matrix Dimensions Right

<img width="727" alt="스크린샷 2024-07-11 오전 11 32 43 (1)" src="https://github.com/user-attachments/assets/751b1d30-62c1-4843-acf4-9db2c5027ef7">

**Z[i] = W [i] * X + b**에서 Layer 1의 Perceptron 수가 3이고 Input Data는 2이기 때문에 가중치 W의 차원 수는 3(Layer 1의 Perceptron 갯수) * 2(Layer 0의 Perceptron 갯수)이다. 

즉, **행렬의 크기 == 차원 수**를 의미한다. 

Layer 0은 Input Data 종류를 의미하고 그 다음 Layer에서는 (Layer 2의 Perceptron 갯수) * (Layer 1의 Perceptron 갯수)이다. 

위 문제에서 Data 수를 1개라고 가정했기 때문에 X의 차원 수는 2 (X1, X2) * 1(Data 갯수)이다. 

Layer 2 계산에서는 X가 a[1]이 되기 때문에, Layer 1의 Perceptron 수 * Data 갯수이다. 

bias는 Perceptron마다 가지는 고유의 값이기 때문에 3(Perceptron 갯수) * 1(고정 값)이다. 

Z[i]는 W[i]와 X[i]의 행렬 곱셈 + b이기 때문에 행렬 곱셈 원리에 따라 3 * 1 즉, (Layer 1의 Perceptron 갯수) * (Input Data의 갯수)가 된다. 

a[i]의 차원 수 == Z[i]와 같고, 행렬 b의 덧셈을 처리할 때에는 BroadCasting이 사용되어 자동 연산된다. 

<img width="565" src="https://github.com/user-attachments/assets/4d2a04d2-2a83-4409-b3a2-f731ad5fa6ce">

Layer 1의 Perceptron 갯수는 3이였지만 다른 Layer에도 계산해야 하기 때문에 일반화를 통해 Layer를 i라 하고 각 Layer의 Perceptron 갯수를 n[i]라 하면 위와 같이 표현된다. 

<img width="565" alt="스크린샷 2024-07-11 오전 11 59 16" src="https://github.com/user-attachments/assets/01616051-0949-44fd-944e-b8c25dccf856">

Gradient Descent을 사용하기 위해 필요한 **매개변수 w, b의 도함수들의 차원 수**도 마찬가지로 알 수 있다. 

<img width="727" alt="스크린샷 2024-07-11 오전 11 32 43" src="https://github.com/user-attachments/assets/eea32e82-567e-48aa-8c7c-e32cf37a9c74">

<img width="727" alt="스크린샷 2024-07-11 오전 11 49 27" src="https://github.com/user-attachments/assets/18d34c22-ee49-4fd3-8a20-a59632f74cdb">

Input Data의 갯수를 m이라고 할 때 Z[i], a[i]의 차원 수는 n[i](입력되는 Perceptron 갯수) * m(Data 갯수)로 표현된다. 

도함수 또한 마찬가지로 바뀌게 된다. 

이때 **실제 Training에서 Update 되는 매개변수 W, b의 차원 수는 Data 수에 영향을 받지 않는다.** 

## Why Deep Representations ?

<img width="719" alt="스크린샷 2024-07-11 오후 1 25 36" src="https://github.com/user-attachments/assets/bff38210-6bb5-4c21-b0ab-debdaffa2635">

NN의 Layer가 깊어질수록, DL이 좀 더 복잡한 일을 수행할 수 있다.

<img width="725" alt="스크린샷 2024-07-11 오후 1 27 28" src="https://github.com/user-attachments/assets/f7a0601a-5bce-43ab-8303-61f9a45b0830">

DL의 Layer가 얕을수록 기하급수적으로 더 많은 계산을 수행해야 하기 때문에 NN의 Layer가 깊어지는 것이 좋다. 

또한 최근 DNN을 사용한 SW가 훌륭한 성능을 발휘한 경우가 많았기 때문에 DNN을 사용하는 사례가 늘어나게 되었다. 

## Building Blocks of Deep Neural Networks

NN 안에서 계산 과정은 **Forward Propagation → Loss → Backward Propagtion** 순서로 이루어진다. 

<img width="570" alt="스크린샷 2024-07-11 오후 1 32 46" src="https://github.com/user-attachments/assets/894662bf-bd02-43ea-bed5-72da44f032a6">

Forward Propagation 과정에서 산출된 Z[l]을 Backward Propagation에서도 사용한다. 

DL의 정확도를 높이기 위해 사용하는 **Gradient Descent에서 필요로 하는 값**은

1. **da[l]**
2. **Z[l]**

이다. 

da[l]은 Forward Propagation 이후 진행하는 Loss 계산 과정을 통해 따로 구하기 때문에 사실상 중요한 값은 Z[l]이다. 

따라서 **Cache**라는 새로운 변수로 정의해 **Z[l]의 값을 저장하고 기억**한다. 

<img width="600" alt="스크린샷 2024-07-11 오후 1 37 57" src="https://github.com/user-attachments/assets/2cd5064e-3948-4071-9905-fcb6d61a776b">

전체 과정은 위와 같다. **Loss Function 계산 과정에서 구한 미분 값 a[l]과 Activation Function의 Input인 Z[l]을 이용해 매개변수 w, b를 Update 해준다.** 

이를 각각의 NN Layer마다 진행해 최종적으로 W[1], b[1]을 갱신해주면 된다. 

매개변수 W와 b를 1번 Update 하는 것을 두고 Backward Propagation 과정을 1번 진행했다 라고 표현한다.  

## Forward and Backward Propagation

Backward Propagation 과정에서 이루어지는 수학적 계산은 아래와 같다. 

<img width="698" alt="스크린샷 2024-07-11 오후 1 41 54" src="https://github.com/user-attachments/assets/10801f9e-cce6-403e-a580-19f821d59077">

**da[l]를 입력했을 때 da[l - 1], dW[l], db[l]을 출력하는 과정**이다. 

Z[l]과 da[l]을 사용해 각각 Perceptron의 매개변수 W, b를 Update 해준다. 

이 과정을 반복해 Layer 1의 매개변수 W, b를 Update 하면 Backward Propagation의 과정이 끝나는 것이다. Update 과정은 아래 공식과 같다. 

<img width="681" alt="스크린샷 2024-07-11 오후 2 05 47" src="https://github.com/user-attachments/assets/0667866d-bf5c-4125-b52b-c4c09145093c">

**Back Propagation 과정에서 Layer L의 매개변수 W[l], b[l] 값을 Update 하는 방식은 도함수 값 dW[l], db[l]을 구해 Update 하는 방식**이다. 

새로운 W[l] = 원래 W[l] = dw[l]

이후 NN Layer에서도 Gradient Descent를 수행하기 위해 da[l - 1] 값을 구해줘야 한다. 

da[l - 1], dW[l], db[l] 값을 구하기 위해서는 dz[l]의 값이 꼭 필요하기 때문에 Forward Propagation에서 저장한 Z[l]과 Loss 계산 과정에서 구한 da[l]을 사용해 dz[l]을 구한다. 

da[l]은 Loss 계산 과정에서 알 수 있는 값 + Backward Propagation을 통해 나머지 da[1 ~ l - 1]의 값을 구하는 것을 말한다. 

<img width="644" alt="스크린샷 2024-07-11 오후 2 06 22" src="https://github.com/user-attachments/assets/7d9f31c4-0eb4-4a03-9e8a-7d96bcf847c8">

이전까지 계산은 Layer마다 Perceptron이 1개일 때 방식이다. 

**실제로 사용하는 NN에서는 Layer마다 Perceptron이 수백-수천개이므로 행렬과 Vector를 적용해 위와 같이 계산**한다. 

np.sum은 행렬의 총합을 의미하는 Python Code이며 수학 기호 ∑와 같은 역할을 수행한다. 

Z[l]을 구하는 과정이 각각의 Perceptron의 산출 값을 총 합해 구하기 때문에 dW, db, da를 구하는 과정에서 m으로 나누어 준다. 

각 Perceptron의 매개변수를 Update 하는 상황에서 다시 m으로 나누어 평균 값을 만들어 주는 것이다. → **m은 Perceptron 갯수**를 의미한다. 

지금까지의 과정이 아래 도식화 된 그림의 윗 부분이다. 

맨 처음 Input Data x를 입력하게 되면 각각의 NN Layer를 거치며 가공이 되어 최종적으로 Y-hat을 출력하며 이 과정이 Forward Propagation이다. 

**ReLU는 Activation Function, Sigmoid는 최종 가공된 숫자 값을 원하는 확률 값으로 바꿔주는 Logistic Function**이다. 

<img width="681" alt="스크린샷 2024-07-11 오후 1 42 02" src="https://github.com/user-attachments/assets/d246c9e6-847b-4a15-aa03-dbbc29ba4878">

**Forward Propagation을 수행한 뒤 Y와 Y-hat을 비교해 da[l]의 값을 구하는 것이 Loss를 계산하는 과정**이고, 이는 위 그림에서 오른쪽 부분이다. 

Loss를 이용해 da[l]을 구했다면 **Gradient Descent를 이용해 Backward Propagation을 진행**하고, 이는 위 그림에서 아래 부분이다. 

Loss 계산 과정에서 구한 da[l]과 Forward Propagation 저장한 Z[l]을 이용해 매개변수 dW[l], db[l], da[l - 1] 값을 구해주고 다시 NN Layer마다 이 과정을 반복해 최종적으로 dW[1], db[1]을 구해준다. 

**dW, db로 매개변수 W, b를 Update** 해주면 Backward Propagation 과정이 끝난다. 

## Parameters VS Hyperparameters

<img width="553" alt="스크린샷 2024-07-11 오후 2 42 24" src="https://github.com/user-attachments/assets/15597a69-9668-42e6-8493-ae5a7df8d884">

**Parameter**

- 가중치 W, bias와 같이 DL Training을 통해 Computer가  수정하는 값

**Hyperparameter**

- NN Layer의 갯수, Learning Rate 등 사람이 직접 설정하는 값

Hyperparameter는 Training 반복 횟수, NN Layer의 깊이, Learning Rate, Training Time, 정규화 매개변수 등 많은 종류가 있다. 

**Hyperparameter의 값에 따라 Training 성능이나 정확도, Training Time, Overfitting 문제 등 다양한 것들이 결정되기 때문에 아주 중요한 변수이다.** 

<img width="699" alt="스크린샷 2024-07-11 오후 2 47 49" src="https://github.com/user-attachments/assets/37b226c1-539d-433e-ae8d-3a181ff3053b">

**Idea → Code → Experiment** 순서로 개발 과정을 반복하게 된다. 

**Idea**

- Hyperparameter를 어떻게 설정할지 연구하는 과정이다.

**Code**

- 정해놓은 Idea를 Programming Code로 바꿔 Model의 성능을 Test 할 수 있도록 준비하는 과정이다.

**Experiment**

- DL Model을 Test, 성능을 확인하는 과정이다.

위 3단계를 반복하면서 가장 높은 성능을 내는 Hyperparameter를 찾아낸다. 

## What does this have to do with the Brain ?

<img width="693" alt="스크린샷 2024-07-11 오후 2 52 30" src="https://github.com/user-attachments/assets/478eb5dc-dd1b-4535-9858-d392ece71bc8">

> The Lecture Notes referred to <a href = 'https://www.deeplearning.ai/'>DeepLearning.AI</a>