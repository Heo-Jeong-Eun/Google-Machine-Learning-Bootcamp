## Neural Networks Overview

NN을 구현하기 앞서 NN의 Overview를 알아 볼 것이다. 

<img width="691" alt="스크린샷 2024-07-05 오후 9 34 16" src="https://github.com/user-attachments/assets/99cc08b6-8ffb-4576-9c88-3e213935c208">

Training Data가 입력 되는 **Input Layer**

결과, 예측치의 값이 산출되는 **Output Layer**

Input Layer과 Output Layer 사이 Layer을 의미하는 **Hidden Layer** 

NN은 위 3가지 Layer로 구성된다. 

위 자료에서는 x1, x2, x3가 Input Layer, Y-hat은 Output Layer이다. 

수학적으로 표현할 때, Input Layer을 [0]으로 2번째 Layer부터 [1]로 표현한다. 

각 Layer에 해당하는 변수들을 표현할 때에도, 변수 위에 []를 붙여 각 Layer에 해당하는 변수(z, w, b, a)를 나타낸다. 

이와 반대로 Back Propagation 과정을 통해 구한 Derivatives 값들은 끝에서부터 시작해 나타낸다. → dz[1], dz[2], …

## **Neural Network Representation**

아래 흰 동그라미를 **Perceptron** or **Neuron**이라고 한다. 

a[0]은 Input x, a[1]은 첫 번째 Layer의 결과값, Sigmoid Function의 값을 나타내고, a[2]는 Y-hat을 나타낸다. 

<img width="691" alt="스크린샷 2024-07-05 오후 9 40 53" src="https://github.com/user-attachments/assets/2fdf7e03-cd4b-415c-9612-c5f3225e9471">
NN 표현 방법

<img width="355" alt="스크린샷 2024-07-05 오후 9 43 44" src="https://github.com/user-attachments/assets/c9aea595-8d93-4bcc-b4ad-9a3f98f053b8">

각 Layer의 맨 위 Perceptron부터 맨 아래 Perceptron까지 순서대로 위와 같이 표현한다. 

즉, (4 * 1) 행렬 a[1] = (a[1](1), a[1](2), …)과 같이 구성된다. 

## Computing a Neural Network’s Output

NN의 결과가 산출되는 과정을 확인하며, 표현 방법에 대해 알아볼 것이다. 

<img width="693" alt="스크린샷 2024-07-05 오후 9 48 14" src="https://github.com/user-attachments/assets/3e944d6c-ff02-4d7c-8c74-31462dfdcd1f">

<img width="693" alt="스크린샷 2024-07-05 오후 9 48 47" src="https://github.com/user-attachments/assets/2ebbd4ac-9e12-4fb5-874e-a8cf3b52bb6f">

Perceptron 하나씩 계산하며 결과를 산출하고, 이 과정을 **Vectorizing**을 통해 아래와 같이 표현할 수 있다. 

<img width="693" alt="스크린샷 2024-07-05 오후 9 50 50" src="https://github.com/user-attachments/assets/9c79aeff-a3e3-4c9d-890d-6434abd37447">

**i번째 Layer라고 가정했을 때 Z[i] = W[i].T * X[i] + b[i]로 표현할 수 있다.** 

또한 각 행렬의 차원 수를 알아보면, W[1]의 차원 수는 4(Layer 1의 Perceptron 수) * 3(Training Data 수)

b의 차원 수는 4(4Layer 1의 Perceptron 수) * 1로 표현된다. → b는 Perceptron당 하나이다. 

<img width="693" alt="스크린샷 2024-07-05 오후 9 53 55" src="https://github.com/user-attachments/assets/04952a79-0dba-4f35-9528-ce0f13cd2337">

행렬의 곱에서 차원 수는 **BroadCasting**을 통해 변할 수 있다. 

## Vectorizing Across Multiple Examples

<img width="693" alt="스크린샷 2024-07-05 오후 9 57 57" src="https://github.com/user-attachments/assets/8391a4f3-bb7e-476c-a007-c00b30279a02">

<img width="693" alt="스크린샷 2024-07-05 오후 9 58 40" src="https://github.com/user-attachments/assets/d18412f9-4d10-44c4-9043-72141b28fdc7">

## Explanation for Vectorized Implementation

Programming 과정에서 비효율적인 계산을 하는 for loop을 쓰지 않고 Algorithm 성능을 향상하기 위해 **Vectorizing**을 한다. 

<img width="693" alt="스크린샷 2024-07-05 오후 9 59 32" src="https://github.com/user-attachments/assets/9385dfe7-b7e9-44b0-a110-5ae87aac3ca1">

W[1] * X 행렬을 나타낸 부분을 살펴볼 때, 행렬의 열은 Input Data x의 갯수를 나타내고, 행은 Input Data x의 차원 수 m을 나타낸다. → x = m * 1 Vector이다. 

여기에 b[1]만 더해주면 z[1] Vector가 만들어진다. 

## Activation Function

Activation Function에는 Non-Linear Function만 사용되는지 알아볼 것이다. 

<img width="693" alt="스크린샷 2024-07-05 오후 10 03 14" src="https://github.com/user-attachments/assets/25a337ac-23bc-46a9-ae12-f8950a8326a5">

**Activation Function을 통해 매개변수 W, b를 수정하고 Loss를 줄일 수 있다.** 

DL에서 주로 사용하는 Activation Function은 Sigmoid Function이나 tanh Function, ReLU Function, Leaky ReLU Function이 자주 쓰인다. 

<img width="693" alt="스크린샷 2024-07-05 오후 10 06 24" src="https://github.com/user-attachments/assets/2662328d-625d-4dec-8df4-bda7a5df49e9">

DL Training에 자주 쓰이는 Activation Function

## Why do you need Non-Linear Activation Functions?

**DL Training에 쓰이는 Activation Function**은 모두 Linear Function이 아닌 **Non-Linear Function의 형태**이다. 

<img width="693" alt="스크린샷 2024-07-05 오후 10 08 22" src="https://github.com/user-attachments/assets/018a16c9-4f43-4dfc-8888-7b1a7a46e531">
NN Training 과정에서 오차를 최소화하는 Gradient Descent을 통해 매개변수를 수정했다. 

**만약 Activation Function이 Linear Function이라면 NN을 거쳐 Training 하는 의미가 사라진다.** 

위 공식처럼 w(1)x + b(1)로 귀결되기 때문에 Training의 의미가 사라지게 되고 Gradient Descent 수정 또한 의미가 없다. 

Sigmoid Function, tanh Function은 Input에 따라 Derivatives의 값이 변한다. 

ReLU, Leaky ReLU는 0을 기점으로 모양이 변하므로 Derivatives의 값을 이용해 매개변수를 수정하는 Gradient Descent을 사용할 수 있게 되며 NN을 통한 Training이 의미 있는 결과를 갖게 된다. 

## Derivatives of Activation Functions

<img width="693" alt="스크린샷 2024-07-05 오후 10 22 44" src="https://github.com/user-attachments/assets/e4435e89-3355-45a4-853c-d18aa4db892b">

<img width="693" alt="스크린샷 2024-07-05 오후 10 23 26" src="https://github.com/user-attachments/assets/4eca70c9-1b07-4231-8ab7-f2a108c4e63d">

<img width="693" alt="스크린샷 2024-07-05 오후 10 23 50" src="https://github.com/user-attachments/assets/4745d7b0-23eb-4a32-aaec-8a8cb2885b92">

## Gradient Descent for Neural Networks

<img width="693" alt="스크린샷 2024-07-05 오후 10 24 32" src="https://github.com/user-attachments/assets/a89f4c25-8ccb-4449-a470-9d16c41fd0aa">

<img width="693" alt="스크린샷 2024-07-05 오후 10 25 13" src="https://github.com/user-attachments/assets/bf68e7b2-50ca-4765-9bad-ca4e91716444">

## Backpropagation Intuition

<img width="693" alt="스크린샷 2024-07-05 오후 10 25 44" src="https://github.com/user-attachments/assets/216fafd5-2b53-44c6-bc6f-5524dde34f9b">

<img width="693" alt="스크린샷 2024-07-05 오후 10 25 59" src="https://github.com/user-attachments/assets/398b9b30-3246-4205-b3aa-2fdc3d62cbab">

<img width="693" alt="스크린샷 2024-07-05 오후 10 26 19" src="https://github.com/user-attachments/assets/3084954b-9633-4c4d-9916-28bb0a267e88">

<img width="693" alt="스크린샷 2024-07-05 오후 10 26 36" src="https://github.com/user-attachments/assets/bc6393a8-2bb6-4bfd-87fe-1ef63c751dfd">

## Random Initalization

<img width="693" src="https://github.com/user-attachments/assets/9ded53e1-2675-4db6-a4ef-d111b9971940">

**매개변수를 초기화 할 때 w, b의 값 중 w를 0으로 초기화하면** Activation Function을 Linear Function로 사용할 때와 마찬가지로 **NN을 거쳐 Training 하는 의미가 없어진다.** 

<img width="693" alt="스크린샷 2024-07-05 오후 10 22 13" src="https://github.com/user-attachments/assets/2cf7ad3c-64c5-48b3-801c-f10c80462d7d">

또한 w를 너무 크게 초기화 하게 되면 Derivatives 값이 0에 가까워 Training의 의미가 퇴색되거나 Function 값이 너무 커져 Training 속도가 느려지는 문제가 생긴다. 따라서 **Random 값에 0.01을 곱해준다.**