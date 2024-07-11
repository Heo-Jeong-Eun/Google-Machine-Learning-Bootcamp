# **Neural Network Basics**

## Binary Classification

**Logistic Regression**은 **Binary Classification**를 위한 Algorithm이다. 

<img width="635" alt="스크린샷 2024-07-03 오후 7 03 41" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/fbbb6ddf-07d9-410d-97dd-cc7edcdcda1e">
Input Image가 있을 때, 고양이인지 아닌지 분류하는 Binary Classification 문제가 있다. 

Image를 인식해 고양이라면 1, 아니면 0으로 Label을 출력한다. 

Image의 RGB Data를 Pixel별로 인식하고 Red, Green, Blue 3개로 분리된 행렬을 사용한다. 

각 RGB의 값들을 하나의 Feature Vector x에 나열하면 64 * 64 * 3 차원으로 나타난다. 

즉 전체 차원은 12288이 되며 x의 차원을 의미하는 nx = 12288이 된다. 

### Notation

<img width="635" alt="스크린샷 2024-07-03 오후 7 11 14" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/4f7036b8-d845-49bb-bb78-ba6b2e17836b">

**Binary Classification의 목표는 Feature Vector x로 표현된 Image를 Input으로 받아 해당 Label의 y가 1인지 0인지 예측하는 분류기를 Training하는 것이다.** 

이 과정에서 사용하는 표기법이 있는데, (x, y)는 쌍으로 표현되며 x는 Feature Vector이고 y는 0 또는 1인 Label이다. 

Training Set은 m개의 Training Example로 구성되며 

<img width="498" alt="스크린샷 2024-07-03 오후 9 49 09" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/a0301ce1-0521-4338-b610-762fe0825968">

부터

<img width="498" alt="스크린샷 2024-07-03 오후 9 50 16" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/9b237e48-1cee-4d7e-9fc5-a2c92af8b727">

까지 나타낼 수 있다. 

이때 m은 Training Example 갯수를 나타낸다. 

NN을 구현할 때 Training Set의 **Input 행렬 X에**  

<img width="498" alt="스크린샷 2024-07-03 오후 9 51 32" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/8020636c-7191-4b2d-8454-db1bee50de26">

위 식을 가져와 **열로 쌓는 것이 더 유리**하다. 

이때 **Input 행렬 X는 n개의 행과 m개의 열을 갖게 되고 nX * m 차원의 행렬**이라는 것을 의미한다. 

**Output Label y 또한 열로 쌓아**

<img width="498" alt="스크린샷 2024-07-03 오후 9 53 05" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/7cc1d093-f59d-4a16-a7fa-180ffbddc986">

위와 같이 정의한다. 이때 **Y는 1 * m 차원의 행렬**이 된다.

즉, **NN을 좀 더 쉽게 구현하기 위해서는 X, Y 모두 열로 쌓는 것이 유리하고 해당 표기법은 Logistic Regression에 사용**한다.

## Logistic Regression

<img width="691" alt="스크린샷 2024-07-03 오후 7 34 18" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/4eb45cf4-b47c-42d7-a268-ebec7f6280f8">

Logistic Regression은 **Output Label Y가 모두 0이거나 1인 경우**, 즉 **Binary Classification** 문제에서 사용하는 Training Algorithm이다. 

<img width="708" alt="스크린샷 2024-07-03 오후 7 52 09" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/afce7fb0-77c1-4add-b5a5-bab38727a0ff">

Logistic Regression은 **Input Feature Vector X에 대해 Output Y가 1이 될 수 있는 확률**을 구하는 것인데, 이때 **Y-hat**은 통계학에서 **추정치**를 의미한다. 

Input Feature X는 n차원의 Vector이다. 

Logistic Regression의 매개변수는 w와 b로 w는 n 차원의 Vector이고, b는 실수이다. Deep Learning에서는 w를 가중치, b를 bias, 편향이라고 부른다. 

이 때 매개변수 w, b를 계산하기 위해서는 w를 전치시켜 마치 Linear Function의 한 종류와 같이 변형해야 한다. 

**w를 전치시킴으로써 열 행렬의 형태로 만들어 주는 것**이다. 이는 Linear Regression 방식에서 자주 사용한다. 

<img width="708" alt="스크린샷 2024-07-03 오후 7 53 21 (1)" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/7b5eb4ed-6e66-4250-81d3-769e9e59deea">

하지만 위의 경우 Binary Classification으로 좋은 Function는 아니다. 

Binary Classification에서 Y-hat값이 확률 값이기 때문에 0과 1사이의 값이여야 하는데 위 Function는 Y값이 1보다 클 수 있고 음수가 될 수도 있기 때문이다. 

따라서 이를 해결하기 위해 **Sigmoid Function**를 사용한다. 

Output은 다음과 같다. 

<img width="716" alt="스크린샷 2024-07-03 오후 9 45 51" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/d5740185-2593-435b-bce7-b0cdde8edad0">

![Untitled](https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/253d4c1e-0be0-46c8-a0cd-87570791ea3c)

위와 같은 Graph 모양을 가지는 Function를 Sigmoid라고 하며 아래와 같이 나타낸다. 

<img width="716" alt="스크린샷 2024-07-03 오후 9 45 22" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/fd6cc56f-3fc5-4bb8-8224-92c478db7373">

이 때 z 값이 매우 크면 e^-z는 0에 가까지고 Sigmoid Function는 1에 가까워진다. 

반대로 z가 매우 작은 값이거나 음수이면 Sigmoid Function는 0에 가까워 진다. 

**Logistic Regression을 구현할 때 중요한 것은 Y가 1을 잘 추정할 수 있도록 최적의 매개변수 w와 b값을 찾는 것이며, 최적의 w, b 값을 찾기 위해서는 매개변수에 따라 달라지는 Y-hat의 정확도를 측정해야 한다.** 

## **Logistic Regression Cost Function**

Logistic Regression Model에서 매개변서 w, b의 값에 따라 달라지는 **Model의 정확도**, 즉 **오차를 측정**하는 것을 **Cost Function**이라고 한다. 

<img width="475" alt="스크린샷 2024-07-03 오후 10 32 55" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/e31a1bd1-bf56-452d-919d-3092ee94724c">

Y-hat == Y = 1에 대한 추정치이며 Sigmoid Function에 

<img width="708" alt="스크린샷 2024-07-03 오후 7 53 21" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/1bf51618-2007-4dc2-b8f3-05f201f0d37f">

값을 대입한 것이다. 

Logistic Regression Cost Function를 통해 Y = Y-hat가 되는 것이 목적이다. 

<img width="757" alt="스크린샷 2024-07-03 오후 10 35 38" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/1eb5beb3-fcc5-4643-9f12-6fba0e061588">

위 공식에서 x는 Training시킬 Image이고 y는 그에 대한 정답이다. 

**하지만 실제 결과와 예측값이 동일한 경우는 없으므로 결과값과 예측값 사이 손실, Loss가 존재한다.** 

따라서 **Loss Function = Y-hat - Y**를 나타낸다. 

![스크린샷 2024-07-03 오후 10 39 04](https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/f6408022-a8ff-4c70-906d-97b533bef7ca)

x와 y가 m개 있기 때문에 이에 따른 Loss도 m개가 나올 것이다. 

보통 m개의 Loss를  제곱해 최소화 하는 방법을 생각하는데, **Loss 제곱의 합을 최소화** 해야 더 정확한 결과을 얻을 수 있다. 

이 때 **Y-hat은 Sigmoid Function 값**이기 때문에 **Natural Logarithm e를 사용**하므로 Graph 모양의 이상을 막기 위해 비슷한 역할을 하는 **Log Loss Function을 사용해 오차를 최소화**한다. 

Y = 1인 경우 Loss = -log(Y-hat)이고, **오차를 최대한 작게 만들기 위해서 Y-hat이 최대한 커야 한다.** 

만약 Y = 0인 경우 반대로 Y-hat은 최대한 작아야 한다.

**Cost Function J(w, b)는 m개의 Loss 총 합을 다시 m으로 나누어 Loss의 평균치를 구하는 Function**이다. 

![Untitled](https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/919b0c4f-ede2-4b13-91ed-ff57b0af57e1)

> Cost Function

Cost Function의 Graph는 위 사진과 같으며 매개변수 w, b를 합했기 때문에 Cost Function J(w, b)를 3차원 상으로 표현하게 되면 아래와 같다. 

<img width="533" alt="스크린샷 2024-07-03 오후 10 54 11" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/7c15cf88-37b3-4f04-bfe3-3f067933333c">

### Cost Function VS Loss Function

**Loss Function이 Single Training Example에 대한 계산**일 때, **Cost Function은 전체 Training Set에 대한 평균 값을 계산한 것**이다. 

## Gradient Descent

**Gradient Descent는 초기 지점에서 Global Optimum으로 가장 빠르게 이동하게 하는 Algorithm이다.** 

Model의 정확도, 오차를 측정하는 Logistic Regression Cost Function에서 오차가 가장 낮은 즉, Function 값이 가장 작은 맨 아래 지점을 향해 가장 빠른 경로로 가도록 만들어 매개변수 w, b를 효율적으로 수정하게 해준다.

우리가 실제 분석에서 만나게 되는 Function들은 닫힌 형태가 아니거나 복잡해 미분 계수와 그 근을 계산하기 어려운 경우가 많고 실제 미분 계수를 계산하는 과정 구현보다 Gradient Descent 구현이 더 쉽다. 

Data 양이 많은 경우 Gradient Descent와 같은 Iterative 한 방법을 사용하면 계산량 측면에서도 효율적이기 때문에 Gradient Descent를 사용한다. 

<img width="666" alt="스크린샷 2024-07-03 오후 11 22 08" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/1ecbb7f7-92b4-4cef-91ba-f0df9c35f2be">

> L : Gradient Descent / R : Cost Function

위 Graph의 수평축은 w축과 b축으로 구성되어 있다. 해당 Graph에서 Cost Function J의 최소값을 나타내는 지점 즉, Global Optimum을 찾고, 과 그에 해당하는 w, b 값을 찾는 것이 목표이다. 

오른쪽 그림에서 Cost Function J는 Convex Function이다. 아래 Graph처럼 여러 개의 최소값 지점을 가지는 것과 대비, 하나의 최소값만 가진다. 

Logistic Regression의 경우 일반적으로 첫 설정은 0부터 시작한다. 

시작 지점을 설정할 때, 변수의 값을 0으로 초기화하고 적절한 변수값을 찾아가는 것이다. 

Cost Function J는 Convex Function이기 때문에 내려가게 되면 결국 한 지점, Global Optimum에서 만나게 된다. 

<img width="687" alt="스크린샷 2024-07-03 오후 11 34 28" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/5db6b96b-3626-4251-a67d-263ea9930e34">

> Loss Graph and Gradient Descent

가장 아래 지점인 Global Optimum에 수렴할 때까지 계속해서 Gradient Descent로 매개변수 w, b 값을 수정해준다. → W = W - α * (dJ(w) / dw)

**α**는 Gradient Descent로 이동하는 거리를 조절하기 위한 변수로 **Learning Rate**를 의미한다. 

![Untitled](https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/74c5b60f-bc63-4538-b7da-a1a09ec14765)

최소값 지점을 목표로 어떤 지점에서 w 값을 정할 때 J(w)의 값은 감소해야 한다. 

예를 들어 어떤 지점에서 w 값이 증가할 때 J(w)도 증가하면 그 지점에서는 w 값을 감소시켜야 한다. 

반대로 어떤 지점에서 w의 값이 감소할 때 J(w)가 증가하면 그 지점에서는 w 값을 증가시켜야 한다. 

즉, 어떤 지점에서 w 값을 증가해야 할지 감소해야 할지 정할 때 Function의 기울기, Derivatives를 활용해 방향을 정한다. 

## Derivatives

<img width="687" alt="스크린샷 2024-07-04 오전 12 05 12" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/d52a19eb-1c4a-4d77-8388-3e28d5d708be">

**Derivatives는** Function a의 접선의 기울기를 나타내며 미분, Differential과 같은 의미이다. 

접선의 기울기란 어떤 지점에서 x의 순간 변화량에 대한 y의 순간 변화량을 말한다. 

## Computation Graph

NN은 구현할 때 순전파, **Forward Propagation**과 역전파, **Back Propagation**으로 나뉜다. 

**Forward Propagation는 NN의 Output을 계산하고 Back Propagation는 기울기 또는 도함수를 계산한다.** 

<img width="687" alt="스크린샷 2024-07-04 오전 12 09 33" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/5d960066-d163-4103-b08a-b38cddea81d0">

Computation Graph는 최적화하려는 Feature Output 변수, 위 경우 J와 같은 변수가 있을 때 유용하다.  

위 예제에서 왼쪽 → 오른쪽으로 계산하게 되면 J를 계산할 수 있다. 

Derivatives를 계산하기 위해서는 오른쪽 → 왼쪽으로 이동하는 과정이 필요하다. 

Computation Graph는 왼쪽 → 오른쪽 계산을 구성한다. 

**즉, Forward Propagation은 Function값을 구하는데 사용하며 Back Propagation은 미분 값을 구하는데 사용된다.** 

## Derivatives with a Computation Graph

Computation Graph에서 Back Propagation을 사용해 미분 값을 구할 것이다. Derivatives의 경우 오른쪽 → 왼쪽으로 계산한다. 

<img width="687" alt="스크린샷 2024-07-04 오전 12 12 34" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/e5738087-df26-463b-b3ae-e6cc6babc875">

![스크린샷 2024-07-04 오전 12 14 45](https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/b41985a8-7f7a-408a-ac20-79e6d6ca97f1)

## Logistic Regression Gradient Descent

Logistic Regression에서 Gradient Descent를 적용하기 위해 Derivatives를 계산하는 방법에 대해 배울 것이다. 

**Derivatives는 Logistic Regression에서 Gradient Descent를 적용시키기 위한 핵심 방정식, Equations이다.** 

그리고 **Derivatives 연산은 Computation Graph를 이용해 계산**한다. 

<img width="640" alt="스크린샷 2024-07-04 오후 2 00 06" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/533f8c22-6025-4aa6-b32b-f3dcbbfcbf7a">

<img width="689" alt="스크린샷 2024-07-04 오후 2 00 26" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/50b99040-9493-4534-8a71-75bc5fdbe595">

기울기, **Derivatives를 활용하는 Gradient Descent를 이용하기 위해** 왼쪽 → 오른쪽으로 계산하는 **Back Propagation**을 사용한다. 

위 공식의 계산 과정을 거친 후 Learning Rate α를 조절해주면 Gradient Descent 적용 준비가 끝난다. 

**Back Propagation 계산을 통해 Derivatives를 구하고 Gradient Descent를 사용해 매개변수 w, b를 갱신한다.** 

**다시 Forward Propagation으로 Logistic Regression - Cost Function을 계산하는 과정을 반복하며 Y-hat과 Y의 오차를 줄여나간다.**  

## Gradient Descent on m Example

이전까지 예시들은 NN이 하나뿐인 Single Training이지만, 실제 DL에서는 여러 개의 NN을 사용하기 때문에 전체 Training Set에서의 Training 방법을 배울 것이다.  

<img width="689" alt="스크린샷 2024-07-04 오후 2 10 13" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/2fdd3f16-da6e-4b63-9b4f-2549dffa1480">

단순히 **Single Training이 m개**라고 생각하면 된다. 

m개의 Loss Function의 총합을  m으로 나누어 평균을 구해주면 Cost Function J의 값이 된다. 

이후 Back Propagation을 통해 Derivate 값의 평균을 구하고, Gradient Descent를 통해 변수를 갱신하면 된다. 계산 과정은 아래와 같다. 

<img width="689" alt="스크린샷 2024-07-04 오후 2 12 34" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/2ee877ac-9639-421b-a733-0793978874ba">

이때 for loop을 통ㅇ해 총합을 구하고 연산하게 되는데 DL에서는 방대한 양의 Data Set을 Training 시키고 수억개의 NN을 이용하기 때문에 for loop을 사용하게 되면 연산량이 많아져 Algorithm 성능이 떨어지게 된다. 

이를 해결하기 위해 for loop 대신 Vectorization을 사용한다. 

## Vectorization

<img width="689" alt="스크린샷 2024-07-04 오후 11 18 14" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/d1ab6dbe-69d4-470f-8fb8-f5d6a1efa88c">

```python
z = np.dot(w.T, x) + b
```

np.dot()은 Numpy Python Libray의 내장 Function으로 w, x를 Vectorization 하는 역할을 수행한다. 

이처럼 **Vector 형태로 계산하게 되면 병렬적 연산**을 하고 이는 CPU보다 GPU일 때 유리하다. 

## More Vectorization Example

<img width="689" alt="스크린샷 2024-07-04 오후 11 29 11" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/36590cb2-ff62-4bcb-acf8-0acd72d5207a">

행렬 A와 Vector v의 곱으로 Vector u를 계산하려고 한다. → Vector = 행렬 = 배열

이때 np.zeros((n, 1))은 0으로 이루어진 n * 1 행렬을 생성한다는 의미이다. 

for loop이 아닌 np.dot()을 사용하게 되면 복잡한 연산 과정 없이 간단하게 표현할 수 있다. 

<img width="689" alt="스크린샷 2024-07-04 오후 11 31 35" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/31e1d1d6-5dc6-49d1-a86f-e63b85df3415">

위 계산도 마찬가지로 np.exp()를 사용해 for loop을 대신한다. 

## Vectorizing Logistic Regression

**Logistic Regression 또한 Vectorizing** 하는 것이 유리하다. 

<img width="689" alt="스크린샷 2024-07-04 오후 11 34 40" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/fe2c6cb0-2864-45b2-8525-9bb0aa90f377">

> Logistic Regression Vectorizing

<img width="689" alt="스크린샷 2024-07-04 오후 11 37 53" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/97ff8fe6-8659-4246-94a4-0192dcc9a440">

> Logistic Regression Vectorizing

## Vectorizing Logistic Regression’s Gradient Computation

마찬가지로 **Gradient Descent도 Vectorizing** 하는 것이 좋다. 

<img width="689" alt="스크린샷 2024-07-04 오후 11 40 40" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/7b214190-5050-41ac-abf1-2bccf15c604a">

<img width="689" alt="스크린샷 2024-07-04 오후 11 40 52" src="https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/d66acca8-d2e6-445d-a026-5493d5db5c43">

## BroadCasting in Python

**BroadCasting**은 **Numpy가 산술 연산 중에 다른 모양 배열을 처리하는 방법**이다. 

![Untitled (3)](https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/27842240-81d7-41fc-884e-eb78e2298a76)

배열 a(3)와 b(1)을 곱하는 경우, 배열의 갯수가 다르기 때문에 원래는 연산이 불가능하다. 

이런 경우 BroadCasting을 사용해 2를 복사하게 되면 연산이 가능해진다. 

BroadCasting을 사용하게 되면 훨씬 더 효율적인 Memory 사용이 가능해진다. 

![Untitled (4)](https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/558f2060-5367-45e2-a3d3-55bba7197f53)

행이 아닌 열에도 BroadCasting은 활용 된다. 만일 후행 차원이 같지 않으면 BroadCasting 오류가 생기게 된다. 

![Untitled (5)](https://github.com/Heo-Jeong-Eun/Google-Machine-Learning-Bootcamp/assets/60500256/a13ff97f-9d4d-43ca-8a3c-b2bc3dd4602d)

BroadCasting이 세로 방향으로만 이루어지는 것은 아니다. 

np.newaxis()로 새로운 축을 삽입해 가로, 세로 방향에서 BroadCasting이 일어나는 경우도 있다.