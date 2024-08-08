# Hyperparameter Tuning, Batch Normalization and Programming Frameworks

## Tuning Process

<img width="719" alt="스크린샷 2024-07-24 오후 5 18 09" src="https://github.com/user-attachments/assets/c016f4a8-0730-4962-8bde-6a6083551828">

**Hyperparameter는 다양하고 각 Hyperparameter의 중요성은 다르다.** 

예를 들어, Adam Optimizer의 Hyperparameter인 β1, β2, ε은 거의 변경하지 않고 기본값을 사용하기 때문에 Tuning을 거의 하지 않아도 된다. 

**Hyperparameter 중 α가 가장 중요**하며, 다음으로는 Hidden Unint, Mini-Batch Size인 Hyperparameter **β가 중요**하고, 그 다음으로는 **Learning Rate Decay가 중요**하다. 

![Untitled](https://github.com/user-attachments/assets/9287ab83-eabb-4c39-b3a8-20fcfe3af2fd)

### **Random Sampling**

ML 개발 초기, 2개의 Hyperparameter가 있는 경우 왼쪽과 같이 Grid 형식으로 Sampling 하는 경우가 많았다. 

위의 경우 Grid 5 * 5의 형태, 즉 25개의 Point로 설정했고 이 방법은 Hyperparameter 갯수가 비교적 적은 경우에 잘 동작한다. 

Hyperparameter 1은 Learning Rate α이고 Hyperparameter 2는 Adam Algorithm이 ε이라고 할 때, α는 매우 중요하고 ε는 거의 중요하지 않다. 

따라서 왼쪽과 같이 Grid 형식으로 Sampling을 무작위로 진행하게 되면 25개의 Model을 Train 했지만 5개의 α값으로만 시도할 수 있다. 

반대로 Sampling을 Random으로 진행했다면 각 25개의 α 값를 모두 시도하게 된다. 

### **Coarse to Find**

Hyperparameter를 Sampling 하는 경우에 Coarse to Find Searching 방법이 자주 사용된다. 

![Untitled (1)](https://github.com/user-attachments/assets/c0a6de45-c96f-41e2-8c67-2de83144eb31)

Coarse to Find는 위와 같이 Sampling 된 Point를 탐색해 파란색 내부에서 잘 동작한다고 알아냈을 때, 파란색 Point로 Zoom-In하여 파란색 공간에 대한 밀도를 높혀 Sampling을 또 다시 Sampling해서 탐색하는 방법이다. 

Coarse to Find Searching은 임의의 Sampling을 진행하고 충분히 탐색한 후 선택적으로 사용하면 된다. 

## Using an Appropriate Scale to Pick Hyperparameters

단순히 임의로 Hyperparameter를 Sampling 하는 것은 유효한 범위에서 탐색하는 것이 아니기 때문에 비효율적이다. 

따라서 **Hyperparameter를 Smapling 할 때 적절한 Scale을 선택함으로써 Hyperparameter Searching Process를 더 체계적으로 조직, 효율적으로 탐색할 수 있다.** 

Hidden Unit의 갯수인 n[l]의 값을 탐색할 때, 잘 동작하는 범위는 50-100라고 가정한다. 

이 경우 Hidden Unit의 갯수를 50-100 사이의 값으로 선택하는 것이 비교적 합리적인 방법이라고 할 수 있다. 

다른 예시로 NN의 Layer의 갯수를 결정할 때, Layer의 갯수가 2-4개가 잘 동작한다고 가정하면 2-4 사이에서 선택하는 것이 합리적이다. 

![Untitled (2)](https://github.com/user-attachments/assets/76f187fd-8621-48eb-b946-8342a7ba07b0)

Learning Rate α를 탐색할 때, 0.0001-1 사이의 값으로 균일화 되게 Sampling을 진행하면 약 90%의 값이 0.1-1 사이에 있을 것이다. 

즉 Sampling의 90%정도를 0.1-1사이의 값에 집중하게 되는 것이고 10% 정도만 0.0001-0.1 사이에 집중하는 것이다. 

![Untitled (3)](https://github.com/user-attachments/assets/f0f86eb6-8259-4243-bfb1-404cd2b9994c)

이러한 방법은 적절하지 않기 때문에 대신 Hyperparameter를 Linear Scale이 아닌 Log Scale을 적용해 탐색하는 것이 적합하다. 

### **Log Scale Sampling**

일부 Hyperparameter는 Sampling 할 때 적절한 Scale 대신 **Log Scale에서 Sampling 하는 것이 더 합리적**이다. → **Learning Rate α와 같은 경우** 

### **β Samling**

**지수 가중 Mean 계산에 사용되는 β**는 **1 - β**의 값을 Log Scale에서 Sampling 해야 한다. 

### **Efficient Exploration**

Log Scale Sampling을 통해 Hyperparameter의 민감도가 높은 영역을 더 밀도있게 탐색할 수 있다. 

## Hyperparameter for Expoenetially Weighted Average

Exponentially Weighted Averages β의 적절한 값이 0.9-0.999 사이 값이라고 하면 탐색해야 하는 범위 또한 0.9-0.999가 된다. 

**같은 범위 내 균일한 탐색을 위해**서는 Linear Scale이 아닌 **Log Scale**로 탐색해야 한다. 

가장 좋은 것은 1 - β의 범위를 찾는 것 즉, 0.001-0.1 사이의 값을 찾는 것이다. 

![Untitled (4)](https://github.com/user-attachments/assets/70255697-8889-401a-aa05-0358e2ce7ffe)

**Linear Scale로 Sampling 하는 것이 안되는 수학적 이유**는 **EWA에서 β값이 1에 가까워질수록 결과에 민감**한데, Linear Sacle Sampling을 하게 되면 β가 대부분 1에 근점한 범위에서 Sampling 할 것이기 때문이다. 

## Hyperparameters Tuning in Practice Pandas VS Caviar

<img width="687" alt="스크린샷 2024-07-23 오후 3 02 10" src="https://github.com/user-attachments/assets/984705ea-2207-4df8-8fba-1dde1339f7f1">

### **Hyperparameter Effectiveness of Intuition, 직관의 유효성**

한 응용 Domain에서 얻은 Hyperparameter Setting에 대한 직관은 주기적으로 Update가 필요하다. 몇 달에 한번씩 Hypermarameter를 다시 Test하거나 재평가 하는 것이 좋다. 

### **Babysit One Model VS Train Many Models in Parallel**

Computing Resourse가 충분하지 않으면 한 Model을 보살피며 Training, 만약 충분하다면 Model을 병렬로 Training 하여 다양한 Hyperparameter Setting을 시도하는 것이 좋다. 

### **Pandas VS Caviar**

한 Model을 보살피며 Training 하는 Pandas 접근 방식과 여러 Model을 Parallel로 Training 하는 접근 방식 중 Computing Resourse에 따라 적절한 방식을 선택할 수 있다. 

## Normalizing Activations in a Network

Batch Normalization은 NN Training을 더 쉽게하고 Hyperparameter Search를 더 효율적으로 만들며 아주 깊은 Network도 쉽게 Training 할 수 있게 해준다. 

![Untitled (5)](https://github.com/user-attachments/assets/859b258b-ea6d-45b6-aaf3-96fcd3455cb9)

### **Input Feature Normalization**

**Logistic Regression이나 Linear Regression에서 Normalization**은 **Input Feature의 Mean과 Variance를 조절하고, 더 동그란 모양으로 Cost를 변경해 Gradient Descent의 속도를 높여준다.** 즉, Training이 더 빠르게 진행된다. 

![Untitled (6)](https://github.com/user-attachments/assets/27a494e7-c45b-4c6a-8245-f411c91c6258)

Depp Model의 경우 Input이 x만 있는 것이 아니라 a[1], a[2] Activation도 있는데, 이 경우 Batch Normalization을 할 수 있는지가 문제이다. 

그리고 이 방법은 w[3], b[3]을 더 빨리 Train하는 방법이고 a[2]는 w[3], b[3]의 Train에 영향을 준다. 

이 방법이 **Batch Normalization**이고 줄여서 Batch Norm이라고도 한다. 

정확하게 이야기하면 a[2]가 아닌 **z[2]**를 **Normalization** 하는 것이다. 

### **Hidden Unit Normalization**

Hidden Layer l에서 Hidden Unit 값 z(1), …, z(m)가 주어졌을 때 Mean과 Variance, Normalization은 아래와 같이 구할 수 있다. 

<img width="350" height="200" alt="스크린샷 2024-07-24 오후 11 34 27" src="https://github.com/user-attachments/assets/3c27178a-2a49-4f6a-8268-774d135a8701">

**Batch Normalization**은 **Input Layer 뿐 아니라 NN의 Hidden Layer 값도 Nomalize** 한다. 

γ와 β는 Model에서 Train하는 Parameter이고 Gradient Descent, Momentum, RMSprop, Adam을 통해 Train 할 수 있다. 

γ와 β 사용하지 않으면 Normalization을 통해 Hidden Unit은 항상 Mean이 0, Variance는 1이 되므로 이는 올바르지 않기 때문에 γ와 β를 통해 ~z의 Mean이 원하는 값이 되도록 한다. 

즉, **각 Hidden Unit이 다른 Mean과 Variance를 갖도록 한다.** 

만약 Activation Function이 Sigmoid이고 Mean이 0, Variance가 1로 Normalization을 했다면 a의 값은 Linear한 영역에 밀집되어 정상적으로 Train 되지 않을 것이다. 

<img width="602" alt="스크린샷 2024-07-24 오후 10 05 30" src="https://github.com/user-attachments/assets/934d6a59-0b6f-493b-a1fd-846dc7814302">

만약 

<img width="602" alt="스크린샷 2024-07-24 오후 10 07 29" src="https://github.com/user-attachments/assets/ddb555db-17f4-4ec4-876d-2d6e829dad9b">

로 설정한다면, 

<img width="448" alt="스크린샷 2024-07-24 오후 10 08 31" src="https://github.com/user-attachments/assets/7bdba907-cc40-4df6-8b22-82cca027fbf7">

가 되는 것과 동일하다. 

![Untitled (7)](https://github.com/user-attachments/assets/11dad3df-3169-415d-887b-7babbb1e053a)

**NN의 여러 Layer에 Batch Norm을 적용해 Training을 더 효율**적으로 만들 수 있다. 

## Fitting Batch Norm into a Neural Network

Batch Normalization은 Z 값을 계산한 후 Activation Function a 과정 이전에 적용된다. 

![Untitled (8)](https://github.com/user-attachments/assets/63589bb3-dbb6-4811-8ffe-d75c5d0196cd)

위와 같은 NN이 있을 때 Batch Norm을 적용하면 FP는 아래와 같이 진행된다. 

![Untitled (9)](https://github.com/user-attachments/assets/03f07394-a77b-46c8-a0a9-63ba07878124)

Input x와 W[1], b[1]를 통해 z[1]을 구하고, β[1], γ[1]를 통해 Batch Norm을 적용해 ~z[1]을 구한다. 

Normalization이 적용된 Activation Function을 통해 a[1]를 구한다. 그리고 다음 Layer에서도 마찬가지로 진행하게 된다. 

<img width="708" alt="스크린샷 2024-07-24 오후 10 17 55" src="https://github.com/user-attachments/assets/b403f21a-66f6-4639-955b-d6a9940d6360">

Parameter는 위와 같다. 여기서 β는 Adam이나 Momentum의 β와는 다르다. 

새로운 Parameter인 γ와 β도 W, b와 같은 방법으로 Train되며 Gradient Descent 뿐 아니라 Adam 등 Optimization Algorithm이 적용되었을 때도 동일하다. 

<img width="708" alt="스크린샷 2024-07-24 오후 10 21 23" src="https://github.com/user-attachments/assets/852285ac-f25e-4f73-ba61-5729cca7b61c">

Mini-Batch에서 Batch Norm을 적용하면 아래와 같이 적용된다. 위에 적용한 Batch Norm과 동일하다. 

주의해야 할 점은 **각 Mini-Batch의 Data만으로 Normalization을 적용해야 한다는 것**이다. 

![Untitled (10)](https://github.com/user-attachments/assets/79b68b1b-1e92-4ae8-ae84-8ade72aee4e8)

그리고 Parameter에서 **b[l]은 무시**될 수 있는데, **Batch Norm에서 Mean 값을 구하고 Input에 Mean 값을 빼기 때문에 상수항을 더하는 것은 아무런 효과가 없기 때문**이다. 

따라서 이 Parameter를 제거하거나 0으로 설정할 수 있다. 

β[l]은 Bias나 이동에 영향을 주는 항이다. Batch Norm은 Mini-Batch에 사용해 Gradient Descent 진행은 아래와 같이 수행된다. 

<img width="708" alt="스크린샷 2024-07-24 오후 10 27 26" src="https://github.com/user-attachments/assets/f6890c5f-6e08-4509-884c-bbc7d286fb3b">

Momentum, MRSprop, Adam 등의 Optimization Algorithm과도 같이 사용이 가능하다.

## Why does Batch Norm Work ?

Batch Norm이 잘 동작할 수 있는 이유에는 두 가지가 있다. 

첫 번째, **Input Feature를 Normalization하여 Mean을 0, Variance를 1로 만들어주고 이것이 Train의 속도를 증가**시킨다. 

모든 Feature를 Normalization 하기 때문에 Input Feature x가 비슷한 범위를 갖게 되고 이로 인해 Train 속도가 증가하는 것이다. 

그리고 이 Normalization은 Input Layer 뿐만 아니라 Hidden Unit에 대해서도 적용한다. 

두 번째, **Batch Normalization을 통해 Model이 Weight의 변화에 덜 민감해지기 때문**이다. 

![Untitled (11)](https://github.com/user-attachments/assets/4ab35d24-47d3-41a2-baa1-8874f000d647)

Logistic Regression에서 모든 Data를 검은 고양이 사진으로 Train 했다고 가정한다. 

이 Model이 오른쪽과 같이 색이 있는 고양이를 판별하려고 하면 Classifier가 잘 동작하지 않을 수 있다. 

![Untitled (12)](https://github.com/user-attachments/assets/1a6d58ac-eee5-417a-8a48-0d62b3117994)

왼쪽 검은 고양이 Sample 일 때 Data 분포이고, 오른쪽은 색깔이 있는 고양이의 Data 분포가 될 수 있다. 

왼쪽의 Sample로만 Train 된 Model의 경우 오른쪽 색깔이 있는 고양이에 대해 일반화가 되어있지 않아 Classification이 잘 되지 않을 수 있다. 

이렇게 Data의 분포가 변하는 것을 **Covariate Shift**라고 부르는데, 만약 Input x의 분포도가 변경되면 Train Algorithm을 다시 Train 시켜야 할 수도 있다. 

Covariate Shift 문제가 Neural Network에는 아래와 같이 적용된다. 

![Untitled (13)](https://github.com/user-attachments/assets/b6dd9267-82d9-4f87-a86f-2e68b9fc1884)

위와 같은 NN이 있다고 할 때, Layer 3을 기준으로 살펴볼 것이다. 

이 Layer에는 W[3], b[3]을 Train 했을 것이다. 

그리고 Layer 3 기준으로 Feature의 값을 이전 Layer에서 계산된 값으로 사용하는데, Hidden Layer 3의 역할은 a1[2], a2[2], a3[2], a4[2]를 Input으로 사용한다. 이 값들을 통해 Y-hat 값으로 Mapping 할 것이다. 

![Untitled (14)](https://github.com/user-attachments/assets/e71e0b6c-22b7-4018-b845-981e1c13b766)

Layer 3, 왼쪽의 Network를 살펴봤을 때, Input x에서부터 Network는 W[1], b[1], W[2], b[2]가 변경되면 a[2]의 값도 변경될 것이다. 

![Untitled (15)](https://github.com/user-attachments/assets/be1e174f-a34c-4d4a-b909-cbe1a98a4869)

Layer 3 기준에서 a[2]는 항상 변하는 값이고 Covariate Shift 문제가 발생하게 된다. 

이때 Batch Normalization을 통해 Hidden Unit의 분포가 변경되는 정도를 줄여주는 것이다. 

정확하게 말하자면 **z[2]를 Normalization 하는 것**이다.  

z1[2], z2[2], z3[2], z4[2]가 이전 Layer의 Parameter들을 Update하면서 변할 수 있는데 Batch Normalization은 이 값들이 변경되더라도 Mean과 Variance는 똑같도록 유지해준다. 

![Untitled (16)](https://github.com/user-attachments/assets/92d04bd0-724f-492c-8613-b6b1f80eeab8)

여기서 γ와 β는 새로 Train 해야 할 Parameter이다. 

결과적으로 Batch Normalization은 Input 값이 변해 생기는 문제를 줄여준다. 

Input의 분포도가 변경될 수 있지만, 변경되는 정도를 줄여주고 결과적으로 다음 Layer에서 Train 해야 하는 양을 줄여주어 더 안정적인 Train이 되도록 한다. 

결과적으로 전체 Network의 Train 속도를 증가시켜주는 효과를 가져온다. 

즉, **Batch Normalization은 이전 Layer의 이동폭이 크지 않도록 똑같은 Mean, Variance 값을 가지게 해서 다음 Layer의 Train을 더 수월하게 해준다.** 

**Batch Normalization은 Regularization의 효과**가 있을 수 있다. 

![Untitled (17)](https://github.com/user-attachments/assets/9ed90eee-b92a-407a-b1bb-caa731a7aaf1)

각 Mini Batch에서는 해당되는 Mini Batch의 Data로만 Mean Variance를 계산했기 때문에 전체 Sample로 했을 구했을 때보다 더 Noisy하고, 결과적으로 ~z[l] 또한 Noisy해 질 것이다. 

그 결과, Dropout과 유사하게 Hidden Layer Activation에 Noisy를 더하게 되는 것이고, 이로 인해서 Regularization의 효과가 발생하는 것이다. 

그러나 Noisy의 정도는 작기 때문에, 큰 효과는 없다. 

만약 **Mini Batch의 Size가 크다면(64->256), Regularization의 효과를 감소**시키는 것이다.

**Batch Norm을 Regularization을 위해 사용하는 것은 권장하지 않는다.**

## Batch Norm at Test Time

<img width="350" height="200" alt="스크린샷 2024-07-24 오후 11 34 27" src="https://github.com/user-attachments/assets/ab85c75a-6253-43e1-ac2b-fcec19f225f5">

> Batch Normalization

Test Time에서는 Mini Batch Size만큼(64, 128, 256)의 Example들을 한번에 처리할 수 있는 Mini Batch가 없을 수 있다. 

그래서 Mean과 Variance을 구하기 위해서는 다른 방법이 필요할 것이다. 

만약 Test에서 1개의 Example 밖에 없는 경우에는, 1개의 값을 가지고 Mean과 Variance을 구하는 것은 적절하지 않기 때문이다.

그래서 Test Time에서 적용하기 위해서는 별도의 Mean과 표준 편차의 예측값을 구해야한다. 

**일반적인 Batch Normalization의 경우에는 Exponetially Weighted Average를 사용하고, 그 예측값은 Mini Batch에서의 Mean 값들으로 구한다.**

![Untitled (18)](https://github.com/user-attachments/assets/cbb5a4e8-1f21-4894-a11f-c42ac522326b)

즉, 특정 Layer에서 각 Mini Batch에서의 *μ*{1}[l], *μ*{2}[l], ⋯*μ*{1}[l], *μ*{2}[l],⋯을 θ로 사용해서 test에서 사용할 *μ*, *σ*2을 구해서, Batch Normalization을 적용하는 것이다.

위 방법이 아니더라도, 전체 Sample을 사용해서 Mean과 Variance의 예측값을 구해서 사용해도 되지만, 보통 EWA를 사용한다. 

**EWA**는 **Train에 사용되는 값들을 사용해서 예측값**을 구하기 때문에 **Running Average**라도 불리기도하며, 꽤 정확하고 Stable한 편이다.

## Softmax Regression

### **Multi Class Classification**

**Softmax Regression**이란 **Lositic Regression의 일반화 된 Version**으로 Binary Classification 대신 **여러 Class를 구분해야 하는 경우 사용하는 Model**이다. 

![Untitled (19)](https://github.com/user-attachments/assets/eb240649-ab39-4057-9a92-00d8bc1a7b9f)

> 고양이, 개, 병아리를 구분하거나 기타 Class를 포함하는 경우 

이때 새로운 표기법을 사용하는데, 대문자 C를 사용해 Category화 하려는 Class의 번호를 나타낸다. 

![Untitled (20)](https://github.com/user-attachments/assets/9f904748-408f-445c-9308-e41cfe829f11)

위와 같은 NN이 있을 때, Output Layer의 Unit 갯수는 4가 될 것이고 일반적으로 C라 표현한다. 

이 Output Layer는 4개의 Class 확률을 나타내는 것이고, 첫 번째 Node는 Other Class일 확률을 결과값으로 나타내고, 두 번째 Unit은 고양이일 확률, 세번째 Unit은 병아리일 확률을 나타낸다. 

4개의 결과값을 나타내기 위해 Output Layer의 결과값 Y-hat은 4 * 1의 차원을 가지게 된다. 

그리고 확률의 총합은 1이여야 하기 때문에 Y-hat에 있는 값들을 모두 합하면 1이 된다. 

이렇게 여러 Class에 대해 확률값을 나타낼 수 있도록 하는 Layer를 **Softmax Layer**라고 한다. 

![Untitled (21)](https://github.com/user-attachments/assets/d394cf17-b74f-4547-9f08-d0398614e6e7)

### **Calculation Softmax Function**

마지막 Layer에서는 Linear 부분을 아래와 같이 계산한다. 

<img width="849" alt="스크린샷 2024-07-19 오후 3 22 18" src="https://github.com/user-attachments/assets/9e239b42-c47e-4eb4-8d51-a0543a2db397">

z[L]을 구한 후 Softmax Activation Function을 적용해야 하는데 이는 일반적인 Activation Function와는 다르다. 

![스크린샷 2024-07-19 오후 3 21 00](https://github.com/user-attachments/assets/95b71195-39a4-46ac-a684-4dcdacdd4750)

a[L]의 차원은 4 * 1 Vector이다. z[L]이 아래와 같이 주어졌을 때, 

<img width="519" alt="스크린샷 2024-07-24 오후 11 53 53" src="https://github.com/user-attachments/assets/3efd1238-fdd0-454e-9fed-2dda36436ad9">

Vector t를 구하면 아래와 같다. 

<img width="519" alt="스크린샷 2024-07-24 오후 11 54 16" src="https://github.com/user-attachments/assets/da706413-d523-4d70-ac66-4cd7352fddfe">

그리고 a[L]은 다음과 같이 구할 수 있다. 

<img width="519" alt="스크린샷 2024-07-24 오후 11 54 46" src="https://github.com/user-attachments/assets/a6feaa76-ed7b-480c-8963-c4206f9d2b55">

이렇게 구해진 확률 값을 모두 더하면 1이 된다. 

**이전 Binary Classification에서는 Activation Function이 Single Row의 값을 Input으로 받았지만, Softmax Regression에서는 Input, Output 모두 Vector 값이다.** 

Hidden Layer가 없는 Softmax Regression의 경우, C = 3

<img width="597" alt="스크린샷 2024-07-24 오후 11 58 20" src="https://github.com/user-attachments/assets/25ee3849-0f14-430c-8953-865e3f0c9990">

Input으로 x1, x2를 가질 때 Model은 아래와 같이 계산된다. 

<img width="597" alt="스크린샷 2024-07-24 오후 11 58 50" src="https://github.com/user-attachments/assets/ad53705b-28f1-4b7f-bd40-ce3adc822a6e">

따라서 C = 3일 때 아래와 같은 차원을 가진다. 

<img width="716" alt="스크린샷 2024-07-24 오후 11 59 25" src="https://github.com/user-attachments/assets/a701fbbd-97ad-4341-b91c-331b6c473489">

Decision Boundary는 Logistic Regression과 유사한데, 각 Class간의 Decision Boundary가 Linear적으로 나타난다. 

결국 2개의 Class 간의 Decision Boundary가 Linear인 것이다.

C = 4, 5, 6인 경우에는 다음과 같이 나타날 수 있다.

<img width="716" src="https://github.com/user-attachments/assets/84845740-0aeb-48b3-b15e-6dc48d8e8ee6">

**DNN에서 Hidden Layer, Hidden Unit이 더 많은 경우 더 복잡한 Non-Linear Decision Boundary를 갖는 것도 가능하다.** 

## Training a Softmax Classifier

Softmax Layer를 사용하는 NN을 어떻게 Train시킬 수 있는지 확인한다. 

이전 Example에서 우리는 z[L]을 사용해서 4개의 Class의 확률을 아래와 같이 구했다.

![Untitled (23)](https://github.com/user-attachments/assets/f2bc78e5-a886-425c-a64a-53fa86b6f22a)

Hardmax의 경우에는 [1 0 0 0] 처럼 값을 가지는데, 가장 큰 확률을 가지는 요소가 1이 되고 나머지는 0이 되는 반면에, Softmax는 각 Class의 확률을 값으로 가지며, Softmax Regression은 C개의 Class를 가진 Logistic Regression을 일반화한 것이라고 할 수 있다. 

만약 C = 2라면, Softmax Regression은 Logistic Regression과 동일한 것이다.

우선 Train을 하는데 필요한 Loss Function Example를 정의해 실제로 Softmax Layer로 어떻게 학습시키는지 알아볼 것이다. 

만약 Sample에 대한 y의 값과 예측값 Y-har이 아래와 같이 예측되었을 때,

<img width="696" alt="스크린샷 2024-07-25 오전 12 08 24" src="https://github.com/user-attachments/assets/b3d4954d-3b4e-4388-bd1f-a7ca4195ebf7">

위와 같은 결과로 예측되었다면 좋지 않은 결과라고 할 수 있다. 

주로 사용되는 Loss Function은 아래와 같다. 

<img width="696" alt="스크린샷 2024-07-25 오전 12 09 17" src="https://github.com/user-attachments/assets/396aec7d-c7a0-46ff-84f7-ac5added4ee7">

위 Example에서 y1 = y3 = y4 = 0, y2 = 1이고 고양이로 분류된 Sample이다. 

결국 Loss Function은 y2에 대한 항만 남고 나머지는 0이 되어 아래와 같이 계산된다. 

<img width="696" alt="스크린샷 2024-07-25 오전 12 10 59" src="https://github.com/user-attachments/assets/dd68fa6a-35be-4cbe-8c44-34883f27ec48">

Loss를 줄이려고 한다면 Y-hat2를 최대한 크게 만들어야 한다. 

지금까지 한 개의 Train Example에 대한 Loss를 구현한 것이고 전체 Train Set에 대해서는 모든 Train Set의 Loss 합을 더해 구하게 된다. 

<img width="696" alt="스크린샷 2024-07-25 오전 12 12 16" src="https://github.com/user-attachments/assets/47ef0536-d070-4f50-91e6-1494ed217f74">

### Gradient Descent with Softmax

![Untitled (24)](https://github.com/user-attachments/assets/396c7d8d-c2a2-4e0f-825a-bb6410ae7eaa)

위와 같은 NN이 있을 때 Output Layer의 결과값은 FP를 통해 z[L] → a[L] = Y-hat을 구하고 구한 값으로 Loss L(Y-hat, y)를 계산한다. 

그리고 BP와 GD를 구해야 하는데 여기서 BP를 위해 초기화 해야하는 값은 Output Layer에서 z의 Derivative 즉, 미분항이며 

<img width="696" alt="스크린샷 2024-07-25 오전 12 15 09" src="https://github.com/user-attachments/assets/b34399a3-a34d-4f4a-87f6-28564f32f613">

와 같이 구할 수 있다. 

마찬가지로 dz[L]은 (4, 1) Vector이고 구한 값을 사용해 BP를 진행해 NN에서 필요한 Dericative를 구하면 된다. 

## Deep Learning Frameworks

![스크린샷 2024-07-23 오후 3 04 56](https://github.com/user-attachments/assets/96a60972-dbd7-4250-ab6d-b8030b9a2b0a)

Python과 Numpy를 사용해 DL Algorithm을 기초부터 구현해 본 경험은 매우 유익하다. 

그러나 더 복잡한 Model, 대규모 Model을 구현할 때 직접 모든 것을 구현하는 것은 실용적이지 않다. 

큰 Application을 구축할 때 직접 행렬 곱셈 Function을 구현하기보다는 **효율적으로 처리해주는 Library를 사용하는 것이 더 바람직**하다. 

DL도 마찬가지로 Framework를 사용하여 Model을 구현하는 것이 더 실용적이다. 

현재 여러 DL Framework가 있으며 각 Framework는 특정 Application에 적합하다. 

Framework 선택 시 몇 가지 기준을 고려해야 한다. 

### **DL Framework Selection Criteria**

1. Ease of Programming의 용이성
    - Network 개발 및 반복 수정, 실제 Production 배포 시 얼마나 쉽게 사용할 수 있는가
2. Run Speed 
    - 대규모 Data Set에서의 Training 속도가 얼마나 효율적인가
3. Open Source 
    - 단순히 Open Source인가를 떠나 SW 관리가 잘 되고 있는지, 단일 기업의 통제 하에 있는지 여부를 확인해야 한다.

일부 회사는 Open Source로 SW를 제공하지만 시간이 지나며 이를 점차 페쇄하거나 독점적으로 전환하는 경우가 있다. 

따라서 Open Source의 상태가 장기적으로 유지될 것인지, 신뢰할 수 있는지 여부를 주의 깊게 살펴봐야 한다. 

단기적으로 선호하는 Programming Language나 Application 분야에 따라 다양한 Framework가 적합할 수 있다. 

수치 선형 대수 Library보다 높은 수준의 추상화를 제공하여 ML Application 개발 시 더 효율적이다. 

## TensorFlow

TensorFlow의 자동 미분 기능과 Computation Graph를 통해 Backward Propagation을 자동으로 처리하여 효율적인 Training이 가능하다. 

Framework를 사용해 복잡한 NN을 쉽게 개발하고 다양한 Optimizer를 사용할 수 있다. 

TensorFlow는 DL Algorithm을 보다 효율적으로 개발하고 사용할 수 있게 도와준다. 

### **Cost Function Optimization**

J(w) = w^2 - 10w + 25라는 Cost Function을 사용한 Optimization 과정으로 이 Function은 (w - 5)^2로 최소값을 가지는 w는 5이다. 

![스크린샷 2024-07-19 오후 6 03 02](https://github.com/user-attachments/assets/66b8ad6d-71cb-483b-ae56-db66bdb46d72)

### **TensorFlow Directions**

1. TensorFlow와 Numpy를 Import하고 Parameter를 w로 초기화한다. 
2. Adam Optimizer를 설정하고 Learning Rate를 0.1로 설정한다. 
3. Cost Function을 정의하고 TensorFlow의 자동 미분 기능을 이용하여 Backward Propagation을 자동으로 처리한다. 

GradientTape을 사용해 Cost Function을 계산하고 Parameter w를 Update한다. 

Cost Function이 Data x, y에 의존하는 경우, Training Data를 TensorFlow Program에 통합할 수 있다. 

Data x를 이용해 Cost Function을 정의하고 Optimizer를 사용해 Training을 진행한다. 

TensorFlow는 Forward Propagation만 구현하면 자동으로 Backward Propagation을 처리한다. 

Computation Graph를 구성해 효율적으로 계산을 수행한다. 

Adam Optimizer 외 다른 Optimizer로 쉽게 교체가 가능하다. 

TensorFlow와 같은 DL Framework를 사용하면 복잡한 NN을 더 쉽게 개발할 수 있다. 

Framework가 제공하는 기능을 통해 빠르고 효율적인 DL Model 개발이 가능하다.