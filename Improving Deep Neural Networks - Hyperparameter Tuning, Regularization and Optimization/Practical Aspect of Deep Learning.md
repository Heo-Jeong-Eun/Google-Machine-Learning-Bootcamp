# Practical Aspect of Deep Learning

## Train / Dev / Test Sets

Training Set, Dev Set, Test Set를 잘 설정하면 좋은 성능을 갖는 NN을 빠르게 찾는데 도움이 된다. 

<img width="712" alt="스크린샷 2024-07-13 오후 2 40 43" src="https://github.com/user-attachments/assets/54679c0d-13a5-478d-aefd-1073211f494d">

NN을 구현할 때는 위와 같이 결정해야 할 Hyperparameter가 많은데 처음부터 적절한 Hyperparameter 값을 선택하는 것은 거의 불가능하다.

따라서 **반복적인 실험 과정을 통해 적절히 Hyperparameter를 바꿔가며 더 좋은 성능을 갖는 NN을 찾아야 한다.** 

![Untitled](https://github.com/user-attachments/assets/d16d7a9c-777d-4270-af43-0fce3261908a)

위와 같이 Data가 있을 때, 일부분을 **Training Set, Dev Set, Test Set**으로 분리할 수 있다. 

Training Set에서 Training Algorithm을 계속 수행하고, Dev Set을 통해 여러 다른 Model 중에서 어떤 Model이 가장 성능이 좋은지 평가한다. 

그리고 가장 적합한 Model을 찾았으면 Test Set을 통해 최종 Model을 평가한다. 

일반적으로 Data Set을 7 : 3으로 Training Set과 Test Set으로 분리하거나 Dev Set 즉, Cross Validation Set가 주어지면 60 : 20 : 20으로 Training, Dev, Test Set으로 분리한다. 

Data가 100, 1000, 10000인 경우 위 비율로 나누는 것이 적합하지만, Big Data의 경우 Data가 수백만이 넘어가기 때문에 이 경우 Dev Set과 Test Set은 전체 Data에서 작은 비율을 차지한다. 

Dev Set이나 Test Set은 Algorithm을 평가하기에 충분하기만 하면 되기 때문에 수백만의 Data가 있을 때 Dev, Test Set은 10000개 정도면 충분하다. 

결과적으로 최대한 적은, 거의 없는 Bias 값이 목표이기 때문에 Test Set을 사용하는데, Bias가 중요하지 않다면 Test Set은 없어도 괜찮다. 

즉, **Training Set로 학습을 진행하고 다른 Model을 시도하고, Dev Set에서 Model을 평가하고 반복을 통해 가장 좋은 Model을 찾는다.** 

이때 Dev Set을 Test Set으로 부르기도 한다. 

위와 같은 방법의 Algorithm은 Bias와 Variance 문제를 더 효율적으로 계산하고 개선할 수 있다. 

## Bias / Variance

<img width="752" alt="스크린샷 2024-07-13 오후 3 01 20" src="https://github.com/user-attachments/assets/fe58f20e-dcb4-4bff-bc8b-b6e5ae3b23cb">

첫 번째 High Bias Graph는 Data가 잘 Fitting 되지 않아 Underfitting 문제를 가지고 있다. 

반면 세 번째 High Variance Graph는 복잡한 분류기에 Fitting 했기 때문에 Data를 완벽히 Fitting시켜 Overfitting 문제를 가지고 있다. 

해당 Graph의 경우 x1, x2 두 개 Input에 대해 Decision Boundary를 그렸기 때문에 시각적으로 확인이 가능하지만 고차원의 Input을 가지는 경우 시각화가 어렵다. 

![Untitled (1)](https://github.com/user-attachments/assets/236626bd-0df5-4790-bae2-3f2df8b73f0f)

Cat Classification이 있고 Input이 고양이인 경우 Y = 1, 아닌 경우 Y = 0으로 나타낸다. 

이때 주목해야 할 것은 **Train Set Error**와 **Dev Set Error**이다. 

일반적으로 사람은 완벽하게 고양이를 인식하기 때문에 Human Error는 0에 가깝다. 

![Untitled (2)](https://github.com/user-attachments/assets/d4980e59-d145-4f4c-8ccd-bb92fd53c235)

만약 Train Set Error가 1%이고 Dev Set Error가 11%인 경우, **Training Set에서는 잘 동작하지만 Dev Set에서는 잘 동작하지 못하는 것**으로 볼 수 있다. 

이 경우 **Training Set에 Overfitting** 했다고 볼 수 있다. 

즉, 일반화 하지 못한 경우로 **High Variance** 문제를 갖고 있다고 할 수 있다. 

두 번째로 Train Set Error가 15%, Dev Set Error 16%라고 할 때 Human Error가 0%이라고 Training하면 해당 Algorithm은 **Training Set와 Dev Set 모두에서 잘 동작하지 못하는 것**이다. 

Training Algorithm이 Data에 잘 Fitting 되지 않은 경우로 **Underfitting** 문제라고 할 수 있다. 

이 Algorithm은 **High Bias** 문제를 갖고 있다고 할 수 있다. 

세 번째로 Training Set Error가 15%, Dev Set Error가 30%인 경우, 두 번째와 마찬가지로 이 경우는 Hight Bias 문제를 갖고 있다고 할 수 있다. 

그리고 Dev Set Error와 Training Set Error의 차이가 크기 때문에 Training Set에 Overfitting해서 High Variance 문제도 가지고 있는 최악의 경우라고 할 수 있다. 

마지막으로 Training Set Error가 0.5%, Dev Set Error가 1%인 경우가 있다. 

이는 꽤 잘 동작한다고 볼 수 있지만 Human Error가 0%에 근접하거나 Optimal Error 즉, Bayes Error가 거의 0%에 근접한다는 가정이 있어야 올바르다. 

만일 Optimal Error 또는 Bayer Error가 15%라면 두 번째 분류기는 나쁜 경우가 아니고, High Bias로 판단하지 않는다. 

즉, 흐릿한 이미지가 있어 사람도 잘 분류하지 못하고 Computer도 잘 분류하지 못하는 경우에는 Bias, Variance 분석 방법이 달라진다. 

**중요한 것은 Training Set Error를 살펴보며 Data가 얼마나 잘 Fitting 하는지 확인하고, Dev Set에서 얼마나 Variance 문제가 발생하는지 확인하는 것이다.** 

그렇다면 High Bias, High Variance는 어떻게 발생하는 것일까

![Untitled (3)](https://github.com/user-attachments/assets/a72de0fa-01f6-40cc-bf7f-d32cf8a1b3ae)

위 Graph에서 보라색 직선으로 분류되는 경우는 Data를 Underfitting 하기 때문에 High Bias를 가지고 있다. 

이러한 분류기는 대부분 선형이고, 선형이기 때문에 Underfitting 하게 된다.

만약 파란색 Graph처럼 Dat를 Overfitting 한다면 Curve Function이거나 2차 Function이기 때문에 High Variance 문제가 발생한다. 

그리고 고차원의 입력을 갖는 경우에도 Overfitting 즉, High Variance 문제를 가질 수 있다. 

그리고 보라색 직선이지만 가운데 두 개의 Sample에 Overfitting 하는 경우에는 High Bias, High Variance 두 가지 문제를 모두 가질 수 있다. 

## Basic “Recipe” for Machine Learning

<img width="638" alt="스크린샷 2024-07-13 오후 4 27 09" src="https://github.com/user-attachments/assets/45ac00c2-375a-47a2-b0e3-ac9d0419e5ac">

**Training Set Error, DEV Set Error를 이용해 Bias, Variance 문제를 진단할 수 있다.** 

이러한 정보가 ML의 System적인 부분을 알 수 있게 해주고 이를 Basic Recipe라고 하며 System적인 접근을 통해 Algorithm 성능에 기여할 수 있다. 

**Basic Recipe**

- 초기 Model을 Training 한 후 해당 Algorithm이 High Bias를 띄고 있는지 확인한다.
- 만약 High Bias 문제가 있다면 더 큰 Network / 다른 구조의 NN을 선택 / 더 길게 Training을 진행 / 더 최적화 된 Algorithm을 선택하는 방법이 이다.
- High Bias를 감소시켰다면 다음으로 High Variance 문제를 확인한다. 이때 Dev Set Error를 통해 확인할 수 있다.
- High Variance 문제가 있다면 더 많은 Data를 수집하는 것이 문제 해결에 가장 좋은 방법이다.
- 만일 Data를 구할 수 없는 상황이라면 Regularization을 시도할 수 있다.
- 적합한 NN 구조를 사용하는 것도 도움이 되며, 이 경우에는 High Bias 문제도 해결이 가능하다.

중요한 것은 High Bias, High Variance인지에 따라 해결 방법이 다르다는 것이다. 

보통 Training Set Error와 Dev Set Error를 통해 High Bias, High Variane를 판단하는데, 우선적으로 문제 여부를 확인하는 것이 해결의 시작이다. 

ML 초기에는 Bias-Variance-Tradeoff에 대해 많은 논의가 있었다. 

Variance가 줄어들면 Bias가 증가하고, Bias가 줄어들면 Variance가 증가할 수 있었기 때문이다. 

DL 초기, 적용 방법이 많이 없을 때 한 가지 요소의 희생 없이는 Bias나 Variance를 줄일 방법이 없었다. 

하지만 최근 Big Data에 맞게 큰 Network를 지속적으로 Training 시킬 수 있어 Bias는 Variance에 큰 영향을 미치지 않고 대부분 줄어든다고 볼 수 있다. 

즉, 한 가지 요소를 해결하는데 다른 요소에 악영향을 주지 않는다는 것이다. 

## Regularization

만일 NN이 Data에 **Overfitting** 하면 그것은 **High Variance** 문제가 있다는 것이고, 가장 먼저 시도해 볼 것은 **Regularization**이다. 

더 많은 Training Data Train 하는 방법이 있지만, 비용 문제가 있어 Regularization을 추가하는 것이 좋다. 

### **Logistic Regularization**

**Logistic Regression**에서 **Cost Function J를 최소화**하는 Parameter w, b를 구하는데 **Cost Function J에 Regularization항을 추가**하면 아래와 같다. 

<img width="688" alt="스크린샷 2024-07-13 오후 5 56 21" src="https://github.com/user-attachments/assets/087c8065-b637-46ea-9442-4872b39055b7">

Logistic Regression, Cost Function J + Regularization항 추가 

여기서 

<img width="688" alt="스크린샷 2024-07-13 오후 5 57 06" src="https://github.com/user-attachments/assets/e9a26d64-2bb5-4432-a646-049506ab1a20">

이고, w의 Norm 아래 첨자 2는 해당 Norm이 **L2 Regularization**이라는 의미이다. 

그리고 b에 대한 Regularization항 

<img width="390" alt="스크린샷 2024-07-13 오후 5 59 20" src="https://github.com/user-attachments/assets/98ee1a8b-2428-4ea4-85e6-da57ada53f4b">

b에 대한 Regularization항

를 추가할 수도 있지만 Parameter b는 단순한 상수이며 Parameter w는 보통 높은 차수이기 때문에 b를 추가해도 의미가 없으므로 생략한다. 

**L2 Regularization이 가장 일반적인 유형의 Regularization**이다. 

L1 Regularization을 사용하게 되면 아래와 같다. 아래 첨자로 1이 붙어있다. 

<img width="594" alt="스크린샷 2024-07-13 오후 6 01 45" src="https://github.com/user-attachments/assets/2aa450ca-30ba-4eb2-b034-c7b9bdd3004f">

L1 Regularization

만약 L1 Regularization을 사용한다면 w Vector 안에 0 값이 많이 존재하게 되어 Sparse 한 상태가 된다. 

Parameter Set이 0이고, Model에 저장하는데 더 적은 Memory를 사용해 Model을 압축하기에 유용하지만 그렇게 자주 사용되지 않고 대체로 L2 Regularization을 많이 사용한다. 

이때 *λ*는 Regularization Parameter이다. *λ*는 보통 Dev Set이나 CV Set을 사용해 적절한 λ 값을 설정하게 된다. 이것이 **Logistic Regression의 Regularization**이다. 

### NN Regularization

**NN에서 Regularization항을 추가한 Cost Function J**는 아래와 같이 정의된다. 

<img width="674" alt="스크린샷 2024-07-13 오후 7 58 49" src="https://github.com/user-attachments/assets/ebee02f1-b2d0-4e29-b35d-f5b07bf7bc3d">

NN에서 Regularization항 추가한 Cost Function J

Regularization에서 Norm은 아래와 같다. 

<img width="634" alt="스크린샷 2024-07-13 오후 7 59 17" src="https://github.com/user-attachments/assets/fa3bb7b3-790c-4a11-bdda-06e8ac636cc9">

Norm

여기서 w[l]은 (n[l], n[l - 1]) 차원으로 Norm은 Frobenius라고 불리며 아래 첨자 F가 붙어서 

![스크린샷 2024-07-13 오후 8 00 46](https://github.com/user-attachments/assets/360a29af-bea5-43b5-bb48-3a979fb49a5e)

로 쓰이기도 한다. 

이제 Regularization항을 Gradient에 추가하는 방법을 알아볼 것이다. 

Regularization항을 추가했기 때문에 dW는 아래와 같이 계산된다. 

<img width="652" alt="스크린샷 2024-07-13 오후 8 02 38" src="https://github.com/user-attachments/assets/748b218c-7eb7-4918-bbe5-b2a70ced4586">

그리고 Parameter는 아래와 같이 Update 된다. 

<img width="652" alt="스크린샷 2024-07-13 오후 8 02 52" src="https://github.com/user-attachments/assets/7ff068cb-3f83-4f58-b755-7c210ddefdc8">

(1 - *αλ / m*)가 1보다 작은 값이기 때문에 L2 Regularization은 **Weight Decay**라고 불리기도 한다. 

## Why Regularization Reduces Overfitting

**Regularization은 High Variance 문제, 즉 Overfitting 문제를 해결하는데 도움**이 된다. 

![Untitled (4)](https://github.com/user-attachments/assets/95b9c588-58fc-4b84-8902-cba932f35ae5)

위와 같은 NN이 있을 때

![Untitled (5)](https://github.com/user-attachments/assets/14cedfa8-6da3-490c-a99c-72bb9f958e9d)

이 NN이 세 번째 Graph처럼 Overfitting 한다고 가정한다. 

그리고 Cost Function J는 아래와 같이 정의된다. 

<img width="687" alt="스크린샷 2024-07-13 오후 8 06 24" src="https://github.com/user-attachments/assets/95cc62df-30db-4d34-b672-dd8d0e3dfa58">

**L2 Norm**은 **Weight 행렬이 너무 크면 줄여주는 역할**을 한다. 

이 역할이 Overfitting을 막아주는 역할은 아래와 같다. 

이때 Lambda를 크게 하면 w값을 0에 가까운 값으로 만들 것이다. W[l] ~= 0이 된다면 Hidden Unit의 영향을 거의 0으로 만들어 버린다. 

NN을 더욱 단조롭게 만들고 단순히 여러 층으로 이루어진 Logistic Regression과 유사하다고 볼 수 있다. 

이 경우 High Bias Graph 쪽으로 이동하게 되는데 이상적으로 Lambda 값을 적당한 값으로 설정해서 두 번째 Graph와 같이 만들 수 있다. 

실제 Hidden Unit이 0이 되는 것은 아니다. 지속적으로 Hidden Unit이 사용되지만 Hidden Unit의 효과와 영향이 줄어드는 것이다. 

![Untitled (6)](https://github.com/user-attachments/assets/9f38877e-290b-4eee-ae5b-be82e6acfc3b)

위와 같은 tanh Function이 있을 때 z의 값이 작으면 빨간색으로 표시된 tanh의 선형 부분만 사용한다. 

Regularization에서 Lambda 값이 크면 W[l] 값이 작아지고 결국 Z[l]의 값도 작아진다. 

<img width="675" alt="스크린샷 2024-07-13 오후 8 19 03" src="https://github.com/user-attachments/assets/44843a1c-8975-4dae-94c1-e03ec540e0a4">

Linear Regression처럼 모든 Layer가 선형적으로 동작하고, 모든 Layer가 Linear하다면 단순한 Linear Network가 될 것이다. 

**Activation이 Linear하기 때문에 Non-Linear Decision이 불가능해 Overfitting 할 수 없게 되는 원리이다.** 

## Dropout Regularization

L2 Regularization 외 **Drop Regularization**이라는 기법도 있다. 

<img width="681" alt="스크린샷 2024-07-13 오후 8 23 35" src="https://github.com/user-attachments/assets/d3388842-9c32-416b-9ed8-acb5233e6ba3">

위와 같은 NN에 Overfitting이 존재할 때, 각 Layer에 Node를 제거하는 확률을 설정한다. 

위 경우 0.5로 설정했는데 이 때 절반의 **Node가 제거**되고 아래와 같이 **감소된 Network로 FP와 BP를 진행**한다. 

![Untitled (7)](https://github.com/user-attachments/assets/f597e687-fd42-461a-a1ea-fd03a4a26601)

무작위로 Node를 삭제하지만 정상적으로 잘 작동한다.

Sample에 대해 **기존보다 더 작은 Network로 Training 하기 때문에 Regularization 효과를 얻는다.** 

### Inverted Dropout

<img width="686" alt="스크린샷 2024-07-13 오후 8 38 11" src="https://github.com/user-attachments/assets/d340b4eb-d1c4-4c43-bb89-b066ef90f392">

1. Dropout Vector d를 생성한다. 
    
    ```python
    d3 = np.random.rand(a3.shape[0], a3.shape[1])
    ```
    
2. d3가 keep_prob보다 작은지 확인한다. 여기서 keep_prob = 0.8이고 그렇다면 Node가 제거될 확률은 0.2가 된다. 
    
    ```python
    	d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
    ```
    
    d3은 0.8의 확률로 1이 되고 0.2의 확률로 0인 Vector가 된다. 
    
3. a3에 d3의 요소를 곱해준다.
    
    ```python
    a3 = np.multiply(a3, d3)
    ```
    
4. keep_prob으로 나누어준다. 
    
    ```python
    a3 /= keep_prob
    ```
    

만약 50개의 Unit이 있는 Layer라면 10개의 Unit은 제거되어 0이 된다. 

<img width="447" alt="스크린샷 2024-07-13 오후 8 33 41" src="https://github.com/user-attachments/assets/60785031-4406-4cf0-9fc4-805faed9949d">

Z[4]는 위와 같고 a[3]이 20%까지 감소한다는 것을 알 수 있다. 

이때 Z[4]는 Expected Value 값을 감소시키지 않기 위해 a3에 0.8을 나누는 것이다. 

그렇다면 a3의 Expected Value는 변하지 않고 Z[4]의 Expected Value도 변하지 않는다. 

keep_prob을 1로 설정하면 Dropout을 적용하지 않는 것과 동일하다. 

d3는 어떤 Unit을 0으로 만들지 결정하는데, FP와 BP에 모두 사용된다. 

이때 주의해야할 것은 **Test Set에서 Dropout을 사용하지 않는 것**이다. 

Test에서 예측할 때 Dropout을 사용하면 예측값이 Random하게 되고 결국 예측값에 Noise만 추가될 뿐이다. 

그리고 keep_prob을 나누어주는 연산을 했기 때문에 Test에서 Dropout을 사용하지 않더라도 Test의 Expected Value는 변하지 않는다. 

## Understanding Dropout

Dropout은 무작위로 Unit을 제거하는 역할을 하는데, Regularization이 어떻게 잘 동작할 수 있을까

**무작위로 Unit을 제거하게 되면 더 작은 NN으로 학습하는 것과 동일하기 때문에 더 작은 NN을 사용하면 Regularization 효과를 가져가는 것처럼 보인다.** 

단일 Unit의 관점에서 볼 때, 

![Untitled (8)](https://github.com/user-attachments/assets/edb80183-e875-496e-9d14-4d2f6fcaf9ee)

NN에서 Dropout을 도입하게 되면 위와 같이 Unit이 제거될 수 있고, 또 다른 Unit이 제거될 수도 있다. 

이런 상황에서 의미있는 결과 값이 되기 위해서는 보라색으로 표시된 **Unit이 어떤 한가지 Feature에 의존하면 안된다.** 

왜냐하면 **Dropout에서는 그 특성이 임의로 제거될 수도 있기 때문**이다.

때문에 **하나의 Input에 큰 Weight를 주지 않고 Weight을 분산**시키게 된다. 

Weight를 분산시키면 **Weight의 Norm을 축소시키는 효과**가 있다. 

따라서 L2 Regularization과 유사하게 Weight를 줄이고 Overfitting 문제를 해결해준다. 

<img width="689" alt="스크린샷 2024-07-13 오후 8 51 17" src="https://github.com/user-attachments/assets/fa4b6908-b39a-41c1-94dd-765c1833c60b">

Input Feature가 3개인 NN

 Dropout을 적용할 때 선택해야 하는 매개변수 중 하나는 **keep_prob**인데, 이 값은 **Layer 별로 설정이 가능**하다. 

예를 들어 Layer 1은 0.7, Layer 2는 0.5, Layer 3은 0.7, Layer 4, 5는 1로 설정할 수 있다. 

1로 설정한 것은 Dropout을 적용하지 않는다는 의미이다. 

W1은 7 * 3 Matrix, W2은 7 * 7 Matrix, W3은 3 * 7 Matrix이고 W2가 가장 큰 Matrix라고 할 수 있다. 

W2 Matrix의 Overfitting을 방지하기 위해 Layer 2의 keep_prob을 낮게 설정할 수 있다. 

즉, Overfitting이 우려되는 Layer에 keep_prob을 낮게 설정하고, 반대인 Layer는 높게 설정하는 것이다. 

keep_prob보다 높게 설정하는 것은 L2 Regularization에서 Lambda 값을 증가시키는 것과 동일하다. → 다른 Layer보다 더 Regularization 하기 위해

Input Layer에도 Dropout을 적용할 수 있긴 하지만 잘 사용하지는 않는다. 

### Dropout Weakness

Dropout을 사용하게 되면 Cross Validation에서 설정해야하는 **Hyperparameter가 증가**한다는 것이 단점이다. 

또한 **Cost Function J가 명확하게 정의되지 않는다.** 매 반복마다 Node를 무작위로 제거하기 때문에 Gradient Descent 성능을  Double Check하기 어렵다. 

## Other Regularization Methods

L2 Regularization, Dropout 외 Overfitting을 줄일 수 있는 방법이 있다. 

![Untitled (9)](https://github.com/user-attachments/assets/d14bf9df-3d53-4dee-a37b-1248706c7ea3)
Data를 더 이상 수집할 수 없는 경우 **Data Augmentation**이라는 것을 사용한다. 

Image를 가로로 뒤집거나 회전, Zoom In, 찌그러뜨리거나 확대해 Data Set을 확보하는 것으로 새로운 Data를 추가하는 것보다 좋지 않지만 시간을 절약해주는 장점이 있다. 

이 방법을 통해 큰 비용 없이 Overfitting을 줄일 수 있다. 

![Untitled (10)](https://github.com/user-attachments/assets/bd147db0-6a90-47af-9bf2-1e9b8261fc71)

**Early Stopping**은 **Gradient Descent를 실행하면서 Training Set Error나 J Function과 Dev Set Error Graph를 그려 w가 Middle Size일 때 중지시키는 방법**이다. 

Training 초기 Parameter w를 0에 가깝게 작은 값으로 초기화하고 Training 하는 과정에서 w는 증가하게 되는데 w의 비율이 중간정도 되는 지점에서 중지한다. 

**ML Process**

1. Cost Function J를 Optimize → Gradient Descent
2. Not Overfit → Regularization

위 두 단계가 서로 영향을 미치지 않도록 **Orthogonalization**이라는 것이 있다. 

Early Stopping을 사용하면 위 **두 단계가 함께 적용**되고, **개별의 문제를 각각 따로 해결할 수 없게 된다.** 

**즉, Overfitting을 해결하기 위해 Gradient Descent를 중간에 멈추기 때문에 최적화를 완벽하게 하지 못하고 J를 더이상 줄이지 못하게 된다.** 

따라서 Early Stopping 대신 L2 Regularization을 사용하는 경우 더 길게 학습하면 되고, 여러 값의 Lambda를 Test 해야하지만 Hyperparameter를 분해하는데 도움이 된다. 

다만 **Early Stopping은 Gradient Descent 한번으로 w 값을 설정할 수 있어 단점이 있지만 종종 사용되는 방법**이다. 

하지만 L2 Regularization을 사용하는 것을 추천한다. 

## Normalizing Inputs

**Training 속도를 높이는 방법** 중 하나로 **Input을 Normalization** 하는 것이다. 

![Untitled (11)](https://github.com/user-attachments/assets/d3e3cf4f-f16f-4f4f-994b-2fa96f726cd7)

두 개의 Input이 있는 경우 Input을 Normalization 하는 방법은 아래 두 단계로 이루어진다. 

1. **평균을 빼거나, 0으로 만든다.** 
    
    <img width="603" alt="스크린샷 2024-07-14 오전 12 29 53" src="https://github.com/user-attachments/assets/5d62ff6b-445f-479f-84bc-edd386e91fed">
    
    위와 같이 평균 *μ*을 구해 x에 해당 값을 빼준다. 
    
    <img width="583" alt="스크린샷 2024-07-14 오전 12 31 15" src="https://github.com/user-attachments/assets/6825d779-2f36-499c-ac6e-cc3793b76f68">
    
    이 과정을 거치면 아래와 같이 Graph가 이동한다. 
    
    <img width="612" alt="스크린샷 2024-07-14 오전 12 32 04" src="https://github.com/user-attachments/assets/358d84bf-b39f-45c0-aa6d-a26a4fea316b">
    
2. **Normalize Variance**
    
    <img width="644" alt="스크린샷 2024-07-14 오전 12 34 56" src="https://github.com/user-attachments/assets/dbfeae90-ded9-4adb-bf5c-cd1881d47af1">
    
    Variance *σ^2*를 구해 표준편차 *σ*를 x에 나누어준다. 
    
    <img width="644" alt="스크린샷 2024-07-14 오전 12 38 07" src="https://github.com/user-attachments/assets/c8a58646-793c-45a0-8b83-a1172fa51ae9">
    
    <img width="644" alt="스크린샷 2024-07-14 오전 12 38 25" src="https://github.com/user-attachments/assets/13f58452-58d3-495b-9d97-3abdccb3ebae">
    
    위와 같이 Normalization 해주게 되면 Graph는 아래와 같이 나오며 x1, x2의 편차가 모두 1이 된다. 
    
    <img width="644" alt="스크린샷 2024-07-14 오전 12 39 21" src="https://github.com/user-attachments/assets/3f6231b2-0dde-4239-b38e-ae67c7be55e3">
    
    만일 이 방법을 이용해 Training Set에 적용한다면 똑같은 *μ, σ*를 Test Set에 적용해야 한다. 
    
    Training Set과 Test Set을 다르게 Normalization 하는 것은 좋지 않다. 
    

### Normalize Input

<img width="644" alt="스크린샷 2024-07-14 오전 12 41 27 (1)" src="https://github.com/user-attachments/assets/56aaf7c1-4830-4d6f-b255-0312494209c4">

표준화 되지 않는 Input Feature를 사용하게 되면 아래와 같은 J Graph를 얻을 수 있다. 

![Untitled (12)](https://github.com/user-attachments/assets/d1144997-22c4-439e-a29d-cf27693fed6e)

만약 x1의 범위가 1-1000이고 x2의 범위가 0-1이면 w1, w2의 범위가 매우 다른 값을 띄게 된다. 

따라서 위와 같은 Graph가 나오게 되고, 해당 Graph의 Contour를 그려보면 아래와 같다. 

![Untitled (13)](https://github.com/user-attachments/assets/5eb48074-8ae0-4829-a1e8-a92038dcd63e)

Feature를 Normalization하면 Cost Function은 평균적으로 더 대칭적인 성향을 띄게 되고, Graph와 Contour는 아래와 같다.

![Untitled (14)](https://github.com/user-attachments/assets/0f5a02f0-8531-4099-ab10-c14b17f749b8)

만약 Normalization이 되지 않은 Feature를 사용하게 되면 매우 작은 Learning Rate를 사용해야 한다. 

Gradient Descent를 수행하면 더 많은 단계를 거쳐 최소값에 도달하기까지 계속 왔다갔다 할 수 있기 때문이다. 

![Untitled (15)](https://github.com/user-attachments/assets/6fefaccc-b1dd-46ce-8ab8-ede62bdb3b19)

**Normalization을 사용해 더 구형의 Contour를 띄고 있다면, 어디에서 시작하더라도 Gradient Descent가 바로 최소값에 도달할 수 있다.** 

![Untitled (16)](https://github.com/user-attachments/assets/24817457-28a5-4edd-a5ca-c7bba83c4d43)

실제 Parameter w가 고차원의 Matrix이기 때문에 Graph로 나타내기는 힘들어 정확하게 전달되지 않는 부분도 있지만, 주로 Cost Function은 더 구형을 띄고 Feature를 유사한 Scale로 맞추게 되면 더 쉽게 최적화 할 수 있다. 

만약 x1이 0-1의 범위를 갖고, x2가 -1-1의 범위, x3이 1-2의 범위를 갖는다면 이 Feature들은 서로 비슷한 범위에 있기 때문에 실제로 잘 동작한다. 

범위에 따라 크게 다른 경우, 평균을 0으로 만들고 Variance을 1로 만든다면 Training이 더 빨라질 수 있다. 

## Vanishing / Exploding Gradients

DNN을 Training 할 때 가장 큰 문제점 중 하나는 Gradient Descent가 매우 작아지거나 커지는 경우이다. 

이것은 NN의 미분항이나 기울기가 매우 작아지거나 커지는 것을 의미하며 이 경우  Training이 매우 까다롭다. 

Vanishing, Exploding Gradient의 문제점이 무엇인지 알아보고 Random Weight Initalization을 통해 문제를 줄이는 것을 알아볼 것이다. 

![Untitled (18)](https://github.com/user-attachments/assets/0501f8cd-974b-40d6-b3a3-db90910cedee)

위 NN은 Unit의 수는 적지만 Deep한 NN의 한 예시이다. 

Activation Function은 g(z) = z로 Linear하며, Parameter b = 0으로 무시한다. 

이 경우 결과 값은 

<img width="644" alt="스크린샷 2024-07-14 오전 12 41 27" src="https://github.com/user-attachments/assets/561bb96d-44a8-4530-8d18-abec0e4be42d">

가 될 것이다. 

여기서 z[1]은

<img width="631" alt="스크린샷 2024-07-14 오전 1 01 44" src="https://github.com/user-attachments/assets/0a095c11-0979-45fd-aab6-44b43307694f">

가 된다. 

만약 W[l]의 값이 1보다 약간 큰 

<img width="631" alt="스크린샷 2024-07-14 오전 1 02 47" src="https://github.com/user-attachments/assets/5fdedf5d-0325-4975-8b5b-1f8a7bf900d3">

이라고 한다면, 결과 값은 아래와 같다. 

<img width="580" alt="스크린샷 2024-07-14 오전 1 03 36" src="https://github.com/user-attachments/assets/c59e62e6-900c-4f80-b4ca-1df36a6fefae">

결국 결과값에 1.5^L이 곱해지므로 매우 Deep한 NN일 경우 결과값 Y-hat이 기하급수적으로 증가할 것이다. 

반대로 1.5를 0.5로 변경하게 되면 0.5^L이 되어 결과값은 매우 작은 값이 될 수 있다. 

결과적으로 w의 비중 즉, Matrix보다 크거나 작은 경우에 따라서 Activation Unit이 매우 커지거나 작아질 수 있다는 것이다. 

기울기가 매우 크거나 작다면 Training이 어렵다. 

특히 기울기가 매우 작은 경우 Gradient Descent에 많은 시간이 걸려 Training이 오래 걸리게 된다. 

## Weight Initalization for Deep Networks

Vanishing, Exploding Gradient 문제를 해결하기 위해 완전하지는 않지만 Random 초기화를 잘하면 어느 정도 문제를 해결할 수 있다. 

<img width="722" alt="스크린샷 2024-07-14 오전 1 09 55" src="https://github.com/user-attachments/assets/4503191d-9e14-407a-8227-454dc9ec1665">

어느 Layer에서 Input Feature의 갯수가 n개라면 z는 위와 같이 나타낼 수 있다. → b는 생략

이 경우 적절한 z값을 갖기 위해 n이 많을 수록 w는 더 작아져야 한다. 

이를 위한 한 가지 방법은 Var(𝑤𝑖) = 1 / n으로 두는 것이다. 

Code 상에서는 특정 Layer에서 다음과 같이 Random 초기화를 할 수 있다. 

```python
W[l] = np.random.randn(shape) * np.sqrt(1 / n[l - 1])
```

만약 Activation Function으로 ReLU를 쓰는 경우 1 / n이 아닌 2 / n을 사용하는 것이 더 잘 동작한다. → (Var(𝑤𝑖) = 2 / n)

이렇게 표준정규분포로 Random 초기화를 하고 np.sqrt(1 / n[l - 1])을 곱하는 것이 w의 Variance을 1 / n으로 만드는 것이다. 

Activation Function에 따라 Variance은 아래와 같이 다르게 설정될 수 있다. 

<img width="551" alt="스크린샷 2024-07-14 오전 1 17 38" src="https://github.com/user-attachments/assets/59dbd7ae-c43f-4abb-b61d-0b7884a53408">

위 방법들은 w의 Variance(편차값)의 기본값을 제시한다. 따라서 Variance이 Hyperparameter일 수 있다. 

## Numerical Approximation of Gradients

BP를 사용하는 경우 **Gradient Checking**이라는 Test를 통해 **BP의 과정이 올바른지 확인할 수 있다.** 

<img width="683" alt="스크린샷 2024-07-14 오후 6 29 45" src="https://github.com/user-attachments/assets/10a573f7-65c4-48bb-b445-82dfeb803d54">

위와 같은 Graph가 갖는 𝑓(𝜃) Functiuon의 기울기를 구할 때, 구하는 지점의 𝜃 = 1이다. 그리고 *ϵ*을 0.01로 두고 **𝜃에서 빼고 더한다. 

**𝜃에서의 기울기를 구하는데, 작은 삼각형으로 구하는 것보다 큰 삼각형을 사용해 구하는 것이 더 정확하다.** 

높이를 너비로 나누어 기울기를 구하는데 아래와 같이 기울기의 근사치를 구할 수 있다. 

<img width="683" alt="스크린샷 2024-07-14 오후 6 35 40" src="https://github.com/user-attachments/assets/36de4065-242b-4640-be4b-9da275595f19">

기울기의 근사치를 구하게 되면 3.0001이 나온다. 

<img width="683" alt="스크린샷 2024-07-14 오후 6 36 47" src="https://github.com/user-attachments/assets/2194aafc-6a54-474f-ba64-06005ff691e7">

f의 미분값은 위와 같고 1에서의 미분값은 3이기 때문에 실제 미분항과 수학적으로 구한 기울기의 오차는 0.0001이다. 

만약 작은 삼각형의 기울기 즉, One-Sided Difference를 구하게 되면 

<img width="663" alt="스크린샷 2024-07-14 오후 6 38 38" src="https://github.com/user-attachments/assets/ad6fff62-36df-45f7-bcc4-279d3cf2a6e3">

이며 계산값은 3.0301로 미분값과 0.0301의 오차를 가지게 된다. 

따라서 미분값의 근사치는 큰 삼각형 즉, Two-Sided Difference이 미분값과 더 유사해 정확도가 높다. 

<img width="663" alt="스크린샷 2024-07-14 오후 6 40 56" src="https://github.com/user-attachments/assets/7a31f52b-165d-42fd-9f59-5bfcab166a76">

미분학에서 기울기 구하는 공식 

## Gradient Checking

Gradient Checking은 BP를 사용할 때 Bug를 찾아줌으로서 문제점을 해결하는 시간을 절약해준다. 

<img width="698" alt="스크린샷 2024-07-14 오후 6 45 31" src="https://github.com/user-attachments/assets/6b703655-775e-4e33-b709-bbe2d4e826eb">

**Gradient Checking**

1. 모든 Parameter를 Vector로 변환 후 연결시켜 Big Vector로 만들어준다. 
2. dW, db를 d𝜃로 만들어서 J(𝜃)의 기울기와 비교한다. 

<img width="698" alt="스크린샷 2024-07-14 오후 6 46 57" src="https://github.com/user-attachments/assets/1aa18676-f3e2-4248-a7f7-cf2dc072ab9f">

Gradient Checking은 위와 같이 이루어진다. 각 i마다 기울기의 근사치를 위 식으로 구한다. 

그리고 BP를 통해 구한 d𝜃와 비교하는데, 아래와 같은 식으로 값을 구한다. 

<img width="678" alt="스크린샷 2024-07-14 오후 6 49 06" src="https://github.com/user-attachments/assets/1534f38c-0e7f-4914-a2b1-cb202819614c">

이때 비교값은 *ϵ* = 10^−7로 설정하며, 이 값은 기울기 근사치를 구할 때 *ϵ*와 관련은 없다. 때문에 비교값보다 작으면 좋은 결과이다. 

만약 10^-5의 값을 가지게 된다면 적당한 결과이지만 다시 살펴볼 필요가 있고, 10^-3으로 크다면 좋지 못한 결과로 다시 살펴보아야 한다. 

즉 ***ϵ*만큼의 값이 나오거나 더 작은 값**이 나온다면 **BP의 과정이 정상**이라는 것이다. 

## Gradient Checking Implementation Notes

<img width="697" alt="스크린샷 2024-07-14 오후 6 53 27" src="https://github.com/user-attachments/assets/4e3135da-0851-4e98-8e6c-ce3ccfeb42d3">

1. 모든 i에 대해 기울기 근사치를 구하는 것은 계산 시간이 매우 오래 걸린다. 따라서 Debugging 할 때만 사용하고, 충분히 오차가 작은 값이 나온다면 Gradient Checking은 끄는 것이 좋다. 
2. Algorithm이 Gradient Checking에 실패했을 때 값이 크게 다른 i를 찾아 어느 Layer에 Bug가 발생하는지 찾을 수 있다. → 항상은 하니지만 Bug를 Tracking 할 수 있다. 
3. Regularization항이 있다면 d𝜃에도 추가해야 한다. 
    
    ![Untitled (19)](https://github.com/user-attachments/assets/51bdb8c8-0117-4549-8ae8-dfcea84db505)
    
4. Dropout에서는 Unit을 임의로 제거하기 때문에 Gradient Checking은 Dropout에서는 동작하지 않는다. → J가 제대로 정의되지 않는다. 
5. 자주 사용하지 않지만 Parameter가 0에 가까울 때, Gradient Descent가 잘 동작할 수도 있다. 
    
    하지만 Gradient Descent를 진행하며 Parameter w, b가 커지면 BP는 오직 w와 b가 0에 가까울 때만 잘 동작할 수 있고, w와 b의 값이 커질수록 정확도가 떨어지게 되는 것이다. 
    
    한 가지 방법은 Gradient Checking을 Random Initalization에서 실행시키고 어느 정도 Network Training을 진행한 뒤 다시 Gradient Checking을 실행하는 것이다.