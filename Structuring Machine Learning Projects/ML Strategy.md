# ML Strategy

## Why ML Strategy ?

<img width="687" alt="스크린샷 2024-07-23 오후 2 57 06" src="https://github.com/user-attachments/assets/791770d9-9600-4c66-8158-170a4a633473">

## **Orthogonalization**

**Orthogonalization**는 하나의 동작이 하나의 기능만 수행하는 것을 의미한다.  

ML System을 구축할 때,  Adjustment Elements가 많다.

효과적인 ML Engineer는 각 Adjustment Elements가 특정한 문제를 해결하는 데 어떻게 기여하는지 명확히 이해하고 있다. 

<img width="690" alt="스크린샷 2024-07-23 오후 3 14 22" src="https://github.com/user-attachments/assets/be939631-6b28-4122-92ab-597463628d03">

Orthogonalization가 적용되어 있다면 Tuning이 훨씬 쉬워지기 때문에 ML System에도 아래와 같이 Orthogonalization을 사용할 것이다. 

![Untitled](https://github.com/user-attachments/assets/e3bf5952-aae4-4633-a4fa-01ff6f2852e7)

**Train Set → Dev Set → Test Set**의 순서로 잘 동작해야 한다. 

Train, Dev, Test가 모두 잘 동작해 실제 상황에서도 잘 동작하는 System이 되도록 해야한다. 

### **ML System Adjustment**

만약 **Trining Set**에서 Cost Function이 잘 맞지 않는다면 **더 큰 Network Train, 더 나은 Optimization Algorithm 사용**으로 이를 해결할 수 있다. → Adam

Algorithm이 Dev Set와 잘 맞지 않는 경우 즉, Train Set에서는 잘 동작하고 **Dev Set에서만 잘 동작하지 않는 경우**에는 **L2 Regularization 사용, 더 큰 Train Set 사용**으로 이를 해결할 수 있다. 

Test Set에서 잘 동작하지 않는 경우 즉, **Dev Set에서는 잘 동작하고 Test Set에서만 잘 동작하지 않는 경우** **Dev Set에 Overfit** 되어있을 확률이 높다. 

이 경우 **이전 단계로 돌아가 더 큰 Dev Set 사용하는 것**이 좋다. 

마지막으로 Test Set에서는 잘 동작하고 **실제 Train되지 않은 새로운 Data에서는 잘 동작하지 않는다**면 **이전 단계로 돌아가 Dev Set을 바꾸거나 Cost Function을 변경**해야 한다. 

**Dev, Test Set의 분포가 잘 설정되어 있지 않거나 Cost Function이 올바르게 계산되고 있지 않은 것**으로 볼 수 있다. 

### Early Stopping

NN을 Train 시키는 경우 Early Stopping은 잘 사용하지 않는다. 

너무 일찍 Train을 멈추게 되면 **Train Set에서 잘 동작하지 않을 수 있지만 Early Stopping은 Dev Set의 성능을 향상시키는데 도움이 된다.** 

즉, Orthogonalization 하지 않아 동시에 2개의 요소에 영향을 준다. 

따라서 사용할 때 주의가 필요하고 그보단 최대한 Orthogonalization된 Tuning 방법을 선택하는 것이 좋다. 

### **ML Strategy**

각 문제에 대한 적절한 Adjustment Elements를 명확히 파악한다. 

병목 지점을 진단하고 이를 해결하기 위한 Adjustment Elements를 사용한다. 

## **Single Number Evaluation Metric**

ML Project에서 Hyperparameter Tuning이나 Train Algorithm을 시도할 때, **Single Real Number Evaluation Metric**가 있으면 어떤 변화가 성능을 개선했는지 빠르게 판단할 수 있다.

Single Real Number Evaluation Metric는 새로운 시도가 이전 아이디어보다 더 나은지 쉽게 비교할 수 있도록 도와줍니다.

<img width="690" alt="스크린샷 2024-07-23 오후 4 17 49" src="https://github.com/user-attachments/assets/c23d5593-05ed-49c1-b2e1-26640ae2745e">

**ML은 Empirical Process**, 경험을 토대로 진행되는 절차를 가진다. 

이 반복적인  과정을 통해 Algorithm을 개선해 나가는 것이다. 

기존에 만들어진 Classifier A와 새로운 Classifier B를 비교하는 Example의 경우, Precision와 Recall을 평가 지표로 사용할 수 있다.

**Precision** 

**정확도, 정밀도** True로 판별한 것들 중 실제로 Ture인 것의 비율이다. 

**Recall** 

**검출율, 재현율** 검출된 결과가 얼마나 정확한지, 검출된 결과들 중 실제로 일치는 것을 말하며 실제 True인 것들 중에서 Ture라고 판별한 비율을 의미한다. 

Precision는 Classifier가 Cat라고 인식한 Image 중 실제 Cat인 비율을 나타내며, Recall은 실제 Cat Image 중 올바르게 인식된 비율을 나타낸다.

Precision와 Recall을 동시에 평가하면 Classifier의 성능을 정확히 판단하기 어려운 경우가 많다.

이를 해결하기 위해 **F1 Score**를 사용하여 **Precision와 Recall을 결합한 Single Real Number Evaluation Metric**를 만든다.

![스크린샷 2024-07-25 오후 4 34 08](https://github.com/user-attachments/assets/8a7029ae-9aea-457b-9fb9-c27862040db2)

F1 Score는 Precision와 Recall의 **Harmonic Mean**으로 계산된다.

<img width="677" alt="스크린샷 2024-07-25 오후 4 35 04" src="https://github.com/user-attachments/assets/fb38f8aa-4a83-4806-819f-a8ad710e5732">

위 Image을 보면 Classifier A가 B보다 더 좋은 F1 Score를 갖는 것을 알 수 있다. 

<img width="723" alt="스크린샷 2024-07-25 오후 4 36 10" src="https://github.com/user-attachments/assets/f7dc889c-f042-4adb-aa0f-69aece7a43f2">

Cat Application을 미국, 중국, 인도 및 기타 지역에서 사용할 때, 2가지 Classifier가 있고 각 대륙별로 다른 값의 Error를 가진다고 가정한다. 

4개의 지표를 가지고 A가 나은지, B가 나은지 결정하기는 쉽지 않다. 

그리고 더 많은 Classifier로 Test하게 되면 어떤 Algorithm이 더 우수한지 빠르게 판단하기 어렵다.

<img width="723" alt="스크린샷 2024-07-25 오후 4 38 01" src="https://github.com/user-attachments/assets/108b2015-16db-499a-b56c-b9253fc2667f">

이 경우, 대륙별 Error로 Mean을 계산해 Single Real Number Evaluation Metric로 사용하면 성능을 빠르게 비교할 수 있다.

### **Efficient Decision Making**

Single Real Number Evaluation Metric를 사용하면 다양한 Idea를 신속하게 Test하고, 어떤 Idea가 더 나은지 쉽게 판단할 수 있다.

이를 통해 ML Algorithm의 성능을 개선하는 반복적인 과정을 가속화할 수 있다.

## **Satisficing and Optimizing Metrics**

ML System을 평가할 때 Single Real Number Evaluation Metric로 모든 중요한 요소를 결합하는 것은 쉽지는 않다.

이러한 경우, **Satisficing Metrics와 Optimizing Metrics**를 설정하는 것이 유용할 수 있다.

<img width="690" alt="스크린샷 2024-07-23 오후 4 21 01" src="https://github.com/user-attachments/assets/289cbdbf-8509-4f25-b7e4-ccd28d01371d">

Cat Classifier의 분류 Precision와 실행 시간을 고려해야 한다고 가정한다. 

**실행 시간을 고려할 때, Precision와 실행 시간을 결합한 Single Real Number Evaluation Metric를 만들 수 있다.** 

예를 들어, **총 Cost는 Precision - 0.5 * 실행 시간으로 정의**할 수 있다.

그러나 이와 같은 공식으로 Precision와 실행 시간을 결합하는 것이 인위적일 수 있다.

대신, Precision를 Optimaze 지표로 설정하고 실행 시간을 만족 지표로 설정하여 실행 시간이 100ms 이하일 때 최대한 높은 Precision를 달성하는 **Classifier**를 선택하는 방법이 있다. 

Precision은 높으면 높을수록 좋지만 실행 시간은 어느정도 만족할 수치가 된다면 더이상 신경쓰지 않아도 된다. 

따라서 **실행 시간의 최대치는 100ms 이내로 설정하고 이 수치를 만족하는 Classifier 중 가장 Precision이 높은 것을 선택하는 것**이다. 

이때 **Precision은 Optimizing Metrics**, **실행 시간은 Satisficing Metrics**라고 한다. 

### **Optimizing and Satisficing Metrics**

**Optimizing Metrics**는 최대한 잘 수행하고자 하는 지표를 의미한다.

**Satisficing Metrics**는 일정 임계값을 넘으면 더 이상 신경 쓰지 않는 지표를 의미한다.

일반적으로 **고려해야하 하는 N개의 Matrix가 있으면 1개를 Optimizing Metrics로 선택**하고 **나머지 N - 1개를 Satisficing으로 선택하는 것이 합리적**이다. 

Trigger Word를 감지하는 System을 구축한다고 가정한다.

Precision를 Optimizing Metrics로 설정하고, 24시간 동안 최대 한 번의 False Positive만 발생하도록 Satisficing Metrics를 설정할 수 있다.

Trigger를 말했을 때 System이 제대로 반응하도록 하고, 그렇지 않을 때는 하루에 한 번 이하로만 잘못 반응하도록 하는 것이다.

즉, **고려해야 할 요소가 여러 가지라면 가장 좋은 결과를 얻고 싶은 한 가지 조건은 Optimizing Metrics로 설정하고 나머지는 Satisficing Metrics를 통해 만족할만한 수치의 범위를 설정하면 된다.** 

## **Train / Dev / Test Distributions**

**Dev Set**은 Development Set으로 **Cross-Validation Set**라고도 불린다. 

일반적으로 ML 업무의 Process는 다양한 Idea를 시도하고 Code화 해 결과를 평가, 그 중 가장 잘 구현 된 것을 선택한다. 

이러한 반복적인 과정을 통해 Dev Set의 성능을 지속적으로 개선하면서 최종적으로 하나를 선택해 Test Set으로 평가해 보는 것이다. 

<img width="690" alt="스크린샷 2024-07-23 오후 4 33 23" src="https://github.com/user-attachments/assets/df9a4443-fc61-4eae-80f4-6d9c7e6bccdc">

Cat Classifier를 구축한다고 가정했을 때 미국, 영국, 유럽 기타 지역, 남미, 인도, 중국, 아시아 기타 지역, 호주 등의 지역에서 Data를 수집한다. 

Dev Set와 Test Set를 설정할 때, 4개의 지역을 선택하여 Dev Set에 넣고, 나머지 4개의 지역을 Test Set에 넣는 방법을 생각해 볼 수 있지만 이는 좋지 않은 생각이다. 

Dev Set와 Test Set가 다른 분포에서 나오는 경우, Dev Set에서 성능을 Optimazation 하는 데 몇 달을 보낸 후에 Test Set에서 성능이 좋지 않다는 것을 깨닫게 될 수 있기 때문이다.

### **Aligning Distributions**

**Dev Set와 Test Set가 같은 분포에서 나오는 것이 중요하다.** 

이렇게 하면 Dev Set에서 성능을 Optimaze하면서 Test Set에서도 좋은 성능을 기대할 수 있다. 

Dev Set와 평가 지표를 설정하는 것은 목표를 설정하는 것과 같다. 

Dev Set에서 좋은 성능을 내기 위해 다양한 Idea를 시도하고 실험을 통해 최적의 Model을 찾게 된다.

<img width="690" alt="스크린샷 2024-07-23 오후 4 33 54" src="https://github.com/user-attachments/assets/4d82b411-ff0e-42b9-8883-9786150ca523">

중간 소득 지역의 대출 승인 Data를 사용하여 Dev Set를 설정한 후, 저소득 지역의 Data로 Test를 수행한 경우가 있다. 

이 경우, 중간 소득 지역과 저소득 지역의 Data 분포가 매우 다르기 때문에, Dev Set에서 Optimaze된 Model이 Test Set에서 잘 작동하지 않았고 이로 인해 많은 시간을 낭비하게 되었다. 

<img width="690" alt="스크린샷 2024-07-23 오후 4 34 08" src="https://github.com/user-attachments/assets/fc20582b-4274-4119-9e94-61946bed635a">

Dev Set와 Test Set를 설정할 때는 미래에 얻을 것으로 예상되는 Data를 반영하고, 그 Data에서 좋은 성능을 내는 것이 중요하다. 

따라서 Dev Set와 Test Set는 같은 분포도를 가지는 Data에서 설정하는 것을 권장한다.

만약 Dev Set과 Test Set이 다른 분포도를 갖는다면 Dev Set에서 얻은 Data의 정보가 Test Set과 많이 다를 수 있다. 

## **Size of Dev and Test Sets**

![Untitled (1)](https://github.com/user-attachments/assets/50984799-925b-4988-a09c-53d1066e98ea)

Dev Set와 Test Set는 동일한 분포에서 나와야 한다. 

이때 그 길이는 얼마나 되어야 할까 ? ML에서 Data의 70 / 30 분할 규칙이나 60% Train, 20% Dev, 20% Test Set로 분할하는 규칙은 과거에는 합리적이었다.  

Data가 수백-수천 개일 때는 이 규칙이 유효했지만, 최근 ML에서는 Data Set 크기가 훨씬 커졌다.

Data가 백만 개인 경우, 98%는 Train Set에, 1%는 Dev Set, 1%는 Test Set에 할당하는 것이 합리적일 수 있다. 백만 개의 Data 중 1%인 1만 개는 Dev Set나 Test Set로 충분하다. 

**DL Algorithm은 많은 Data를 필요로 하므로, 큰 Data Set에서는 Trining Set에 더 많은 비율을 할당하는 것이 일반적**이다.

### **Test Set Size**

**Test Set의 목적은 최종 System의 성능을 평가하는 것**이다. 

Test Set는 System의 전체 성능에 대해 높은 신뢰성을 제공할 만큼 충분히 커야 한다. 

단, 몇 백만 개의 Example가 필요하지 않을 수 있으며, 1만 개 정도로 Data가 충분할 수도 있다.

Application에서는 최종 System의 전체 성능에 대해 높은 신뢰성이 필요하지 않을 수 있다. 

이 경우, Train Set와 Dev Set만 있어도 괜찮을 수 있다. 

만약 최종 System의 성능을 실제로 평가하지 않고 바로 배포할 경우, Train Set와 Dev Set만 사용하고 Test Set는 생략할 수 있다.

## **When to Change Dev / Test Sets and Metrics**

Dev Set을 설정하고, Evaluation Metrics을 설정하는 것은 System을 구축하는데 목표를 이룰 수 있도록 방향성을 제시해주는 것과 같다. 

하지만 Project를 진행하는 도중에 이러한 설정이 잘못되어서 잘못된 방향으로 가고 있다는 것을 뒤늦게 깨닫는 경우도 있다. 

이러한 경우에는 목표를 이동해야하고 결국 **Dev Set이나, Evaluation Metrics을 다시 설정**해야 한다. 

![스크린샷 2024-07-23 오후 5 18 38](https://github.com/user-attachments/assets/e891fe5e-fc79-44ac-861e-31aeab9ca87f)

예를 들어 Cat Classifier를 만들고 있고 Metrics은 Classification Rrror를 사용한다고 가정한다. 

A Algorithm과 B Algorithm이 각각 3%와 5%의 Error를 보인다. 

처음에는 A Algorithm이 더 나아 보이지만, 실제로는 A Algorithm이 Pornographic Image를 많이 통과시키는 문제가 있다. 

따라서 회사와 사용자 입장에서는 B Algorithm이 더 나은 선택이 된다. 

이 Example에서 평가 지표는 A Algorithm이 더 나은 것처럼 보이게 하지만, 실제로는 B Algorithm이 더 나은 결과를 제공한다. 

이 경우 **Evaluation Metric나 Dev Set, Test Set를 변경해야 한다.**

![Untitled (2)](https://github.com/user-attachments/assets/36ef4924-6c89-40e8-b9b1-9670aa345071)

기존의 Classification Error는 위와 같다. 이 식에서 Sum 기호 다음 부분은 실제 y와 일치하지 않는 예측의 갯수를 Count 한 것이다.

이런 식 즉, 잘못된 Classification Error 지표는 Pornographic Image를 Non-Pornographic Image와 동일하게 보기 때문에 제대로 평가되지 않을 것이다. 

이를 해결하기 위해서는 아래와 같이 **wi**를 추가하면 된다. 

![Untitled (3)](https://github.com/user-attachments/assets/ab583997-874b-4ba8-90d8-7a173d0c86a0)

**가중치를 추가하는 것**으로 Image가 Pornographic Image라면 10이라는 Weight를 주는 것이다. 

Weight는 Pornographic Image를 Dev Set, Test Set을 거쳐 Labeling해 도입할 수 있다. 

그리고 이때 Normalization을 위해 mdev가 아닌 ∑im(i)로 나누어 주었다. 

이처럼 Metrics가 원하는 순서로 선호도를 평가하지 않는다면 새로운 Evaluation Metrics의 도입을 생각해야 한다. 

### **Orthogonalization**

<img width="690" alt="스크린샷 2024-07-23 오후 5 19 00" src="https://github.com/user-attachments/assets/657464d3-17ed-4942-8edd-833c605daafc">

이 경우 **Orthogonalization을 적용**할 수 있는데, 이때 문제를 독립적인 단계로 나누는 것이 중요하다. 

첫 번째 단계는 Evaluation Metric를 정의하는 것이다. Classifier를 평가하기 위해 Metrics을 잘 정의하는 것에 집중한다. 

두 번째 단계는 이 지표에서 잘 수행하는 방법을 찾는 것이다. 

즉, Target을 Plan하는 것과 Tunning 하는 단계를 분리해 생각하는 것이다. 

Evaluation Metric를 정의하는 것은 목표를 설정하는 것과 같고, 이후에는 목표에 맞게 성능을 조정하는 것이 두 번째 단계이다.

### **Defining the Metric**

Cat Classifier A와 B가 각각 3%와 5%의 Error를 보이는 경우를 예로 들었을 때 고품질의 Image를 기반으로 한 Dev Set에서는 A가 더 나은 성능을 보일 수 있지만, 실제 사용 환경에서는 B가 더 나은 성능을 보일 수 있다. 

이 경우 현재의 Evaluation Metric와 Dev Set가 실제 Application 성능을 반영하지 못하는 문제이다. 따라서 Evaluation Metric나 Dev 및 Test Set를 실제 사용 환경을 반영하도록 변경해야 한다.

<img width="690" alt="스크린샷 2024-07-23 오후 5 19 20" src="https://github.com/user-attachments/assets/d38663f0-c61c-4b1c-a937-8c6d7b99d293">

## **Why Human-Level Performance?**

**최근 몇 년 동안 많은 ML Team이 Human Level의 성능과 비교하는 것에 대해 이야기하고 있다. 왜 그럴까 ?**

첫 번째 이유는 DL의 발전 덕분에 ML Algorithm이 갑자기 훨씬 더 잘 작동하게 되었고, 많은 응용 분야에서 ML Algorithm이 실제로 Human Level의 성능과 경쟁할 수 있게 되었기 때문이다. 

두 번째 이유는 인간이 할 수 있는 일을 목표로 하는 것이 ML System을 설계하고 구축하는 작업 흐름을 훨씬 더 효율적으로 만든다. 이러한 설정에서는 Human Level의 성능과 비교하거나 이를 모방하려고 시도하는 것이 자연스럽다.

<img width="690" alt="스크린샷 2024-07-23 오후 5 19 39" src="https://github.com/user-attachments/assets/c9744a09-b54c-418a-aa88-79ecc980ebe3">

x축이 시간이고, 이 시간이 여러 달 또는 여러 해 동안 문제에 대해 작업하는 시간이라고 가정한다. Human Level의 성능에 접근할 때까지 진행 속도가 비교적 빠르다. 

하지만 시간이 지나면서 Algorithm이 Human Level의 성능을 초과하면 Precision 향상이 실제로 느려질 수 있다. 

그리고 아마도 더 좋아지겠지만 Human Level의 성능을 초과한 후에는 성능이 계속 향상될 수 있지만, Precision가 급격히 상승하는 경향은 종종 느려진다. 

최종적으로는 이론적 최적 성능 수준에 도달하는 것이 목표이다.

### **Bayes Optimization Error**

시간이 지남에 따라 더 큰 Model과 더 많은 Data를 사용하여 Algorithm을 계속 Train하면 성능이 Bayes Optimization Error에 접근할 수 있지만, 이 값이 이론적으로 최소한의 Error이며, ML System은 이 값을 절대로 넘어설 수 없다.

즉, 위 Graph에서 보라색 선은 긴 시간이 지나도 Bayer Error에 도달할 수 없는 것이다. 

**Bayes Optimization Error는 가장 완벽한 Error 수준을 의미한다.** 

어떤 함수가 x에서 y로 Mapping될 때, 일정 수준의 정확성을 초과할 수 없다. 

예를 들어 음성 인식에서는 일부 Audio가 너무 시끄러워서 정확한 전사를 알아내는 것이 불가능할 수 있다. 

또는 Cat 인식에서는 일부 Image가 너무 흐릿해서 아무도 Cat가 있는지 없는지 알 수 없을 수 있다. 따라서 최적의 Precision 수준이 100%가 아닐 수 있다. 

Bayes Optimization Error는 **x에서 y로 Mapping되는 가장 완벽한 이론적 함수**이다. 

문제에 대해 몇 년 동안 작업하더라도 Bayes Optimization Error를 초과할 수 없습니다. 

Human Level의 성능에 도달할 때까지 진행 속도가 빠르고, 이를 초과한 후에는 느려질 수 있다. 

그 이유는 두 가지이다.

첫 번째, 많은 작업에서 Human Level의 성능이 Bayes Optimization Error와 크게 다르지 않기 때문이다. 

사람들은 Image에서 Cat를 찾거나 Audio를 듣고 전사하는 데 매우 능숙하다. 

따라서 Human Level의 성능을 초과할 때 개선할 여지가 많지 않을 수 있습니다. 

두 번째, ML Algorithm이 Human Level의 성능보다 떨어질 때 적용할 수 있는 몇 가지 도구가 있기 때문이다. 

예를 들어, 사람이 잘하는 작업에서는 Label Data를 수집하거나 수동 Error 분석을 수행하여 Algorithm을 개선할 수 있다. 

인간이 더 나은 성능을 보일 때는 이러한 도구를 활용하기가 더 쉽다. 

그러나 Algorithm이 인간보다 더 잘할 때는 이러한 도구를 적용하기가 어렵다.

ML Algorithm이 사람의 작업을 복제하려고 할 때 매우 유용하며, 사람 수준의 성능을 따라잡거나 초과할 수 있는 이유 중 하나이다. 

또한, 인간이 특정 작업에서 얼마나 잘 수행할 수 있는지 알면 Bias와 Variance을 얼마나 줄여야 할지 더 잘 이해할 수 있다. 

<img width="690" alt="스크린샷 2024-07-23 오후 5 19 55" src="https://github.com/user-attachments/assets/3793bee5-7f23-4cbc-af5d-8442cd9944b4">

> ML이 인간 Level의 성능에 미치지 못하면 위와 같은 방법들을 사용할 수 있다. 

## **Avoidable Bias**

<img width="690" alt="스크린샷 2024-07-23 오후 6 04 31" src="https://github.com/user-attachments/assets/7b9ee7e8-e964-487e-922c-4233d38db33a">

ML Algorithm이 Train Set 에서 좋은 성능을 내기를 원하지만 너무 잘할 필요는 없다.

<img width="690" alt="스크린샷 2024-07-23 오후 6 04 51" src="https://github.com/user-attachments/assets/d8fed76d-45b8-4466-aa37-c74701da3a8c">

주어진 Image에서 인간이 거의 완벽한 Precision를 가지고 있다고 가정한다. 

이 경우 Human Level의 Error는 1%이다. 이 상황에서 Train Algorithm이 8%의 Train Error와 10%의 Dev Error를 가지고 있다면 Train Data Set에서 더 나은 성능을 내도록 해야 한다. 

인간이 Train Data Set에서 얼마나 잘하는지와 Algorithm의 성능 간에 큰 차이가 있음을 보여준다. 

이 경우 바이어스를 줄이는 데 초점을 맞춰야 한다. 더 큰 NN을 훈련시키거나 더 오랫동안 Train을 시도해보는 등의 방법을 사용할 수 있다.

Human Level의 성능이 1%가 아닌 경우, 다른 Data Set에서 Human Level의 Error가 7.5%라고 가정한다. 

이 경우 인간도 Cat Image을 정확하게 분류하지 못한다. 

이 상황에서는 Train Error와 Dev Error가 동일하더라도 Train Data Set에서 이미 충분히 잘하고 있을 수 있다. 이 경우 Bias을 줄이는 데 초점을 맞추어야 한다. 

Regularization을 시도하거나 더 많은 Train Data를 확보하는 방법을 사용할 수 있습니다.

### **Bayes Error**

Cat Classification Example에서 Human Level의 **Error를 Bayes Error의 추정치**로 생각해볼 수 있다. 

인간이 Computer Vision 작업에서 매우 능숙하기 때문에 Bayes Error와 크게 다르지 않기 때문이다.

Human Level의 Error가 Bayes Error보다 더 나쁘지만 크게 다르지 않을 수 있다.

<img width="690" alt="스크린샷 2024-07-23 오후 6 05 17" src="https://github.com/user-attachments/assets/5f0c21ad-db7e-49c7-9fa6-09f899b1d08c">

인간 Level의 성능이 얼마인지에 따라 Train Error, Dev Error에서는 각각 다른 방법을 사용해 Bias와 Variance을 감소시키기 위해 노력한다. 

예를 들어, Human Level의 Error가 1%인 경우, 8%의 Train Error는 매우 높기 때문에 Bias를 줄이는 데 집중해야 한다. 

반면, Human Level의 Error가 7.5%인 경우, Train Error가 8%라면 이는 크게 나쁘지 않으며, 대신 2%의 Dev Error와의 차이를 줄이는 데 집중해야 한다.

### **Terminology Avoidable Bias and Variance**

여기서 **Bayes Error와 Train Error의 차이를 Avoidable Bias**라고 부른다. 

Train Error를 Bayes Error와 가까워지도록 개선하는 것이 목표이고 Overfitting 하지 않는 이상 Bayes Error보다 낮아질 수는 없다. 

이 용어는 최소한의 Error가 있다는 것을 의미하고 Bayes Error가 7.5%인 경우 이 Error 수치 이하로 개선될 수 없다는 것을 포함하고 있다. 

특히 두 번째 Example에서 Avoidable Bias는 0.5%이고 Train Error, Dev Error와 차이는 2%이기 때문에 2%를 줄이는 것이 더 큰 효과를 가져온다. 

## **Understanding Human-Level Performance**

Bayes Error는 현재 또는 미래에 달성할 수 있는 최상의 Error이다. 

그리고 **Human Level Error는 Bayes Error를 추정하는 방법이 될 수 있다.** 

<img width="696" alt="스크린샷 2024-07-24 오후 3 16 15" src="https://github.com/user-attachments/assets/89231937-d052-4d16-9881-ec99f5b4ef24">

**Example Medical Image Classification**

- 일반적인 사람이 이 작업에서 3%의 Error를 낸다고 가정한다.
- 숙련된 의사는 0.7%의 Error를 낸다.

- 일반적인 의사는 1%의 Error를 낸다.
- 숙련된 의사 팀은 0.5%의 Error를 낸다.

이때 Human Level의 Error는 어떻게 정의할 수 있을까 ?

바로 **Error의 추정치를 Bayes Error로 접근하는 것**이다. 

경험 많은 의사 팀이 0.5%의 Error를 가지고 있기 때문에 Bayes Error는 0.5%이거나 0.5% 이하이다. 

따라서 0.5%의 Error를 일으킬 수 있기 때문에 Optimal Error는 0.5%이거나 낮다고 할 수 있다. 

물론, 경험이 많은 의사 팀의 규모가 더 커서 0.5%보다 낮을 수도 있지만, Optimal Error는 0.5%보다 낮을 수 없고, 이런 상황에서 교수님의 의견으로는 0.5%를 Bayes Error의 추정치로 지정할 것이다. 

이런 방법으로 0.5%를 Human Level Performance로 정의한다. 

<img width="696" alt="스크린샷 2024-07-24 오후 3 20 32" src="https://github.com/user-attachments/assets/7b2160c3-cb84-45a3-b2af-3622e6724e74">

1. Train Error가 5%이고 Dev Error가 6%일 때, Human Level의 성능을 1%, 0.7%, 0.5% 중 어느 것으로 정의하더라도 Avoidable Bias는 약 4%이다. 
    
    이 경우, Bias 감소 기술에 집중해야 한다.
    
    어떤 값으로 지정해도 Avoidable Bias는 4-4.5% 값을 가지고 Train Error와 Dev Error의 차이는 1% 값을 가진다.
    
2. Train Error가 1%이고 Dev Error가 5%일 때, Avoidable Bias는 0%에서 0.5% 사이이며, 4%의 Variance 문제에 집중해야 한다.
    
    Human Level Error를 어떤 값으로 설정해야 하는지는 중요하지 않다. 
    
3. Train Error가 0.7%이고 Dev Error가 0.8%일 때, Human Level의 성능을 0.5%로 정의하면 Avoidable Bias는 0.2%이며, 이는 Variance 문제보다 더 크다. 
    
    만약 0.7%를 Bayes Error의 추정치로 정한다면 Avoidable Error는 0%가 되고 Variance를 줄이는 방법에 집중할 수 있다. 
    

### **Bias and Variance Analysis**

**Bayes Error의 추정치를 더 정확하게 알면, Avoidable Bias와 Variance를 더 정확히 추정할 수 있게 되고, 어디에 중점을 둘 것인지 결정을 내릴 수 있을 것이다.**

즉, Bias 감소 또는 Variance 감소에 집중할지 빠르게 결정할 수 있다는 의미이다. 

이때 **Human Level의 성능을 초과한 경우**에는 **Bias와 Variance를 제거하기 어렵다.** 

## **Surpassing Human-Level Performance**

<img width="690" alt="스크린샷 2024-07-24 오후 2 52 47" src="https://github.com/user-attachments/assets/4ffb7ab0-b0e0-424b-aa70-de0403750acc">

**Example Human and Algorithm Performance**

- Team of Humans 0.5%
- Train Error 0.6%

- One Human 1%
- Dev Error 0.8%

위 경우, Avoidable Bias는 0.5%이다. 따라서 Bias는 최소 0.1%이고, Variance은 0.2%이다. 이 경우, Variance를 줄이는 데 집중해야 할 수도 있다.

**Harder Example**

- Team of Humans 0.5%
- Train Error 0.3%

- One Human 1%
- Dev Error 0.4%

이 경우, **Avoidable Bias를 평가하는 것은 더 어려워진다.** 

Train Error가 0.3%라면, 이것이 Overfitting으로 인한 것인지, 아니면 Bayes Error가 0.1%인지, 혹은 0.2%인지, 아니면 0.3%인지 알 수 없다. 

이 경우, **Bias를 줄일지 Variance을 줄일지 결정하는 것이 어려워진다.**

Algorithm이 인간의 성능을 초과했다면, 인간의 직관을 통해 Algorithm을 개선하는 방법을 찾는 것이 더 어려워진다. 

따라서 더 나은 방향을 제시하는 도구들이 잘 작동하지 않게 된다. 

<img width="690" alt="스크린샷 2024-07-24 오후 3 03 26" src="https://github.com/user-attachments/assets/23a29ba2-0610-4702-a932-b114767459dd">

이 Example들은 구조화된 Data를 Train하는 문제들로, 방대한 Data에 접근할 수 있는 것이 유리하다. 

## Improving Your Model Performance

Train Algorithm 성능이 효과가 있기 위해서는 2가지 전제 조건이 필요하다. 

![Untitled (4)](https://github.com/user-attachments/assets/5a07c4c7-d859-468a-aea5-6e9d480bfd96)

첫 번째로는 Train Set에 잘 Fitting 되어야 한다는 것이다. 이는 낮은 Avoidable Bias를 갖는것과 동일한 의미이다. 

두 번째로 Dev Set, Test Set에서도 잘 Fitting 되어야 한다는 것이다. Variance가 나쁘지 않음을 의미한다. 

즉, **ML의 성능을 향상시키고자 한다면 Train Error와 Bayes Error의 차이를 보고 Avoidable Bias 문제를 확인한 후 Dev Error, Train Error의 차이를 살펴보면 된다.** 

<img width="696" alt="스크린샷 2024-07-24 오후 3 07 41" src="https://github.com/user-attachments/assets/7c32953f-d808-4127-9ba5-56f533bab87f">

**Avoidable Bias를 줄이기 위해서는 더 큰 Model Train, 더 긴 시간 Train, 더 나은 Optimization Algorithm을 사용하는 방법이 있다.** 

또 다른 방법으로는 더 좋은 구조를 사용하는 것으로 더 좋은 Hyperparameter, Activation Function을 사용하거나 Layer의 갯수, Hidden Unit의 갯수를 바꾸는 방법이 있을 수 있다.  

- 더 나은 Optimazation Algorithm 사용 → Adam
- 더 나은 NN Architecture 또는 Hyperparameter 찾기
- 새로운 Model Architecture 시도 → RNN, CNN

**Variance를 줄이기 위해서는 더 많은 Data를 수집하거나 Regularization을 시도하는 방법이 있다.** 

그리고 Avoidable Bias와 동일하게 NN 구조를 변경할 수 있고 Hyperparameter를 변경해 개선할 수 있다. 

- Regularization 기법 사용 L2 Regularization, Dropout
- Data Augmentation 사용
- 다양한 NN Architecture, Hyperparameter Search

## Carrying Out Error Analysis

Train Algorithm을 개발할 때 Train Algorithm이 Human Level 성능에 미치지 못한다면 Error를 수작업으로 점검하는 것이 도움이 될 수 있다. 

이런 Process를 **Error Analysis**라고 한다. 

Cat Classifier를 위한 Train Algorithm을 개발하고 있고 그 결과로 90%의 정확도에 도달했다고 가정하면, Dev Set에서 10%의 Error를 가지고 있다고 볼 수 있다. 

이때 Train Algorithm이 분류한 Example을 보면 아래와 잘못 분류된 것을 찾을 수 있다. 

![Untitled (5)](https://github.com/user-attachments/assets/5cb12e40-74b1-4bac-8db7-69558257d52e)

이 경우 Dog Data를 더 수집하거나 Dog에 특화한 Feature를 Design 하는 방법으로 Algorithm을 강화할 수 있다. 

Algorithm에서 Dog을 잘못 인식하는 것을 수정하는데 수개월의 시간이 걸릴 수도 있다. 

게다가 위의 방법들이 실제로 도움이 되는지 알 수 없으며, 그 방법을 사용해서 수개월이 지난 후에서야 성능이 좋아지지 않았다는 것을 발견 할지도 모른다. 

이런 상황에서 Error Analysis는 어떤 방법이 시도해볼만한 가치가 있는지에 대해 Guide Line을 제시할 수 있다. 

대략 100개의 Mis-Label된 Dev Set Example을 뽑아서 수작업으로 검사한다. 

그리고, 100개의 잘못 분류된 Dev Set Example중에서 5%가 Dog라고 가정해보자. 

즉, 잘못 분류된 100개의 Dev Set Example 중에서 5개만이 Dog라는 것이고, 이것은 Dog을 제대로 인식하도록 수정을 했더라도 100개 중에서 5개만이 제대로 분류된다는 의미이다. 

Error의 5%만이 수정되어서, Dev Set Error를 10%에서 9.5%까지만 줄일 수 있다는 의미이다.
합리적으로 생각했을 때, Dog을 제대로 인식하도록 강화하는 방법은 효율적이지 못하다는 것이다.

즉, Error Analysis를 수행하면, Dog 인식 문제에 투자하는 것이 얼마나 도움이 되는지와 같은 Guide Line을 제시해준다.

이번에는 100개의 잘못 분류된 Example에서 50장이 Dog라고 가정해보면, Dog 인식 문제에 시간을 투자하는 것이 조금 더 효율적으로 느껴질 수 있다. 

이와 같은 경우에는 만약 Dog 인식 문제를 해결한다면, Dev Error를 10%에서 5%까지 감소시킬 수 있기 때문이다. 따라서 시도할만한 방법이 될 수 있다.

수작업이지만, 짧은 시간의 분석으로 어떤 방법에 집중해야 하는지 알려주며 훨씬 더 좋은 결정을 내릴 수 있고, 시간을 절약하게 해준다.

**Error Analysis에서 여러가지 Error 원인들을 Parallel하게 분석해서 평가할 수도 있다.**

![Untitled](https://github.com/user-attachments/assets/5dbb4b46-1a29-4e7d-b4c3-6d596857e942)

성능을 개선하기 위한 다양한 Idea를 가지고 Error Analysis를 진행하는 것이다.

![Untitled (6)](https://github.com/user-attachments/assets/716e7040-693e-498e-b455-02ad36565cbf)

Row는 잘못 인식된 Image들의 Index에 해당하고, Column은  Dog 인식 문제, Great Cat 인식 문제, 흐릿한 Image 문제에 해당한다. 

그래서 잘못 인식된 Image가 어떠한 문제를 가지고 있는지 확인하고, 필요하다면 Comment도 남겨둔다. 

기억해야될 것은 이 잘못 분류된 Image들이 Dev Set Example에서 추출되었다는 것이다.

Error Analysis를 진행하고 Image Review가 끝났다면, 100개의 Example에서 각 문제점이 몇 %를 차지하는지 수치로 정리한다. 

여기서는 Dog 인식 문제가 8%, Great Cat 인식 문제가 43%, 흐릿한 Image 문제가 61%로 나타났다. 

이렇게 진행하다보면, 또 다른 문제점을 발견할 수도 있는데, 이런 경우에는 Column에 추가해서 다시 확인하고, 몇 %를 차지하는지 확인한다.

이 절차를 모두 끝내면, 어떤 문제에 대해서 개선을 하는 것이 가장 효율적인지 추정할 수 있다. 

위와 같은 경우에는 흐릿한 Image에서 많은 Error가 있고, Great Cat에 대한 Error도 꽤 많았다. 

따라서, 분석 결과가 무조건 흐릿한 Image 문제를 개선해야된다는 아니다. 

이 분석 결과는 직접적으로 답을 내려주는 것이 아니라, **어떤 것을 선택해야되는지 수치적인 Guide Line을 제시하는 것**이다. 

Dog 인식 문제를 개선하면 8%까지밖에 개선을 하지 못한다거나, Great Cat 인식 문제를 개선하면 43%정도 개선이 가능하다라는 것을 말해주는 것이다. 

이 분석은 각 문제에 따라서 성능을 개선하는 부분에 있어서 특정한 한계치를 나타내고 있는데, 여기서는 Great Cat에 대해서 개선하는 방법과 흐릿한 Imgae에 대한 개선을 하는 방법 중에 선택하는 경우가 있을 수 있고, 또는 여유가 많다면 두 개로 나누어서 하나는 Great Cat 인식 문제를 개선하고, 다른 하나는 흐릿한 Image 인식 문제를 개선하는 방법을 선택할 수도 있다.

이렇게 수작업으로 분석하는 것은 시간이 꽤 짧게 소요되는데, 그럼에도 불구하고 우리가 해야되는 것들 중에서 우선순위를 정하는데 많은 도움을 주고, 접근 방법에 대한 효율을 수치적으로 측정할 수 있다.

## Cleaning-Up Incorrectly Labeled Data

Supervised Learning의 Data는 Input x와 Output y Label로 이루어져 있는데, Data를 확인하다가 결과값 y가 다르다는 것을 확인해서 Data가 잘못되었다는 것을 찾는다면 어떻게 될까 ? 이런 Data의 Label을 수정하는 것이 효과적일까 ?

Cat Classifier에서 Image가 Cat인 경우에는 y는 1이고, Cat이 아닌 경우에는 y는 0의 값을 갖는다.

![Untitled (7)](https://github.com/user-attachments/assets/3206cd5d-e3e1-4731-9569-3f6926432a9d)

위와 같은 경우에는 Dog는 1로 잘못 Label 되어 있다. 

이런 경우에 Mislabeled Example이라는 용어를 사용하지 않고, **Incorrectly Labeled Example**이라고 한다. → Mislabeled Example은 Train Algorithm이 잘못된 y값으로 Label할 때를 사용한다. 

여기서는 Training Set이나 Dev, Test Set에 있는 Data Set에서 잘못 Label된 y값은 사실 0이어야 되며, Data Set을 Lable한 사람이 잘못 Input했을 수 있다. 

이렇게 Data에 잘못 Label된 Example이 있으면 어떻게 해야할까 ?

우선 Training Set에서 고려해보면, DL Algorithm은 Training Set에서 Random Error에 대해서 상당히 견고하다는 것이 밝혀져있다. 

따라서, 이렇게 잘못 Label된 Example들이 고의적으로 발생한 것이 아니라, 단순 실수에 의한 것이고 Random으로 발생되었다고 한다면, 이 Error는 그대로 두어도 괜찮고, 고치는데 많은 시간을 쏟을 필요가 없다. 

물론 Training Set를 살펴보면서 직접 검사하고 수정해도 나쁠 것은 없다. 

하지만, 전체 Data Set의 크기가 충분히 크고, Error의 %가 너무 크지만 않다면, 굳이 수정하지 않아도 괜찮으며, 실제로 Training Set의 Label에서 Error가 있음에도 정상적으로 동작하는 ML Algorithm들이 많다.

![Untitled](https://github.com/user-attachments/assets/3ac7a850-aaf4-4ce0-865d-13f313a93e3d)

하지만 DL Algorithm Systemic Errors에 대해서는 문제가 발생할 수 있다. 

예를 들어서, Data Set의 Label을 담당하는 사람이 지속적으로 흰색 Dog를 Cat으로 Label했다면, Classifier가 모든 흰색 Dog를 Cat으로 분류하도록 배울 수 있기 때문에 문제가 될 수 있다.

만약 Dev Set이나 Test Set에서 잘못 Label된 Example들이 있다면 어떨까 ?
이런 문제가 걱정된다면, 추천하는 방법은 Error Analysis를 진행하면서, Category를 추가해서 잘못 Label된 Example의 %를 구하는 것이다.

![Untitled (9)](https://github.com/user-attachments/assets/59d6b45c-39db-4baa-ae1d-1f223425b882)

잘못 Label된 Example의 %를 구한다. 

그리고 이 문제를 개선해서 Dev Set에서 Algorithm의 성능이 현저하게 개선될 수 있다면 Label이 잘못된 것을 수정하라고 이야기할 수 있지만, 그 성능이 크게 바뀌지 않는다면 효율적이지 않다.
전체적인 Dev Set Error를 살펴보고 결정하는 것을 추천한다. 

이전 Example처럼 10%의 dev Error를 가지고 있고, 잘못 Label된 Image를 수정해서 개선한다면 10% Error 중에서 6%, 즉, 0.6%를 개선할 수 있다는 것이다. 

따라서 Label을 수정한다고 해도 9.4%까지밖에 성능이 개선되지 않는다는 것이고, 이는 매우 비중이 작다고 볼 수 있다.

만약 잘못 Label된 Example의 비율이 30%이고, Dev Error가 2%라면, Error의 30%는 0.6%이고, Label을 수정하면 Error를 1.4%까지 줄일 수 있기 때문에 꽤 비중이 높다고 할 수 있고, 시도할만한 방법이라고 할 수 있다.

![Untitled](https://github.com/user-attachments/assets/5180e396-a55b-4e7a-bd35-561854e18ae1)

Dev Set의 주목적은 여러 Classifier 사이에서 어떤 것을 선택할 지 도움을 주는 것이다. 

만약 Classifier A가 2.1% Error를 가지고 있고, Classifier B가 1.9%의 Error를 가지고 있다면, Classifier B를 선택할 수 있을 것이다. 

하지만 Error 중에서 잘못된 Label로 인한 Error가 0.6%라고 한다면, 더이상 B가 더 낫다고 할 수 없다. 

그렇다면, 직접 Dev Set에서 잘못 Label된 부분을 고치는 것은 좋은 선택이 될 것이다.

![Untitled (10)](https://github.com/user-attachments/assets/c2b0c256-01dd-4df0-9ed8-16d913518da5)

위 Image에서 두 가지 Example를 살펴보면, 오른쪽 Example는 Dev Error가 2%이고, Dev Error에서 잘못된 Label로 인한 Error가 0.6%이기 때문에 Algorithm에 큰 영향을 끼치고 있다. 

반면에, 왼쪽 Example는 10%의 Dev Error에서 0.6%만이 잘못된 Label로 인한 Error이며, Algorithm에 끼치는 영향이 더 적다고 볼 수 있다.

만약 Dev Set에서 Label을 수작업으로 검사하고 직접 고치려고 한다면, 다음의 Guide Line을 추가적으로 고려하는 것을 권장한다.

![Untitled (11)](https://github.com/user-attachments/assets/3e7f6fc0-8859-48c5-9d62-14e3568485e2)

첫 번째로 어떤 방식을 적용하든, Dev Set과 Test Set이 계속해서 같은 분포도를 갖도록 해야한다. 

따라서, Dev Set에서 어떤 방법을 사용해서 수정한다면, Test Set에도 동일하게 적용해야한다. → Dev Set과 Test Set은 같은 분포를 갖도록 해야한다.

두 번째는 Train Algorithm에서 잘못 분류한 Example뿐만 아니라, 제대로 분류한 Example들도 포함해서 살펴보아야 한다. 

98%의 정확도를 나타낼 때, 틀린 2%만 고치는 게 쉽고, 98%를 검토하는 것은 어렵지만, 고려해볼만 하다. → 올바르게 Label 되지 않아서 맞춘 것이 있을 수 있다. 

세 번째는 Train Data에도 Dev, Test Data에 적용한 방법을 동일하게 적용할 수 있지만, 적용하지 않아도 된다. 

이전에 이야기했듯이 Train Data를 수정하는 것은 덜 중요한 편이고, 많은 노력을 투자하지 않아도 된다. 

따라서 Train과 Dev, Test Set의 분포도가 조금 달라도 된다.

## **Build Your First System Quickly, then Iterate**

만일 완전히 새롭게 Train Algorithm App을 작업한다면 추천하는 방법은 우선 System을 먼저 빠르게 만들고, 반복 Test 업무를 진행하는 것이다.

<img width="400" src="https://github.com/user-attachments/assets/6ed2692b-92b1-4b5f-97e1-613f5b14160b">

음식 인식 System을 Example로 들어보면, Train Algorithm을 개발하는데 여러 가지 방법들이 있고, 우선 순위를 정할 수 있는 것들 또한 많다. 

예를 들어서, 시끄러운 주변환경에서 잘 동작하는 음성 인식 System을 만들 수 있는 특정한 기술이 있거나, 억양이 있는 음성이 더 잘 인식되도록 하는 방법, 어린이들의 음성 인식 방법 등이 있을 수 있다. 

일반화시켜서 이야기하자만, ML App에서 접근할 수 있는 방법이 50가지가 될 수 있는데, 이 방법들은 모두 합리적이고 System 개선에 도움이 될 수 있는 부분이다.

문제는 정확히 어떤 방법을 선택해서 집중하느냐이다.

해당 분야에서 수년간 개발했더라도, 새로운 App을 만드는 경우에는 쉽게 선택하지 못할 수 있다. 

따라서, 새롭게 개발하는 경우에는 우선 System을 빠르게 만들고, 반복 Test를 진행하는 것을 추천한다.

- **Set up Dev, Test Set and Metric**
- **Build Initial System Quickly**
- **Use Bias, Variance Anylysis & Error Analysis to Priority Next Steps**

구체적으로 우선 Dev, Test Set과 Metric을 설정하는 것이다. 

결국에는 목표를 어디에 둘지 설정하는 과정이다. 만약 잘못된 경우에는 언제든지 변경할 수 있다. 

일단, 목표를 설정하고, 그 다음에 ML System을 우선 만든다. 

Traing Set를 수집하고, Train한 다음 결과를 살펴본다. 

그리고 Dev, Test Set과 Metric을 통해서 얼마나 잘 동작하는지 살펴보고 이해하는 것이다. 

그런 다음에 Bias, Variance Analysis나 Error Analysis를 사용해서 다음 단계에 대한 우선순위를 지정해서 개선할 수 있다.

요약하자면, 초기 System으로 우선 Train을 완료하고, Train 완료된 System을 통해 Bias, Variance를 조절하고, Error Analysis를 통해서 Error를 살펴보고 많은 접근 방법 중에서 우선순위를 정해서 다음 과정을 진행하는 것을 반복하는 것이다.

물론 이런 Guide Line이 덜 적용되는 특정 분야도 있을 수 있다. 

만약 얼굴 인식 System을 만든다면, 방대한 자료를 토대로 처음으로 복잡한 System을 만드는 것도 괜찮을 것이다. 

하지만, 처음으로 문제를 해결해 나가는 과정이라면 너무 많은 생각을 하지 말고, 복잡하지 않게 우선 빠르게 System을 구축하고 이것을 바탕으로 System의 우선순위를 정해서 개선해 나가는 방식으로 접근하는 것을 추천한다.

## **Training and Testing on Different Distribution**

DL은 Train Set이 충분할 때 가장 잘 동작한다. 

이런 이유로 많은 Engineer들이 단순히 Data를 최대한 많이 수집해 Training Set으로 사용하는 결과를 초래한다. 

하지만 이렇게 사용하는 Data 중에 일부, 혹은 많은 Data들이 Dev, Test Set과 같은 분포도를 가지지 않을 수 있다.

만약 우리가 사용자들이 핸드폰으로 찍은 Image를 Upload 시키는 App을 만들고, App을 통해 Upload한 Image들이 Cat인지 아닌지 분류하고 싶다고 가정해보자. 

그렇다면 우리는 두 가지 종류의 Data를 얻게 된다. 

하나는 우리가 정말 관심있어하는 분포의 Data이고, 다른 하나는 Upload된 Image의 Data인데, 해당 Image는 Frame이 낮거나 희미해 보일 수 있다.

<img width="692" alt="스크린샷 2024-08-01 오후 4 54 55" src="https://github.com/user-attachments/assets/8f9ab118-d832-4aef-8b90-7d24ea5ebe5f">

Web에서 고해상도의 전문가들이 찍은 Cat Image을 충분히 Download받을 수 있고, App 사용자는 많지 않아서 App으로 Upload된 Image은 적다고 한다. 

10,000개의 Image들을 App을 통해서 얻을 수 있고, Web을 통해서는 200,000개의 Image를 받을 수 있을 수도 있다.

이런 경우 최종 System이 과연 App Image의 분포에서 잘 동작하게 될까 ? 아닐 가능성이 높다.

우리가 목표로 하는 Image는 10,000개 정도로 많지 않아서 이것만을 사용해서는 안되고, 200,000개의 Web Image를 사용하는데, Web Image들이 도움이 될 것 같지만, 우리가 목표로하는 App을 통해 Upload 되는 Image는 200,000개의 Web Image와 다른 분포도를 가지고 있다.

이런 경우에 한 가지 시도할 수 있는 방법은 두 가지의 Data를 합해서 임의로 섞어서, Train Dev, Test Set으로 나누어서 Train 하는 것이다.

즉, 210,000개의 모든 Image을 임의로 섞은 다음에, 205,000개는 Training Set으로, 나머지 2500, 2500개의 Image는 Dev, Test Set으로 사용하는 것이다.

이 방법은 장점도 있지만, 단점도 있다.

장점은 Train, Dev, Test Set을 같은 분포도를 가지게 할 수 있지만, 단점으로는 Train의 결과가 Web Image의 분포도를 따를 수 있다는 것이다. 

즉, Train의 목표가 App을 통해 Upload 된 Image를 분류하는 것이 아니라, Web Image를 분류하는 데에 더 집중이 된다는 것이다. 따라서 다음 방법을 추천한다.

![Untitled (13)](https://github.com/user-attachments/assets/5b437214-834a-48e5-8b8b-f53039391cf1)

여전히 Web Image가 200,000개이고 App을 통해 Upload된 Image가 10,000개일 때, Training Set으로는 Web Image 200,000개와 추가로 App을 통해 Upload된 Image 5,000개를 사용해서 구성한다. 

그리고 나머지는 Dev, Test Set으로 나눈다. 

여기서 핵심은 Dev, Test Set은 우리가 중점적으로 집중해야하는 Data 분포도를 갖는 Image인 것이다. 

즉, App에서 Upload된 Image을 분류하는 것이 목적이기 때문에, Dev, Test Set의 분포도를 App을 통한 Image 분포도를 갖도록 하는 것이다. 

하지만, 여전히 Training Data와 Dev, Test Set의 분포도는 서로 다르다. 

그러나 이렇게 나누어진 Data에서의 결과가 장기적으로 훨씬 더 좋은 성과를 보인다.

<img width="692" alt="스크린샷 2024-08-01 오후 4 55 24" src="https://github.com/user-attachments/assets/8e040ee2-a502-4392-a1a3-d49fd7aaad5b">

다른 Example로 Rearview Mirror 음성 인식 기기를 살펴보면, 우리는 어떤 Data를 사용해서 이 음성 인식 System을 Train 해야할까? 

위에 나타난 방법처럼 Training Set, Dev, Test Set을 구성할 수 있는 방법이 다양하게 있다. 

Training을 위해서 축적된 음성 인식 Data들을 구매할 수 있고, 해당 Data를 가지고 있을 수도 있다.예를 들어서 이런 Data들이 500,000개가 있다고 가정한다.

그리고 Dev, Test Set의 경우에는 실제 Rearview Mirror 음식 인식된 음성 Data가 Training Set보다 적게 있다고 하자. 대략 20,000개가 있다. 

운전을 하면서 말하는 것이기 때문에 Training을 위해 수집된 Data와 실제 Rearview Mirror 음성 인식된 Data의 분포도는 매우 다를 것이다. 

하지만 우리가 목표로 두고 관심을 가지는 Data는 바로 실제 Rearview Mirror Data이고, 우리는 이 Data들이 작 동작되고 Dev, Test에서 잘 되도록 하는 것이 목표이다.

그래서 방금 전의 Example처럼, 이번에도 Training Set을 위한 Data에서 추가로  Rearview Mirror 음성 Data의 절반인 10k를 Train에 사용한다. 

그리고 나머지 10k는 5k, 5k씩 나누어서 Dev와 Test Set으로 사용할 수 있다.

## **Bias and Variance with Mismatched Data Distributions**

Train Algorithm의 Bias와 Variance를 추정하는 것은 다음 단계로 어떤 업무를 수행해야할 지에 대한 우선순위를 결정하는데 큰 도움을 준다. 

하지만, Train Algorithm의 Training Set과 Dev, Test Set이 서로 다른 분포도를 가지고 있다면 Bias, Variance를 분석하는 방법이 달라진다.

계속해서 Cat 인식 Example로 살펴보자. 

이 인식 기능에서 인간은 거의 완벽에 가까운 능력을 보여서, Bayes Error가 거의 0%에 가깝다는 것을 의미한다. 

그리고 Error Analysis를 하기 위해서 Training Error, Dev Error를 구한다.

<img width="300" src="https://github.com/user-attachments/assets/04b37124-61fc-45e8-be6c-2c3628412b2d">

위 경우에는 Training Error가 1%이고, Dev Error가 10%라고 해보자. 

그렇다면 이 경우 평소와 같다면 Variance의 문제가 심해서 Train Algorithm이 일반화를 잘못시킨다는 것을 의미한다. 

하지만, Training Set과 Dev Set의 분포도가 서로 다르기 때문에, Variance 문제라고 단정지을 수 없다. 

Training Set의 Data들은 고화질의 Image로 아주 쉽게 분류가 가능했지만, Dev Set은 흐릿한 Image로 쉽게 분류하지 못했을 수 있다는 것이다.

이런 Error Analysis 상황에서 Training Error와 Dev Error를 비교할 때, Training Set과 Dev Set의 분포도가 서로 다르기 때문에, Error의 차이가 얼마만큼 Algorithm이 Dev Set의 분포도를 Train하지 못해서 생기는 오류인지, 정말 Variance 문제로 인해서 발생하는 Error인지 구분하기가 쉽지 않다.

이 문제를 해결하기 위해서 우리는 **Training-Dev Set**이라고 새롭게 Data Set을 정의하고 사용하는 것이 도움이 될 수 있다. 

Training-Dev Set은 Training Set에서 일부 추출한 Data인데, Training Set과 동일한 분포도를 가지게 되지만, 이 Data는 Train을 위한 Data는 아니다.

<img width="250" src="https://github.com/user-attachments/assets/23a32d0d-61a8-4090-b07c-1ba5cab65209">

위와 같이 Data를 분배하게 되고, Train과 Train-Dev는 서로 같은 분포도를 갖고, Dev와 Test가 서로 같은 분포도를 갖는다. 

물론 Train, Train-Dev와 Dev, Test의 분포도는 서로 다르다.

이렇게 분배한 후에 Training Set으로만 Train시키고, Error Analysis를 진행하기 위해서 Training Error, Training-Dev Error, Dev Error를 구한다.

<img width="300" src="https://github.com/user-attachments/assets/e5e29601-dd80-48f5-9988-1c4f187d8cf2">

위 두가지 Example를 살펴보자.

왼쪽 Example는 training Error가 1%, Training-Dev Error가 9%, Dev Error가 10%이다. 

결과를 보면 Training → Training-Dev Error에서 많이 증가한 것을 볼 수 있고, 같은 분포도를 갖더라도 일반화가 잘 되지 않았다는 것을 볼 수 있기 때문에, 이 문제는 Variance 문제가 있다고 볼 수 있을 것이다.

오른쪽 Example는 Training Error는 1%, Training-Dev Error는 1.5%, Dev Error는 10%이다. 

이 경우 Training과 Training-Dev Error의 차이가 크지 않으므로, Low Variance를 갖는다고 할 수 있으며, Dev Error에서 많이 증가한다. 

이런 경우가 전형적인 Data Mismatch문제를 갖고 있다고 할 수 있다.

다음 두가지 Example를 더 살펴보자.

<img width="300" src="https://github.com/user-attachments/assets/9c3502cd-749d-49d4-87b8-7b1415f2196a">

이번에는 Human Error까지 포함했는데, 왼쪽 Example는 Human Error가 0%, Training Error는 10%, Training-Dev Error가 11%, Dev Error가 12%이다.

Bayes Error가 0%라는 의미인데, Training Error와의 차이인 Avoidable Bias가 매우 큰 것을 볼 수 있고, Bias 문제가 있다고 볼 수 있다.

오른쪽 Example는 동일하게 Human Error가 0%, Training Error 10%, Training-Dev Error 11%, 그리고 Dev Error가 20%이다. 

이 경우에는 두 가지의 문제점이 존재한다. 첫 번째로 Bayes Error와 Training Error의 차이가 크기 때문에 Avoidable Bias문제가 존재하고, 두 번째는 Training-Dev Error와 Dev Error의 차이가 크기 때문에 Data Mismatch의 문제도 있다고 볼 수 있다.

핵심은 **Human-Level Error, Training Error, Training-Dev Error, Dev Error를 확인**하고, 이 **오류들의 차이를 분석**해서 **Avoidable Bias Problem이나 Variance, Mismatch문제가 있는지 확인하는 것**이다.

![Untitled (18)](https://github.com/user-attachments/assets/176ba999-59fe-407c-b537-4a6230605f46)

추가로, **Test Error를 구해서 얼마나 Dev Set에 Overfitting 되었는지 확인하는 척도**로 사용할 수 있다.

그리고 Training Error보다 Dev, Test Error가 더 잘 동작할 수도 있는데, 이런 경우는 Dev, Test Set보다 Training Set이 Train하기에 더 힘든 Data 경우에 발생할 수 있다.

더 일반화해서, Rearview Mirror 음성 인식 System Example를 가지고 살펴보도록 하겠다.

![Untitled (19)](https://github.com/user-attachments/assets/1a34b749-7eda-4619-ad32-bde89cc2c531)

가로 축은 General Speech Recognition, 즉, 일반적인 음성 인식 System에서 가지고 온 Data가 있고, Rearview Mirror Speech Data, Rearview Mirror를 통한 음성 인식 Data가 있다. 

세로 축에는 Human Level Error, Train Algorithm이 Train한 Examples의 Error, 그리고 Train하지 않은 Examples의 Error가 있다. 

각 칸의 의미는 위와 같고, 이전에 살펴본 Example와 같이 동일하게 분석해서 Avoidable Bias, Variance, Data Mismatch 문제를 분석하는 것이다.

추가적으로 Rearview Mirror Speech Data에서 Human Level Error와 Error on Examples Trained on의 값들을 구해서 분석하는 경우에 부가적인 Insight를 제공하기도 한다. 

위 Example에 따르면, Rearview Mirror 음성 인식 Data는 일반 음성 인성 인식 Data보다 Human Level Error가 더 크기 때문에, 보통 인식하기 더 어렵다는 것을 볼 수 있다.

## **Addressing Data Mismatch**

![Untitled (20)](https://github.com/user-attachments/assets/15461ee6-b46b-4b3e-9b99-06e7e5fdffaf)

Error Analysis로 Mismatch 문제가 있다는 것을 알았을 때, 위와 같은 방법이 있다.

첫 번째로는 **Training Set과 Dev, Test Set이 어떤 차이가 있는지 수작업으로 Error Analysis를 수행하는 것**이다. 

**Test Set에 Overfitting하는 것을 피하려면, Dev Set에서만 Error Analysis를 수행해야 한다.**

예를 들어서, Rearview Mirror 음성 인식 System을 개발하고 있다면, Training Set과 Dev Set을 비교해서 Dev Set이 보통 소음이 더 심하고, Car 소음도 많다는 점을 발견할 수도 있다. 

이처럼 Training Set이 Dev Set과 어떻게 다른지 파악할 수 있다면, 이후에 Traininig Data를 Dev Set과 더 유사하게 만들 수 있는 방법을 찾을 수 있다.

두 번째 방법은 **Dev, Test Set과 유사한 Data를 만들거나 또는 수집하는 것**이다. 

Car 소음이 가장 큰 원인이라고 발견했다면, Car 내에서 소음이 심한 Data를 Train 할 수 있다.

위 방법들은 Systematic한 방법이 아니고, Insight를 얻는 것이 보장되지 않는 것처럼 생각될 수 있지만, 수작업으로 이렇게 Error 분석하는 것은 종종 많은 문제들을 해결하는데 도움이 될 수 있다.

이렇게 분석을 해서 결국 Training Data를 Dev Set와 더 유사하게 만드는 것이 목표라면 우리가 할 수 있는 방법은 무엇이 있을까 ? 

우리가 사용할 수 있는 방법 중에 하나는 인공적으로 Data를 합성하는 것이 있다. → **Artificial Data Synthesis**

![Untitled (21)](https://github.com/user-attachments/assets/464ef144-adb0-4ff1-ba49-ce37b22032ff)

우리가 깨끗한 Audio를 많이 녹음했다고 가정하고, 차 소음 Data를 가지고 있다면, 이 두 개의 음성을 합성해서 시끄러운 차 안에서 말하는 것과 같은 효과를 낼 수 있다.

실제로 밖에 나가서 시간을 쏟으면서 엄청난 양의 음성 자료를 모을 필요없이, 인공적인 Data 합성을 통해서 더 빨리 Data를 생성할 수 있다.

하지만, 이 방법을 사용할 때 한 가지 주의할 점이 있다. 

만약 우리가 10,000시간 동안 소음없이 깨끗하게 녹음된 Data가 있고, Car 소음 Data가 1시간짜리가 있다면, Car 소음 Data를 10,000번 반복시켜서 합성할 수 있다. 

실제 Car에서는 다양한 소음들이 있지만, 이 경우에는 우리가 1시간짜리 소음에 대해서만 Train을 진행하게 된다면, 한 시간짜리 Car 소음에 Overfitting할 수도 있다. 10,000시간 동안의 Car 소음을 수집하는 것이 가능할지는 모르겠지만, 10000 시간의 Car 소음으로 합성을 한다면 더 좋은 성능을 낼 수 있을 것이다.

즉, 합성을 통해서 만든 Data들이 전체의 일부분이 될 수도 있다는 것이다. 

따라서 **합성한 Data에 Overfitting할 위험이 존재하고, 합성할 때에 이 부분을 유의**해야 한다.
	
## **Transfer Learning**

DL의 강력함은 한 가지 Task에서 Train한 내용은 다른 Task에 적용을 할 수 있다는 것이다. 

예를 들어서, Neural Network(NN)이 Cat와 같은 Image을 인식하도록 Train했을 때, 여기서 Train한 것을 가지고 부분적으로 X-ray Image를 인식하는데 도움이 되도록 할 수 있다. 

이것이 바로 **Transfer Learning**이라고 한다.

![Untitled (22)](https://github.com/user-attachments/assets/ed5d69cd-c5d7-48c3-a097-9e4a3fa73fdc)

Image 인식 기능을 NN으로 Train을 했다고 해보자. x는 Image이고, y는 Some Object이다. 

Image는 Cat나, Dog, 또는 Bird 등이 될 수 있다. 

이렇게 Train한 NN을 사용해서 Transfer한다고 표현하는데, Cat와 같은 Image을 인식하도록 Train한 것이 X-ray Scan을 읽어서 방사선 진단에 도움이 될 수 있다.

Transfer Learning은 다음과 같이 적용한다.

우선, Train한 NN Network의 마지막 Output Layer를 삭제하고, 마지막 Layer의 Parameter, Weight(and Bias)도 삭제한다. 

그리고, 마지막 Layer를 새로 만들고, 무작위로 초기화된 Weight를 만든다. 

이렇게 생성한 Layer들을 통해서 진단의 결과값을 나타내는 것이다.

구체적으로 설명하자면, Image 인식 업무에 관련해서 Parameter를 Train을 시킨 것을 사용하는 것이고, 이 Train Algorithm을 방사선 Image에 Transfer 한다.

Data Set (x, y)를 방사선 Image로 바꾸어 주고, 마지막 Output Layer의 Parameter W[L], b[L]을 초기화시킨다. 

그리고 새로운 Data Set에서 NN을 다시 Train시키면 된다.

<img width="300" src="https://github.com/user-attachments/assets/f6d9baef-5485-47dd-9aa7-bf1a4a2ef84d">

기존 Train Algorithm을 가지고 다시 Train하는 경우에 방법은 새로운 Data Set에 따라 두 가지의 방법이 있다.
만약, 방사선 Image Data Set이 많지 않다면, 마지막 층의 Parameter만 초기화하고 나머지 Layer의 Parameter는 고정시켜서 Train시킬 수 있다. 

만약, Data가 충분히 많다면, 나머지 Layer에 대해서도 다시 Train시킬 수 있다.

모든 Layer의 Parameter를 다시 Training시키는 경우에, Image 인식 기능에 대해서 첫번째 Train을 Pre-training이라고 부른다.→ NN의 Parameter를 Pre-Initialize or Pre-Training하기 위해서 Image 인식 Data를 사용하기 때문이다. 

그리고, 방사선 Image에 Train시키는 두번째 단계를 종종 **Fine Tuning**이라고 한다.

이렇게 Image 인식 기능에서 Train한 내용을 방사선 Image를 인식하고 진단하는 것으로 Transfer 시킨 것이다. 

이것이 가능한 이유는 Edges를 감지하거나, Curve를 감지, 또는 Positive Objects를 감지하는 Low Level Features(특성) 때문이다. 

Image 인식을 위한 Data Set에서 Image의 구조, Image가 어떻게 생겼는지에 대해서 즉, Image들의 부분부분들을 인식하도록 Train한 것들이 방사선 Image 인식에도 유용할 수 있다.

다른 예제인, 음성 인식 System으로 다시 살펴보자.

![Untitled (24)](https://github.com/user-attachments/assets/1ab88d0d-b977-4ed6-9d95-f137b49e5044)

우리가 음성 인식 System을 Train했다고 한다면, Input x는 Audio가 될 것이고, y는 Ink Transcript가 될 것이다. 

이제 Wake Word(Trigger Word)를 감지하는 System을 만드려고 할 때, NN의 마지막 Layer를 삭제하고 새로운 Output Layer를 만들어야 한다. 

종종, 마지막 Output Layer를 하나로만 생성하는 게 아니라, 여러 개의 새로운 Layer를 생성할 수도 있다. 

그리고, 우리가 Train시키고자 하는 Data를 얼마나 보유하고 있느냐에 따라서, 새롭게 생성한 Layer만 Train하느냐 또는 더 많은 Layer들을 Train하느냐 선택할 수 있다.

위의 Examples들로부터 우리는 Transfer Learning는 언제 적용해야되는지 파악할 수 있을 것이다.

Transfer하고자 하는 곳의 Data가 많고, Transfer하려고 하는 곳에 Data가 적을 때 효과적일 것이다.

예를 들어, Image 인식 업무를 위한 Data가 1,000,000가 있으면, 이 정도 Data는 Low Level의 특성을 배우기에 충분할 것이다. 

하지만, 방사선 Image 인식을 위한 Data는 매우 적게, 100개의 Image만이 있다고 하자. 

그렇다면, Image 인식 기능에서 Train한 것들이 방사선 Image로 Transfer되어서 사용되면, 방사선 Image의 Data가 많지 않더라도 도움이 될 수 있다.

음성 인식 기능도 마찬가지다. 음성 인식 기능에서 10000시간의 Data로 Train을 했지만, Trigger Word감지를 위한 Data는 1시간 분량밖에 되지 않을 때, 음성 인식 기능에서 어떻게 듣는지 배우는 부분들이 Trigger Word감지 기능을 Train하는데 도움을 줄 수 있다.

당연히 반대의 경우에는 Transfer Laerning이 불가능하다.

정리하자면, Transfer Learning은 Task A와 Task B가 같은 Input(Image나 음성같은)으로 구성되어 있고, Task A의 Data가 Task B의 Data보다 훨씬 더 많은 경우에 가능하다. 

조금 더 추가하자면, Task A에서 Train하는 Low Level 특성이 Task B를 Train하는데 도움이 될 수 있다고 판단되는 경우에 사용할 수 있다.

## **Multi-Task Learning**

Transfer Learning이 순차적으로 Task A를 Train하고 Task B로 넘어가는 절차가 있었다면, **Multi-task Learning은 동시에 Train을 진행**한다. 

NN이 여러가지 Task를 할 수 있도록 만들고, 각각의 Task가 다른 Task들을 도와주는 역할을 한다.

우리가 Autonomous Driving Car를 만든다고 생각해보자. 

Autonomous Driving Car는 보행자나 다른 Car, 정지 표지판, 신호등 등을 잘 감지해야 한다.

![Untitled (25)](https://github.com/user-attachments/assets/834bbb73-101d-4463-9f36-79d95b986b42)

위 왼쪽 Image를 보면, 정지 표지판과 Car이 있고, 보행자나 신호등은 보이지 않는다. 

이 Image가 Input x(i)라고 한다면, Output y(i)는 하나의 Label이 아닌 4개의 Label이 필요할 것이다. 

만약 더 많은 것들을 감지하려고 한다면, 4개가 아니라 더 많은 Label을 가질 수 있을 것이다.

그러면 y(i)는 4 x 1 Vector가 되고, Data Set 전체를 참조하면, y Matrix는 오른쪽 아래처럼 4 x m Matrix가 된다.

우리는 y값을 예측하기 위해서 NN을 Train시키는 것이고, Input x를 가지고, output y-hat를 구하는게 목적이다.

![Untitled (26)](https://github.com/user-attachments/assets/c7e53af9-7fc9-4730-9463-ae5f7cf72844)

위의 Neural Network를 Train하기 위해서는 NN의 Loss를 정의해야 한다.

예측값은 4 x 1 Vector인 y-hat이기 때문에, Loss는 다음과 같이 구할 수 있다.

<img width="744" alt="스크린샷 2024-08-01 오후 6 11 55" src="https://github.com/user-attachments/assets/f991166b-3a5f-44a2-971e-75040e998483">

여기서 Image는 복수의 Label을 가질 수 있다. 

따라서 Image가 보행자나 Car, 정지 표지판, 신호등으로 판단하는 것이 아니라 하나의 Image에 대해서 보행자, Car, 정지 표지판, 신호등이 있는지 판단하는 것이고 여러 개의 물체가 같은 Image에 존재할 수 있다는 것이다. 

여기서 결과는 4개의 Label을 가지고, Image가 4개의 물체를 포함하고 있는지 알려주는 역할을 한다.

우리가 사용할 수 있는 또 다른 방법은 4개의 NN을 각각 Training 시키는 것이다. 

하지만 NN의 초반 특성들 중의 일부가 다른 물체들과 공유될 수 있다면, 이렇게 4개의 NN을 각각 Training 시키는 것보다, **한 개의 NN을 Train시켜서 4개의 일을 할 수 있도록 하는 것이 보통 더 좋은 성능을 갖는다.**

부가적인 내용으로, 이제까지 Train을 하기 위한 Data의 Label이 모두 달려있는 것처럼 설명했지만, **Multi-Task Learning은 어떤 Image가 일부 물체에 대해서만 Label이 되어 있더라도 잘 동작**한다.

<img width="300" src="https://github.com/user-attachments/assets/c8b0a8c4-0a09-4fad-93f2-cb205cb8fbe8">

위 Label처럼 Label하는 사람이 귀찮거나, 실수로 Label하지 않았아서 물음표로 나타나더라도 Algorithm을 Train시킬 수 있다. 

이런 경우에 Loss를 구할 때에는 물음표는 제외하고 y의 값이 0이거나 1인 것들만 취급해서 Loss를 구한다.

Multi-Tasking은 언제 사용할 수 있을까 ?

![Untitled (28)](https://github.com/user-attachments/assets/78452520-dc9a-4526-99be-6427347edc9c)

첫 번째로 각각의 task들을 Train할 때, Lower Level 특성을 서로 공유해서 유용하게 사용될 수 있는 경우에 가능하다.

두 번째는 필수 규칙은 아니고 항상 옳지도 않지만 각각의 Task의 Data Set의 양이 유사할 때 가능하다. 

엄격하게 적용되는 것은 아니지만, Multi Task Learning이 효과가 있으려면, 보통 다른 Task들의 Data의 합이 하나의 Task의 양보다 훨씬 많아야 한다.

마지막, 충분히 큰 NN에서 Train시키는 경우에 잘 동작한다. 

Rich Carona 연구원은 Multi-Task Learning이 각각의 NN으로 Train하는 것보다 성능이 좋지 않다면, NN이 충분히 크기 못하는 경우라는 것을 발견했다.

실제로 Multi-Task Learning은 Transfer Learning보다 훨씬 더 적게 사용되며, Transfer Learning의 경우에는 Data는 적지만, 문제 해결을 위해 사용되는 경우를 자주 보게 된다.

예외적으로 Computer Vision Object Detection 영역에서는 Transfer Learning보다 Multi-Task Learning이 더 자주 사용되며, 개별적인 NN으로 Train하는 것보다 더 잘 동작한다.

## **What is End-to-End Deep Learning**

End-to-End Deep Learning은 여러 단계의 Process를 거치는 것들을 하나의 NN으로 변환하는 것이다. 

![Untitled (29)](https://github.com/user-attachments/assets/1f1cd83f-dd91-4940-b78f-af5c5666e208)

음성 인식을 예로 들어보면, Input x인 Audio를 사용해서 Output y에 Mapping하면 Audio Clip이 글로 옮겨지게 되는 것이다. 

이전에 음성 인식은 많은 단계가 필요했다. 먼저 Audio Hand-Designed Features를 추출한다. 

MFCC라는 Algorithm을 사용할 수 있는데, 이 Algorithm은 Audio Hand-Designed Feature와 Low-Level Feature를 추출하는 Algorithm다. 

그리고 ML을 적용해서 Phonemes를 추출한다. 그 다음에 음소들을 묶어서 Words 형태로 만들고, 글로 나타낸다.

반대로 End-to-End Deep Learning은 하나의 거대한 Neural Network를 Train시켜서, Audio 입력하고 Direct로 문자로 출력하도록 하는 것이다.

End-to-End Learning이 효과를 나타내기 시작하면서, 많은 연구원들이 개별적인 단계를 설정하기 위해서 Pipe-Line을 설계하는 시간을 줄이게 되었다.

End-to-End Deep Learning은 기존과 달리 중간 단계들을 우회한다. 

End-to-End Deep Learning이 잘 동작하기 위해서는 많은 양의 Data가 필요하다.

예를 들어서, 음성 인식을 위한 3000시간의 Data가 있을 때, 기존의 중간 과정이 있는 Pipe-Line에서는 잘 동작하지만, End-to-End Deep Learning은 10,000시간 또는 최대 100,000시간의 많은 Data가 있는 경우에 잘 동작한다. 

따라서, Data가 작다면 전통적인 Pipe-Line을 통한 Train이 실제로 더 잘 동작한다.

만약 중간 정도 양의 Data가 있다면 Pipe-Line과 End-to-End Deep Learning의 중간 방법을 사용할 수 있다.

![Untitled (30)](https://github.com/user-attachments/assets/84e8db8d-64d2-4f0f-9af1-81ed898f23fc)

얼굴 인식 예제로 Pipeline 방식과 End-to-End Deep Learning을 조금 더 살펴보도록 하자.

![Untitled (31)](https://github.com/user-attachments/assets/66b3b557-4697-44ae-b9d6-36e921a42624)

위 System은 Camera가 문에 접근하는 사람의 얼굴을 확인하고, 인식하면 회전식 출입문을 자동으로 통과시켜주는 System이다. 

따라서 ID Card 등을 들고다니지 않아도, 건물 안으로 들어갈 수 있다. 

이 System은 어떻게 구축하면 될까?

가능한 방법 중의 한 가지는 Camera Capture한 Image를 인식하는 것이다. 

사람이 출입문에 다가가면 Camera가 Capture하고 Capture한 Image를 Input x로 사용하고, 사람의 ID인 Ouput y로 Mapping하는 것이다.

위 방법의 문제점은 출입문에 접근하는 사람이 다양한 방향에서 접근할 수 있다는 것이다. 

위 Image에서 녹색 위치거나 파란색 위치일 수도 있고, 보라색처럼 매우 가까이 있어서 얼굴이 너무 크게 보일 수도 있다. 

즉, 이 System을 구축하기 위해서 사용할 수 있는 최선의 방법은 End-to-End Deep Learning 방식으로 Raw Image를 사용해서 NN으로 ID를 알아내는 것이 아니라, 여러 단계로 나누는 것이다.

![Untitled (32)](https://github.com/user-attachments/assets/d74050e2-cb08-4d5e-b8a8-03cc0a756aa1)

먼저, 첫 번째 감지기는 사람의 얼굴이 Image의 어디에 있는지 알아낸다. 

그리고 그 사람의 얼굴을 발견하면, 그 Image 부분을 자르고 확대해서 얼굴이 가운데로 오게 한다. 

그런 다음에 이 Image를 NN에 Input으로 사용해서 Train을 해서 동일인물인지 비교하는 방법 등을 통해서 사람의 ID를 파악할 수 있다.

두 번째 접근 방법인 두 개의 Train Algorithm을 사용하는 것이 전체적으로 더 좋은 성능을 보여준다.

왜 두 번째 방법이 더 효과적일까?

우선 각각의 Algorithm이 해결하고자하는 문제가 단순하다는 것이다. 

두 번째는 각각의 Algorithm Train을 위한 Data가 충분히 많다는 것이다. 

특히나 Face Detection을 위한 Data는 매우 많다. 

그리고, 동일 인물인지 아닌지 알아내기 위한 Data도 무수히 많다.

대조적으로 End-to-End Deep Learning 방식으로 모든 것을 동시에 Train하려고 시도한다면, x → y로 Mapping되는 Data가 매우 적을 것이다. 

따라서 End-to-End Deep Learning 방법으로 해결하기 위해서 충분한 Data는 없지만, 실제로 단계를 나누어서 하위 문제를 해결하기 위한 Data는 충분하므로, 단계를 나누어서 해결하는 것이 더 효과적이다. 

물론, 필요한 Data가 충분하다면, 아마도 End-to-end 접근 방법이 더 효과적일 것이다.

![Untitled (33)](https://github.com/user-attachments/assets/e2cd4c3b-b9e6-44e4-8c22-fd65dfa4438b)

기계 번역을 예로 들어보면, 전통적으로 기계 번역 System은 복잡한 Pipe-Line으로 구성되어 있는데, 영어 Text를 가지고 여러 단계를 거쳐서, 프랑스어로 번역하게 된다.

오늘 날에는 꽤 많은 Matching Data가 있기 때문에, 이런 경우에는 End-to-End Deep Learning이 기계 번역에서 매우 효과적일 수 있다.

또 다른 예시로, 아이의 손 X-ray 사진을 가지고, 나이를 추정하는 System을 살펴보자. 해당 System은 아이가 정상적으로 성장하고 있는지 확인하는데 사용될 수 있다.

먼저 Image를 살펴보면서 뼈 부분, 뼈 조각의 위치, 다른 뼈의 길이를 알면 아이들의 평균 손 뼈 길이표를 대조해서 아이의 나이를 추정할 수 있고, 단계를 나누는 것이 꽤 효과가 있다. 

반대로, End-to-End 접근 법으로 Image에서 아이들의 나이를 바로 Mapping한다면, 잘 동작하지 않을 수 있고 매우 많은 Data가 필요할 것이다. 

첫 번째 접근 방법에서 각 단계는 비교적 간단한 문제이고, 많은 Data가 필요하지 않을 수 있다. 

따라서 충분한 Data를 모으기 전까지는 첫 번째 접근 방법이 End-to-End 방법보다 더 효과적일 수 있다.

## **Whether to Use End-to-End Learning**

언제 End-to-End Deep Learning을 사용할 수 있을까 ? 

![Untitled (34)](https://github.com/user-attachments/assets/7021bb9e-d644-4ee7-ab3c-aee6de45a7f4)

**장점**

1. 완벽한 Train을 통해서 Data가 말을 하는 것처럼 할 수 있다. 
    
    우리가 충분히 많은 Data를 가지고 있다면, X → Y에 Mapping되는 기능을 만들어낼 수 있다. 
    
    예를 들어서, 이전의 음성 인식 System에서는 단어의 기본 음절 단위를 가지고 해석을 하는데, Train Algorithm이 음성학적으로 생각하지 않고, 음성 표현이 바로 해석된다면, 전체적인 성과는 더 좋아질 것이다.
    
2. 수작업이 줄어든다. 이 경우에는 Work-Flow가 단순해지고, 중간 과정들을 설계하는데 많은 시간을 투자하지 않아도 된다.

**단점**

1. Data가 많이 필요하다. X-Y Mapping을 바로 하기 위한 Data가 필요하다. 
    
    이전 강의에서는 하위 작업인 얼굴 인식과 얼굴을 식별하기 위한 Data들은 많았지만, Image를 바로 식별하기 위한 Data는 거의 없었다. 
    
    그래서 이런 System을 훈련시키기 위해서는 입력과 출력이 모두 필요한 Data가 필요하다.
    
2. 유용하게 수작업으로 설계된 Component들을 배제한다는 것이다. 
    
    Data가 많지 않다면, Train Set으로 얻을 수 있는 것이 적고, 잘 설계된 Hand-Designed Component를 사용하는 것이 더 좋은 성능을 낼 수 있다.
    
![Untitled (35)](https://github.com/user-attachments/assets/4fa87f86-54ba-42c9-9e86-d7bb281ff9bb)

End-to-End Deep Learning을 적용하기 위해서 중요한 질문은, x → y로 Mapping시키기 위해 필요한 Data가 충분히 많은가의 여부이다. 

이전 강의에서 봤듯이, 뼈를 인식하는 것과 인식 뼈를 토대로 나이를 추정하는 것은 Data가 많이 필요하지 않을 수 있지만, 뼈 사진을 통해서 바로 나이를 Mapping하는 것은 복잡한 문제처럼 보이고, 많은 Data가 필요할 것이다. 

다른 Example로 Autonomous Driving 차는 어떻게 만들 수 있을까 ?

![Untitled (36)](https://github.com/user-attachments/assets/b42c6ded-7d5f-4b76-a6f2-c98862089ab2)

Image, Radar, LiDAR를 통해서 주변 사물들을 인지하고, 그 다음에 어떤 길로 갈지 결정한다. 

그런 다음에 적절한 Steering과 가속,제동 명령을 실행한다. 

이 Example가 보여주는 것은 ML이나 DL을 사용해서 구성 요소들을 개별적으로 Train하고 적용할 때, 할 수 있는 작업에 따라 Train하려는 X → Y Mapping 유형을 신중하게 선택해야한다는 것이다.

대조적으로 End-to-End Deep Learning 방식은 Image를 입력하고, Steering을 직접 출력하는 것이고, 접근법이 꽤 흥미롭다. 

하지만, Data Availability와 NN을 통해 배울 수 있는 유형을 고려할 때, 이것은 실제로 가장 유망한 접근 방법, 최선의 접근 방법이 아니라고 생각한다.

**따라서 End-to-End Deep Learning이 정말 잘 동작하고, 효과가 있을 수 있지만, 어디에 적용할 수 있는지 알고 있어야 한다.**