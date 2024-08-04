# Optimization Algorithms

## Mini-Batch Gradient Descent

**Batch**는 한번에 처리하는 Data의 묶음을 의미, 중괄호를 사용해 아래와 같이 표현할 수 있다 .

![스크린샷 2024-07-16 오후 3.45.36.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/e7ad7dc7-d1ef-44df-bdb3-d9c925f0ef4e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.45.36.png)

```
{Batch 번호} [Layer 번호] (Example 번호)
```

일반적인 Gradient Descent는 **Batch Gradient Descent**라고 부르며 Parameter를 Update 할 때 모든 Training Example m에 대해 Gradient를 계산한 다음 Parameter를 Update 하게 된다. 

즉, Gradient Descent를 진행하기 전에 전처리 과정인 Gradient 계산이 필요하고 m이 커질수록 느려지게 된다. 

이때 Vectorization를 활용하면 m이 크더라도 효율적으로 계산을 할 수 있다. 

하지만 m이 너무 크다면 Training Example을 반복하는데 오랜 시간이 걸릴 것이다. 

**모든 Training Example m에 대해서가 아니라 중간에 Parameter를 Update하면 훨씬 더 빠르게 Algorithm을 진행할 수 있다.** 

우선 Training Example을 **Mini-Batch**라고 부르는 **Baby-Training Set**으로 나누어 준다. 

m = 5,000,000인 Training Set이 있을 때 각 1000개의 Example을 갖도록 나누게 되면 Mini-Batch는 총 5000개가 된다. 

최종적으로 x{5000}가 되고 Y에 대해서도 동일하게 나누어 준다. 

![스크린샷 2024-07-16 오후 3.41.30.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/7f09cca0-756a-4eff-b8c1-e0ce9d96f39e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.41.30.png)

각각의 X(nx, m)의 Mini-Batch들은 (nx, 1000)의 차원을 갖게 되고, Y(1, m)은 (1, 1000)의 차원을 갖게 된다. 

정리하면 Mini-Batch t = X{t}, Y{t}이며, t = Mini-Batch의 Index이다. 

여기서 **Batch**는 모든 Example들에 대해서 한 번에 처리하는 우리가 알고 있는 일반적인 **Gradient Descent를 의미**하며, **Mini-Batch Gradient Descent**는 **Mini-Batch를 한번에 처리해서 Parameter를 Update** 한다는 것을 의미한다.

**Batch Gradient Descent**

- 전체 Training Data를 하나의 Batch로 보고 Training 하는 것 → Batch Size는 m

**Mini-Batch Gradient Descent**

- 전체 Data를 여러 개의 Batch로 나눈 것

### Epoch

모든 Training Example을 한번에 Gradient Descent 하지 않고, Batch를 순회하며 천천히 Gradient Descent 시켜준다. 이 과정을 **Epoch**라고 한다. 

```
1 Epoch = 1 Pass Through Data
```

![for loop 1번 == 1 Epoch, 해당 코드에서 한 번의 Gradient Descent에서 총 5000 Epoch를 진행한다. 따라서 전체 Iteration을 위한 또 하나의 for loop가 필요하다. ](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/8a126602-3021-4f0a-a55a-15b9800c5c6b/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.05.55.png)

for loop 1번 == 1 Epoch, 해당 코드에서 한 번의 Gradient Descent에서 총 5000 Epoch를 진행한다. 따라서 전체 Iteration을 위한 또 하나의 for loop가 필요하다. 

대체로 m이 매우 클 때 Mini-Batch는 Batch보다 훨씬 빠르게 진행된다. 

Mini-Batch는 전체 Data를 나누어 훨씬 더 빠르게 자주 Update하는 방법이다. 

**전체 Data를 Mini-Batch Size만큼 나누게 되면 그만큼 계산량이 줄어 더 빠르게 Parameter를 Update 할 수 있어 이론상 Mini-Batch Size 배수만큼 더 빠른 Training이 가능하다.** 

![스크린샷 2024-07-16 오후 3.41.52.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/f7e81381-4f9f-40cc-983b-6ba33e686ef6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.41.52.png)

## Understanding Mini-Batch Gradient Descent

전체 Training Set에 Gradient Descent를 진행하게 되면 반복할 때마다 Cost가 계속해서 감소할 것을 기대한다. 

만약 한 번의 Iteration에서 Cost가 증가한다면 어딘가 문제가 발생했음을 의미한다. 

Learning Rate가 매우 큰 경우 이러한 현상이 발생할 수 있다. 

![스크린샷 2024-07-16 오후 4.23.37.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/c03d45bc-05a2-4a6c-b145-5fd93822cd5d/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.23.37.png)

Mini-Batch Gradient Descent의 경우 매 반복마다 X{t}, Y{t}를 처리하게 되는데 이 값을 통해 J{t}를 구하게 되고, 매 반복마다 다른 Training Set에서 Train을 진행하는 것과 동일하다. 

이렇게 Training을 진행하게 되면 Cost Function은 오른쪽과 같이 Noisy한 모습을 보이게 될 것이다. 

**Mini-Batch Size가 1인 경우 Stochastic Gradient Descent**으로 각 Example 하나가 Mini-Batch가 되는 것이다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/35245ece-1793-4408-b7e9-b2b7c0fb1c2e/Untitled.png)

위 두 가지의 경우 최적의 값을 찾아가는 모습은 아래와 같다. 파란색이 Batch Gradient Descent이고 보라색이 Stochastic Gradient Descent을 나타낸다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/37b3abd4-417d-4aad-bf4b-0431c0afd8c5/Untitled.png)

실제로 우리가 사용하는 **Mini-Batch Size는 1-m 사이의 값**이 될 것이다. 

**Mini-Batch Size를 m으로 설정**하면 매 Iteration마다 아주 많은 Training Set을 처리하게 된다.

m이 너무 크다면 긴 시간이 소요, m이 작다면 Batch Gradient Descent를 그냥 사용해도 괜찮다. 

반대로 **Mini-Batch Size를 1로 설정**하면 하나의 Example만 처리해도 Parameter를 Update할 수 있다는 점이 좋다.

Noise가 많이 발생할 수 있지만 더 작은 Learning Rate를 사용해 개선시키거나 감소시킬 수 있다. 

하지만 Stochastic Gradient Descent는 Sample을 처리하는 점에서 Vectorization의 효율을 잃어버리는 단점이 있다. 

따라서 **실제로 가장 잘 동작하는 것은 Mini-Batch Size가 너무 크거나 작지 않은 경우**이다. 

가장 빠른 Training 속도를 보여주며 두 가지 장점이 있다. 

1. Stochastic Gradient Descent와 비교했을 때 Vectorization을 활용한 빠른 속도를 잃지 않는다. 
    
    1000개씩 처리하는 것이 1개씩 처리하는 것보다 빠르다.
    
2. 전체 Training Set m을 처리할 때까지 기다릴 필요 없이 Parameter Update인 Gradient Descent를 수행할 수 있다. 
    
    한 번의 Iteration으로 5000 Epoch의 Gradient Descent Step을 수행한다. 
    

![스크린샷 2024-07-16 오후 4.39.11.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/6bb2b56a-ae17-48f0-8238-fefc752e8b7f/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.39.11.png)

적절한 Mini-Batch Size를 선택하면 최적값을 찾아갈 때 왼쪽 Graph와 같이 진행된다. 

매 Gradient Descent Step에서 최소 값을 향한다는 보장은 없지만 전체 과정에서 최소값을 향해 진행하는 경향이 있다. 변동이 너무 크다면 언제든디 Learning Rate를 감소시켜 개선할 수 있다. 

**만약 Mini-Batch Size가 1-m 사이의 값이면 어떻게 그 값을 선택할 수 있을까**

1. Training Set의 크기가 작다면 Batch Gradient Descent를 사용한다. m의 크기가 작다면 Mini-Batch Gradient Descent는 의미없다. → n ≤ 2000
2. m이 충분히 크다면 Mini-Batch Size는 64-512가 보편적이다. 보통 64, 128, 256, 512로 사용되는데 Conputer Memory의 Layout과 접근 방식에 따라 달라지고 2의 지수의 값을 가질 때 더 빨리 실행된다. 
    
    2^n으로 선택, 만약 1000이라면 1024를 권장한다. 
    
3. 모든 Mini-Batch X{t}, Y{t}가 CPU, GPU Memory에 있도록 한다. 이는 Application이나 Training Sample의 크기에 따라 변할 수 있지만 CPU, GPU에 들어가지 않는 Mini-Batch를 처리하는 경우 성능이 저하되고 악화되는 것을 볼 수 있다.  

**Mini-Batch Size**가 또 하나의 **Hyperparamter**이며, **2의 지수값을 몇 개 시도해보고 Algorithm을 효율적으로 만들어주는 최적의 한 가지 값을 선택**하면 된다. 

## Exponentially Weighted Averages

**Exponentially Weighted Averages**를 통계학에서는 Exponentially Moving Averages, **EMA**라고 한다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/ad03f3d2-1ac0-474e-8154-25dea8434f1f/Untitled.png)

London의 날씨 예보를 보면서 EMA 예시를 들어볼 것이다. 1년동안 London의 날씨를 표로 나타내면 오른쪽과 같다. 

정형화 되어 있지 않고 날씨 Data를 모두 나타내어 Nosiy 하게 보인다. 

날씨 변화 Trend를 구하고 싶은 경우 Local 평균, 온도에 대한 이동 평균값을 구하기 위해서는 아래와 같이 구하면 된다. 

![스크린샷 2024-07-17 오후 2.22.56.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/367c34ba-027f-432a-aa84-797d73257a3e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.22.56.png)

이렇게 구하면 다음과 같이 빨간색 Graph로 나타낼 수 있고, 각 days의 온도를 Moving Average로 나타낸 EMA를 구할 수 있다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/2e2f4e45-594a-4c3d-b16e-e7bcde02a2b2/Untitled.png)

위 공식의 0.9를 β로 변경해서 나타내면 아래와 같다. 

![스크린샷 2024-07-17 오후 2.26.56.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/40df6acc-baec-48e3-a264-8637ad670bf9/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.26.56.png)

여기서 Vt는 대략적으로 (1 / 1 - β)days의 평균 기온이 된다. 

즉, β = 0.9는 10일동안 평균 기온과 비슷하다고 보면 된다. Graph로 나타내면 빨간색 선이 된다.  

만약 β = 0.98로 지정된다면 지난 50일간의 평균 기온과 비슷하다. Graph로 나타내면 초록색 선이 된다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/f8648c31-b9bc-4208-b3c9-a23281023943/Untitled.png)

β = 0.5라면 2일간의 평균 기온이기 때문에 노란색 Graph처럼 매우 Nosiy한 결과를 얻을 것이고, Outliner에 취약하게 된다. 하지만 기온이 변하는 것을 더 빨리 반영시켜준다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/6b305d4c-18f4-4af6-952f-768dd36cebe3/Untitled.png)

## Understanding Exponentially Weighted Averages

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/51419c31-bb87-4468-abd9-4b970f4c93bc/Untitled.png)

![스크린샷 2024-07-17 오후 2.30.47.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/3589b2f8-4240-4296-a8f9-f8c0ffca5d08/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.30.47.png)

위 식에서 β의 값을 0.9, 0.98, 0.5로 설정했을 때 위와 같은 빨간색, 초록색, 노란색 Graph를 얻을 수 있었는데 조금 더 수학적으로 일별 평균 기온을 어떻게 산출하면 아래와 같다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/833f4aed-f215-499c-b331-2d1b597a478d/Untitled.png)

![V100](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/20e72221-e4b0-4b9c-ac93-f9674a6cb9dc/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.33.34.png)

V100

일별 기온을 기하급수적으로 감수하는 함수에 곱해 더하는 것이다. 아래 두 Graph 곱의 합을 의미한다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/185057c6-6ac7-4e06-818c-65df066f871a/Untitled.png)

이런 점 때문에 Exponentally Weighted Average라고 부른다. 

여기서 0.9의 10승을 하게 되면 그 값이 0.35정도가 되는데 이 값은 대략 1 / e이다. 

다시 말해 지수 Graph가 1 / 3정도로 감소하는데 약 10일 정도가 소요되고 β가 0.9인 경우에 직전 10간의 기온만 집중하여 EMA를 구하는 것과 같다. 

일반적으로 ε을 사용해서 (1 - ε)^1 / ε = 1 / e로 나타내며 β가 0.9였다면 ε의 값은 0.1일 것이다. 

반면 β의 값이 0.98이라고 한다면 0.98의 50승이면 대략 1 / e와 비슷한 값이 되고, 이 값은 50일간 대략적인 평균치라고 보면 된다. 

대략적인 평균 일수를 나타낸 것이기 때문에 정식 수학적인 표현은 아니다. 

![스크린샷 2024-07-17 오후 2.52.43.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/afc6007e-6027-4c27-a8ef-53f97aa4e018/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.52.43.png)

v0은 0으로 초기화하고 그 다음에 첫째 날의 v1을 구하고 다음에 v2를 구하게 된다. 

구현을 하게 되면 오른쪽과 같은데 하나의 vθ를 가지고 갱신하면서 EMA를 구할 수 있다. 

기본적으로 단 한 줄의 Code로 단 하나의 값만 Memory에 저장하면 되기 때문에 아주 적은 양의 Memory가 사용되고 효율성이 높은 것이 장점이다. 

하지만 위 공식은 정확한 평균 계산 방식은 아니다. 

정확한 평균은 최근 10일이나 50일의 기온을 더하고 10이나 50으로 나누면 더 좋은 추정치를 얻을 수 있지만 이 방법은 많은 Memory가 필요로 하며 구현이 복잡하고 연산의 부담이 커지는 단점이 있다. 

이러한 이유로 ML에서 EMA가 많이 사용된다. 

## Bias Correction in Exponentially Weighted Average

EMA를 구현하는 방법에 대해서 배웠는데, 여기서 Bias Correction이라고 하는 세부적인 기술이 있다. 이 방법은 EMA를 조금 더 정확하게 계산할 수 있도록 해준다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/e861780f-81b2-4706-8949-338906d51366/Untitled.png)

이전에 β의 값이 0.9와 0.98인 Graph를 보았는데, 실제로 위와 같은 Graph가 나오지는 않는다. 

β가 0.98일 경우 실제 Graph는 아래와 보라색과 같다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/f5ad6143-af6d-4097-b85c-7c659f9d6e85/Untitled.png)

보라색 Graph는 매우 낮은 곳에서 시작한다. EMA를 구할 때 v0 = 0으로 초기화 하고 진행한다. 

따라서 v1 = 0.98v0 + 0.02θ1을 계산할 때 0.98은 무시된다. 

v1은 0.02θ1이 되는 것이고, 화씨 40도인 경우 v1은 0.02 * 40으로 8이 되어 훨씬 더 작은 값이 나오게 된다. 

첫 번째 날 온도로는 좋은 평균치가 아니고 그 다음날도 마찬가지이다. 

특히 초반 부분에 대해서는 좋은 평균치가 아니게 되는데 이런 평균치를 보완해주기 위한 방법이 **Bias Correction**이다. 

이 방법은 vt 대신 vt / 1 - βt를 사용하는 것이다. 

t = 2일 때를 살펴보면 아래와 같다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/be1b5134-e1aa-4000-af0f-45f2169b0f4d/Untitled.png)

초기 EMA 값들을 1 - βt로 보정해주게 되고 t값이 충분히 커지면 βt는 0으로 수렴하게 되고 Bias Correction의 영향은 사라지게 된다. 

이 방법을 통해 초반부 EMA 값을 보정해주고 보라색 Graph가 초록색 Graph에 맞춰 들어가도록 해준다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/8a58496a-ce12-4718-98d5-a954593e6daa/Untitled.png)

ML에서는 EMA를 구현할 때 Bias Correction을 보통 신경쓰지 않고 적용하지 않는다. 

Bias 된 평균치가 구해지는 초기 구간을 기다리거나 Bias 된 구간부터 시작해도 되기 때문이다. 

Bias Correction은 더 좋은 평균치를 일찍 구하는데 도움이 된다. 

## Gradient Descent with Momentum

**Momentum, Gradient Descent with Momentum**이라고 불리는 Algorithm이 있다. 

이 Algorithm은 **일반적인 Gradient Descent보다 항상 더 빠르게 동작**한다. 

Algorithm의 **기본 Idea는 Gradient Descent의 EMA를 구하고 이 값을 이용해 Parameter W를 Update하는 것**이다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/eb3dea42-d656-47af-9b7a-a66108ce62b4/Untitled.png)

위와 같은 Cost Function을 최적화시키려고 한다고 할 때, Gradient Descent나 Mini-Batch Gradient Descent를 사용하면 파란색 Graph처럼 빨간색의 최소값을 향해 왔다갔다 진동을 하면서 접근하게 된다. 

**이때 생기는 진동폭이 Gradient Descent를 느리게 만든다.** 

만약 훨씬 더 큰 Learning Rate를 사용하는 경우 보라색처럼 발산할 수도 있다. 

진동하는 폭을 감소시키기 위해서 세로축에는 Slower Learning, 가로축에는 Faster Learning을 원할 것이다. 

이는 Gradient Descent Momentum을 통해 감소시킬 수 있다. 

**Momentum Algorithm**은 아래와 같다. Layer를 나타내는 위첨자 [l]은 생략한다.

![스크린샷 2024-07-17 오후 3.27.43.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/c6243d15-55b0-4554-88ce-12e9bbac923c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.27.43.png)

현재 Mini-Batch에서 dW와 dB를 구한다. 

그 다음에 dW와 dB에 대해 EMA를 구하고, 구한 EMA를 가지고 W와 b Parameter를 Update한다. 이 과정을 통해 Gradient Descent를 더욱 Smooth하게 해준다. 

**세로축의 변동 평균은 거의 0**이 되고, **가로축의 변동 평균이 꽤 큰 값**이 되어 아래와 같이 빨간색 Graph처럼 **최소값을 향해 접근**하게 된다. 

![스크린샷 2024-07-17 오후 3.31.04.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/73932d39-7748-443a-b2ad-62f82620044d/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.31.04.png)

Algorithm에서 dW와 db는 가속도의 역할을 하고, VdW, Vdb는 속도의 역할을 하게 되는 것이다. 

**Momentum은 이전 Step에서 구한 dW와 db를 가지고 Update 되는 것이 아니라 이전 Iteration들의 dW와 db의 평균치를 가지고 Update 되는 것**이다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/83bd5d06-b8ef-4ddc-a2fd-3205da681479/Untitled.png)

위와 같이 정리할 수 있고, 여기서 Hyperparameter는 α, β 2가지이다. 

보통 β의 값은 0.9가 Default이며 이것은 지난 10번의 Gradient Descent의 평균치를 의미한다. 

그리고 Bias Correction을 적용할 수 있는데 이는 실제로 잘 사용되지 않는다. 

단지 10번의 Iteration 이후에는 Warmming-Up 되어 더이상 Bias Correction을 적용할 필요가 없기 때문이다. 

그리고 **vdW는 dW와 동일한 차원**을 갖고, **vdb는 db와 동일한 차원**을 갖는다. 

추가적으로 논문을 살펴보면 (1 - β)가 생략되어 나타나기도 한다. 

vdW = βvdW + dW로 나타난다. 이 경우 α 값이 그에 상응하는 값으로 설정되어야 하는데 α는 vdW, vdb의 Scaling에도 영향을 주기 때문에 Return 해야 할 수도 있어서 주로 생략되지 않은 공식을 더 선호한다. 

## RMSprop

또 다른 최적화 Algorithm으로 RMSprop이라는 Algorithm이 있다. 

이 Algorithm은 Root Mean Square Propagation의 약자인데 이 방법을 사용해 Gradient Descent의 속도를 증가시킬 수 있다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/1e8ee698-dc9c-4bfb-a2d4-7aa5ff3882e1/Untitled.png)

**세로축을 Parameter b라고 하고, 가로축을 Parameter W라고 할 때 세로축은 더 느리고 가로축은 더 빠르게 Training 하기를 원한다.** 

이때 RMSprop Algorithm은 아래와 같다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/a4613075-266a-4420-a66e-a822b721ae2e/Untitled.png)

**RMSprop Algorithm**은 현재 Mini-Batch에 대해 dW, db를 구하고 VdW가 아닌 SdW 표기를 사용해 구하게 되는데 Momentum과 유사하지만 **dW와 db를 제곱해 SdW, Sdb를 구한다.** 

이는 EMA의 제곱 평균을 구하는 것을 의미하고 이렇게 구한 SdW, Sdb를 가지고 Parameter를 아래와 같이 Update 하게 된다. 

![스크린샷 2024-07-17 오후 3.54.18.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/e4f8adf0-c1f1-447d-bb45-4dcef5813343/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.54.18.png)

위 예시의 가로축은 Parameter W, 세로축은 Parameter b라고 했는데 가로축의 변동은 증가시키고 세로축의 변동은 늦추고 싶기 때문에 **SdW는 비교적 작은 값**이 되어야 하고, **Sdb는 비교적 큰 값**이 되어야 한다. 

실제 미분항을 보면 가로축에서보다 세로축이 더 큰 것을 볼 수 있고, 기울기는 b 방향으로 더 크게 된다. 

가로축의 변통폭은 무뎌지고, 세로축의 변동폭은 커지도록 도와주고 아래와 초록색 Graph와 같은 모양으로 Update가 진행된다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/ffbf8066-f3c3-4d90-85e9-c8606b4b8861/Untitled.png)

Parameter를 간단히 W, b로 나타냈는데 실제로는 고차원의 Parameter 공간에 있고 변동을 무디게 하려고 하는 세로축의 차원은 w1, w2, w17과 같은 Parameter Set의 합일 수 있고, 가로축의 차원은 w3, w4 등 Parameter Set의 합으로 나타낼 수 있다. 

**RMSprop Hyperparameter = β2**

**Momentum Hyperparametre = β1**

RMSprop에서 Root SdW가 0에 가까워지면 이 값이 나누어지면서 폭발적인 값을 가지게 된다. 

이 값을 안정적이게 하기 위해 실제로 RMSprop을 도입할 때 분모에 아주 작은 ε 값을 더하게 된다. 

어떤 값이 되어도 상관 없지만 이상적인 값은 10^-8이 기본값이고 아래와 같다. 

![스크린샷 2024-07-17 오후 4.04.21.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/c869d098-6b7c-4e33-b25e-3359ea1a02f3/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.04.21.png)

이것이 **RMSprop**이며 **Momentum과 비슷하게 Gradient Descent를 진행할 때 진동을 감소시키는 효과가 있다.** 

이렇게 진동을 감소시키면 더 큰 Learning Rate α를 사용할 수 있고, Training Algorithm의 속도를 증가시킬 수 있다. 

## Adam Optimization Algorithm

**Adam Algorithm**은 **Momentum과 RMSprop을 합친 것**이다. 

![스크린샷 2024-07-17 오후 4.07.56.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/3743c0ae-b873-422c-82bf-ff956db0a5f9/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.07.56.png)

Momentum과 RMSprop을 각각 적용하면 **추가적으로 Adam에서는 Bias Correction도 같이 진행하게 된다.** 

이렇게 구한 

![스크린샷 2024-07-17 오후 4.08.46.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/2415c55f-1168-4102-bc8a-b7f8c63aaf44/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.08.46.png)

를 가지고 Parameter W, b를 위와 같이 Update 하면 된다. 

이 Algorithm은 아주 다양한 NN Network에서 매우 효과가 있음이 증명되었다. 

Adam Algorithm에서는 몇 개의 Hyperparameter가 있는데, **Learning Rate인 α는 아주 중요하고 Tuning이 필요한 Parameter**이다. 

따라서 다양한 범위의 값을 시도해 어떤 값인지 찾아야 한다.  

β1의 기본 설정값은 0.9이고 β2의 값은 Adam 논문을 따르면 0.999를 권장한다. 

그리고 ε의 값은 10^-8을 권장하지만 이 값은 성능에 거의 영향이 없기 때문에 굳이 바꿀 필요는 없다. 

**Adam Algorithm을 사용할 때 보통 기본값을 많이 사용해 β1, β2, ε 값을 잘 변경하지 않는다.** 

**α의 값만 여러가지 시도를 통해 어떤 값이 잘 동작하는지 확인하면 된다.** 

## Learning Rate Decay

**Algorithm을 더 빠르게 Training 시킬 수 있는 방법** 중 하나는 Learning Rate를 시간이 지나면서 천천히 감소시키는 것이다. 이를 **Learning Rate Decay**라고 한다. 

Mini-Batch Gradient Descent를 사용한다고 가정했을 때, Training을 반복하면서 최소값으로 향하는 경향이 있지만 정확하게 최소값으로 수렴하지는 않을 것이다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/5803cabd-3b65-4c72-bfbe-baf027aab82e/Untitled.png)

위와 같이 최소값 주변을 맴돌면서 절대로 수렴하지 않는다. 이는 **Learning Rate를 어떤 값으로 고정시켰고 Noise가 존재하기 때문**이다. 

하지만 만약 Learning Rate α를 천천히 감소시킨다면 초기 반복에서는 비교적 큰 Learning Rate로 빠르게 Training이 가능할 것이다. 

그리고 Learning Rate가 점점 감소하면서 Step이 점차 작아질 것이다. 

결국에는 최소값 부근에서 매우 좁은 범위를 왔다 갔다 할 것이다. 이전처럼 큰 Learning Rate를 가지고 최고값 주변을 맴도는 것과 비교하면 훨씬 낫다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/e6979593-db77-4394-aea3-357302e487d8/Untitled.png)

**Learning Rate 구현**은 아래와 같다. 

![스크린샷 2024-07-17 오후 4.29.30.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/27151920-f404-471a-a61f-b8719ffe0dc8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.29.30.png)

여기서 **Epoch는 각 Mini-Batch Set에 대해서 진행하는 것을 의미**하고 **α0은 초기 Learning Rate**이며 **Decay Late와 α0은 Hyperparameter**가 된다. 

α0 = 0.2, Decay Rate = 1인 경우 아래와 같다. 

![스크린샷 2024-07-17 오후 4.31.38.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/b2b29c8e-4496-43d6-8b77-7520d43a065e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.31.38.png)

만약 **Learning Rate Decay** 방법을 사용한다면 **다양한 값으로 α0과 Decay Rate를 바꿔보며 잘 동작하는 값을 선택**해야 한다. 

이 외 방법은 아래와 같다. 

![스크린샷 2024-07-17 오후 4.33.00.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/fe458f0d-9eb1-467c-bcb9-dafc66370f8e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.33.00.png)

Learning Rate Decay는 최후의 수단으로 나중에 사용해 볼 방법이다.

보통 α를 어떤 값으로 정해 Tuning 하는 것이 큰 효과가 있고, 필요한 경우 Learning Rate Decay를 사용하는 것을 권장한다. 

## The Problem of Local Optima

초기 DL 분야에서 최적화 Algorithm이 좋지 않은 Local Optima로 수렴하는 것에 대한 우려가 있었다. 

![초기 Local Optima에 대한 우려 ](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/4065593f-e754-40ad-b724-aaf834707476/Untitled.png)

초기 Local Optima에 대한 우려 

위와 같은 경우는 Local Optima가 매우 많이 존재할 것이고 Training Algorithm이 Global Minimum이 아닌 Local Optima에 수렴하는 경우 쉽게 발생할 것이다. 

위와 같이 Graph를 그리게 되면 다수의 Local Optima를 쉽게 만들 수 있다는 것을 발견할 수 있지만 실제로 이 Graph는 올바르지 않다. 

NN Network를 새로 만들게 되면 기울기가 0인 지점에서 항상 Local인 것은 아니고 Cost Function에서 기울기가 0인 대부분의 지점들은 Saddle Point이다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/a1cde5cc-2966-43a3-9eb2-456d1fceeae7/Untitled.png)

즉, **대부분의 경우에는 위 Graph처럼 Saddle Point가 되고 실제 Local Minimum이 되는 확률은 매우 낮다.** 

따라서 **Local Optima는 크게 문제 되는 부분이 아니다.** 

문제는 Plateaus이다. 이는 Training 속도를 저하시킬 수 있다. Plateaus 함수의 기울기 값이 0에 근접한 범위를 뜻하며 아래 그림과 같은 Graph를 의미한다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b8076850-8628-41ea-9b5e-a3111230d0da/e4f62b5a-b830-45e5-abc1-76cd86660297/Untitled.png)

기울기가 0이거나 0에 근접하기 때문에 표면이 매우 완만해서 Plateaus 구간을 빠져나오는데 많은 시간이 소요될 수 있다. 

1. 비교적 큰 NN에서 학습하는 이상 Local Optima 문제가 발생할 확률은 매우 낮다. 
2. Plateaus가 문제되어 Training의 속도를 저하시킬 수 있는데, 이런 문제는 Momentum, RMSprop 또는 Adam Algorithm과 같은 Algorithm이 Plateaus를 빠져나오는 속도를 높여줄 수 있다.