
# How does ML work? 머신 러닝의 학습 원리
### what is the basic idea of machine learning / 머신 러닝의 기본적인 학습 원리
--- 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```


```python
# load, and quick check data
house_price = pd.read_csv('data/house_price_train.csv')
house_price = house_price[['LotArea', 'SalePrice']]

plt.plot(house_price['LotArea'], house_price['SalePrice'], '.')
```




    [<matplotlib.lines.Line2D at 0x29671f039b0>]




![png](/img/posts/How%20does%20ML%20work_files/How%20does%20ML%20work_2_1.png)



```python
# To make it easier to understand, I pull out 15 random houses,
# we will use these house only for this post
sample_data = house_price.sample(15, random_state=3)
plt.plot(sample_data['LotArea'], sample_data['SalePrice'], '.')
plt.xlabel('Area', fontsize=12, color='blue')
plt.ylabel('Price', fontsize=12, color='blue')
```




    <matplotlib.text.Text at 0x296722e4128>




![png](/img/posts/How%20does%20ML%20work_files/How%20does%20ML%20work_3_1.png)



```python
# Simple function to draw lien
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
```

## 3 lines represent 3 different prediction models
as we can see, green is the best fit line out of 3. we can tell green line fits best by instict, but how did we know? why not red line or even yellow?
## 아래 세가지 색깔의 선은 서로 다른 예측 모델을 의미합니다.
우리는 본능적으로 초록선이 자료를 예측하는데 가장 적합하다는 것을 알 수 있습니다. 하지만, 우리가 '본능적'으로 생각하는 사이 우리는 어떠한 과정을 통해 이러한 결론을 낼 수 있었을까요?


```python
# Check out our naive prediction line
plt.plot(sample_data['LotArea'], sample_data['SalePrice'], '.', markersize=8, label='_nolegend_')
plt.xlabel('Area', fontsize=12, color='blue')
plt.ylabel('Price', fontsize=12, color='blue')
graph('5*x', range(0, 20000))
graph('20*x', range(0, 20000))
graph('30*x', range(0, 20000))
plt.legend(['5', '20', '30'])
```




    <matplotlib.legend.Legend at 0x29672beec50>




![png](/img/posts/How%20does%20ML%20work_files/How%20does%20ML%20work_6_1.png)


---
# 머신러닝의 원리
만약 우리가 'W' 값을 알고있다면, 우리는 각각의 'x'값에 대한 'y' 값을 예측할 수 있습니다.
문제는 곱셈의 역함수가 없다는 것인데요.(곱셈의 역함수를 구해내는 것은 무척이나 까다로워 쓸 수 없습니다.) 
이 문제를 해결하기 위해 우리는 좌측의 'y'값을 우측으로 넘긴 후 'W' 값을 바꿔가며 가장 최소의 결과('Error')를 얻는 'W'값을 찾아내는 것입니다.

---
# What we are trying to do in Machine Learning
If we know the 'W' we can predict 'y' value when we have 'X' value.
problem is multiplication does not have inverse, so we move y value to the right side
and keep the left side as 0. Now we find 'W' that make smallest 'Error'

![Fig-1](/img/posts/How%20does%20ML%20work_files/Fig_1.png)
![Fig-2](img/posts/How%20does%20ML%20work_files/Fig_2.png)
---
## See actual example / 데이터 예제
plot shows sum of errors on 15 house prices for each 'W' from 0 to 50
as you can see error starts from around 200k, hits zero, and goes down to -200k

그래프에서 볼 수 있듯, 에러는 약 20만에서 시작해 0을 찍고, 약 -20만까지 떨어진다.


```python
# Check the error based on W
plt.xlabel('W', fontsize=12)
plt.ylabel('Error', fontsize=12)

for w in range(0, 50):
    predict = w * sample_data['LotArea']
    error = np.mean(sample_data['SalePrice'] - predict)
    plt.scatter(w, error)
```


![png](/img/posts/How%20does%20ML%20work_files/How%20does%20ML%20work_11_0.png)


## Negative Error? / 에러가 음수?
Negative error does not make sense, it happens when our predict house price was bigger than actual price. to solve this problem, simply we square the error, aka MSE(Mean Squared Error)

음수 에러는 말이 안되죠? 이 음수 에러는 단순히 우리의 예상 가격이 실제 가격보다 큰 경우 발생하는 현상입니다. 이러한 현상을 없애기 위해 보통 머신 러닝에서는 에러를 제곱하여 계산하며 이걸 MSE(Meas Squared Error/제곱한 에러의 평균) 이라고 부릅니다. 


```python
# Check the squared error based on W
plt.xlabel('W', fontsize=12)
plt.ylabel('Error', fontsize=12)

for w in range(0, 50):
    predict = w * sample_data['LotArea']
    error = np.mean((sample_data['SalePrice'] - predict)**2)
    plt.scatter(w, error)
```


![png](/img/posts/How%20does%20ML%20work_files/How%20does%20ML%20work_13_0.png)

