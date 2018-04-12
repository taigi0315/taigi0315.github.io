---
layout: post
title: "Ridge , Lasso Regression Implementation"
date: 2018-04-10
excerpt: " Implementing Ridge/Lasso Regression from scratch"
tags: [Machine Learning, Regression, Ridge, Lasso, Python]
comments: true
feature: "../assets/img/posts/RidgeLasso/ridgelasso.png"
---
<center><b><a href="https://github.com/taigi0315/chois-ml-note/tree/master/Ridge_Lasso_Regression" target="_blank"><img src="../assets/img/GitHub-Mark.png"><font size="4">Git Repo Link</font></a></b></center>

<font size="3">
In this post, I am going to share the how I implemented Ridge/Lasso regression using python
I am going to use house price data for testing the model.
I won't talk about detail information of what Ridge/Lasso regression is, and how it work.
I have learned from <b><a href="https://www.coursera.org/learn/ml-regression" target="_blank"> Coursera ML course</a></b>, and some other pages I found from google search.<br />
Alright, Let's start ! 
</font>

---

## Load libraries & Data

```python
# Importing libraries
import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

%matplotlib inline
```

```python
# Load & Quick look the data
house_price = pd.read_csv('kc_house_data.csv')
plt.scatter(house_price['sqft_living'], house_price['price'])
house_price.head()
```
## Quick look 

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>...</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>20141013T000000</td>
      <td>221900.0</td>
      <td>3</td>
      <td>...</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>20141209T000000</td>
      <td>538000.0</td>
      <td>3</td>
      <td>...</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>20150225T000000</td>
      <td>180000.0</td>
      <td>2</td>
      <td>...</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>20141209T000000</td>
      <td>604000.0</td>
      <td>4</td>
      <td>...</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>20150218T000000</td>
      <td>510000.0</td>
      <td>3</td>
      <td>...</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>


<img src="../assets/img/posts/RidgeLasso//Ridge%20Regression%28Gradient%20Descent%29_1_1.png" width="75%">

---

## Data prep
Nothing fancy, simple data prep
* convert year-built and year-renovated to number of years since year-built and year-renovated<br />
..* ex) year-built:2010 => 8    year-renovated:2000 => 18
* Drop off some features we don't want to use.<br />

```python
# converting 'built' and 'renovated' data
house_price['age'] = 2018 - house_price['yr_built']
for index, row in house_price.iterrows():
    if house_price['yr_renovated'][index] == 0:
        house_price.loc[index, 'age_renovated'] = house_price.loc[index, 'age']
    else:
        house_price.loc[index, 'age_renovated'] = 2018 - house_price.loc[index, 'yr_renovated']
```


```python
# Dropping the features we don't need
drop_fields = ['id', 'date', 'long', 'sqft_living15', 'sqft_lot15', 'zipcode', 'lat', 'long', 'grade', 'view', 'yr_built', 'yr_renovated']
house_price = house_price.drop(drop_fields, axis=1)
```


```python
# Check features & data shape after processing.
print ('Shape of data: ' , house_price.shape)
print ('List of features: ', *house_price.columns.values, sep='\n')
```

    Shape of data:  (21613, 12)
    List of features: 
    price
    bedrooms
    bathrooms
    sqft_living
    sqft_lot
    floors
    waterfront
    condition
    sqft_above
    sqft_basement
    age
    age_renovated

## Correlation coefficient
A correlation coefficient is a way to put a value to the relationship. Correlation coefficients have a value of between -1 and 1. A “0” means there is no relationship between the variables at all, while -1 or 1 means that there is a perfect negative or positive correlation (negative or positive correlation here refers to the type of graph the relationship will produce). <a href="http://www.statisticshowto.com/what-is-correlation/" target="_blank"> Check more </a>


```python
# correlation with price feature
print((house_price.corr()['price']).sort_values(ascending=False))
```

    price            1.000000
    sqft_living      0.702035
    sqft_above       0.605567
    bathrooms        0.525138
    sqft_basement    0.323816
    bedrooms         0.308350
    waterfront       0.266369
    floors           0.256794
    sqft_lot         0.089661
    condition        0.036362
    age             -0.054012
    age_renovated   -0.105755
    Name: price, dtype: float64

---
## Splitting data 
We need 'Train', 'Validation', 'Test' dataset
<u>Train</u> the model with <b>Train</b> data<br />
Find the best <u>hyper parameter</u> with <b>Valid</b> data<br />
Measure the model <u>accuracy </u>result of model with <b>Test</b> data

```python
# split the data set
[train, test] = train_test_split(house_price, test_size= 0.2)
[train, valid] = train_test_split(train, test_size= 0.2)
print('Train :', train.shape, '\nValid: ', valid.shape, '\nTest :', test.shape)
```

    Train : (13832, 12) 
    Valid:  (3458, 12) 
    Test : (4323, 12)



```python
# Splitting data
train_X = train.loc[:, train.columns != 'price']
train_y = train['price']
valid_X = valid.loc[:, valid.columns != 'price']
valid_y = valid['price']
test_X = test.loc[:, test.columns != 'price']
test_y = test['price']
print('X: ', train_X.shape, '\ny: ', train_y.shape)
```

    X:  (13832, 11) 
    y:  (13832,)

---

## Building Ridge Regression Model


```python
class RidgeRegression(object):
    def __init__(self, learning_rate=1e-5, l2_penalty=1e-1, verbose=False, iteration=1e3):
        self.weights = None
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.cost_history = []
        
    def predict(self, X):
        y_pred= np.dot(X, self.weights)
        return(y_pred)

    def calculate_cost(self, y, y_pred):
        cost = np.sum((y - y_pred)**2) + self.l2_penalty*np.sum(self.weights ** 2)
        return cost

    def fit(self, X, y, learning_rate, l2_penalty, iteration, verbose):
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.iteration = iteration
        # Case : 1 feature input data
        if len(X.shape) == 1: 
            self.weights = 0
        else: 
            self.weights = np.zeros(X.shape[1])
        for iter in range(int(iteration)):
            y_pred = self.predict(X)
            error = y - y_pred
            # store cost history for printing
            cost = self.calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            # weight update
            self.weights += learning_rate*((np.dot(X.T, error) - l2_penalty*self.weights))
            # print progressing
            if verbose == True:
                sys.stdout.write("\rProgress: {:2.1f}".format(100 * iter/float(iteration)) \
                                    + "% ... Cost: " + str(cost))
                sys.stdout.flush()
            
    def l2_penalty_tuning(self, train_X, train_y, valid_X, valid_y, l2_penalty):
        # uses self.iteration, self.learning.
        lowest_cost = None
        best_l2_penalty = None
        print("Tuning Penalty...")
        for index, penalty in enumerate(l2_penalty_values):
            # train the model with training data
            self.fit(train_X, train_y, l2_penalty = penalty, learning_rate=self.learning_rate, iteration=self.iteration, verbose=False)
            # calculate the cost with valid data 
            y_pred = self.predict(valid_X)
            cost = np.sum((valid_y - y_pred)**2)
            if (best_l2_penalty == None or cost < lowest_cost):
                lowest_cost = cost
                best_l2_penalty = penalty
            print("[%d/%d] Penalty: %.5f    Cost: %.5e" %(index, len(l2_penalty), penalty, cost))
        print ("----------------")
        return [lowest_cost, best_l2_penalty]
    
    def r2_score(self, X, y):
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SSTO = np.sum((y - y.mean()) ** 2)
        return (1 - (SSE / float(SSTO)))
```

---

## Basic model training

```python
ridge_model = RidgeRegression()
ridge_model.fit(train_X, train_y, learning_rate=3e-14, l2_penalty=10, iteration=1e3, verbose=True)

    Progress: 99.9% ... Cost: 9.76122384115e+14

print (ridge_model.weights)

    [  1.96330456e-01   1.38999246e-01   1.48281001e+02  -9.43053455e-02
       8.90600051e-02   3.51312203e-03   1.97854058e-01   1.18154919e+02
       3.01260826e+01   3.11042156e+00   2.81224194e+00]

plt.plot(ridge_model.cost_history, label="Training cost")
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost', fontsize=14)

plt.scatter(train_X['sqft_living'], train_y)
plt.scatter(train_X['sqft_living'], ridge_model.predict(train_X))
plt.xlabel('sqft_living', fontsize=14)
plt.ylabel('price', fontsize=14)
```
<figure class="half">
<img src="../assets/img/posts/RidgeLasso/Ridge%20Regression%28Gradient%20Descent%29_12_1.png" width="75%">
<img src="../assets/img/posts/RidgeLasso/Ridge%20Regression%28Gradient%20Descent%29_13_1.png" width="75%">
</figure>

## finding best parameter

```python
print ("R2_score : ", ridge_model.r2_score(train_X, train_y))

    R2_score :  0.4813598252417063

l2_penalty_values = np.logspace(-4, 4, num=5)
[lowest_cost, best_penalty] = ridge_model.l2_penalty_tuning(train_X, train_y, valid_X, valid_y, l2_penalty = l2_penalty_values)
print("Best Penalty : %.5f   Cost : %.5e " %(best_penalty, lowest_cost))

    Tuning Penalty...
    [0/5] Penalty: 0.00010    Cost: 2.59980e+14
    [1/5] Penalty: 0.01000    Cost: 2.59980e+14
    [2/5] Penalty: 1.00000    Cost: 2.59980e+14
    [3/5] Penalty: 100.00000    Cost: 2.59980e+14
    [4/5] Penalty: 10000.00000    Cost: 2.59980e+14
    ----------------
    Best Penalty : 0.00010   Cost : 2.59980e+14 
```

## Model with best L2 penalty


```python
best_model = RidgeRegression()
best_model.fit(train_X, train_y, l2_penalty=best_penalty, learning_rate=3.5e-14, iteration=5e3, verbose=True)

    Progress: 100.0% ... Cost: 9.29356215433e+14

plt.scatter(test_X['sqft_living'], test_y)
plt.scatter(test_X['sqft_living'], best_model.predict(test_X))
plt.xlabel('sqft_living', fontsize=12)
plt.ylabel('price', fontsize=12)
model.r2_score(test_X, test_y)

    0.47562133488271474
```
<img src="../assets/img/posts/RidgeLasso/Ridge%20Regression%28Gradient%20Descent%29_17_1.png" width="75%">


# LASSO regression coordinate descent

## Normalize features
In the house dataset, features vary wildly in their relative magnitude: `sqft_living` is very large overall compared to `bedrooms`, for instance. As a result, weight for `sqft_living` would be much smaller than weight for `bedrooms`. This is problematic because "small" weights are dropped first as `l1_penalty` goes up. 

To give equal considerations for all features, we need to **normalize features**. we divide each feature by its 2-norm so that the transformed feature has norm 1.


```python
# split the data set
[train, test] = train_test_split(house_price, test_size= 0.2)
[train, valid] = train_test_split(train, test_size= 0.2)
print('Train :', train.shape, '\nValid: ', valid.shape, '\nTest :', test.shape)

    Train : (13832, 12) 
    Valid:  (3458, 12) 
    Test : (4323, 12)
```


```python
# Splitting data into input(X) / output(y)
train_X = train.loc[:, train.columns != 'price']
train_y = train['price']
valid_X = valid.loc[:, valid.columns != 'price']
valid_y = valid['price']
test_X = test.loc[:, test.columns != 'price']
test_y = test['price']
print('X: ', train_X.shape, '\ny: ', train_y.shape)

    X:  (13832, 11) 
    y:  (13832,)
```

```python
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return(normalized_features, norms)
```

```python
# normalize data
[train_X, train_norms] = normalize_features(train_X)
print('Norms : ', *train_norms, sep='\n')
train_X.head()

    Norms : 
    410.795569596
    265.084774176
    268050.032751
    4908561.04402
    187.010694881
    10.3923048454
    407.87743257
    232777.796375
    61897.0640095
    6521.78840503
    6250.50325974
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <td>...</td>
      <th>sqft_basement</th>
      <th>age</th>
      <th>age_renovated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9202</th>
      <td>0.012172</td>
      <td>0.003772</td>
      <td>0.012087</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.001687</td>
      <td>0.001760</td>
    </tr>
    <tr>
      <th>13395</th>
      <td>0.007303</td>
      <td>0.003772</td>
      <td>0.003805</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.007820</td>
      <td>0.008159</td>
    </tr>
    <tr>
      <th>4909</th>
      <td>0.007303</td>
      <td>0.005659</td>
      <td>0.006267</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.010273</td>
      <td>0.010719</td>
    </tr>
    <tr>
      <th>7891</th>
      <td>0.012172</td>
      <td>0.010374</td>
      <td>0.008954</td>
      <td>...</td>
      <td>0.019387</td>
      <td>0.006747</td>
      <td>0.007039</td>
    </tr>
    <tr>
      <th>967</th>
      <td>0.009737</td>
      <td>0.010374</td>
      <td>0.009513</td>
      <td>...</td>
      <td>0.017933</td>
      <td>0.004907</td>
      <td>0.005120</td>
    </tr>
  </tbody>
</table>
</div>



## Implementing Coordinate Descent with normalized features
We seek to obtain a sparse set of weights by minimizing the LASSO cost function
```
SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|).
```
(By convention, we do not include `w[0]` in the L1 penalty term. We never want to push the intercept to zero.)

The absolute value sign makes the cost function non-differentiable, so simple gradient descent is not viable (you would need to implement a method called subgradient descent). Instead, we will use **coordinate descent**: at each iteration, we will fix all weights but weight `i` and find the value of weight `i` that minimizes the objective. That is, we look for
```
argmin_{w[i]} [ SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|) ]
```
where all weights other than `w[i]` are held to be constant. We will optimize one `w[i]` at a time, circling through the weights multiple times.  
  1. Pick a coordinate `i`
  2. Compute `w[i]` that minimizes the cost function `SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|)`
  3. Repeat Steps 1 and 2 for all coordinates, multiple times
 
---

```python
class LassoRegression():
    def __init__(self):
        self.weights = None
        self.l1_penalty = None
        self.iteration = None
        self.tolerance = None

    def predict(self, feature_matrix):
        predictions = np.dot(feature_matrix, self.weights)
        return(predictions)

    def lasso_coordinate_descent_step(self, i, X, y):
        # compute prediction
        prediction = self.predict(X)
        # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
        ro_i = (X.iloc[:,i] * (y - prediction + self.weights[i]*X.iloc[:,i]) ).sum()
        if i == 0: # intercept -- do not regularize
            new_weight_i = ro_i 
        elif ro_i < -self.l1_penalty/2.:
            new_weight_i = ro_i + (self.l1_penalty/2)
        elif ro_i > self.l1_penalty/2.:
            new_weight_i = ro_i - (self.l1_penalty/2)
        else:
            new_weight_i = 0.

        return new_weight_i

    def fit(self, X, y, l1_penalty, tolerance=1e-1, verbose=False):
        self.l1_penalty = l1_penalty
        self.verbose = verbose
        self.tolerance = tolerance
        self.weights = np.zeros(X.shape[1])
        
        converge = True    
        print_index = 0
        iter = 0
        while(converge):
            max_change = 0
            iter += 1
            changes = []
            for i in range(len(self.weights)):
                old_weights_i = self.weights[i]
                self.weights[i] = self.lasso_coordinate_descent_step(i, X, y)
                #print "new weight = %d" %weights[i]
                this_change = self.weights[i] - old_weights_i
                changes.append(this_change)
                max_change =  max(np.absolute(changes))
                print_index += 1
                if(verbose == True and print_index % 500 == 0):    
                    print("max change : %.3f" %(max_change) )
                            
            if (max_change < self.tolerance or iter > 1e3) :
                converge = False
            
    def r2_score(self, X, y):
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SSTO = np.sum((y - y.mean()) ** 2)
        return (1 - (SSE / float(SSTO)))
    
    def l1_penalty_tuning(self, train_X, train_y, valid_X, valid_y, l1_penalty, tolerance=10):
        lowest_cost = None
        best_l1_penalty = None
        print("Tuning Penalty...")
        for index, penalty in enumerate(l1_penalty_values):
            self.fit(train_X, train_y, l1_penalty = penalty, tolerance=tolerance, verbose=False)
            cost = sum((valid_y-self.predict(valid_X))**2)
            if (best_l1_penalty == None or cost < lowest_cost):
                lowest_cost = cost
                best_l1_penalty = penalty
            print("[%d/%d] Penalty: %.5f    Cost: %.5e" %(index, len(l1_penalty), penalty, cost))
        print ("----------------")
        return [lowest_cost, best_l1_penalty]
```

## Basic model training

```python
lasso_model = LassoRegression()
lasso_model.fit(train_X, train_y, l1_penalty=1, tolerance=10, verbose=True)

    max change : 502650.747
    max change : 187854.178
    max change : 63732.160
    max change : 21520.989
    max change : 7043.920
    max change : 2357.197
    max change : 576.635
    max change : 257.082
    max change : 7.224
    max change : 27.868

plt.scatter(train_X['sqft_living'], train_y)
plt.scatter(train_X['sqft_living'], lasso_model.predict(train_X))
plt.xlabel('sqft_living', fontsize=12)
plt.ylabel('price', fontsize=12)
```

<img src="../assets/img/posts/RidgeLasso/Ridge%20Regression%28Gradient%20Descent%29_27_1.png" width="75%">

## Denormalization / finding best parameter
```python
# since we normalized train data, we have to normalize test data as well.
normalized_valid_X = valid_X / train_norms

# since train data is normalized, normalize the result weight to use it for test data set.
l1_penalty_values = np.logspace(1, 5, num=5)
[lowest_cost, best_penalty] = lasso_model.l1_penalty_tuning(train_X, train_y, normalized_valid_X, valid_y, l1_penalty = l1_penalty_values, tolerance=1e3)

print("Best Penalty : %.3f   Cost : %.5e " %(best_penalty, lowest_cost))

    Tuning Penalty...
    [0/5] Penalty: 10.00000    Cost: 2.02350e+14
    [1/5] Penalty: 100.00000    Cost: 2.02350e+14
    [2/5] Penalty: 1000.00000    Cost: 2.02347e+14
    [3/5] Penalty: 10000.00000    Cost: 2.02332e+14
    [4/5] Penalty: 100000.00000    Cost: 2.02206e+14
    ----------------
    Best Penalty : 100000.000   Cost : 2.02206e+14 
```

## Model with best parameter

```python
best_lasso_model = LassoRegression()
best_lasso_model.fit(train_X, train_y, l1_penalty=best_penalty, tolerance=10, verbose=True)

    max change : 521786.869
    max change : 192029.341
    max change : 53811.536
    max change : 18840.320
    max change : 11643.193
    max change : 9937.226
    max change : 3718.359
    max change : 1354.504
    max change : 223.197
    max change : 379.711
    max change : 143.864
    max change : 30.024
    max change : 32.148
    max change : 12.069

plt.scatter(test_X['sqft_living'], test_y)
plt.scatter(test_X['sqft_living'], best_lasso_model.predict(test_X/train_norms))
plt.xlabel('sqft_living', fontsize=12)
plt.ylabel('price', fontsize=12)
```

<img src="../assets/img/posts/RidgeLasso/Ridge%20Regression%28Gradient%20Descent%29_31_1.png" width="75%">

## Conclusion..
I found ridge regression model has best fit when the penalty is really low (as zero).
I double checked my implementation, but couldn't find the problem.
if you can give me any advice for this, please leave me comment or mail it to me :)
I think I should add a part that showing the weight / penalty plot to show impact of penalty, but I was too lazy..
I believe you can figure out the differece once you understand both and implement it from scratch!
Hope you liked this post!

<img src="../assets/img/id-card.png" width="100%">


