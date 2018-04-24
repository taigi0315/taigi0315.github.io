---
layout: post
title: "Build Logistic Classifier"
date: 2018-04-10
excerpt: "Implementing Logistic classifier using python, test model on Amazon product review."
tags: [Machine Learning, Python, Logistic, classifier, Amazon]
comments: true
feature: "../assets/img/posts/logistic-classifier/amazon.jpg"
---

<span style="color:#5B2C6F; font-weight: bold; font-family: Georgia; font-size:2em"> Implementing Logistic Classifier &<br />Test on Amazon Product Review data set</span><br />
## This post will cover
* How to implement `Logistic Classifier` using python.
* Test my model on `Amazon Baby Product Review` Data set. 
(Model will classify the sentiment of review, if it is positive, or negative.)
* Apply `L2 penalty` on Logistic model.
* Compare the coefficient&accuracy of models with different L2 penalties.
* Visualize the result.

---
## Load data & libraries
```python
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from math import sqrt
import json
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
# pd.read_csv intelligently converts input to python datatypes.
products = pd.read_csv("amazon_baby_subset.csv")
products = products.astype(str)
print ('Shape : ', products.shape)

    Shape :  (53072, 4)
```

--- 

## Data preprocessing

```python
# Change format of feature
products['rating'] = products['rating'].astype(int)
products['sentiment'] = products['sentiment'].astype(int)
# fill in N/A's in the review column
products = products.fillna({'reveiw':''}) 
```


```python
# Write a function remove_punctuation that takes a line of text and removes all punctuation from that text
def remove_punctuation(text):
    import string
    return text.translate(string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)
products.head()
```

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>sentiment</th>
      <th>review_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>1</td>
      <td>All of my kids have cried non-stop when I trie...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>5</td>
      <td>1</td>
      <td>We wanted to get something to keep track of ou...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lamaze Peekaboo, I Love You</td>
      <td>One of baby's first and favorite books, and it...</td>
      <td>4</td>
      <td>1</td>
      <td>One of baby's first and favorite books, and it...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SoftPlay Peek-A-Boo Where's Elmo A Children's ...</td>
      <td>Very cute interactive book! My son loves this ...</td>
      <td>5</td>
      <td>1</td>
      <td>Very cute interactive book! My son loves this ...</td>
    </tr>
  </tbody>
</table>
</div>

---

### Instead of using all sets of words in entire review, we are going to classify the sentiment of review using this important words(193 words) only.
```python
# read "important_words.json" file
with open('important_words.json') as data_file:
    important_words = json.load(data_file)

print('Number of importatnt words : ', len(important_words))

    Number of importatnt words :  193
```

<b>Bag of Words</b><br />
Turn `John likes to watch movies. Mary likes movies too.` to <br />
`{"John":1,"likes":2,"to":1,"watch":1,"movies":2,"Mary":1,"too":1};`
```python
# `Bag of Words` in 2 line! (Python *.*)
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))
```

```python
# Split train/valid data set
train_data = products.sample(frac=0.8)
validation_data = products.drop(train_data.index)
```

```python
# Input(X) : 1(bias) + 193(word) columns, value means how many `word` in the review
# Output(y) : Sentiment(0: Negative, 1:Positive)
def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    features_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return(features_matrix, label_array)
```


```python
feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment')
print ('Input feature(X) : ', feature_matrix_train.shape, 'Output(y) : ', sentiment_train.shape)

    Input feature(X) :  (42458, 194) Output(y) :  (42458,)
```

---

## Building a Logistic Classifier Model.
Cost function of logistric classifier with l2 penalty(red).<br />
$$\ell\ell(\mathbf{w}) = \sum_{i=1}^N \Big( (\mathbf{1}[y_i = +1] - 1)\mathbf{w}^T h(\mathbf{x}_i) - \ln\left(1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))\right) \Big) \color{red}{-\lambda\|\mathbf{w}\|_2^2} $$

```python
class logistic_classifier():
    def __init__(self):
        self.coefficients = np.zeros(1)
        self.l2_penalty = 0;
        self.iteration = 501
        self.learning_rate = 0
    
    def predict_probability(self, feature_matrix):
        #Take dot product of feature_matrix and coefficients
        score = np.dot(feature_matrix, self.coefficients)
        #Compute P(y_i = +1|x_i, w) using the link function
        predictions = 1.0/(1 + np.exp(-score))
        return predictions
    
    #Compute derivative of log likelihood with respect to a single coefficient
    def feature_derivative_with_L2(self, errors, feature, coefficient, feature_is_constant):
        #Compute the dot product of errors and feature(without L2 penalty)
        derivative = np.dot(errors, feature)
        
        #add L2 penalty term for any feature that isn't the intercept
        if not feature_is_constant:
            derivative -= 2 * self.l2_penalty * coefficient
        return derivative
    
    def compute_log_likelihood_with_L2(self, feature_matrix, sentiment):
        indicator = (sentiment == +1)
        scores = np.dot(feature_matrix, self.coefficients)
        lp = np.sum((indicator-1) * scores - np.log(1. + np.exp(-scores))) - self.l2_penalty*np.sum(self.coefficients[1:]**2)
        return lp
    
    def fit(self, feature_matrix, sentiment, learning_rate, l2_penalty, iteration):
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.iteration = iteration
        self.coefficients = np.zeros(feature_matrix.shape[1])
        print (self.l2_penalty, self.iteration, self.learning_rate)
        for itr in range(iteration):
            #Predict P(y_i = +1|x_1,w) using your predict_probability() function
            predictions = self.predict_probability(feature_matrix)

            #compute indicator value for (y_i = +1)
            indicator = (sentiment==+1)

            #Compute the errors as indicator - predictions
            errors = indicator - predictions
        
            for j in range(len(self.coefficients)): #loop over each coefficient
                is_intercept = (j==0)
                #Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
                #compute the derivative for coefficients[j]. Save it in a variable called derivative
                derivative = self.feature_derivative_with_L2(errors, feature_matrix[:,j], self.coefficients[j], is_intercept)
                #add step size times the derivative to the current coefficient(l2_penalty is already added)
                self.coefficients[j] += learning_rate * derivative

            #Checking whether log likelihood is increasing
            if (itr <= 100 and itr %10 ==0) or \
                (itr <= 1000 and itr %100 ==0) or (itr <= 10000 and itr %1000 ==0) or itr % 10000 ==0:
                    lp = self.compute_log_likelihood_with_L2(feature_matrix, sentiment)
                    print ('iteration %*d : log likelihood of observed labels = %.8f' % \
                    (int(np.ceil(np.log10(iteration ))), itr, lp))
       
    def get_accuracy(self, feature_matrix, sentiment):
        #compute scores using feature_matrix, coefficients
        scores = np.dot(feature_matrix, self.coefficients)
        #threshold scores by 0
        positive = scores > 0
        negative = scores <= 0
        scores[positive] = 1
        scores[negative] = -1

        correct = float((scores == sentiment).sum())
        total = float(len(sentiment))
        accuracy = float(correct / total)
        return accuracy
```

---

## Test model on data with different L2 penalty

```python
learning_rate = 5e-6
iteration = 501

l2_penalty = 0
model_0_penalty = logistic_classifier()
model_0_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)

    0 501 5e-06
    iteration   0 : log likelihood of observed labels = -29287.06390223
    iteration  10 : log likelihood of observed labels = -28088.62554003
        .
        .
        .
    iteration 400 : log likelihood of observed labels = -21260.57132828
    iteration 500 : log likelihood of observed labels = -20977.57766506
```

```python
l2_penalty = 5
model_5_penalty = logistic_classifier()
model_5_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```


```python
l2_penalty = 10
model_10_penalty = logistic_classifier()
model_10_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```


```python
l2_penalty = 1e2
model_1e2_penalty = logistic_classifier()
model_1e2_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```


```python
l2_penalty = 1e3
model_1e3_penalty = logistic_classifier()
model_1e3_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```


```python
l2_penalty = 1e5
model_1e5_penalty = logistic_classifier()
model_1e5_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```

## Compare / Visualize the results

```python
#but we gonna use this DataFrame
table = pd.DataFrame({'word': important_words, 
                      'l2_penalty_0': model_0_penalty.coefficients[1:],
                      'l2_penalty_5': model_5_penalty.coefficients[1:],
                      'l2_penalty_10': model_10_penalty.coefficients[1:],
                      'l2_penalty_1e2': model_1e2_penalty.coefficients[1:],
                      'l2_penalty_1e3': model_1e3_penalty.coefficients[1:],
                      'l2_penalty_1e5': model_1e5_penalty.coefficients[1:]})
```

### Word with big coefficient means it is 'Positive' or 'Negative' word.
```python
table = table.sort_values(['l2_penalty_0'], ascending=[0])
table = table[['word', 'l2_penalty_0', 'l2_penalty_5', 'l2_penalty_10', 'l2_penalty_1e2', 'l2_penalty_1e3', 'l2_penalty_1e5']]
positive_words = table[1:6]['word']
negative_words = table[-6:-1]['word']
print ('Positive words : \n', positive_words)
print ('Negative words : \n', negative_words)
```

    Positive words : 
     3        love
    7        easy
    2       great
    33    perfect
    82      happy
    Name: word, dtype: object
    Negative words : 
     99          thought
    96            money
    168        returned
    105    disappointed
    113          return
    Name: word, dtype: object
table.head()

### We can find the coefficient is getting smaller with larger penalty

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>l2_penalty_0</th>
      <th>l2_penalty_5</th>
      <th>l2_penalty_10</th>
      <th>l2_penalty_1e2</th>
      <th>l2_penalty_1e3</th>
      <th>l2_penalty_1e5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>loves</td>
      <td>1.079773</td>
      <td>1.068865</td>
      <td>1.058152</td>
      <td>0.894294</td>
      <td>0.006031</td>
      <td>0.006031</td>
    </tr>
    <tr>
      <th>3</th>
      <td>love</td>
      <td>1.073764</td>
      <td>1.064262</td>
      <td>1.054938</td>
      <td>0.913068</td>
      <td>0.008937</td>
      <td>0.008937</td>
    </tr>
    <tr>
      <th>7</th>
      <td>easy</td>
      <td>1.032797</td>
      <td>1.023669</td>
      <td>1.014713</td>
      <td>0.878385</td>
      <td>0.008436</td>
      <td>0.008436</td>
    </tr>
    <tr>
      <th>2</th>
      <td>great</td>
      <td>0.765449</td>
      <td>0.759323</td>
      <td>0.753307</td>
      <td>0.661158</td>
      <td>0.006930</td>
      <td>0.006930</td>
    </tr>
    <tr>
      <th>33</th>
      <td>perfect</td>
      <td>0.714365</td>
      <td>0.706664</td>
      <td>0.699097</td>
      <td>0.582801</td>
      <td>0.003026</td>
      <td>0.003026</td>
    </tr>
  </tbody>
</table>
</div>

---

```python
plt.rcParams['figure.figsize'] = 10, 6

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table[table['word'].isin(positive_words)]
    table_negative_words = table[table['word'].isin(negative_words)]
    del table_positive_words['word']
    del table_negative_words['word']
    
    for i in range(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].as_matrix().flatten(),
                 '-', label=positive_words.iloc[i], linewidth=4.0, color=color)
        
    for i in range(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].as_matrix().flatten(),
                 '-', label=negative_words.iloc[i], linewidth=4.0, color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()


make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=[0, 5, 10, 1e2, 1e3, 1e5])
```

<img src="../assets/img/posts/logistic-classifier/Logistic%20Classifier%20Scratch%20Implementation_23_0.png" width="90%">

```python
train_accuracy = {}
train_accuracy[0] = model_0_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[5] = model_5_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[10] = model_10_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[1e2] = model_1e2_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[1e3] = model_1e3_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[1e5] = model_1e5_penalty.get_accuracy(feature_matrix_train, sentiment_train)
print (train_accuracy)

    {0: 0.7699844552263413, 5: 0.7699844552263413, 10: 0.7699137971642565, 
        100.0: 0.7690894531065995, 1000.0: 0.7244571105563145, 100000.0: 0.7244571105563145}
```


```python
validation_accuracy = {}
validation_accuracy[0] = model_0_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[5] = model_5_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[10] = model_10_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[1e2] = model_1e2_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[1e3] = model_1e3_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[1e5] = model_1e5_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
print (validation_accuracy)

    {0: 0.7658752590917656, 5: 0.7654983983418127, 10: 0.7654041831543245, 
        100.0: 0.7634256642170718, 1000.0: 0.7218767665347654, 100000.0: 0.7218767665347654}
```
### Plot accuracy on training and validation sets over choice of L2 penalty.

```python
plt.rcParams['figure.figsize'] = 10, 6

sorted_list = sorted(train_accuracy.items(), key=lambda x:x[0])
plt.plot([p[0] for p in sorted_list], [p[1] for p in sorted_list], 'bo-', linewidth=4, label='Training accuracy')
sorted_list = sorted(validation_accuracy.items(), key=lambda x:x[0])
plt.plot([p[0] for p in sorted_list], [p[1] for p in sorted_list], 'ro-', linewidth=4, label='Validation accuracy')
plt.xscale('symlog')
plt.axis([0, 1e5, 0.70, 0.78])
plt.legend(loc='lower left')
plt.rcParams.update({'font.size': 18})
plt.tight_layout
```

<img src="../assets/img/posts/logistic-classifier/Logistic%20Classifier%20Scratch%20Implementation_27_1.png" width="90%">

## Conclusion..
Logistic classifier was bit tricky because of `likelihood`.<br />
(To me, It seems like backward! (which it is) tricky to understand)<br />
Getting the sentiment of text is pretty cool! (Now you can go ahead grab tweet, get sentiment of it)<br />
Hope you liked this post!

<img src="../assets/img/id-card.png" width="100%">