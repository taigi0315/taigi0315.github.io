---
layout: post
title:  "Predict Stock Price"
date:   2015-08-10
excerpt: "Predict Apple stock price using Neural Network"
project: true
tag:
- machine learning
- R
- neural network
- z score
feature: "../assets/img/projects/stock-price-predict/apple-stock1.jpeg"
comments: true
---

## Intro
<font size="3">
Predicting stock market price is one of the most attractive prediction problems.<br />
The fact is the stock prices are measured by needs and support which means it is decided by personal decision.<br />
This is why predicting stock prices has been and remains unsolved.
</font>
***

<figure class="half">
<img src="../assets/img/projects/stock-price-predict/wordcloud.png">
<img src="../assets/img/projects/stock-price-predict/worldstockexchanges.png">
</figure>

***

## Data
<font size="3">
Project uses variety of data that is related with target stock price(Apple Inc) in many perspectives.<br />
Project assumes that our target stock price has a connection with global economic data.<br />
Data is accumulated from Quandl, USA government, and Yahoo finance.<br />
* Index Data
* Community Data
* Currency Data
</font>

### Index Data
<ul>
<li>SwissMarket</li>
<li>STI</li>
<li>SP500 </li>
<li>Russell1000</li>
<li>NASDAQ</li>
<li>Kospi</li>
<li>CAC40</li>
<li>ATX</li>
<li>Nikkei225</li>
<li>Treasury 5 year</li>
<li>Treasury 30 year</li>
</ul>

### Commmodity Data
<ul>
<li>Gold price in USD</li>
<li>Silver price in USD</li>
<li>Oil price in USD</li>
</ul>

### Currency Data
<img src="../assets/img/projects/stock-price-predict/applesupply-2.png" width="75%">
<center><font size="2" colot="gray"> Apple Global Supply </font></center>
<ul>
<li>Europe Euro to USD</li>
<li>China Yen to USD</li>
<li>Korea Won to USD</li>
<li>Colombian Peso to USD</li>
<li>Japanese Yen to USD</li>
<li>Malaysian Ringgit to USD</li>
<li>Thai Baht to USD</li>
</ul>

### Others
<ul>
<li>Window size</li>
<li>Month</li>
<li>Other Stocks in S&P500 which has high or low Z-Score with Apple stock</li>
</ul>
<a href="https://www.youtube.com/watch?v=4VCnhPMnv-4" target="_blank" width="50%">
	<img src="../assets/img/projects/stock-price-predict/relatedStock-img.png">
</a>
<center><font size="2" colot="gray">S&P500 Relationship based on Z-Score</font></center>

***

## Model
<img src="../assets/img/projects/stock-price-predict/nn.png" width="65%">
<b>Neural Network</b>
<font size="3">
-Input : today’s feature values<br />
-Output : tomorrow’s target price in 2 classes(positive / negative)<br />
</font>
	

***

## Result
<img src="../assets/img/projects/stock-price-predict/test1.png" width="75%">
<center><font size="2" colot="gray"> Accuracy with different iteration </font></center>
<img src="../assets/img/projects/stock-price-predict/test2.png" width="75%">
<center><font size="2" colot="gray"> Accuracy with different window size </font></center>
<img src="../assets/img/projects/stock-price-predict/test3.png" width="75%">
<center><font size="2" colot="gray"> Accuracy with different size of hidden layer</font></center>

***

## Conclusion
<img src="../assets/img/projects/stock-price-predict/result.png" align="left" width="35%">

<font size="3">
&nbsp; The highest accuracy I found in multiple experiments was 76.48% <br />
&nbsp; Add ‘Month columns’ actually increase performance of program in most of experiment<br />
&nbsp; Neural network is really strong but it also needs a lot of experiments to achieve its best performance<br />
</font>
<br />
<br />