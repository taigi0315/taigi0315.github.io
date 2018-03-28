---
layout: post
title:  "Gold Price Predict"
date:   2016-04-06
excerpt: "Predict gold price using Linear Regression and virtual trading using Greedy Algorithm."
project: true
tag:
- machine learning
- R
- linear regression
feature: "../assets/img/projects/gold-price-predict/goldprice.jpg"
comments: true
---
<center><b><a href="https://github.com/taigi0315/GoldPricePrediction-Linear-Regression-_R" target="_blank"><img src="../assets/img/GitHub-Mark.png"><font size="4">Git Repo Link</font></a></b></center>
# About Project
<font size="3">
Project uses gold price data from Quandl<br />
Script grabs certain days of gold price(controlled by windows size parameter), and predicts next day gold price using linear regression algorithm<br />
Using Greedy algorithm, script decide selling the gold it is holding, or purchase gold with cash balance it has.
Script runs certain period of time, and return the result of trading.
</font>

***

# Parameters
<font size="3">
<ui>
<li>winLen = number of days that will be used for prediction</li>
<li>cashBalance = starting balance in USD</li>
<li>goldBalance = starting gold balance</li>
<li>cashRate = percentage of investment (1 = invest 100% of cash balance)</li>
<li>growthRate = threshold of price growth for decision making</li>
</ui>
</font>

***


<img src="../assets/img/projects/gold-price-predict/gold-price-table.jpg" width="95%">


<img src="../assets/img/projects/gold-price-predict/gold-price-graph.jpg" width="95%">

