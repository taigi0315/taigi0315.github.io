---
layout: post
title:  "Is lot empty?"
date:   2015-07-10
excerpt: "Classify if parking lot is empty using Logistic Regression model"
project: true
tag:
- machine learning
- R
- logistic regression
- cross validation
feature: "../assets/img/projects/parking-lot/lot.jpeg"
comments: true
---
<center><b><a href="https://github.com/taigi0315/Parking-Lot-Classifier" target="_blank"><img src="../assets/img/GitHub-Mark.png"><font size="4">Git Repo Link</font></a></b></center>

# About Project
<font size="3">
Project uses parking lot image data(PKlot) from Federal University of Parana<br />
Script gray scale the original image and crop specific lot area out of whole image.<br />
Logistic model is trained using 10 fold cross validation.<br />
Script returns predicted value from 0 to 1.<br />
Using threshold(0.5) we can convert predicted value to 0(empty lot) or 1(car in lot)<br />
</font>

***

<img src="../assets/img/projects/parking-lot/project_3_info1.jpg" width="95%">

<img src="../assets/img/projects/parking-lot/project_3_info2.jpg" width="95%">

