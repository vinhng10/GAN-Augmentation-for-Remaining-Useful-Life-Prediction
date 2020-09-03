# GAN Augmentation for Remaining Useful Life Prediction

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Introduction
This repository demonstrates my attempt to apply GAN augmentation to improve remaining useful life (RUL) prediction. This work is done as part of my bachelor thesis, [Toward Deep Learning in Remaining Useful Life Estimation](https://github.com/vinhng10/GAN-Augmentation-for-Remaining-Useful-Life-Prediction/blob/master/Thesis.pdf), at Aalto University. 
GAN aumentation has been shown to improve performance of some machine learning model by introducing more data to training. The goal of this project is to exammine if GAN augmentation can also enhance RUL prediction in predictive maintenance.

## Experiment
The Colab sample experiement can be found [here](https://colab.research.google.com/drive/1NY-4ISTnyFUqXeZmycdIn2zZEXiVXVaZ?authuser=2).

## Dataset
The dataset used in this experiment is C-MAPSS dataset from Prognostic and Health Management Challenge in 2008. The detail description and download link of the dataset can be found [here](https://github.com/makinarocks/awesome-industrial-machine-datasets/tree/master/data-explanation/PHM08%20Challenge%20on%20this%20dataset).

## Architecture
In this project, conditional GAN is used to model time series data distribution. Both of the generator and discriminator are conditioned on a target RUL sampled from uniform distribution U(0, 1). The architecture is inspired from this [paper](https://arxiv.org/pdf/1706.02633.pdf).
![](https://github.com/vinhng10/GAN-Augmentation-for-Remaining-Useful-Life-Prediction/blob/master/images/rsz_screenshot_from_2020-09-03_22-05-35.png?raw=true)

## Result
This table shows the final result of the experiment. Although the improvement has small margin and is not consistent among all 4 subdatasets of C-MAPSS dataset, it can still shows promissing direction for future research, where more sophisticated GAN architectures and training techniques can be applied.
![](https://raw.githubusercontent.com/vinhng10/GAN-Augmentation-for-Remaining-Useful-Life-Prediction/master/images/rsz_3screenshot_from_2020-09-03_21-23-06.png)
