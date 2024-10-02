# Arabic Digit Classification using Machine Learning Techniques

This project focuses on implementing and comparing various **machine learning algorithms** and **feature extraction techniques** to recognize handwritten Arabic digits. The goal is to explore how different classifiers and feature extraction methods affect the recognition accuracy, especially in challenging scenarios such as noisy data.

## Table of Contents
- [Introduction](#introduction)
- [Machine Learning Techniques](#machine-learning-techniques)
- [Feature Extraction Methods](#feature-extraction-methods)
- [Robustness to Noise](#robustness-to-noise)
- [Results and Analysis](#results-and-analysis)


## Introduction

Handwritten digit classification is a critical task in the field of machine learning, with applications ranging from **optical character recognition** to **automated systems** for data entry. This project compares several **classification algorithms** and **feature extraction techniques** to determine the best approach for recognizing handwritten Arabic digits. The evaluation includes tests under both normal and noisy conditions to assess model robustness.

## Machine Learning Techniques

We implemented and compared multiple **machine learning classifiers** to evaluate their performance on Arabic digit classification:

1. **k-Nearest Neighbors (k-NN)**: 
   A simple yet effective classifier where classification is based on the majority label among the k-nearest neighbors. We optimized the value of k using a validation set.
   
2. **Bayesian Classifier**: 
   A probabilistic model that assigns labels based on the likelihood of the input features under each class, considering both prior and likelihood probabilities.

3. **Decision Tree**: 
   A non-parametric model that splits the dataset into subsets based on feature values, creating a tree-like decision structure. This method is intuitive but prone to overfitting without proper tuning.

4. **Random Forest**: 
   An ensemble learning method that improves decision tree performance by training multiple trees on random subsets of the data and features, then aggregating their predictions.

## Feature Extraction Methods

Effective feature extraction is crucial for improving the performance of machine learning algorithms. Two primary techniques were explored:

1. **Zoning**: 
   The image is divided into a grid (9x9), and the average pixel values in each zone are computed. This produces a feature vector of 81 elements, capturing regional patterns in the image.

2. **Horizontal & Vertical Histograms**: 
   This method computes the distribution of pixel intensities across the horizontal and vertical axes, creating a feature vector of 99 elements. It captures the overall shape and structure of the digits.

## Robustness to Noise

In practical applications, images are often subject to noise. To evaluate how the classifiers handle such scenarios, **salt and pepper noise** was introduced to the validation set. We then tested the classifiers on this noisy data to gauge their robustness. The models were reevaluated to find the optimal parameters (such as **k** in k-NN) for this noisy setting.

## Results and Analysis

The performance of the classifiers and feature extraction methods was evaluated on both clean and noisy datasets. Key findings include:

- **Zoning** was the more effective feature extraction method, consistently outperforming the **Histogram** method.
- **Random Forest** was the most accurate classifier, achieving **86.30% accuracy** with clean data when paired with the Zoning method.
- **k-NN** and **Decision Trees** also showed strong performance, but they were sensitive to the choice of hyperparameters.
- On noisy data, performance degraded significantly, with the best accuracy dropping to **9.75%**, indicating a need for more advanced noise-handling techniques or robust feature engineering.

| Classifier        | Zoning Accuracy | Histogram Accuracy |
|-------------------|-----------------|--------------------|
| 1-NN              | 73.60%          | 37.60%             |
| k-NN              | 80.20%          | 45.80%             |
| Bayes             | 66.00%          | 42.50%             |
| Decision Tree     | 71.20%          | 41.00%             |
| Random Forest     | 86.30%          | 51.50%             |

On noisy data, the accuracy dropped significantly:

| Classifier        | Zoning Accuracy | Histogram Accuracy |
|-------------------|-----------------|--------------------|
| 1-NN              | 8.70%           | 8.70%              |
| k-NN              | 9.70%           | 9.30%              |
| Bayes             | 8.70%           | 10.80%             |
| Decision Tree     | 9.70%           | 9.70%              |
| Random Forest     | 10.80%          | 8.70%              |
