# Face Recognition Using Various Feature Extraction and Classification Techniques

This project presents a comprehensive face recognition system utilizing advanced **feature extraction methods** and a diverse set of **machine learning classifiers**. The system is built using the **ORL Face Database** and investigates the impact of dimensionality reduction through **Principal Component Analysis (PCA)**, along with different classification techniques to enhance recognition accuracy.

## Table of Contents
- [Introduction](#introduction)
- [Machine Learning Techniques](#machine-learning-techniques)
- [Feature Extraction Methods](#feature-extraction-methods)
- [Impact of PCA and Sample Size](#impact-of-pca-and-sample-size)
- [Robustness to Noise](#robustness-to-noise)
- [Results and Analysis](#results-and-analysis)

## Introduction

Face recognition is widely used in applications such as **biometrics**, **security systems**, and **human-computer interaction**. This project explores different combinations of **classification algorithms** and **feature extraction techniques** to optimize face recognition accuracy using the **ORL Face Database**. The experiments also focus on the trade-offs between computational efficiency and recognition performance, with an emphasis on scalability and robustness.

## Machine Learning Techniques

This project compares multiple **machine learning classifiers**, each with unique characteristics:

1. **k-Nearest Neighbors (k-NN)**:
   - A simple, yet powerful classification method based on proximity in feature space. Different values of k are tested to optimize performance.
   
2. **Bayesian Classifier**:
   - A probabilistic classifier that utilizes the likelihood of features given a class to predict the most probable class based on **Bayesian statistics**.
   
3. **Decision Tree**:
   - A rule-based model that creates a tree-like structure to make classification decisions. While intuitive, it can overfit without proper tuning.

4. **Random Forest**:
   - An ensemble method that improves upon Decision Trees by using a collection of them to enhance accuracy and reduce overfitting through **bootstrap aggregating (bagging)**.

5. **Parzen Window**:
   - A non-parametric classifier that estimates probability density functions to make predictions based on feature distributions.

## Feature Extraction Methods

Feature extraction is critical to improve classification accuracy. Two main techniques were employed:

1. **Mutual Information**:
   - A statistical measure that quantifies the amount of information one variable (feature) provides about another. Features with high mutual information scores are deemed more relevant for classification.

2. **Pearson Correlation**:
   - Measures the linear relationship between features and class labels. Highly correlated features are selected for classification.

**PCA** was also used for **dimensionality reduction**, reducing the number of features while retaining the most variance from the data.

## Impact of PCA and Sample Size

To optimize feature representation, PCA was employed with varying numbers of components (from 30 to 70). Key findings include:
- As the number of PCA components increases, classification time increases but accuracy may decrease, especially with excessive components.
- The effect of different training sample sizes was also evaluated, showing that more training data significantly improves accuracy, particularly for k-NN and Parzen Window classifiers.

## Robustness to Noise

The system’s robustness was tested by introducing **salt and pepper noise** to 10% of the test data. Results show that all classifiers experienced a notable drop in accuracy with noisy data, highlighting the sensitivity of the models to noise.

## Results and Analysis

| Feature Extractor / Classifier | Mutual Information Accuracy | Pearson Correlation Accuracy |
|-------------------------------|-----------------------------|------------------------------|
| Self-Written k-NN              | 82.50%                      | 59.50%                       |
| Python k-NN                    | 82.50%                      | 59.50%                       |
| Parzen Window                  | 82.50%                      | 62.00%                       |
| Self-Written Bayes             | 69.50%                      | 51.00%                       |
| Decision Tree                  | 43.00%                      | 27.00%                       |
| Random Forest                  | 78.50%                      | 56.00%                       |

Overall, **Mutual Information** proved to be the better feature extraction method, and **k-NN** and **Parzen Window** were the best performing classifiers, each achieving **82.50% accuracy** with **Mutual Information** features.

### Performance Under Noisy Data:

The introduction of noise drastically reduced accuracy for all classifiers. The k-NN classifier dropped to **4.00%**, and the overall average accuracy decreased significantly across all models.
