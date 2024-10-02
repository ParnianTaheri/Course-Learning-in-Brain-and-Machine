# Comparison Between Error Backpropagation and Genetic Algorithm for XOR Problem

This project compares two different techniques—**Error Backpropagation** and **Genetic Algorithm (GA)**—for solving the **XOR classification problem** using a simple **Multilayer Perceptron (MLP)** model. The goal is to evaluate the effectiveness, speed, and convergence behavior of both methods in optimizing the neural network weights.

## Table of Contents
- [Introduction](#introduction)
- [Multilayer Perceptron (MLP) Overview](#multilayer-perceptron-mlp-overview)
- [Optimization Techniques](#optimization-techniques)
  - [Error Backpropagation](#error-backpropagation)
  - [Genetic Algorithm](#genetic-algorithm)
- [Results and Analysis](#results-and-analysis)


## Introduction

The **XOR problem** is a classic problem in machine learning, known for its non-linearly separable nature, which cannot be solved with a simple linear classifier. This project addresses the XOR problem using an **MLP** classifier with a **2:2:1 architecture** and compares two optimization methods: **Error Backpropagation** and **Genetic Algorithm**. Both methods are evaluated based on their convergence speed and accuracy in training the MLP.

## Multilayer Perceptron (MLP) Overview

A **Multilayer Perceptron (MLP)** is a type of artificial neural network that consists of multiple layers of neurons:
- **Input Layer**: 2 neurons for the binary inputs of the XOR problem.
- **Hidden Layer**: 2 neurons with sigmoid activation functions.
- **Output Layer**: 1 neuron for the binary output of the XOR classification.

The task is to adjust the weights of the MLP so that it correctly classifies the XOR problem.

## Optimization Techniques

### Error Backpropagation

**Error Backpropagation** is a supervised learning algorithm used for training neural networks. The weights are updated by minimizing the **Mean Squared Error (MSE)** between the predicted and actual outputs. The algorithm works as follows:
1. Randomly initialize the weights.
2. Calculate the output error and propagate it backward through the network to adjust the weights.
3. Continue this process for a fixed number of **epochs** or until the error reaches a predefined threshold.

In this project:
- The learning rate (**η**) is used to control weight updates.
- Training stops when the error falls below **0.03** or after **1000 epochs**.

### Genetic Algorithm

A **Genetic Algorithm (GA)** is a population-based optimization technique inspired by natural selection. The GA evolves a population of candidate solutions by applying crossover and mutation operations. The steps are:
1. Initialize a population of random solutions (weights).
2. Evaluate the fitness of each individual using **1/MSE** as the fitness function.
3. Perform **crossover** and **mutation** to generate new solutions.
4. Replace the least fit individuals in the population and repeat for a fixed number of generations or until convergence.

In this project:
- Each solution (chromosome) has **9 genes**, representing the 9 weights of the MLP.
- **Crossover** and **mutation** are applied randomly to generate new offspring.

## Results and Analysis

Both **Error Backpropagation** and **Genetic Algorithm** successfully train the MLP to classify the XOR problem. However, their performance differs:

- **Error Backpropagation** converges after a larger number of iterations but shows steady and predictable progress.
- **Genetic Algorithm** converges faster with fewer iterations but depends on random population initialization and may vary in efficiency across runs.

### Key Findings:
- **Genetic Algorithm** tends to converge faster but can be computationally expensive due to population management.
- **Error Backpropagation** is slower but more stable for small-scale problems like XOR.

| Method               | Convergence Speed | Iterations to Convergence |
|----------------------|-------------------|--------------------------|
| Error Backpropagation | Moderate          | Higher                    |
| Genetic Algorithm     | Fast              | Lower                     |
