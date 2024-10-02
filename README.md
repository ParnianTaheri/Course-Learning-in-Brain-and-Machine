# Learning-in-Brain-and-Machine

This course provides a comprehensive overview of machine learning principles, feature extraction, and brain-based learning methods. It combines foundational machine learning techniques with advanced neural network architectures inspired by biological systems, offering insight into both supervised and unsupervised learning paradigms.

---

## 1. Principles of Machine Learning

### 1.1 **Supervised Learning**
Supervised learning focuses on training models using labeled data to make predictions or classifications. The key concepts include feature extraction, selection, and classification.

- **Basic Concepts**:
  - Overview of supervised learning and its applications in classification and regression tasks.

- **Feature Extractor**:
  - **Linear Approaches**:
    - **Principal Component Analysis (PCA)**: A linear technique to reduce dimensionality by transforming data to a new set of variables (principal components).
    - **Karhunen-Loeve Expansion**: A statistical method for representing data in a reduced number of dimensions.
    - **Independent Component Analysis (ICA)**: A computational method for separating a multivariate signal into additive, independent components.
    - **Factor Analysis**: A technique to uncover the latent structure in a set of observed variables.
    - **Discriminant Analysis**: A method for classifying data by finding a linear combination of features that separates classes.

  - **Nonlinear Approaches**:
    - **Kernel PCA**: An extension of PCA that uses kernel methods to perform nonlinear dimensionality reduction.
    - **Multidimensional Scaling (MDS)**: A technique that visualizes the level of similarity of individual cases in a dataset.

  - **Neural Networks**:
    - **Feed-Forward Neural Networks**: A simple type of artificial neural network where connections between nodes do not form a cycle.
    - **Self-Organizing Map (SOM)**: An unsupervised learning technique that projects high-dimensional data into lower dimensions.

- **Feature Selector**:
  - Methods and techniques to identify the most relevant features from the dataset to improve model performance.

- **Classification**:
  - Based on the **concept of similarity**:
    - **One-nearest neighbor (1-NN)**: A simple classification technique that assigns a class based on the closest data point.
  - Based on the **probabilistic approach**:
    - **k-nearest neighbor (k-NN)**: A classification technique that assigns a class based on the majority of the k-nearest points.
    - **Parzen Classifier**: A non-parametric technique that estimates the probability density function of the data.
  - Based on **optimizing certain error criteria**:
    - **Multi-Layer Perceptrons (MLPs)**: A class of feed-forward neural networks used for learning non-linear mappings.

---

### 1.2 **Unsupervised Learning**
In unsupervised learning, models are trained on data without explicit labels, focusing on finding hidden patterns and structures.

- **Clustering**: Grouping data points based on similarity.
- **K-Means Clustering**: A popular clustering algorithm that partitions data into k distinct clusters.
- **Self-Organizing Maps (SOM)**: A neural network that performs clustering by organizing the data in a topological map.

---

### 1.3 **Radial Basis Function (RBF) Networks**
- **RBF Networks**: A type of neural network that uses radial basis functions as activation functions, particularly useful in pattern recognition tasks.

---

### 1.4 **Hmax Model**
- **Hmax Model**: A computational model of the human visual cortex that simulates object recognition through hierarchical processing.

---

### 1.5 **Deep Learning**
Deep learning focuses on building neural networks with multiple layers to automatically learn hierarchical feature representations.

- **Convolutional Neural Networks (CNNs)**:
  - **Main Structure**: The architecture of CNNs, which includes convolutional layers, pooling layers, and fully connected layers.
  - **Convolutional Kernel/Filter**: The core element of CNNs that slides over the input data to detect patterns.
  - **Non-linearity**: Functions like ReLU used to introduce non-linearity in the network.
  - **Common Deep Learning Networks**: Examples of popular deep learning architectures (e.g., AlexNet, VGG, ResNet).

---

## 2. Brain-Based Learning Methods
This section explores learning methodologies inspired by the human brain, presenting research papers on brain-based learning systems, especially in vision and object recognition.

### Key Topics:
1. **Artificial Vision Using Multi-layered Neural Networks**:
   - Neocognitron and its advances in mimicking the visual processing in the brain.
   
2. **Learning Invariant Object Recognition in the Visual System**:
   - Techniques for training systems to recognize objects despite transformations in the visual field.

3. **STDP-based Spiking Deep Convolutional Neural Networks for Object Recognition**:
   - Implementation of biologically inspired Spiking Neural Networks (SNN) for vision tasks.

4. **Biologically Motivated Learning Mechanisms for Visual Feature Extraction**:
   - Stable learning mechanisms inspired by human cognition to handle facial categorization.

5. **Unsupervised Learning of Visual Features through Spike Timing Dependent Plasticity**:
   - A method for unsupervised learning of visual features using biologically motivated neural processes.

---

This course provides a balanced approach between machine learning fundamentals and brain-based neural network techniques, fostering a deep understanding of both artificial and natural learning systems.

