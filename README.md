## cs440@uiuc - intro to artifical intelligence
- AI Snake Bot
- Neural Net Pytorch Image Classification

## Overview

This project implements a shallow neural network (1980s-style) in PyTorch / NumPy to classify images into 4 categories: **ship, automobile, dog, frog**. The goal is to train the model on a subset of CIFAR-10 (resized / filtered) and evaluate it on a held-out development set.

### Key Features

* A neural network with one hidden layer (up to 200 hidden units)
* Activation function: **Sigmoid** or **ReLU**
* Cross entropy loss
* Data standardization (subtract mean, divide by std)
* Reporting accuracy, confusion matrix, and parameter count
* Uses PyTorch `DataLoader` for batching
* Runs fully on CPU (no GPU / CUDA)

Expected dev-set accuracy: around **0.62**, depending on hyperparameters.

---
```

### File Roles

* **reader.py** – Loads and preprocesses the dataset (returns tensors for train/dev sets).
* **neuralnet.py** – Core of the assignment. Defines the `NeuralNet` class, including `__init__()`, `forward()`, `step()`, and `fit()`.
* **mp9.py** – Entry point script that trains and evaluates the model.
* **utils.py** – Helper utilities such as dataset wrapping (`get_dataset_from_arrays`).
* **data/** – Contains the preprocessed CIFAR subset.
* **outputs/** – Optional folder for logs, model checkpoints, or plots.

---

## Design & Architecture

### Neural Network

* **Input dimension:** 31 × 31 × 3 = 2883
* **Hidden layer:** up to 200 neurons
* **Activation:** Sigmoid or ReLU
* **Output layer:** 4 logits (for 4 classes)

Mathematically:
[
F_W(x) = W_2 \sigma(W_1 x + b_1) + b_2
]

**Loss Function:** CrossEntropyLoss (includes internal softmax)

---

### Training Loop & Optimization

#### `fit()`

Coordinates model training:

* Standardizes data (subtract mean, divide by std)
* Iterates for multiple epochs
* Evaluates accuracy and confusion matrix at the end

#### `step()`

Performs one training iteration:

* Zero gradients → Forward pass → Compute loss → Backpropagate → Update weights

#### `forward()`

Defines how inputs are transformed through the layers into logits.

---

### Data Handling

* Mean and standard deviation are computed **only on training data**.
* The same normalization is applied to dev data.
* Uses `get_dataset_from_arrays()` to convert arrays into PyTorch datasets.

---

### Evaluation & Metrics

After training, the model reports:

* **Accuracy** (fraction of correct predictions)
* **Confusion Matrix** (4×4 table of actual vs predicted classes)
* **Parameter Count** (number of trainable parameters)

---

## Hyperparameters & Tuning

* **Hidden units:** ≤ 200
* **Activation:** Sigmoid or ReLU
* **Learning rate:** adjustable (typically 1e-2 to 1e-3)
* **Epochs & batch size:** specified by CLI arguments
* **Standardization:** required for convergence

---

## Usage

```bash
python3 mp9.py -h
```

Example run:

```bash
python3 mp9.py --epochs 50 --batch_size 32
```

Expected output:

* Training loss per epoch
* Dev accuracy
* Confusion matrix
* Parameter count

---

## Design Decisions & Insights

* **ReLU** preferred for faster convergence.
* **Hidden layer size** tuned for balance between accuracy and overfitting.
* **Learning rate** chosen empirically.
* **Normalization** performed in `fit()` to ensure data consistency.

---

## Results

* Dev accuracy: *~0.62 (varies by run)*
* Common misclassifications: dog ↔ frog
* Model converges after ~40 epochs
* Could be improved by adding more layers or regularization

---

## Dependencies

* Python 3.x
* PyTorch
* NumPy
* (Optional) Matplotlib

`requirements.txt` example:

```
torch
numpy
matplotlib
```

---

