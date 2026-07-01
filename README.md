# Image-classification of retinopathy from RetinaMNIST Dataset

---

## Introduction

This is a small demo to show the training of a **binary image classification model** on the RetinaMNIST dataset. The model classifies **28x28 retinal fundus images** into two groups: images with negligible signs of retinopathy and images with noticeable signs of retinopathy.

The goal of this project is to keep the setup simple, while still using a well-known CNN architecture and comparing the effect of different optimizers.

---

## The Dataset

The **RetinaMNIST** dataset is a subset of the popular **MedMNIST2D** dataset. It consists of **1,600 samples**, divided into:

| Split | Number of Images |
|---|---:|
| Training | 1,080 |
| Validation | 120 |
| Testing | 400 |

Each image has a size of **28x28 pixels** with **3 color channels**.

The original dataset is labeled as an **ordinal regression task** with five classes from `0` to `4`. For this demo, I converted it into a binary classification task:

| Original Labels | Binary Label | Meaning |
|---|---:|---|
| `0-1` | `0` | negligible signs |
| `2-4` | `1` | noticeable signs |

---

## CNN Architecture

The base model is **ResNet18** from the models provided by `torchvision`. I used ResNet18 because it is a well-known CNN architecture, but it still had to be adjusted for the very small **28x28 RetinaMNIST** images.

**Changes to the architecture**

- The images are passed to the model at their original **28x28 resolution**.
- The original ResNet18 `7x7` stride-2 input convolution is replaced by a `3x3` stride-1 convolution.
- The maxpool layer is removed, because it would downsample the already small images too aggressively.
- The final fully connected layer is replaced by a single-output linear layer
- Because this is a binary classification task, I used `BCEWithLogitsLoss` as Loss Function to train the model 

This keeps the model close to ResNet18, while making it more suitable for the small input size.

---

## Training Setup

The dataset is loaded with the `medmnist` python library and the `retinamnist` keyword. It is later converted to binary labels during loading. The training images receive a small random rotation as data augmentation, while validation and test images only receive tensor conversion and normalization.

I compared two optimizers out of curiosity, the difference in performance is shown later:

- **Adam**: Adaptive Moment Estimation
- **SGD**: Stochastic Gradient Descent with momentum

To reduce overfitting through extensive training, I implemented early stopping to stop training when the validation loss stops improving. After training, the best validation model is saved and evaluated on the test set.

Because the model uses `BCEWithLogitsLoss`, the output is a raw logit. Therefore, the classification threshold is not fixed by the model itself and can be tuned on the validation set.

---

## Performance and Optimization

In this demo, **SGD slightly outperformed Adam** on the test set. SGD reached a test accuracy of **76.0%** at the best validation threshold, while Adam reached **71.8%**. SGD also stopped earlier, after **8 epochs** instead of **10 epochs**.

| Metric                         |    Adam |     SGD |
|--------------------------------|--------:|--------:|
| Epochs needed for training     |   10/20 |    8/20 |
| Best Threshold                 |    0.25 |    0.35 |
| Validation Acc at Threshold    |  0.7917 |  0.7833 |
| Training Loss                  |  0.4765 |  0.4972 |
| Training Accuracy              |  0.7620 |  0.7194 |
| Validation Loss                |  0.5003 |  0.4690 |
| Validation Accuracy            |  0.7917 |  0.7833 |
| Test Acc at Threshold `0.50`   |  0.6875 |  0.7425 |
| **Test Acc at Best Threshold** | **0.7175** | **0.7600** |

The tuned threshold improved the test accuracy for both optimizers. This is especially visible for Adam, where the test accuracy increased from **68.8%** to **71.8%**.

---

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train.py
```

Evaluate the trained model:

```bash
python evaluate.py
```

The best model weights are saved in the `weights/` directory.
