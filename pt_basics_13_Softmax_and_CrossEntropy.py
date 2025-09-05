"""
    !=> Hedef: 
        => Softmax ve CrossEntropyLoss gibi işlemlerin hesaplanmasının pytorch ile ne kadar kolay olduğunun gösterilmesi
"""
import multiprocessing
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

#! SOFTMAX---------------------------------------
# NUMPY
def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = Softmax(x)
print(f"Softmax /w Numpy: {outputs}")

# Pytorch
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(f"Softmax /w Torch: {outputs}")

#! CrossEntropy----------------------------------
# NUMPY
def CrossEntropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

Y = np.array([1, 0, 0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = CrossEntropy(Y, Y_pred_good)
l2 = CrossEntropy(Y, Y_pred_bad)
print("\n=> Cross Entropy Loss 1 /w Numpy :\n",
    l1
)
print("\n=> CrossEntropy Loss 2 /w Numpy :\n",
    l2
)

# Pytorch

# Not One-Hot
# No Softmax(need logits)
# Already to be Apply: LogSoftmax + NLLLoss
lossFn = nn.CrossEntropyLoss()

Y = torch.tensor([0]) # class:0
Y_pred_good = torch.tensor([[0.7, 0.2, 0.1]]) # 1x3
Y_pred_bad = torch.tensor([[0.1, 0.3, 0.6]]) # 1x3

l1 = lossFn(Y_pred_good, Y)
l2 = lossFn(Y_pred_bad, Y)

print("\n=> Cross Entropy Loss 1 /w Torch :\n",
    l1.item()
)
print("\n=> CrossEntropy Loss 2 /w Torch :\n",
    l2.item()
)

# Tahmin
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print("\n=> Predictions :\n",
    predictions1, predictions2
)

# Pytorch Multiple Class
# Not One-Hot
# No Softmax(need logits)
# Already to be Apply: LogSoftmax + NLLLoss
lossFn = nn.CrossEntropyLoss()

Y = torch.tensor([2, 0, 1]) # classes
Y_pred_good = torch.tensor([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1]]) # 3x3
Y_pred_bad = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.2, 0.1, 0.7]]) # 3x3

l1 = lossFn(Y_pred_good, Y)
l2 = lossFn(Y_pred_bad, Y)

print("\n=> Multiple Class Cross Entropy Loss 1 /w Torch :\n",
    l1.item()
)
print("\n=> Multiple Class CrossEntropy Loss 2 /w Torch :\n",
    l2.item()
)

# Tahmin
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print("\n=> Multiple Class Predictions :\n",
    predictions1, predictions2
)
