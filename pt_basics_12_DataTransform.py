"""
    !=> Hedef: 
        => Yüklenen veriyi işlemlerden geçirme (transform)

        => Veri yüklendiği zaman istenilen boyutta, ölçekte, tipte veya formatta olmayabilir.
"""
import multiprocessing
from typing import Any
import torch
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


#! Prepare Data--------------------------------------------------------------------------------------------------------
"""
    => Veri yüklenirken veya yüklendiğinde ön işlemlerden(transformlardan) geçirmek isteyebiliriz.
    => Bazı dtransform metodlarını, DataLoader' a vererek veya torchvision'un hazır verisetlerini çekerken belirtebiliriz.
    ?=> Hazır Pytorch Transformlar: https://pytorch.org/vision/stable/transforms.html
"""

#Torchvision ile veri yükleme
dataset = torchvision.datasets.MNIST(
    root="./dataset", 
    transform=torchvision.transforms.ToTensor(),     # Bazı transformlar ile veri yüklenirken ön işlemden geçirebiliriz.
    download=True
)

class WineDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        Datafile = np.loadtxt("./dataset/wine/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.NSamples = Datafile.shape[0]
        self.DataX = Datafile[:, 1:]    # İlk sütun hariç: İlk sütun class değerlerini belirtecek şekilde düzenlenmiş.
        self.DataY = Datafile[:, [0]]   # class

        self.Transform = transform


    # Python Indexer: x[0] => index:0
    def __getitem__(self, index):
        sample =  self.DataX[index], self.DataY[index]

        # Transfrom Uygula
        if self.Transform:
            sample = self.Transform(sample)

        return sample
    

    # Python len(x) => length: 1000
    def __len__(self):
        return self.NSamples


#! Custom Transform Methods
class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform():
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

#! 0-Bir adet transform kullanırken
# dataset = WineDataset(transform=ToTensor())
# features, labels = dataset[0]
# print(type(features), type(labels))


#! 1-Birden fazla transform kullanırken
composedTransforms = torchvision.transforms.Compose(
    [
        ToTensor(),     # 0
        MulTransform(2) # 1
    ]
)
dataset = WineDataset(transform=composedTransforms)

features, labels = dataset[0]
print(type(features), type(labels))



















