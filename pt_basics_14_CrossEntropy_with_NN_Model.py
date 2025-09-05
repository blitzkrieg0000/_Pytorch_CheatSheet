"""
    !=> Hedef: 
        => Softmax ve CrossEntropyLoss gibi işlemlerin hesaplanmasının pytorch ile ne kadar kolay olduğunun gösterilmesi

        => Bazı CrossEntropyLoss formüllerini yapay sinir ağında kullanırken dikkat edilmesi gereken yerler

        => Örneğin CrossEntropyLoss ile Multiclass classification yaparken veri, raw(logits) şeklinde, softmax uygulanmadan hesaplamaya girer.

        => Örneğin BinaryCrossEntropyLoss ile Binary sınıflandırma yaparken veriye sigmoid uygulanıp, veri o şekilde hesaplamaya girer.
"""
import torch
import torch.nn as nn


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


# MultiClassClassification => CrossEntropyLoss-----------------------------------------------------
class NeuralNetworkModelMultiClass(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super().__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(input_size, num_classes)


    def forward(self, x):
        out = self.Linear1(x)
        out = self.relu(x)
        out = self.Linear2(x)

        # Çıkışta Softmax kullanılmadı.
        return out

model = NeuralNetworkModelMultiClass(input_size=28*28, hidden_size=5, num_classes=5)
criterion = nn.CrossEntropyLoss() # Kendisi zaten Softmax uygular


# BinaryClassification => BinaryCrossEntropyLoss-----------------------------------------------------
class NeuralNetworkModelBinaryClass(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(input_size, 1)


    def forward(self, x):
        out = self.Linear1(x)
        out = self.relu(x)
        out = self.Linear2(x)

        # Çıkışta Sigmoid kullanıldı.
        out = torch.sigmoid(out)
        return out


model = NeuralNetworkModelBinaryClass(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss() # Kendisi sigmoid uygulamaz.








