"""
    !=> Hedef: 
        => Torch kütüphanesini kullanarak logistic regresyon modeli eğitebilmek
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import matplotlib.pyplot as plt


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


#! Prepare Data--------------------------------------------------------------------------------------------------------
bc = datasets.load_breast_cancer()
original_x, original_y = bc.data, bc.target

n_samples, n_features = original_x.shape

x_train, x_test, y_train, y_test = model_selection.train_test_split(original_x, original_y, test_size=0.2, random_state=278)

standartScaler = StandardScaler()
x_train = standartScaler.fit_transform(x_train)
x_test = standartScaler.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


#! Create Model--------------------------------------------------------------------------------------------------------
# Logistic Regression=> wx + b, sigmoid
class MyLogisticRegressionModel(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.Linear = nn.Linear(n_input_features, 1)


    def forward(self, x):
        return torch.sigmoid(self.Linear(x))


model = MyLogisticRegressionModel(n_features)


#! Loss & Optimizer----------------------------------------------------------------------------------------------------
criterion = nn.BCELoss()    # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#! Training Loop-------------------------------------------------------------------------------------------------------
EPOCHS=1000

for epoch in range(EPOCHS):
    # forward
    y_predicted = model(x_train)
    loss = criterion(y_predicted, y_train)

    # backward
    loss.backward()

    # updates
    optimizer.step()

    # zeroing grads
    optimizer.zero_grad()


    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")


# Training bittiğine göre graph trackingi engellemek için "torch.no_grad" context manager'ını kullanıyoruz.
with torch.no_grad():
    predicted = model(x_test)
    predicted = predicted.round()
    acc = predicted.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy {acc:.4f}")







