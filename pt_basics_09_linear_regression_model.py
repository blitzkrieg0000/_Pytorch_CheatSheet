"""
    !=> Hedef: 
        => Torch kütüphanesini kullanarak lineer regresyon modeli eğitebilmek
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


#! Prepare Data--------------------------------------------------------------------------------------------------------
original_x, original_y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=278)

# Tensore çevir
data_x = torch.from_numpy(original_x.astype(np.float32))
data_y = torch.from_numpy(original_y.astype(np.float32))
data_y = data_y.view(data_y.shape[0], 1)


n_samples, n_features = data_x.shape


#! Create Model--------------------------------------------------------------------------------------------------------
input_sizes = n_features
output_size = 1
model = nn.Linear(input_sizes, output_size)


#! Loss & Optimizer----------------------------------------------------------------------------------------------------
LEARNING_RATE = 1e-2
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


#! Training Loop-------------------------------------------------------------------------------------------------------
EPOCHS = 100
for epoch in range(EPOCHS):
    # forward
    y_predicted = model(data_x)

    # loss
    loss = criterion(y_predicted, data_y)

    # backward
    loss.backward()

    # update weights
    optimizer.step()

    # Clean gradient values
    optimizer.zero_grad()


    if (epoch+1)%10 == 0:
        print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")


# plot
predicted = model(data_x).detach().numpy()
plt.plot(original_x, original_y, "ro")
plt.plot(original_x, predicted, "b")
plt.show()





