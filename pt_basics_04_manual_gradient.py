"""
    !=> Hedef: 
        => Torch kütüphanesindeki Tensor tipinin graph özelliğini kullanarak basit bir model için loss a göre model ağırlıklarını güncelle
        
        => Büyük derin öğrenme modellerini anlamak için temel
"""

import torch
import numpy as np

# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


# Linear Regression------------------------------
# f = w * x

data_x = np.array([1, 2, 3, 4], dtype=np.float32)
data_y = np.array([2, 4, 6, 8], dtype=np.float32)

# Weight
w = 0.0

#! 1-) Manual Model prediction
def Forward(x):
    return w * x


#! 2-) Manual MSE(Mean Squared Error) Loss
def Loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


#! 3-) Manual Gradient
# MSE = 1/N * (w*x - y)**2      # MSE değeri ile loss hesaplanır ve loss'un ağırlığa göre türevi bize gradient değerini verir.
# dL/dw = 1/N * 2x * (w*x - y)  # MSE loss formülünün türevi bize optimizasyon için gerekli türevi verir.
def Gradient(x, y, y_predicted):
    return (2*x * ( y_predicted-y)).mean()



# Training
LEARNING_RATE = 0.01
EPOCHS = 50

print("\n=> Eğitimden önce tahmin: f(5) = :\n",
    f"{Forward(5):.3f}"
)


LEARNING_RATE = 0.01
EPOCHS = 50


for epoch in range(EPOCHS):
    # forward pass
    y_pred = Forward(data_x)

    # loss
    loss = Loss(data_y, y_pred)

    # gradients
    dw = Gradient(data_x, data_y, y_pred)

    # update weights
    w -= LEARNING_RATE * dw     # Ağırlığı, learning rate ve türev değerine bağlı güncelleriz

    if epoch % 5 == 0:
        print(f"Epoch={epoch+1}: w={w:.3f}, loss={loss:.8f}")



print("\n=> Eğitimden sonra tahmin: f(5) = :\n",
    f"{Forward(5):.3f}"
)



























