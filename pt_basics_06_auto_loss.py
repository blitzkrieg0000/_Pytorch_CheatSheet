"""
    !=> Hedef: 
        => Torch kütüphanesini kullanarak model eğitebilmek
"""


import torch
import torch.nn as nn


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


# Linear Regression------------------------------
# f = w * x

data_x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
data_y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)


# Weight
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


#! 1-) Manual Model prediction
def Forward(x):
    return w * x


print("\n=> Prediction Before Training: f(5) = :\n",
    f"{Forward(5):.3f}"
)

# Training
LEARNING_RATE = 0.01
EPOCHS = 100

lossFN = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=LEARNING_RATE)  #Ağırlığı biz belirlediğimiz için [w] parametresini girdik, sonraki bölümde tam otomatik yapacağız.

for epoch in range(EPOCHS):
    # forward pass
    y_pred = Forward(data_x)

    # loss
    loss = lossFN(data_y, y_pred)

    #! Gradients = Backward pass
    # Eğer Pytorchta yapılan işlemlerin ardından çıkan sonuçta .backward() metodunu kullanırsak,
    # işlemi oluşturan değişkenlerin requires_grad=True parametresi olanlarına göre zincir türev kuralı uygulanacaktır.
    # Örnekte, "loss" değişkeni meyadana gelinceye kadar, "Forward()" + "Loss()" metodları uygulanmıştır. 
    # "Forward()" içerisinde" ise "w" değişkeninin parametresi requires_grad=True olduğu için, Loss'un w'ye türevinin sonucu hesaplanmıştır.
    loss.backward() # dl/dw        

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()


    if epoch % 10 == 0:
        print(f"Epoch={epoch+1}: w={w:.3f}, loss={loss:.8f}")



print("\n=> Prediction After Training: f(5) = :\n",
    f"{Forward(5):.3f}"
)





























