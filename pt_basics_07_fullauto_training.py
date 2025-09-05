"""
    !=> Hedef: 
        => Torch kütüphanesini kullanarak model eğitebilmek
"""
import torch
import torch.nn as nn
from torchinfo import summary
# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True



#! Prepare Dataset-----------------------------------------------------------------------------------------------------
data_x = torch.tensor([[1], [2], [3] ,[4]], dtype=torch.float32)
data_y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

test_x = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = data_x.shape
print("\n=> Veri sayısı :\n",
    n_samples
)
data_x = data_x.to(DEVICE)
data_y = data_y.to(DEVICE)
test_x = test_x.to(DEVICE)


#! Create Model--------------------------------------------------------------------------------------------------------
output_size = input_size = n_features
model = nn.Linear(input_size, output_size)
model = model.to(DEVICE)
print(model)

summary(model, input_size=(1, n_features))

print("\n=> Prediction Before Training: f(5) = :\n",
    f"{model(test_x).item():.3f}"
)


#! Training------------------------------------------------------------------------------------------------------------
LEARNING_RATE = 0.01
EPOCHS = 1000

lossFN = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)   # Artık modelimizdeki ağırlıkların güncelleneceğini pytorch biliyor.





for epoch in range(EPOCHS):
    # 1-) Forward pass
    y_pred = model(data_x)

    # 2-) Loss hesapla
    loss = lossFN(data_y, y_pred)

    #! 3-) Gradients = Backward pass
    # Eğer Pytorchta yapılan işlemlerin ardından çıkan sonuçta .backward() metodunu kullanırsak,
    # işlemi oluşturan değişkenlerin requires_grad=True parametresi olanlarına göre zincir türev kuralı uygulanacaktır.
    # Örnekte, "loss" değişkeni meyadana gelinceye kadar, "Forward()" + "Loss()" metodları uygulanmıştır. 
    # "Forward()" içerisinde" ise "w" değişkeninin parametresi requires_grad=True olduğu için, Loss'un w'ye türevinin sonucu hesaplanmıştır.
    loss.backward() # dl/dw        

    # 4-) Ağırlıkları güncelle
    optimizer.step()

    # 5-) zero gradients: Hesaplanmış türevleri bir sonraki adım için sıfırla.
    optimizer.zero_grad()


    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"Epoch={epoch+1}: w={w[0][0].item():.3f}, loss={loss:.8f}")



print("\n=> Eğitimden Sonra Tahmin: f(5) = :\n",
    f"{model(test_x).item():.3f}"
)

