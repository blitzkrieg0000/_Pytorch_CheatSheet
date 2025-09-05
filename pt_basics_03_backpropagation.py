import torch

# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
#torch.backends.cuda.matmul.allow_tf32 = True


x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)   # Bu ağırlık değeri olacak ve bu ağırlığa göre gradient alınacak.


#! Forward Pass, Compute Loss
# Modeli çalıştır ve loss hesapla.
# PyTorch otomatik olarak "w" değeri için Local Gradient lerin formüllerini çıkarır ve hafızada tutar. => requires_grad=True

y_predicted = w * x
loss = (y_predicted - y)**2     # Squared Loss

print("\n=> Loss :\n",
    loss
)


#! Backward Pass
# Modeli güncellemek için forwardda ölçülen gradientleri ve loss u kullan. Loss un ağırlığa göre türevini hesapla
loss.backward()                 # Formülleri hesapla ve gradientleri ölç
print("\n=> gradients :\n",
    w.grad                      # d_loss/d_w nin değeri
)


# Update Weights: Ağırlıkları güncelle...

