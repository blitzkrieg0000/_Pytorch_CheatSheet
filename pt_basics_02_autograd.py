"""
    !=> Hedef: 
        Torch kütüphanesindeki Tensor tipinin graph özelliği ve türev(gradient) alma çalışması
"""

import torch

# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
#torch.backends.cuda.matmul.allow_tf32 = True


# Tensor'un gradyanını hesaplamak için "requires_grad" parametresini ayarlıyoruz.
# Bu parametre sayesinde yapılan işlemlerden gradyan hesaplaması için daha sonra işlenmek üzere graphlar çıkarılır.
# grad_fn isimli metod bu parametre sayesinde arkaplanda çalışır ve (gradyanların)türevlerin formülünü çıkarır ama biz isteyene kadar hesaplamaz.
x = torch.randn(3, requires_grad=True)  
print("\n=>  :\n",
    x
)

y = x + 2   # Node 1
print("\n=> Node 1 :\n",
    y
)

z = y*y*2   # Node 2
print("\n=> Node 2 :\n",
    z
)

z = z.mean()   # Node 3
print("\n=> Node 3 :\n",
    z
)


#! Calculate Gradients
# Bu graphlar hesaplanıyor.
z.backward()    # dz/dx  -> z'nin x'e göre türevi(gradyanı)


# Gradients, Jacobian, Hessian: https://carmencincotti.com/2022-08-15/the-jacobian-vs-the-hessian-vs-the-gradient/
print("\n=> Hesaplanan Gradyanlar :\n",
    x.grad                              # = Jacobian Matrix (.) Gradient Vector
)

# Eğer çıkan değer skaler değilse bu şekilde gradyan hesaplardık.
# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# z.backward(v)



#! NO TRACKING GRADIENT METOD: 1
# x.requires_grad_(False)

#! NO TRACKING GRADIENT METOD: 2
# x.detach_()

#! NO TRACKING GRADIENT METOD: 3
with torch.no_grad():
    y = x + 2
    print("\n=> No Grad :\n",
        x                        # requires_grad=True olmadığını görürüz.
    )



#! Örnek bir model: Her backward işleminde gradyanlar öncekiler ile toplanır. Buna dikkat edilmesi gerekir.
# Epoch değerini değiştirerek kontrol ediniz.
weights = torch.ones(4, requires_grad=True)


for epoch in range(3):
    model_output = (weights*3 + 1).sum() #dummy model
    
    model_output.backward()

    print(f"\n=> Grad Sum {epoch} :",
        weights.grad
    )
    
    # weights.grad.zero_()    # Hesaplanan gradyanlar sıfırlanır. Böylece toplanmaz. Aşağıda örnek verilmiştir.



#! Ağırlıkların sıfırlanması için örnek: Optimization Function adımından sonra sıfırlanması gerekir.
# Örneğin ağırlıkları optimize ettikten sonra gradyanları sıfırlarız. Böylece her döngüde toplanmaz.
optimizer = torch.optim.SGD([weights], lr=0.001)
optimizer.step()    # Tek bir optimizasyon adımı gerçekleştirir.
optimizer.zero_grad() # Daha sonra gradyanların sıfırlanması gerekir.


#! Ağırlıkların sıfırlanması için örnek: Backward işleminden sonra sıfırlanması gerekir.
# weights = torch.ones(4, requires_grad=True)
# z.backward()
# weights.grad.zero_()


#! KISACA TAKİP EDİLEN GRADYAN ÖLÇÜMLERİNİN SIFIRLANMASI GEREKEN YER HER EPOCH SONRASIDIR.

