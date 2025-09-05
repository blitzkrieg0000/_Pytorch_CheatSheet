import torch
import numpy as np

# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
#torch.backends.cuda.matmul.allow_tf32 = True


print(torch.cuda._get_device(0))

exit()

# =================================================================================================================== #
#                                       Basit Tensor İşlemleri                                                        #
# =================================================================================================================== #
#TODO PyTorchTutorial01 ek olarak çalışılmıştır.

#! Tensor tanımlama
x = torch.empty(2, 2, 2, 3)
x = torch.rand(2, 2)
x = torch.zeros(2, 2)
x = torch.ones(2,2)
x = torch.eye(5, dtype=torch.float16)
x = torch.tensor([2.5, 0.5, 0.6])

print("\n=> Boş Tensor Yaratma :\n",
    x,
    x.dtype,
    x.size(),
    x.shape
)


#! Tensor işlemleri
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x + y
z = torch.add(x, y)
y.add_(x)               # Inplace operation

z = x - y
z = torch.sub(x, y)
y.sub_(x)

z = x * y
print("\n=> Carpma- :\n", z)
z = torch.mul(x, y)
y.mul_(y)

z = x / y
z = torch.div(x, y)


# dot
z = x @ y
print("\n=> Carpma- :\n", z)
z = torch.dot(x, y)
x.dot(y)



#! Slice işlemleri
x = torch.rand(5, 3)
print(x[:, 0])
actual_value = x[0, 0].item() # Tek bir değer için, Tensor tipi yerine sayı değeri döndürür.



#! Reshape
x = torch.rand(4, 4)
print("\n=> Yeniden boyutlandırma :\n",
    x.view(16).shape,
    x.view(2, 2, 4).shape,
    x.view(-1, 8).shape,    # PyTorch otomatik olarak -1 değerinin ne olması gerektiğine karar verir. 4 x 4 => 2 x 8
    sep="\n"
)



#! Numpy array'e çevirme
x = torch.ones(5)
np_array = x.numpy()
x.add_(1)                   # Çevrilen numpy arrayin referansı aynıdır. Referans ile tip dönüşür kopyalanmaz.

print("\n=> Numpy Referans :\n",
    type(x),
    x,
    type(np_array),
    np_array,
    sep="\n"
)


x = torch.ones(5)
np_array = x.numpy()
x = x.add(1)               # Ancak böyle bir durumda Tensor obje üzerine başka bir Tensor obje yazmış oluruz.

print("\n=> Numpy Value :\n",
    type(x),
    x,
    type(np_array),
    np_array,
    sep="\n"
)



#! Numpy arrayi ayrıyeten çevirme
# 1
a = np.ones(5)
b = torch.from_numpy(a)                 # Referans ile çevirir.

a += 1
print("\n=> Reference Numpy Convert :\n",
    b
)

# 2
a = np.ones(5)
b = torch.Tensor(a)                     # Her zaman kopyalar

a += 1
print("\n=> Value Numpy Convert :\n",
    b
)


#! GPU da çalışma
x = torch.ones(5, device=DEVICE)

print("\n=> GPU da çalışma :\n",
    "Bu cihazda çalışıyor: ",
    x.device
)

# GPU ya atama
y = torch.ones(5)
y_gpu = y.to(device=DEVICE)     
print("\n=>  :\n",
    y.device,
    y_gpu
)


#! Bir takım işlemler
# Sıralı memory kullanma ve channel_last hale getirme
# Torch'un bazı neural network modülleri hariç, torch channel_first olarak çalışır.
# Ancak bazı performans durumlarında channel_last olarak çalışması sağlanabilir.
# Channel last veya first demek direkt olarak fiziksel matris boyutu demek değildir.
# NHWC (64, 1080, 1920, 3)(channel last) <---> NCHW (64, 3, 1080, 1920)(channel first) olarak adlandırılır.
# Bir veri NCHW formatında bir veri olup, memory olarak channel_last davranabilir.
#? memory_format=torch.channels_last yalnızca 4D matrisler ile kullanılır.
#? https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html

x = torch.rand(64, 3, 224, 224)
  # Sıralı hale getirme, .contiguous() çok önerilmez bunun yerine .to parametresi önerilir.


print("\n=> İşlem 1  :\n",
    x.shape,            # Fiziksel düzeni
    x.device,           # Çalıştığı aygıt memorysi
    x.stride(),         # Memoryde veriler arasındaki aralık (Logic düzen diyebiliriz.)
    x.is_contiguous()   # x memory de sıralı mı
)

x = x.to(memory_format=torch.channels_last)
# x = x.to(device=DEVICE)   # isteğe bağlı çalıştığı cihazı değiştir.

print("\n=>  İşlem 2 :\n",
    x.shape,                # Shape aynı kalacaktır.
    x.device,
    x.stride(),
    x.is_contiguous()       # Ve "torch.channels_last" dan dolayı düzensiz olarak tutulacaktır.
)


#! TORCH GRAD
# Özel olarak bir model üretirken, bir tensor'ün "requires_grad" parametresini True yaparsak, bu tensor için izleme başlatılacak ve mümkünse önceden türev hesabı yapılacak.
# Şuan anlaşılmasa da, sinir ağları ile uğraşırken, ağırlık güncelleme için tüm ağırlıkların birbirine göre zincir türevi alındığında gerekli oluyor.
# Zaten tüm ağırlıklar için bu değer otomatik olarak True ile başlatılıyor. Model dışı tensor tanımlamalarında bu değer varsayılan olarak False dır.
x = torch.ones(5, requires_grad=True)









