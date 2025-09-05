"""
    !=> Hedef: 
        => Model veya Ağırlıkları dosyaya kaydetme

        => Kaydetme metodlarının avantaj ve dezavantajları
"""
import torch
import torch.nn as nn


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


"""
    !=> torch.save() metodu: Lazy Model Save-Load
        => Tüm ağırlıkları ve model yapısını kaydeder. 
        => Ancak model ağırlıklarını tekrar yüklerken, kod düzeni aynı olmak zorundadır.
        => Ağırlıklar python.pickle serialization ile kaydedildiğinden; tekrar kullanılmak istendiğinde kod üzerindeki importlar gereklidir.
        => Bu yapı bir nevi koda bağımlılıklarına bağımlıdır.
        => Çok özel mimarileri kaydederken yardımcı olabilir.
        => torch.load(PATH) ile geri yüklenip kullanılabilir.

    !=> torch.save(model.state_dict())
        => Bu yapı sadece ağırlıkları kaydeder.
        => Model dictionary şeklinde serialize edilir ve okunması kolay şekilde kaydedilir.
        => Model yüklenmek istendiğinde, model oluşturulur ve model.load_state_dict(torch.load(PATH)) ile ağırlıklar yüklenir.
        => Bu yapıda bağımlıdır ancak, modelin nasıl oluşturulduğuna ve kütüphanelere bağımlı değildir.
        => Modelin çeşitli kısımlarını checkpoint olarak bildiğimiz dictionary yapısında da kaydedebiliriz.
"""



#! Create Model--------------------------------------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_input_features=6)
# Model Eğitilir...



#! Metod 1-) Save All--------------------------------------------------------------------------------------------------
for param in model.parameters():
    print(param)

# Tüm modeli kaydet ve yükle
FILE = "model.pth"
torch.save(model, FILE)

loaded_model = torch.load(FILE)
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)



#! Metod 2-) Sadece Ağırlıkları kaydet---------------------------------------------------------------------------------
FILE = "model.pth"
torch.save(model.state_dict(), FILE)

print(model.state_dict())
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE)) # Yüklenmiş dictionary dosyasını alır
loaded_model.eval()
print(loaded_model.state_dict())



#! Extra: state_dict ile modelin checkpointini alabiliriz. Örneğin optimizer'ın çeşitli parametrelerini custom dictionary oalrak kaydedip tekrar yükleyebiliriz.
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Custom Dictionary
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

print(optimizer.state_dict())
FILE = "checkpoint.pth"
torch.save(checkpoint, FILE)

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
epoch = checkpoint['epoch']

model.eval()
# - ya da -
# model.train()

print(optimizer.state_dict())
"""
    => Modeli yükledikten sonra "model.eval()" moda veya training yapacaksak "model.train()" moda almamız gerekir.
    =>  Çünkü bazı sinir ağı katmanları, train ve evaluation aşamalarında farklı davranışlar sergiler.
    =>  Örneğin "dropout" ve "batch normalization" katmanları bu katmanlara örnektir.
    => Hatta dağıtık sistemler ile eğitim yaparken (16 x Nvidia Tesla A100 gibi ^-^) batch normalization katmanı tüm gpularda eğitimde olan modellerdeki Exponential Moving Average(EMA) değerine göre çalışır. İleri bir konudur.
"""



#! Kekstra: Eğer GPU da kaydedilmiş bir ağırlık dosyası, CPU için yüklenecekse "map_location" a cihaz adı verilir.
# DEVICE: "cpu"
# Ya da tam tersi ise "cuda:0" olarak ilk nvidia gpu adı verilebilir.
model.load_state_dict(torch.load("weight/model.pth", map_location=DEVICE))
