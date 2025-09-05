"""
    !=> Hedef: 
        => Transfer Learning ile model eğitimi
        
        => Bazı durumlarda, zaten eğitilmiş bir ağın ağırlıkları ile başlamak, eğitim açısından tutarlılık için; aynı zamanda daha çabuk hedefe varmak açısından kritik derecede önemlidir.
        
        => Örneğin herhangi bir CNN ağının convolution kısmı hariç, son katmanı yeniden eğitilerek, az sayıda iş ile modelin amacı değiştirilebilir.
        
        => Eğitilmiş bir modelin, başka bir amaç uğruna değiştirilip, kullanılmasına FineTuning denir.

        => Bazen ağırlıkları kaldığı yerden devam ettiririz ve modeli benzer amaçlar için kullanırız.  Buna TransferLearning denir.

        => Zaten eğitilmiş bir modelin tamamını eğitmek istemeyebiliriz. Bunu sağlamak adına bazı katmanlarını dondururuz. Yani gradyan hesabı yaptırmayız. Aynı zamanda optimizer metoduna bu eğitmek istemediğimiz ağırlıkların referansını vermeyiz. Böylece işler hafifler ve hızlanır.
"""
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True



#! PARAMS--------------------------------------------------------------------------------------------------------------
# Önceden eğitilmiş bir ağı tekrar eğitirken veya kullanırken, önceden eğitilmiş verisetinin ön işleme adımlarını dikkate almamız gerekir.
# Önceden işlenmiş bir verilerin "ortalama" ve "standart sapma" değerlerini referans alarak yeni verileri de o hale getirmemiz, eğitim tutarlılığı açısından önem teşkil eder.
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
EPOCHS = 25
LEARNING_RATE = 0.001


#! PREPARE DATASET-----------------------------------------------------------------------------------------------------
# hymenoptera_data
# ├── train/
# │   ├── ants/
# │   └── bees/
# └── val/
#     ├── ants/
#     └── bees/
DATA_TRANSFORMS = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),  # Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

DATASET_PATH = "dataset/hymenoptera_data"
image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_PATH, x), DATA_TRANSFORMS[x]) for x in ["train", "val"]}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=multiprocessing.cpu_count()) for x in ["train", "val"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
CLASS_NAMES = image_datasets["train"].classes

print(f"Classes: {CLASS_NAMES}")

def ShowImage(inp, title):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean   # destandardization
    inp = np.clip(inp, 0, 1) # min 0, max 1
    plt.imshow(inp)
    plt.title(title)
    plt.show()

## Örnek veri al ve görselleştir
# Training verisinden bir örnek al
inputs, classes = next(iter(dataloaders["train"]))

# Batch halinde olan verileri "grid" haline getir.
out = torchvision.utils.make_grid(inputs)

ShowImage(out, title=[CLASS_NAMES[x] for x in classes])



#! TRAIN---------------------------------------------------------------------------------------------------------------
def TrainModel(model, criterion, optimizer, scheduler, num_epochs=25):
    """
        Model eğitimi için bir metod
    """
    tic = time.time()

    # model ağırlıklarının kopyalanması(pass by value)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Her epochta training; ardından da validation için bir loop
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Modeli Training moda ayarla.
            else:
                model.eval()   # Modeli evaulate moda ayarla. [Model öğrenmesini(ağırlık güncellemesini ve gradyan hesabını) test için durdur.]

            running_loss = 0.0
            running_corrects = 0

            # Tüm verileri modele göster
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # forward
                # Sadece "train" fazında isek; yani modeli eğitiyorsak, tracking'i aktif et. Böylece otomatik gradyan hesabını açıp kapatırız.
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)    # En yüksek sonuca karar ver [0.7, 0.3] => Index 0 max
                    loss = criterion(outputs, labels)

                    # backward + optimize : Sadece eğitim fazındaysak, otomatik ölçülen gradyanları hesaplayıp, ağırlıkları güncelle
                    if phase == "train":
                        optimizer.zero_grad()   # "w.grad" propertylerini sıfırlıyoruz ki her türevde sıfırlansın, yoksa pytorch yeni türev değerlerini "w.grad" üzerine ekler.
                        loss.backward()         # Tüm ağırlıkların ayrı ayrı, "loss" fonksiyonuna göre türevi alınır ve tüm ağırlıkların "w.grad" propertysine ayrı ayrı ekleme yapılır. dLoss/dW
                        optimizer.step()        # "w.grad" propertylerine göre her ağırlık kendi içinde güncellenir. Ağırlıkları güncellemek demek modelin eğitilmesi demektir.


                # Sonuçlar
                running_loss += loss.item() * inputs.size(0)        # Loss * Batch size Toplam loss'u görebilmek adına, sonuçları belirginleştirmek adına yapılmııştır. Loss formülüyle alakalıdır.
                running_corrects += torch.sum(preds == labels.data) # tahmin edilen vektör ile gerçek labels vektörünü karşılaştır ve True olan değerleri topla: Kaç tanesi doğru?

            if phase == "train":
                scheduler.step()


            epoch_loss = running_loss / dataset_sizes[phase]              # Veri setine göre, gözlemlemek adına loss değeri çıkart
            epoch_acc = running_corrects.double() / dataset_sizes[phase]  # Veri setine göre, ne kadarı doğru tahmin edildiğini çıkart
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Modeli kopyala (deep copy)
            if phase == "val" and epoch_acc > best_acc:                   # Önceki accuracy en iyi accuracyden büyükse model ağırlıklarını hafızada tut ve "deep copy" ile değerlerini kopyala. Böylece referanstan etkilenme.
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - tic
    print(f"Eğitim süresi: {time_elapsed // 60:.0f}dk {time_elapsed % 60:.0f}s")
    print(f"En iyi Accuracy oranı: {best_acc:4f}")


    # Eğitim boyunca elde edilen en iyi accuracy'ye sahip model ağırlıklarını modele yükle
    model.load_state_dict(best_model_wts)
    return model



##! Amaç 1-) ConvNet Modelini Finetuning ile eğitim
# Önceden eğitilmiş ağırlıklara sahip modeli yükle ve Fully Connected katmanını(regresyon kafasını) amaca bağlı olarak değiştir.
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features                     # FullyConnected layerdaki giriş boyutunu öğren
model.fc = nn.Linear(num_features, len(CLASS_NAMES))    # Bu çıkışın önüne 2 dense' e sahip bir "Linear" katman (hidden layer) daha yerleştir çünkü 2 adet sınıfımız var.
model = model.to(DEVICE)                                # Model çalışılacak cuda/cpu cihaza tanıtılmıştır. Örneğin Nvidia Cuda cihazının memory'sine bu değişkenler kopyalanır ve orada çalışılır.

## Optimizer & Criterion(Loss)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)    #! Modelin tüm ağırlıkların referansları optimizer'a tanıtılmıştır. Yani tüm ağırlıklar güncellenecek.

## Step-LR-Decays: LearningRate parametresini gamma oranı ile belirli bir stepten sonra azaltarak, daha yavaş ancak etkili bir eğitim yapmış oluruz.
# 7 stepte gama oranı 0.1 olacak şekilde learning rate'i azaltacaktır.
# Learning rate scheduling, optimizer'ın ağırlık güncellemesinden sonra uygulanmalıdır.
# Kod şu örnekteki gibi olmalıdır. Örneğin her 7 stepte; örnekte her 7 epochta bir "learning rate scheduler" ın adım sayısı artacaktır.:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = TrainModel(model, criterion, optimizer, step_lr_scheduler, num_epochs=EPOCHS)



##! Amaç 2-) Feature Extraction kısmı hariç(Convolutional Layers daki ağırlıklar dondurulacak). ConvNetin sadece regresyon kafasının(son katmanında tam bağlı katmanların) eğitilmesi
# Son katman harici, tüm ağın eğitimini donduruyoruz. Böylece sadece son katman eğitilecek.
# requires_grad = False ile istediğimiz parametreleri donduruyoruz. Böylece backward() aşamasında ağırlık güncellemesine uğramayacaklar ve sadece son katman eğitilecek.
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Yeni parametreler varsayılan olarak requires_grad=True ile işaretlidir. Önceki parametreleri zaten dondurduk(requires_grad=False)
num_features = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_features, len(CLASS_NAMES))
model_conv = model_conv.to(DEVICE)

## Optimizer & Criterion(Loss)
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)  #! Modelin tüm parametreleri hariç, sadece eğitmek istediğimiz ağırlıkları, optimizer metoduna tanımlıyoruz.

# Decay LR: Her 7 adımda bir LearningRate 0.1 gama oranında azalacaktır.
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = TrainModel(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=EPOCHS)


"""
    !=> NOT: Pytorch'un model eğitiminde çalışma mantığı
        
        => Pytorch'un geliştirdiği "auto graph" ve "graph tracking" yöntemleri, kod çalışınca arkaplanda çalışmaktadır.
        
        => Örneğin bir tensor değişkeni oluşturulduğunda "requires_grad=False" olarak işaretlidir. Eğer bu değeri True yaparsanız. Tracking başlar.

        => Tracking maaliyetli bir yapıdır ve düşük seviyeli bir kodlama dili ile yazılmıştır.

        => Modeli oluşturduğunuzda, model içerisindeki öğrenilebilir parametreler izlenir. 
        
        => Modelin ağırlıklarını optimizer'a tanıttığınızda, eğer modelin ağırlıklarında "grad" propertysi dolu ise, önerilen optimizasyon metoduna göre ağırlıklar daha sonra güncellenecektir.

        => Gördüğünüz üzere, model bir tahmin yapar ve bize bir obje döndürür (feed forward). Bu değer genellikle etiket tahmin (y_pred) değeridir.

        => Modelin başından beri "requires_grad=True" olan ağırlıklardan bu yana yapılan işlemler takip edilmiştir.

        => Modelin tahmin ettiği "y_pred" değeri izlenmeye devam edilmektedir. 
        
        => Bu değer daha sonra "Loss" metoduna girerek, "y_pred" ve "y_actual" değerlerine göre kayıp oranı hesaplanır.

        => Artık her bir "requires_grad=True" ağırlık için tahmini bir graph şu şekildedir:
            ##!=> Graph W(n): [Giriş Değeri] => [Convolution İşlemleri...] => [Aktivasyon Fonksiyonu...] => ... => [Sınıflandırma/Regresyon Katmanı] => [Loss Hesabı]

        => Loss hesabına kadar kadar gelirken "requires_grad=True" olan "w" ağırlık değerleri izlenmiş; "dLoss/dW" (loss'un ağırlıklara göre türevi) nin zincir kuralına göre türev formülleri hesaplanmıştır.

        => "loss.backward()" yapıldığında her sınıflandırma çıkışı için loss'un tüm ağırlıklara göre türevi, bu alınan türev formülleri(graph) lar yardımıyla hesaplanır.
            #? Dikkat edilirse tüm yapılan işlemler "w" ağırlıklarına bağlı ve son işlem loss işlemidir. Yani zinci kuralı ile türevlenebilir bir yapı vardır. Her işlem birbirini etkiler.

        => Hesaplanan "dLoss/dW" türev değerleri her bir ağırlık için "w.grad" propertysine eklenir.

        => Artık optimizer.step() yapıldığında optimizer kendisine tanıtılan model ağırlıklarının "w.grad" parametresini, optimizasyon metoduna uygun olarak okur ve günceller. (backward)
        
"""
