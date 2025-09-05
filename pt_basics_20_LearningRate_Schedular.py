"""
    !=> Hedef: 
        => Yapay zeka eğitimlerinde ağırlık güncellemesine en çok etki eden parametre Learning Rate'dir.
        => Learning Rate'in fazla veya az olması modelin, hedeften uzaklaşmasına, hedefi sürekli atlamasına, hedefe bir türlü yaklaşamamasına sebebiyet verebilir.
        => Bu durumda her epochta learning rate yeniden ayarlanarak, duruma göre dinamiklik sağlanır ve model öğrenmesi iyileştirilir.
        => Aşağıda birden fazla kural tabanlı ve matematiksel formüle dayanan Learning Rate Schedularlar verilmiştir.
        => Schedularlar, optimizasyon fonksiyonuna yeni Learning Rate' bildirir ve model bu yeni değere göre eğitilir.
        => Genellikle eğitimin ilerleyen safhalarında learning rate in yavaş yavaş düşürülmesi hedeflenir ve model öğrenmesi önemli ölçüde iyileşir.
        => Aşağıdaki örnekte her epoch, step sayısı olarak temel alınmıştır. Ancak veriseti büyükse bir epochtaki batch sayısıda, döngüde step olarak kullanılabilir.
        => schedular.step() veya schedular.step(validation_loss), step bitince çağırılır ve yeni learning rate optimizer'a bildirilmiş olunur.
        ?=> https://pytorch.org/docs/stable/optim.html
"""
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LRScheduler

# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

# PARAMS
LEARNING_RATE = 0.1


#! Create Model--------------------------------------------------------------------------------------------------------
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


##* Learning Rate Schedulers-------------------------------------------------------------------------------------------
## Method 1
# lambda1 = lambda epoch: epoch / 10
# schedular = LRScheduler.LambdaLR(optimizer, lambda1)         # Belirlenen lambda metoduna göre return edilen değer, eski değer ile çarpılarak kullanılır.

## Method 2
# lambda2 = lambda epoch: 0.95
# schedular = LRScheduler.MultiplicativeLR(optimizer, lambda2)   # Önceki epochtaki lr değeri, belirlenen bir değer ile çarpılarak yeni epochta kullanılır.

## Method 3
# schedular = LRScheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Her "n" step saydıktan sonra, gamma parametresi ile LR değerini çarpar.
# schedular = LRScheduler.MultiStepLR (optimizer, milestones=[30, 80], gamma=0.1)  # Her "milestone" steplerine kadar saydıktan sonra, gamma parametresi ile LR değerini çarpar.

## Method 4
# Exponetial olarak learning rate'i düşürür.
# schedular = LRScheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1, verbose=False)

## Method 5
# Bir formüle göre learning rate'i ayarlar.
# schedular = LRScheduler.CosineAnnealingLR(optimizer, T_max=, eta_min=0, last_epoch=-1, verbose=False)

## Method 6
# Özel bir learning rate düşürücüdür.
# Eğer patience miktarı kadar ilerleyince modelde iyileşme olmazsa, learning rate'i düşürür. Kural tabanlı bir learning rate schedulardır.
#* schedular.step(validation_loss) ile kullanılır.
schedular = LRScheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, threshold=0.0001, threshold_mode="rel", cooldown=0, min_lr=0, eps=1e-08, verbose=False)

## -7-
# Sabit frekanslı iki sınır arasındaki cycle a göre learning rate ayarlanır.
# schedular = LRScheduler.CyclicLR(optimizer, base_lr=0.1, max_lr=0.1, step_size_up=2000, step_size_down=None, mode="triangular", gamma=1.0, scale_fn=None, scale_mode="cycle", cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=False)

## -8-
# Sabit frekanslı iki sınır arasındaki cycle a göre learning rate ayarlanır.
# schedular = LRScheduler.OneCycleLR(...)


print(optimizer.state_dict())
for epoch in range(5):
    # loss.backward() ...           # Gradyanlar hesaplandı.
    optimizer.step()                # Ağırlıklar LR'ye göre güncellendi.
    # validation_loss = validate()  # Validation yapıldı.
    schedular.step()                #! İş bitince Schedular ile Learning Rate güncellendi.

    print("\nLearning Rate: ", optimizer.state_dict()["param_groups"][0]["lr"], "\n")
