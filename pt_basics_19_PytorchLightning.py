"""
    !=> Hedef:
        => Pytorch Lightning işimizi kolaylaştıran eğitim odaklı bir pytorch API dır.
        
        => Pytorch Lightning model eğitimine odaklanır.

        => Uyarılar ve öneriler ile kodumuzu destekler.

        => Kod optimizasyonuna önem verir.

        => Yüksek seviye kodlama ile işi bir kaç adımda bitirmeyi hedefler.

        !=> Entegre bir tensorboard desteği vardır. #=> $ tensorboard --logdir lightning_logs/

        !=> DEVICE ile kod gpu da mı cpu da mı kontole gerek yoktur. Kendisi otomatik cuda'yı seçer.

        !=> Manual olarak for döngülerine ihtiyaç yoktur.

        !=> Optimizer adımlarına ihtiyaç yoktur.

        !=> Hesaplanan "w.grad" sıfırlamasına ihtiyaç yoktur.

        !=> Kendi başına validation loop'a gerek yoktur.
        
        ?=> Resmi Dökümantasyon: https://lightning.ai/docs/pytorch/stable/starter/introduction.html
"""
import multiprocessing
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#Pytorch Lightning API
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer as LightningTrainer


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


#! PARAMS--------------------------------------------------------------------------------------------------------------
INPUT_SIZE = 784    # 28x28 MNIST Dataset
HIDDEN_SIZE = 100
CLASSES = 10        # MNIST Dataset Digits: 0-9
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


#! Create Model--------------------------------------------------------------------------------------------------------
class MyLitNeuralNetwork(pl.LightningModule):                      #! nn.Module
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Relu = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_size, num_classes)

        self.validation_step_outputs = []
    
    def forward(self, x):
        out = self.Linear1(x)
        out = self.Relu(out)
        out = self.Linear2(out)
        return out
    

    #! Set Training Loop
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        images = images.reshape(-1, 28*28)

        ## Forward
        predicted = self(images)
        loss = F.cross_entropy(predicted, labels) 

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
    

    #! Set Optimizer
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)



    #! Set Training Loop
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        images = images.reshape(-1, 28*28)

        ## Forward
        predicted = self(images)
        loss = F.cross_entropy(predicted, labels) 
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}


    #! After Validation
    def on_validation_epoch_end(self):
        average_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        print({"val_loss": f"{average_loss.item():.4f}"})


    #! Set Training Dataset: dataset ve loader iki metoda da ayrılabilir. API dökümantasyonunu kontrol edin.
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = torchvision.datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
        return DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=multiprocessing.cpu_count())


    #! Set Validation Dataset
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        val_dataset = torchvision.datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)
        return DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=multiprocessing.cpu_count())




if '__main__' == __name__:
    trainer = LightningTrainer(max_epochs=EPOCHS, fast_dev_run=False)   #, deterministic=True
    model = MyLitNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, CLASSES)
    trainer.fit(model)  # Tensorflowa benzer bir yapı