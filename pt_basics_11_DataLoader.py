"""
    !=> Hedef: 
        => Torch kütüphanesini kullanarak veri yükleme işlemleri yapma

        => Tüm verisetini tek seferde yükleme işlemini yapmak için kullanılabilir.
          Böylece küçük veriseti ile çalışıyorsak; formatları .csv, .txt vb. ise kullanışlıdır.

        => Python Indexer yapısı ile bir class hazırlanır ve "torch.utils.data.Dataset" den kalıtılır.
            Daha sonra "torch.utils.data.Dataloader" ile veriyi işleme, karıştırma, batchlere ayırma gibi
            işlemleri kolaylıkla yönetebiliriz.
"""
import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


#! Prepare Data--------------------------------------------------------------------------------------------------------
class WineDataset(Dataset):
    def __init__(self):
        super().__init__()
        Datafile = np.loadtxt("./dataset/wine/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.DataX = torch.from_numpy(Datafile[:, 1:])    # İlk sütun hariç: İlk sütun class değerlerini belirtecek şekilde düzenlenmiş.
        self.DataY = torch.from_numpy(Datafile[:, [0]])   # class
        self.NSamples = Datafile.shape[0]


    # Python Indexer: x[0] => index:0
    def __getitem__(self, index):
        return self.DataX[index], self.DataY[index]


    # Python len(x) => length: 1000
    def __len__(self):
        return self.NSamples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=multiprocessing.cpu_count(), pin_memory=True) # Indexer Object
dataiter = iter(dataloader)         # Generator(make iterable) Object

#! Dummy Training Loop-------------------------------------------------------------------------------------------------------
EPOCHS=2
TOTAL_SAMPLES = len(dataset)
STEPS = math.ceil(TOTAL_SAMPLES/4)  # Iterations
print(TOTAL_SAMPLES, STEPS)


for epoch in range(EPOCHS):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward -> backward -> update

        if(i+1)%5 == 0:
            print(f"epoch: {epoch+1}/{EPOCHS}, step {i+1}/{STEPS}, inputs: {inputs.shape}")

