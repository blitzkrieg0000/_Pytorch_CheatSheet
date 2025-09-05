"""
    !=> Hedef: 
        => 
"""
import torch
import torch.nn as nn


# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

import torch
from torch.utils.data import DataLoader, Dataset

# Örnek bir veri kümesi oluştur
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Özel bir collate_fn işlevi tanımla
def custom_collate_fn(batch):
    # Verileri bir araya getir ve tensöre dönüştür
    data = torch.stack(batch, dim=0)
    return data

# Veri kümesini oluştur
data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
dataset = CustomDataset(data)

# DataLoader oluştur ve collate_fn'i tanımla
dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn, num_workers=4)

# DataLoader'ı kullanarak veri yükleme
for batch in dataloader:
    print(batch)


