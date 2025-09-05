# => Data Pipeline
# https://www.bitswired.com/en/blog/post/introduction-to-torchdata-the-best-way-to-load-data-in-pytorch


import torch
from torch.utils.data import IterableDataset, DataLoader

# Basit bir örnek veri kaynağı
data_source = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]


# DataPipe'i kullanarak veri işleme
def transform_example(item):
    return item * 2

# IterableDataset oluşturun
class MyIterableDataset(IterableDataset):
    def __init__(self, data_pipe, transform=None):
        self.data_pipe = data_pipe
        self.transform = transform

    def __iter__(self):
        for item in self.data_pipe:
            if self.transform:
                item = self.transform(item)
            yield item


if '__main__' == __name__:
    # DataLoader oluşturun
    my_dataset = MyIterableDataset(data_pipe=data_source, transform=transform_example)

    dataloader = DataLoader(my_dataset, batch_size=1, num_workers=2, multiprocessing_context="spawn")

    # Verileri döngüye alın
    for batch in dataloader:
        print(batch)
