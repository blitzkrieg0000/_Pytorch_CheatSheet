# => Data Pipeline
# https://www.bitswired.com/en/blog/post/introduction-to-torchdata-the-best-way-to-load-data-in-pytorch

import math
import torch
from torch.utils.data import DataLoader, IterableDataset


# IterableDataset oluşturun
class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))


if "__main__" == __name__:
    # DataLoader oluşturun
    my_dataset = MyIterableDataset(start=0, end=6)

    dataloader = DataLoader(my_dataset, batch_size=1, num_workers=2)

    # Verileri döngüye alın
    for batch in dataloader:
        print(batch)
