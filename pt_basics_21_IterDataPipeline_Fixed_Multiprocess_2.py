# => Data Pipeline
# https://www.bitswired.com/en/blog/post/introduction-to-torchdata-the-best-way-to-load-data-in-pytorch

import math
import torch
from torch.utils.data import DataLoader, IterableDataset


class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        return iter(range(self.start, self.end))
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=0, end=6)



# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


# Mult-process loading with the custom `worker_init_fn`
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))