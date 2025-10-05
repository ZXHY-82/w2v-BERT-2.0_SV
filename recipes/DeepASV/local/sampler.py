from collections import defaultdict
import torch, random, copy
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class WavBatchSampler(object):
    def __init__(self, dataset, dur_range=None, spk_per_batch=32, shuffle=False, batch_size=1, 
                 drop_last=False, distributed=True):
        self.dur_range = dur_range
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        if distributed:
            self.sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequentialSampler(dataset)
          
    def _renew(self):
        if self.dur_range == None:
            return [], None
        else:
            return [], random.uniform(self.dur_range[0], self.dur_range[1])

    def __iter__(self):
        batch, dur = self._renew()
        for idx in self.sampler:
            if self.dur_range == None:
                batch.append((idx))
            else:
                batch.append((idx, dur))
            if len(batch) == self.batch_size:
                yield batch
                batch, dur = self._renew()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
     


        
