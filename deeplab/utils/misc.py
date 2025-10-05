import os
import torch
import random
import numpy as np


def second_to_timeformat(seconds):
    m,s = divmod(seconds, 60)
    h,m = divmod(m, 60)   
 
    return "%dh:%02dm:%02ds" % (h, m, s)
    

def set_random_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_random_seed(worker_seed)


def trim_time_interval(t1, t2, min_t1, max_t2):

    if max(t1,t2)<min(min_t1,max_t2) or max(min_t1,max_t2)<min(t1,t2):
        return 0, 0

    trim_t1 = max(t1, min_t1)
    trim_t2 = min(t2, max_t2)

    return trim_t1, trim_t2

def count_model_parameters(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel() if p.requires_grad else 0
        
    return num_params