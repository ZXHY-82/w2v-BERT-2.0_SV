import math
import torch

def WarmupLR_withStepDecay(optimizer, warmup_step, decay_step, gamma=0.1):
    """
    创建一个带有预热和步长衰减的学习率调度器
    
    Args:
        optimizer (torch.optim.Optimizer): 优化器对象
        warmup_step (int): 预热步数，在预热阶段学习率线性增加
        decay_step (int): 衰减步数，每经过decay_step步，学习率乘以gamma；如果为0，则预热后学习率保持不变
        gamma (float, optional): 衰减系数，默认为0.1
        
    Returns:
        torch.optim.lr_scheduler.LambdaLR: 学习率调度器对象
    """
    def lr_lambda(cur_step):
        if cur_step < warmup_step:
            factor = (cur_step+1) / (warmup_step+1)
        else:
            if decay_step > 0:
                factor = gamma ** ((cur_step-warmup_step) // decay_step)
            else:
                factor = 1.0

        return factor
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return scheduler



def WarmupLR_withStepDecay_groups(optimizer, 
                            warmup_steps: list, 
                            decay_steps: list, 
                            gammas: list):
    """
    创建支持不同 param group 使用不同 warmup/decay 策略的学习率调度器

    Args:
        optimizer (torch.optim.Optimizer): 优化器对象
        warmup_steps (List[int]): 每个参数组的预热步数
        decay_steps (List[int]): 每个参数组的衰减步数
        gammas (List[float]): 每个参数组的衰减系数

    Returns:
        torch.optim.lr_scheduler.LambdaLR: 学习率调度器对象
    """

    assert len(optimizer.param_groups) == len(warmup_steps) == len(decay_steps) == len(gammas), \
        "Length of warmup_steps, decay_steps, and gammas must match number of param groups"

    def get_lambda_fn(warmup_step, decay_step, gamma):
        def lr_lambda(cur_step):
            if cur_step < warmup_step:
                return (cur_step + 1) / (warmup_step + 1)
            elif decay_step > 0:
                return gamma ** ((cur_step - warmup_step) // decay_step)
            else:
                return 1.0
        return lr_lambda

    # 为每个 param group 创建对应的调度策略
    lr_lambdas = [
        get_lambda_fn(warmup, decay, gamma)
        for warmup, decay, gamma in zip(warmup_steps, decay_steps, gammas)
    ]

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)
    return scheduler


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        min_lr,
        max_lr,
        warmup_epoch,
        fix_epoch,
        step_per_epoch,
    ):
        self.optimizer = optimizer
        assert min_lr <= max_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_step = warmup_epoch * step_per_epoch
        self.fix_step = int(fix_epoch * step_per_epoch)
        self.current_step = 0.0

    def set_lr(self,):
        new_lr = self.clr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        new_lr = self.set_lr()
        self.current_step += 1
        return new_lr

    def clr(self, step):
        if step < self.warmup_step:
            # return self.min_lr + (self.max_lr - self.min_lr) * \
                # (step / self.warmup_step)
            return self.max_lr * (step / self.warmup_step)
        elif step >= self.warmup_step and step < self.fix_step:
            # warmup and cosine decrease
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                (1 + math.cos(math.pi * (step - self.warmup_step) /
                (self.fix_step - self.warmup_step)))
        else:
            return self.min_lr



import math

class WarmupStepDecayScheduler:
    def __init__(
        self,
        optimizer,
        max_lr,
        warmup_epoch,
        decay_epoch,
        gamma,
        add_factor,
        step_per_epoch,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_epoch = warmup_epoch
        self.warmup_step = warmup_epoch * step_per_epoch
        self.decay_epoch = decay_epoch
        self.step_per_epoch = step_per_epoch
        self.gamma = gamma
        self.add_factor = add_factor
        self.current_step = 0

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, step=None):
        if step is not None:
            self.current_step = step

        new_lr = self.clr(self.current_step)
        self.set_lr(new_lr)

        self.current_step += self.add_factor
        return new_lr

    def clr(self, step):
        # 当前是第几个 epoch
        epoch = step // self.step_per_epoch - self.warmup_epoch

        if step < self.warmup_step:
            # warmup 线性增长
            return self.max_lr * (step / self.warmup_step)
        else:
            # warmup 完成后，每 decay_epoch 个 epoch 衰减 0.1
            decay_factor = epoch // self.decay_epoch
            return self.max_lr * (self.gamma ** decay_factor)



    
