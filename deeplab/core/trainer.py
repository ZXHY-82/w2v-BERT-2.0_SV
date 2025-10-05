import os
import time
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import ExitStack
from hyperpyyaml import load_hyperpyyaml
from deeplab.utils.misc import second_to_timeformat, seed_worker, count_model_parameters
from deeplab.utils.fileio import init_output_dir, read_json, save_json
import warnings
warnings.filterwarnings('ignore')


class Trainer():
    """
    训练器类，用于管理深度学习模型的训练过程
    
    主要功能：
    1. 初始化训练环境（分布式训练、混合精度训练等）
    2. 管理数据加载、模型训练和验证
    3. 处理学习率调度、梯度累积等训练策略
    4. 记录训练日志和指标
    """
    def __init__(
        self,    
        local_rank, 
        is_distributed,
        yaml_path,
        exps_tag,
        ):
        self.local_rank = local_rank
        self.is_distributed = is_distributed
        self.init_logs_list = []
        self.init_epoch_idx = 0
        self.exps_tag = exps_tag

        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl') 
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.device = torch.device('cuda')

        with open(yaml_path, 'r') as f:
            self.yaml_strings = f.read()
            self.hparams = load_hyperpyyaml(self.yaml_strings)
            self.print('INFO: Load hparams from: {}'.format(yaml_path))

        if self.hparams['use_amp']:
            self.amp_scaler = torch.cuda.amp.GradScaler()
            self.print('INFO: Using mixed-precision training')

        if self.hparams['use_gradient_clipping']:
            self.print('INFO: Using gradient clipping: max_norm=1 norm_type=2')

        if self.hparams['cudnn_benchmark']:
            torch.backends.cudnn.benchmark = True
            self.print('INFO: Using torch.backends.cudnn.benchmark=True')
            
        if 'find_unused_parameters' in self.hparams:
            self.find_unused_parameters = self.hparams['find_unused_parameters'] 
        else:
            self.find_unused_parameters = False

        if 'wandb_cfgs' in self.hparams:
            self.use_wandb = True
        else:
            self.use_wandb = False

        self.iters_to_accumulate = self.hparams['gradient_accumulation']
        self.print('INFO: Using gradient accumulation: {}'.format(self.iters_to_accumulate))

        self.max_iters_per_epoch = self.hparams['max_iters_per_epoch'] 
        if self.max_iters_per_epoch:
            self.print('INFO: Max iterations per training epoch: {}'.format(self.max_iters_per_epoch))

        start_time = time.perf_counter()
        self.prep(self.hparams)
        self.print('INFO: Preparation Time: ', second_to_timeformat(time.perf_counter()-start_time))

        self.print('INFO: Stats of Model Parameters')
        for k,v in self.modules.items():
            params = count_model_parameters(v)
            self.print('      {}: {:.4f}/M'.format(k, params/1e6))


    def initialize_training(self):
        if not hasattr(self, 'train_sampler') and not hasattr(self, 'train_batch_sampler'):
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)  if self.is_distributed else None 
        if not hasattr(self, 'valid_sampler') and not hasattr(self, 'valid_batch_sampler'): 
            self.valid_sampler = DistributedSampler(self.valid_dataset, shuffle=False) if self.is_distributed else None

        if hasattr(self, 'train_sampler'):
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=self.hparams['batch_size'],
                shuffle=bool(self.train_sampler is None),
                sampler=self.train_sampler,
                num_workers=self.hparams['num_workers'],
                collate_fn=self.collate_fn if hasattr(self, 'collate_fn') else None,
                worker_init_fn=seed_worker,
                pin_memory=True,
                )
        elif hasattr(self, 'train_batch_sampler'):
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset, 
                batch_sampler=self.train_batch_sampler,
                num_workers=self.hparams['num_workers'],
                collate_fn=self.collate_fn if hasattr(self, 'collate_fn') else None,
                worker_init_fn=seed_worker,
                pin_memory=True,
                )

        if hasattr(self, 'valid_sampler'):
            self.valid_dataloader = DataLoader(
                dataset=self.valid_dataset, 
                batch_size=self.hparams['valid_batch_size'] if 'valid_batch_size' in self.hparams else self.hparams['batch_size'],
                shuffle=False,
                sampler=self.valid_sampler,
                num_workers=self.hparams['num_workers'],
                collate_fn=self.collate_fn if hasattr(self, 'collate_fn') else None,
                worker_init_fn=seed_worker,
                )
        elif hasattr(self, 'valid_batch_sampler'):
            self.valid_dataloader = DataLoader(
                dataset=self.valid_dataset, 
                batch_sampler=self.valid_batch_sampler,
                num_workers=self.hparams['num_workers'],
                collate_fn=self.collate_fn if hasattr(self, 'collate_fn') else None,
                worker_init_fn=seed_worker,
                )

        if not self.is_distributed:
            for k,v in self.modules.items():
                if self.is_trainable_module(v):
                    self.modules[k] = nn.DataParallel(v.cuda())
                else:
                    self.modules[k] = v.cuda()
        else:
            for k,v in self.modules.items():
                if self.is_trainable_module(v):
                    v = nn.SyncBatchNorm.convert_sync_batchnorm(v)
                    self.modules[k] = DDP(
                        v.to(self.local_rank), 
                        device_ids=[self.local_rank], 
                        output_device=self.local_rank, 
                        find_unused_parameters=self.find_unused_parameters,
                        )
                else:
                    self.modules[k] = v.to(self.local_rank)

        if 'prune_opt' in self.hparams:
            pgs = [
                {
                    'params': [p for n, p in self.modules['spk_model'].module.student_front.named_parameters() if "log_alpha" not in n],
                    'lr': self.hparams['prune_opt']['distill_lr'],
                    'weight_decay': 0.0,
                    'name': 'main_params',
                },
            ]
            if self.hparams['prune_opt']['reg_lr'] is not None:
                pgs.append({
                    'params': [p for n, p in self.modules['spk_model'].module.student_front.named_parameters() if "log_alpha" in n],
                    'lr': self.hparams['prune_opt']['reg_lr'],
                    'weight_decay': 0.0,
                    'name': 'log_alpha',
                })
                pgs.append({
                    'params': [p for n, p in self.modules['lambda'].named_parameters()],
                    'lr': -self.hparams['prune_opt']['reg_lr'],
                    'weight_decay': 0.0,
                    'name': 'lambda',
                })
            self.optimizer = torch.optim.AdamW(pgs)
        else:
            trainable_params = []
            for module in self.modules.values():
                trainable_params += list(filter(lambda p:p.requires_grad, module.parameters()))
            self.optimizer = self.hparams['optimizer'](params=trainable_params)


        if 'scheduler_lmft' in self.hparams:
            self.scheduler_lmft = self.hparams['scheduler_lmft'](optimizer=self.optimizer, step_per_epoch=len(self.train_dataloader))
        else:
            self.scheduler_lmft = None

        if 'scheduler' in self.hparams:
            self.scheduler = self.hparams['scheduler'](optimizer=self.optimizer)
        else:
            self.scheduler = None

        
    def fit(self):
        self.initialize_training()

        if not self.is_distributed or dist.get_rank()==0:
            timestamp = time.strftime('%y%m%d%H%M%S', time.localtime(time.time())) 
            ckpts_dir = os.path.join(self.hparams['output_dir'], 'checkpoints', self.exps_tag+timestamp)
        else:
            ckpts_dir = None 
        if self.is_distributed:
            ckpts_dir = self.sync_string(ckpts_dir, src=0)
        logs_path = os.path.join(ckpts_dir, 'logs.json')
        yaml_path = os.path.join(ckpts_dir, 'train.yaml')
        self.ckpts_dir = ckpts_dir

        if not self.is_distributed or dist.get_rank()==0:
            init_output_dir(yaml_path)
            with open(yaml_path, 'w') as f:
                f.write(self.yaml_strings)

            if self.use_wandb:
                wandb.init(
                    project=self.hparams['wandb_cfgs']['project'], 
                    name=self.exps_tag+timestamp,
                    )
                if self.hparams['wandb_cfgs']['watch']:
                    wandb.watch(
                        models=list(self.modules.values()), 
                        log=self.hparams['wandb_cfgs']['log'], 
                        log_freq=self.hparams['wandb_cfgs']['log_freq'],
                        )

        logs_list = self.init_logs_list
        epoch_idx = self.init_epoch_idx
        for epoch_idx in range(self.init_epoch_idx+1, self.init_epoch_idx+self.hparams['num_epochs']+1):
            logs = self.train_one_epoch(epoch_idx, ckpts_dir, logs_list)
            logs_list.append(logs)
            ckpt_path = os.path.join(ckpts_dir,'ckpt_{}.pth'.format(str(epoch_idx).zfill(4)))
            
            if not self.is_distributed or dist.get_rank()==0:
                save_json(logs_path, logs_list)
                ckpt_data = dict(modules={}, epoch_idx=epoch_idx)
                for k,v in self.modules.items():
                    if isinstance(v, (nn.DataParallel, DDP)):
                        ckpt_data['modules'][k] = v.module.state_dict()
                    else:
                        ckpt_data['modules'][k] = v.state_dict()
                torch.save(ckpt_data, ckpt_path)

            if self.is_distributed: 
                dist.barrier()

            if hasattr(self, 'exec_training_plan'):
                exec_data = self.exec_training_plan(logs['epoch_idx'], logs['train_logs'], logs['valid_logs'], ckpt_path) 
                if (exec_data is not None) and (not self.is_distributed or dist.get_rank()==0):
                    with open(os.path.join(ckpts_dir,'exec.txt'), 'a+') as f:
                        lines = [line+'\n' for line in exec_data]
                        f.writelines(lines)

                if self.is_distributed: 
                    dist.barrier()


    def train_one_epoch(self, epoch_idx, ckpts_dir, logs_list):
        start_time = time.perf_counter()
        train_logs = dict()
        for module in self.modules.values():
            module.train()
        
        if self.is_distributed:
            if hasattr(self, 'train_sampler'):
                self.train_dataloader.sampler.set_epoch(epoch_idx)
            elif hasattr(self, 'train_batch_sampler'):
                self.train_dataloader.batch_sampler.set_epoch(epoch_idx)

        for iter_idx, inputs in enumerate(self.train_dataloader, start=1):
            inputs = self.scatter_data(inputs)
            
            if self.hparams['use_amp']:
                with ExitStack() as stack:
                    if self.is_distributed and iter_idx%self.iters_to_accumulate!=0:
                        contexts = [stack.enter_context(module.no_sync()) for module in self.modules.values()]
                    with torch.amp.autocast('cuda', torch.bfloat16):
                        predictions = self.compute_forward(inputs, stage='train')
                        loss_dict = self.loss_fn(inputs, predictions, stage='train')
                        loss = sum(loss_dict.values()) / self.iters_to_accumulate # loss regularization
                    self.amp_scaler.scale(loss).backward()

                if iter_idx % self.iters_to_accumulate == 0:
                    if self.hparams['use_gradient_clipping']:
                        self.amp_scaler.unscale_(self.optimizer) 
                        self.clip_gradient()
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()  
                    self.optimizer.zero_grad()
                    
            else: 
                with ExitStack() as stack:
                    if self.is_distributed and iter_idx%self.iters_to_accumulate!=0:
                        contexts = [stack.enter_context(module.no_sync()) for module in self.modules.values()]
                    predictions = self.compute_forward(inputs, stage='train')
                    loss_dict = self.loss_fn(inputs, predictions, stage='train')
                    loss = sum(loss_dict.values()) / self.iters_to_accumulate # loss regularization
                    loss.backward()
                
                if iter_idx % self.iters_to_accumulate == 0:
                    if self.hparams['use_gradient_clipping']:
                        self.clip_gradient()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
            train_logs = self.update_logs(train_logs, loss_dict)
            train_logs = self.update_logs(train_logs, self.eval_fn(inputs, predictions))
            used_time = time.perf_counter() - start_time
            self.output_logs(epoch_idx, iter_idx, used_time, train_logs)

            # lmft
            if iter_idx%self.iters_to_accumulate==0 and self.scheduler_lmft:
                self.scheduler_lmft.step()
            
            # items_save
            if self.hparams['items_save'] and iter_idx % self.hparams['item_save_steps'] == 0:
                valid_logs = self.validate_once(epoch_idx)
                if self.is_distributed:
                    valid_logs = self.gather_logs(valid_logs)
                used_time = time.perf_counter() - start_time
                self.output_logs(epoch_idx, iter_idx, used_time, train_logs, valid_logs)

                ckpt_path = os.path.join(ckpts_dir,'ckpt_{}_{}item.pth'.format(str(epoch_idx).zfill(4), str(iter_idx)))
                if not self.is_distributed or dist.get_rank()==0:
                    logs_path = os.path.join(ckpts_dir, 'logs.json')
                    logs_list.append({'epoch_idx':epoch_idx, 'items':iter_idx, 'valid_logs':valid_logs}) 
                    save_json(logs_path, logs_list)
                    ckpt_data = dict(modules={}, epoch_idx=epoch_idx)
                    for k,v in self.modules.items():
                        if isinstance(v, (nn.DataParallel, DDP)):
                            ckpt_data['modules'][k] = v.module.state_dict()
                        else:
                            ckpt_data['modules'][k] = v.state_dict()

                    torch.save(ckpt_data, ckpt_path)

                if self.is_distributed: 
                    dist.barrier()


            if self.use_wandb and self.hparams['wandb_cfgs']['watch']:
                if not self.is_distributed or dist.get_rank()==0:
                    wandb.log(dict(running_loss=loss))
                            
            if self.max_iters_per_epoch and iter_idx>=self.max_iters_per_epoch:
                break
                
        valid_logs = self.validate_once(epoch_idx)
        if self.scheduler:
            self.scheduler.step()

        if self.is_distributed:
            train_logs = self.gather_logs(train_logs)
            valid_logs = self.gather_logs(valid_logs)
        used_time = time.perf_counter() - start_time
        self.output_logs(epoch_idx, iter_idx, used_time, train_logs, valid_logs)

        for k,v in train_logs.items():
            train_logs[k] = float(np.mean(v))
        for k,v in valid_logs.items(): 
            valid_logs[k] = float(np.mean(v))
        if self.use_wandb:
            if not self.is_distributed or dist.get_rank()==0:
                wandb.log(valid_logs)

        logs = {'epoch_idx':epoch_idx, 'train_logs':train_logs, 'valid_logs':valid_logs}

        return logs
    
    
    def validate_once(self, epoch_idx):
        if hasattr(self, 'exec_before_valid'):
            self.exec_before_valid(epoch_idx)         
            if self.is_distributed: 
                dist.barrier() 

        valid_logs = dict()
        for module in self.modules.values():
            module.eval()
        
        if self.is_distributed:
            if hasattr(self, 'valid_sampler'):
                self.valid_dataloader.sampler.set_epoch(epoch_idx)
            elif hasattr(self, 'valid_batch_sampler'):
                self.valid_dataloader.batch_sampler.set_epoch(epoch_idx)
            
        with torch.no_grad():
            for iter_idx, inputs in enumerate(self.valid_dataloader, start=1):
                inputs = self.scatter_data(inputs)
                predictions = self.compute_forward(inputs, stage='valid')
                loss_dict = self.loss_fn(inputs, predictions, stage='valid')
                loss = sum(loss_dict.values()) 
                valid_logs = self.update_logs(valid_logs, loss_dict)
                valid_logs = self.update_logs(valid_logs, self.eval_fn(inputs, predictions))
        
        return valid_logs


    def load_checkpoints(self, ckpt_path):
        ckpt_data = torch.load(ckpt_path, map_location=self.device)
        self.print('INFO: Loaded checkpoints from: {}'.format(ckpt_path))
        # load modules 
        for key, module in self.modules.items():
            if key not in ckpt_data['modules']:
                self.print('      {}: <Not founded in checkpoint data>'.format(key)) 
        
            elif key == 'classifier':
                curr_state_dict = module.state_dict()
                ckpt_state_dict = ckpt_data['modules'][key]
                curr_len = len(curr_state_dict['weight'])
                ckpt_len = len(ckpt_state_dict['weight'])
                if curr_len == ckpt_len:
                    module.load_state_dict(ckpt_state_dict)
                    self.print('      {}: <All weights matched>'.format(key)) 
                elif ckpt_len > curr_len:
                    curr_state_dict['weight'] = ckpt_data['modules'][key]['weight'][:curr_len]
                    module.load_state_dict(curr_state_dict)
                    self.print('      {}: <LMFT All weights matched>'.format(key)) 
                else:
                    curr_state_dict['weight'][:ckpt_len] = ckpt_data['modules'][key]['weight']
                    module.load_state_dict(curr_state_dict)
                    self.print('      {}: <Partial weights matched>'.format(key)) 

            else:
                curr_state_dict = module.state_dict()
                ckpt_state_dict = ckpt_data['modules'][key]
                mismatched = False
                for k in curr_state_dict.keys():
                    if k in ckpt_state_dict and curr_state_dict[k].shape==ckpt_state_dict[k].shape:
                        curr_state_dict[k] = ckpt_state_dict[k] 
                    else:
                        mismatched = True
                module.load_state_dict(curr_state_dict)
                if mismatched:
                    self.print('      {}: <Partial weights matched>'.format(key)) 
                else:
                    self.print('      {}: <All weights matched>'.format(key)) 

        self.init_epoch_idx = ckpt_data['epoch_idx']
        self.init_logs_list = read_json(os.path.join(os.path.dirname(ckpt_path), 'logs.json'))[:self.init_epoch_idx]
        if self.is_distributed:
            dist.barrier()


    def is_trainable_module(self, module):
        for _, param in module.named_parameters():
            if param.requires_grad:
                return True

        return False


    def scatter_data(self, batch_data):
        for k,v in batch_data.items():
            batch_data[k] = v.to(self.device, non_blocking=True)

        return batch_data
    
    
    def clip_gradient(self):
        for module in self.modules.values():
            nn.utils.clip_grad_norm_(parameters=module.parameters(), max_norm=1.0, norm_type=2.0)


    def sync_string(self, string, src=0):
        object_list = [string] if dist.get_rank() == src else [None]
        dist.broadcast_object_list(object_list, src=src)

        return object_list[0]


    def update_logs(self, logs, item_dict):
        for k,v in item_dict.items():
            if k not in logs:
                logs[k] = []
            if isinstance(v, torch.Tensor):
                v = v.item()
            logs[k].append(v)
        
        return logs

   
    def collect_tensor(self, tensor):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        return torch.cat(output_tensors, dim=0)


    def gather_logs(self, logs):
        dist.barrier()
        for k,v in logs.items():
            instant = self.logging_config[k]['instant'] if 'instant' in self.logging_config[k] else False
            v = v[-1] if instant else np.mean(v)
            tensor = torch.Tensor([v]).to(self.device)
            tensor = self.collect_tensor(tensor).mean()
            logs[k] = [tensor.item()]

        return logs


    def output_logs(self, epoch_idx, iter_idx, used_time, train_logs, valid_logs=None):
        used_time = second_to_timeformat(used_time) 
        train_metas = ['Train ']
        for k,v in train_logs.items():
            assert k in self.logging_config, 'Item ({}) has not been set in Logging Config.'.format(k)
            if self.logging_config[k]['visible']:
                decimal = self.logging_config[k]['decimal']
                instant = self.logging_config[k]['instant'] if 'instant' in self.logging_config[k] else False
                v = v[-1] if instant else np.mean(v)
                train_metas.append(('{}:{:.%df} '%decimal).format(k, v))
                 
        if valid_logs is None:
            text = "\rE:{:3d} ({}/{}) | {}| {}".format(
                epoch_idx, iter_idx, len(self.train_dataloader), "".join(train_metas), used_time)
            self.print(text, end='')
            
        else:
            valid_metas = ['Valid ']
            for k,v in valid_logs.items():
                assert k in self.logging_config, 'Item ({}) has not been set in Logging Config.'.format(k)
                if self.logging_config[k]['visible']:
                    decimal = self.logging_config[k]['decimal']
                    valid_metas.append(('{}:{:.%df} '%decimal).format(k, np.mean(v)))
            text = "\rE:{:3d} ({}/{}) | {}| {}| {}".format(
                epoch_idx, iter_idx, len(self.train_dataloader), "".join(train_metas), "".join(valid_metas), used_time)
            self.print(text, end='\n')


    def print(self, content, *args, **kws):
        if not self.is_distributed or dist.get_rank()==0:
            print(content, *args, **kws)
            
        return content


        