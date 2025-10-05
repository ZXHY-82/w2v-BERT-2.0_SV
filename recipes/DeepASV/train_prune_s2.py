import os, sys, argparse, traceback
sys.path.append('../..')
sys.path.append('../../deeplab/pretrained/audio2vector/module/transformers/src')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import accuracy_score
from deeplab.core.trainer import Trainer
from local.dataset import Train_Dataset, Valid_Dataset
from local.sampler import WavBatchSampler
from tqdm import tqdm

class LocalTrainer(Trainer):

    def prep(self, hparams):
        self.train_dataset = Train_Dataset(hparams)
        self.valid_dataset = Valid_Dataset(hparams)

        self.train_batch_sampler = WavBatchSampler(
            self.train_dataset, 
            hparams['dur_range'], 
            shuffle=True, 
            batch_size=hparams['batch_size'], 
            drop_last=True, 
            distributed=self.is_distributed,
            )
        self.valid_batch_sampler = WavBatchSampler(
            self.valid_dataset,
            shuffle=False,
            batch_size=hparams['valid_batch_size'],
            drop_last=False,
            distributed=self.is_distributed,
            )

        self.modules = hparams['modules']
        self.warmup_steps = hparams['warmup_steps']
        self.target_sparsity = hparams['target_sparsity']
        self.cur_steps = 0
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.l1_loss = nn.L1Loss()

        self.dtype = torch.bfloat16 if hparams['use_amp'] else torch.float32

        self.logging_config = {
            'loss_l1':dict(decimal=4, visible=True),
            'loss_cos':dict(decimal=4, visible=True),
            'lr':dict(decimal=8, visible=True, instant=True),
            'val_loss_l1':dict(decimal=4, visible=True),
            'val_loss_cos':dict(decimal=4, visible=True),
            'val_lr':dict(decimal=8, visible=True, instant=True),
            }

        num_spks = len(self.train_dataset.spk_ids)
        num_utts = len(self.train_dataset.utt_list)
        if hparams['speed_perturbation'] is not None:
            num_spks += num_spks * len(hparams['speed_perturbation'])

        self.print('INFO: Num. Spks: {}'.format(num_spks))
        self.print('INFO: Num. Utts: {}'.format(num_utts))
        if hparams['speed_perturbation'] is not None:
            self.print('INFO: Speed Perturbation: ', hparams['speed_perturbation'])

    def get_target_sparsity(self):
        if self.cur_steps >= self.warmup_steps:
            self.cur_steps += 1
            return self.target_sparsity
        self.cur_steps += 1
        return self.target_sparsity * (self.cur_steps / self.warmup_steps)

    def compute_forward(self, inputs, stage):
        x_teacher, x_student = self.modules['spk_model'](inputs['aud_inputs'])
        predictions = {'x_teacher':x_teacher, 'x_student':x_student}
        return predictions

    
    def loss_fn(self, inputs, predictions, stage):
        loss_l1 = self.l1_loss(predictions['x_student'], predictions['x_teacher'])
        loss_cos = -self.cos_sim(predictions['x_student'], predictions['x_teacher']).mean()       
        return dict(loss_l1=loss_l1, loss_cos=loss_cos)
        
        
    def eval_fn(self, inputs, predictions):           
        return dict(lr=self.optimizer.param_groups[0]['lr'])
        

    def validate_once(self, epoch_idx):
        valid_logs = dict()
        for module in self.modules.values():
            module.eval()
        if self.is_distributed:
            self.valid_dataloader.batch_sampler.set_epoch(epoch_idx)

        l1_list = [] 
        cos_list = []

        with torch.autocast('cuda', self.dtype):
            with torch.no_grad():
                for inputs in tqdm(self.valid_dataloader):
                    inputs = self.scatter_data(inputs) 
                    x_teacher, x_student = self.modules['spk_model'](inputs['aud_inputs'])
                    loss_l1 = self.l1_loss(x_student, x_teacher)
                    loss_cos = -self.cos_sim(x_student, x_teacher).mean()
                    l1_list.append(loss_l1)
                    cos_list.append(loss_cos)


        if self.is_distributed:
            gathered_l1 = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_l1, l1_list)
            merged_valid_l1 = []
            for l1 in gathered_l1:
                merged_valid_l1.extend(l1)
            gathered_cos = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_cos, cos_list)
            merged_valid_cos = []
            for cos in gathered_cos:
                merged_valid_cos.extend(cos)

        l1 = torch.mean(torch.stack(l1_list))
        cos = torch.mean(torch.stack(cos_list))


        valid_logs = self.update_logs(valid_logs, dict(val_lr=self.optimizer.param_groups[0]['lr'],
                                                       val_loss_l1=l1, val_loss_cos=cos))
                    
        return valid_logs

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_distributed", default=False, type=bool)
    parser.add_argument("--yaml", type=str, default='')
    parser.add_argument("--pretrain", type=str, default='')
    parser.add_argument("--tag", type=str, default='')
    args = parser.parse_args()
    
    try:
        trainer = LocalTrainer(
            local_rank=int(os.environ["LOCAL_RANK"]) if args.is_distributed else -1,
            is_distributed=args.is_distributed,
            yaml_path=args.yaml,
            exps_tag=args.tag,
        )

        if args.pretrain:
            trainer.load_checkpoints(args.pretrain)
            
        trainer.fit()
        
    except Exception:
        print(traceback.format_exc())

    finally:
        if args.is_distributed:
            dist.destroy_process_group()






