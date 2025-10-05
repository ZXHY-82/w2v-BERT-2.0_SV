# encoding: utf-8
import os, sys, argparse, traceback
sys.path.append('../..')
sys.path.append('../../deeplab/pretrained/audio2vector/module/transformers/src')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import tempfile
from sklearn.metrics import accuracy_score
from deeplab.core.trainer import Trainer
from deeplab.metric.eer import get_eer
from deeplab.utils.fileio import save_trial
from local.dataset import Train_Dataset, Valid_Dataset
from local.sampler import WavBatchSampler


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
        self.embd_dim = hparams['embd_dim']
        self.dtype = torch.bfloat16 if hparams['use_amp'] else torch.float32

        self.logging_config = {
            'loss':dict(decimal=4, visible=True),
            'lr':dict(decimal=8, visible=True, instant=True),
            'acc':dict(decimal=4, visible=True),
            'eer':dict(decimal=4, visible=True),
            }

        num_spks = len(self.train_dataset.spk_ids)
        num_utts = len(self.train_dataset.utt_list)
        if hparams['speed_perturbation'] is not None:
            num_spks += num_spks * len(hparams['speed_perturbation'])

        assert len(self.modules['classifier'])==num_spks, 'Length of classifier must be equal to: ({}).'.format(num_spks)
        self.print('INFO: Num. Spks: {}'.format(num_spks))
        self.print('INFO: Num. Utts: {}'.format(num_utts))
        if hparams['speed_perturbation'] is not None:
            self.print('INFO: Speed Perturbation: ', hparams['speed_perturbation'])
       

    def compute_forward(self, inputs, stage):
        emb_output = self.modules['spk_model'](inputs['aud_inputs'])
        
        if len(emb_output.shape) == 3:
            bsz, seqlen, _ = emb_output.shape
            emb_output = emb_output.reshape(bsz * seqlen, -1)
            inputs['spk_labels'] = inputs['spk_labels'].unsqueeze(1).repeat(1, seqlen).reshape(bsz * seqlen)

        spk_output = self.modules['classifier'](input=emb_output, label=inputs['spk_labels'])
        predictions = {'emb_output':emb_output, 'spk_output':spk_output}

        return predictions

    
    def loss_fn(self, inputs, predictions, stage):
        loss = F.cross_entropy(input=predictions['spk_output'], target=inputs['spk_labels'])
        
        return dict(loss=loss)
        
        
    def eval_fn(self, inputs, predictions):   
        spk_true = inputs['spk_labels'].detach().cpu().tolist()
        spk_pred = predictions['spk_output'].argmax(dim=-1).detach().cpu().tolist()
        acc = accuracy_score(y_true=spk_true, y_pred=spk_pred)
        
        return dict(lr=self.optimizer.param_groups[0]['lr'], acc=acc)
        

    def validate_once(self, epoch_idx):
        valid_logs = dict()
        for module in self.modules.values():
            module.eval()
        if self.is_distributed:
            self.valid_dataloader.batch_sampler.set_epoch(epoch_idx)

        valid_embd = torch.zeros(len(self.valid_dataset.scp_list), self.embd_dim).to(self.device)
        with torch.autocast('cuda', self.dtype):
            with torch.no_grad():
                for inputs in self.valid_dataloader:
                    inputs = self.scatter_data(inputs) 
                    utt_idx = inputs['utt_labels'].item()
                    emb_output = self.modules['spk_model'](inputs['aud_inputs'])
                    if len(emb_output.shape) == 2:
                        valid_embd[utt_idx] = emb_output[0]
                    else:
                        valid_embd[utt_idx] = emb_output[0, -1]
                    
        if self.is_distributed:
            dist.all_reduce(valid_embd, op=dist.ReduceOp.SUM)
            dist.barrier()

        utt2embd = {}
        for utt_idx in range(len(self.valid_dataset.scp_list)):
            k = self.valid_dataset.scp_list[utt_idx]['reco']
            v = valid_embd[utt_idx].unsqueeze(0).float().detach().cpu().numpy()
            utt2embd[k] = v

        with tempfile.TemporaryDirectory() as temp_dir:
            trial_path = os.path.join(temp_dir, 'valid.trial')
            save_trial(trial_path, self.valid_dataset.trial_list)
            eer = get_eer(utt2embd, trial_path)[0]
            valid_logs = self.update_logs(valid_logs, dict(lr=self.optimizer.param_groups[0]['lr'],eer=eer))
                
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






