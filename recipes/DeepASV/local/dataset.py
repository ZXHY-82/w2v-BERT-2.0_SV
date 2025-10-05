import os, sys 
sys.path.append(os.path.split(__file__)[0])
import torch
import numpy as np
from deeplab.utils.fileio import load_audio
from deeplab.utils.corpus import load_musan_dict, load_rirs
from deeplab.dataio.audio import truncate_audio_random, add_noise_from_musan_dict, add_reverberation, speed_augmentation
from data_pipe import prepare_scp_and_trial_list


class Train_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, hparams):
        super(torch.utils.data.Dataset, self).__init__()
        self.repeat = hparams['training_loop']
        self.sr = hparams['sample_rate']
        self.speed_perturbation = hparams['speed_perturbation']
        self.data_aug = hparams['data_aug']
        self.musan_dict = load_musan_dict(hparams['musan_path'])
        self.rirs_list  = load_rirs(hparams['rirs_path'])

        spk2utt = {}
        for group_id, data_loading_fn in enumerate(hparams['train_data']):
            for spk_id, utts in data_loading_fn().items():
                unique_key = '{}-{}'.format(group_id, spk_id)
                assert unique_key not in spk2utt, 'Duplicated keys has been detected.'
                spk2utt[unique_key] = utts
                
        self.spk2utt = {k:list(set(v)) for k,v in spk2utt.items()}
        self.spk_ids = sorted(list(self.spk2utt.keys()))
        self.spk_num = len(self.spk_ids)
        
        utt_list = []
        for idx, spk_id in enumerate(self.spk_ids):
            for utt in self.spk2utt[spk_id]:
                utt_data = dict(wav_path=utt, spk_label=idx, speed_shift=None)
                utt_list.append(utt_data)
                if self.speed_perturbation is not None:
                    for idx_shift, speed_shift in enumerate(self.speed_perturbation, start=1):
                        utt_data = dict(wav_path=utt, spk_label=idx+idx_shift*len(self.spk_ids), speed_shift=speed_shift)
                        utt_list.append(utt_data)
                        
        self.utt_list = sorted(utt_list, key=lambda x:x['spk_label'])
        
        
    def __len__(self):
        
        return len(self.utt_list) * self.repeat
    
    
    def __getitem__(self, idx_data):
        idx, dur = idx_data
        if len(self.utt_list) > 0:
            idx = idx % len(self.utt_list)
        
        utt_data = self.utt_list[idx]
        signal = load_audio(utt_data['wav_path'], self.sr)[0]
        signal = truncate_audio_random(signal, int(dur*self.sr))

        if utt_data['speed_shift'] is not None:    
            signal = speed_augmentation(signal, self.sr, utt_data['speed_shift'])
            
        if self.data_aug:
            aug_type = np.random.choice(['none', 'noise', 'reverb'])
            if aug_type == 'noise':
                signal = add_noise_from_musan_dict(signal, self.sr, self.musan_dict, prob=1.0, snr=[5,20])
            if aug_type == 'reverb':
                signal = add_reverberation(signal , self.sr, self.rirs_list, prob=1.0)
        
        aud_inputs = torch.from_numpy(signal).float()
        spk_labels = torch.tensor(utt_data['spk_label']).long()
    
        inputs = {'aud_inputs':aud_inputs, 
                  'spk_labels':spk_labels}
        
        return inputs


class Valid_Dataset(torch.utils.data.Dataset):
    

    def __init__(self, hparams):
        super(torch.utils.data.Dataset, self).__init__()
        self.sr = hparams['sample_rate']
        self.max_len = int(hparams['max_valid_dur'] * hparams['sample_rate'])
        
        self.scp_list = []
        self.trial_list = []
        for group_id, data_dict in enumerate(hparams['valid_data']):
            scp_list, trial_list = prepare_scp_and_trial_list(data_dict['scp_path'], data_dict['trial_path'], group_id)
            self.scp_list += scp_list
            self.trial_list += trial_list
        
        self.scp_list = sorted(self.scp_list, key=lambda x:x['reco'])
                
    
    def __len__(self):
        
        return len(self.scp_list)
    
    
    def __getitem__(self, idx): 
        scp_data = self.scp_list[idx]
        signal = load_audio(scp_data['wav_path'], self.sr)[0][:self.max_len]
        
        aud_inputs = torch.from_numpy(signal).float()
        utt_labels = torch.tensor(idx).long()
        
        inputs = {'aud_inputs':aud_inputs, 
                  'utt_labels':utt_labels
                  }
        
        return inputs

        