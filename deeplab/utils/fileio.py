import os
import numpy as np
import scipy.io.wavfile as sciwav
import soundfile as sf
import json
import torchaudio
from hyperpyyaml import load_hyperpyyaml, dump_hyperpyyaml


def init_output_dir(output_path):
    target_dir = os.path.split(output_path)[0]
    if len(target_dir) > 0 and (not os.path.exists(target_dir)):
        os.makedirs(target_dir, exist_ok=True)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f) 
    return data


def save_json(path, data):
    init_output_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    

def read_hyperyaml(path):
    with open(path, 'r') as f:
        data = load_hyperpyyaml(f.read())
    return data


def save_hyperyaml(path, yaml_data):
    init_output_dir(path)
    with open(path, 'w') as f:
        dump_hyperpyyaml(yaml_data, f)


def load_audio(path, sample_rate=16000, channels=0):
    signal, sr = sf.read(path)

    if sample_rate and sample_rate != sr:
        effects = [['remix', '1'], ['lowpass', f'{sample_rate//2}'], ['rate', f'{sample_rate}'],]
        signal, sr = torchaudio.sox_effects.apply_effects_file(path, effects)
        signal = signal.squeeze(0).numpy()

    if len(signal.shape)==2 and channels!='all':
        signal  = signal[:, channels]
    return signal, sr

def load_concatenated_audio_by_rttm(rttm_list, path, sr=16000, channels=0, min_dt=0):
    signal_list = []
    for rttm_data in rttm_list:
        if rttm_data['dt'] >= min_dt:
            p1 = int(rttm_data['st'] * sr)
            p2 = int(rttm_data['st'] * sr + rttm_data['dt'] * sr)
            signal = load_audio(path, sr, channels)[0][p1:p2]
            signal_list.append(signal)
    
    if len(signal_list) > 0:
        return np.concatenate(signal_list, axis=0)


    
def load_scp(path, sep='\t'):
    scp_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            scp_data = line.strip('\n').split(sep)
            scp_list.append(dict(reco=scp_data[0],wav_path=scp_data[1]))
    return scp_list

    
def save_scp(path, scp_list, sep='\t'):
    init_output_dir(path)
    with open(path, 'w') as f:
        for scp_data in scp_list:
            line = scp_data['reco'] + sep + scp_data['wav_path'] + '\n'
            f.writelines(line)


def load_trial(path):
    trial_list = []
    with open(path, 'r') as f:
        for line in f:
            key, utt1, utt2 = line.strip('\n').split(' ')
            trial_list.append(dict(key=key, utt1=utt1, utt2=utt2))
    return trial_list


def save_trial(path, trial_list):
    init_output_dir(path)
    with open(path, 'w') as f:
        for trial_data in trial_list:
            line = '{} {} {}\n'.format(trial_data['key'], trial_data['utt1'], trial_data['utt2'])
            f.writelines(line)





