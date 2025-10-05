import os, sys
sys.path.append('..')
sys.path.append('../../..')
sys.path.append('../../../deeplab/pretrained/audio2vector/module/transformers/src')
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
import torch
from tqdm import tqdm
from deeplab.utils.fileio import read_hyperyaml, load_audio
import warnings
warnings.filterwarnings("ignore")

scp_path = '/work/zl389/workspace/LLM_ASV/data/test_vox/vox1-o/wav_copy.scp'
scp_list = [line.strip().split('\t') for line in open(scp_path)]

ckpt_path = '/work/zl389/workspace/LLM_ASV/recipes/DeepASV/results/audio2vec_new/checkpoints/exp_w2v_vb2_lmft_lora_adapter_mfa_merge_0.23_0.14/ckpt_0026_3500item.pth'


ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)

hparams = read_hyperyaml(path=os.path.join(os.path.dirname(ckpt_path), 'train.yaml'))
modules = hparams['modules']

for key, module in modules.items():
    if key == 'classifier':
        continue
    curr_state_dict = module.state_dict()
    ckpt_state_dict = ckpt_data['modules'][key]
    mismatched = False
    for k in curr_state_dict.keys():
        if k in ckpt_state_dict and curr_state_dict[k].shape==ckpt_state_dict[k].shape:
            curr_state_dict[k] = ckpt_state_dict[k] 
        else:
            mismatched = True
    module.load_state_dict(curr_state_dict)
    module = module.eval().cuda()

    if mismatched:
        print('      {}: <Partial weights matched>'.format(key)) 
    else:
        print('      {}: <All weights matched>'.format(key)) 


sr = hparams['sample_rate']
max_len = int(hparams['max_valid_dur'] * hparams['sample_rate'])


utt2embd = {}
dtype = torch.bfloat16 if hparams['use_amp'] else torch.float32
print(dtype)
with torch.autocast('cuda', dtype):
    with torch.no_grad():   
        for scp_data in tqdm(scp_list, ncols=80):
            utt = scp_data[0]
            if utt in utt2embd:
                print('Warning: duplicated utt key.')
            wav_path = scp_data[1]
            signal = load_audio(wav_path, sr)[0][:max_len]
            aud_inputs = torch.from_numpy(signal).float()
            utt2embd[utt] = modules['spk_model'](aud_inputs).float().detach().cpu().numpy()


from deeplab.metric.eer import get_eer
trial_path = '/work/zl389/workspace/LLM_ASV/data/test_vox/vox1-o/trials'
eer, threshold, _, _ = get_eer(utt2embd, trial_path)
print('EER: {:.4f}%'.format(eer*100))