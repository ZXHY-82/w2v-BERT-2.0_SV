import os, sys
sys.path.append('../../..')
sys.path.append('../../../deeplab/pretrained/audio2vector/module/transformers/src')

import torch
from deeplab.pretrained.audio2vector.api import AudioEncoder
import json

ckpt_path = '/work/zl389/workspace/LLM_ASV/publish_code/recipes/DeepASV/results/checkpoints/prune_s2/ckpt_0040.pth'
ckpt = torch.load(ckpt_path)
ckpt_data = ckpt['modules']['spk_model']

model = AudioEncoder(
    'facebook/w2v-bert-2.0',
    False,
    None,
    None,
    'config_prune_stu_0.8.json'
).encoder.eval()

student_params = sum( p.numel() for p in model.parameters()) / 1e6
print(student_params)

cur_state_dict = model.state_dict()
for k in cur_state_dict.keys():
    s_k = 'student_front.encoder.' + k
    if s_k in ckpt_data and cur_state_dict[k].shape == ckpt_data[s_k].shape:
        cur_state_dict[k] = ckpt_data[s_k]
    else:
        print(f'{k}_is_mismatch')
model.load_state_dict(cur_state_dict)

student_params = sum( p.numel() for p in model.parameters()) / 1e6
print(student_params)

ori_ckpt = torch.load('/work/zl389/workspace/LLM_ASV/publish_code/recipes/DeepASV/results/checkpoints/vox2_251005145628/ckpt_0002.pth')
encoder_ckpt = {}

cur_state_dict = model.state_dict()
for k in cur_state_dict.keys():
    new_k = 'front.encoder.' + k
    encoder_ckpt[new_k] = cur_state_dict[k]


ori_ckpt['modules']['spk_model'] = encoder_ckpt
torch.save(ori_ckpt, os.path.join(os.path.dirname(ckpt_path), 'prune_dis.pth'))


