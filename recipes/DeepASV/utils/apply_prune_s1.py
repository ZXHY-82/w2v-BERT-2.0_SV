import os, sys
sys.path.append('../../..')
sys.path.append('../../../deeplab/pretrained/audio2vector/module/transformers/src')

import torch
from deeplab.pretrained.audio2vector.api import AudioEncoder
import json


ckpt_path = '/work/zl389/workspace/LLM_ASV/publish_code/recipes/DeepASV/results/checkpoints/prune_s1/ckpt_0020.pth'
ckpt = torch.load(ckpt_path)
ckpt_data = ckpt['modules']['spk_model']

model = AudioEncoder(
    'facebook/w2v-bert-2.0',
    False,
    None,
    None,
    'config_prune_stu.json'
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

config = model.prune()

student_params = sum( p.numel() for p in model.parameters()) / 1e6
print(student_params)

cur_state_dict = model.state_dict() # after prune 

for k in cur_state_dict.keys():
    s_k = 'student_front.encoder.' + k
    if s_k in ckpt_data:
        ckpt_data[s_k] = cur_state_dict[k]
    else:
        print(f'{k}_is_mismatch')
ckpt['modules']['spk_model'] = ckpt_data
torch.save(ckpt, os.path.join(os.path.dirname(ckpt_path), 'prune_update.pth'))

config_path = '/work/zl389/workspace/LLM_ASV/publish_code/deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/config_prune_stu.json'
with open(config_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for key in config.keys():
    data[key] = config[key]

data['prune'] = False

with open(os.path.join(os.path.dirname(config_path), 'config_prune_stu_0.8.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
