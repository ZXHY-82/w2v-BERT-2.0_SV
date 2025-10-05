import os, sys
sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../deeplab/pretrained/audio2vector/module/transformers/src')
sys.path.append(os.path.split(__file__)[0])
from calflops import calculate_flops

from local.spk_model import Audio2Vec_based_Adapter
from deeplab.pretrained.audio2vector.api import create_lora_config
import torch


peft_config = create_lora_config(
    model_type='w2v-bert',
    r=64,
    lora_alpha=128,
    target_modules=["linear_q", "linear_v"],
    lora_dropout=0.0,
    bias='none')

model = Audio2Vec_based_Adapter(
    model_name='facebook/w2v-bert-2.0', 
    frozen_encoder=True,
    n_mfa_layers=-1,
    pooling_layer='ASP', 
    peft_config=peft_config,
    encoder_config='config_prune_tea.json'
    )
model.eval()

ckpt_path = '/work/zl389/workspace/LLM_ASV/publish_code/recipes/DeepASV/results/checkpoints/vox2_251005144134/ckpt_0001.pth'
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
ckpt_data = ckpt['modules']['spk_model']

cur_state_dict = model.state_dict()
for k in cur_state_dict.keys():
    if k in ckpt_data and cur_state_dict[k].shape == ckpt_data[k].shape:
        cur_state_dict[k] = ckpt_data[k]
    else:
        print(f'{k}_is_mismatch')
model.load_state_dict(cur_state_dict)


input_shape = (1, 16000)
calculate_flops(
    model=model, 
    input_shape=input_shape,
    print_detailed=False,
    print_results=True,
    )

model.front.encoder = model.front.encoder.merge_and_unload()


input_shape = (1, 16000)
calculate_flops(
    model=model, 
    input_shape=input_shape,
    print_detailed=False,
    print_results=True,
    )

update_state_dict = model.state_dict()
ckpt['modules']['spk_model'] = update_state_dict
torch.save(ckpt, os.path.join(os.path.dirname(ckpt_path), 'merge_lora.pth'))