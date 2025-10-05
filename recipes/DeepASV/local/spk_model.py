import os, sys
sys.path.append('../../..')
sys.path.append(os.path.split(__file__)[0])
sys.path.append('../../../deeplab/pretrained/audio2vector/module/transformers/src')
import torch
import torch.nn as nn
from deeplab.pretrained.audio2vector.api import AudioEncoder
try:
    from .modules import pooling_v2
    from .modules.ecapa_tdnn import ECAPA_TDNN
except ImportError:
    from modules import pooling_v2
    from modules.ecapa_tdnn import ECAPA_TDNN

class Audio2Vec_based(nn.Module):
    def __init__(
        self,
        model_name='openai/whisper-tiny', 
        frozen_encoder=True,
        bnb_config=None,
        peft_config=None,
        encoder_config=None,
        n_mfa_layers=1,
        pooling_layer='ASP', 
        embd_dim=256,
        dropout=0,
        ):
        super(Audio2Vec_based, self).__init__()
        
        self.front = AudioEncoder(
            model_name, 
            frozen_encoder,
            bnb_config,
            peft_config,
            encoder_config,
            )
        self.drop = nn.Dropout(dropout) if dropout else None

        if n_mfa_layers == -1:
            self.n_mfa_layers = self.front.n_hidden_states
        else:
            self.n_mfa_layers = n_mfa_layers
        assert 1 <= self.n_mfa_layers <= self.front.n_hidden_states, \
            'Invalid Input: n_mfa_layers'
        
        feat_dim = self.front.d_model * self.n_mfa_layers
        
        self.pooling = getattr(pooling_v2, pooling_layer)(
            feat_dim, 
            self.front.d_model,
            )
        
        self.bottleneck = nn.Linear(
            feat_dim * self.pooling.expansion, 
            embd_dim,
            )

    def forward(self, x):
        x = self.front(x)
        if self.n_mfa_layers == 1:
            x = x.last_hidden_state
        else:
            x = torch.cat(
                x.hidden_states[-self.n_mfa_layers:], 
                dim=-1,
                )
            
        x = self.pooling(x)    
        if self.drop: 
            x = self.drop(x)
        x = self.bottleneck(x)

        return x


class Audio2Vec_based_Adapter(nn.Module):
    def __init__(
        self,
        model_name='openai/whisper-tiny', 
        frozen_encoder=True,
        bnb_config=None,
        peft_config=None,
        encoder_config=None,
        n_mfa_layers=1,
        pooling_layer='ASP', 
        embd_dim=256,
        adapter_dim=128,
        dropout=0,
        ):
        super(Audio2Vec_based_Adapter, self).__init__()
        
        self.front = AudioEncoder(
            model_name, 
            frozen_encoder,
            bnb_config,
            peft_config,
            encoder_config,
            )
        self.drop = nn.Dropout(dropout) if dropout else None

        if n_mfa_layers == -1:
            self.n_mfa_layers = self.front.n_hidden_states
        else:
            self.n_mfa_layers = n_mfa_layers
        assert 1 <= self.n_mfa_layers <= self.front.n_hidden_states, \
            'Invalid Input: n_mfa_layers'
        
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.front.d_model, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.ReLU(True),
                nn.Linear(adapter_dim, adapter_dim),
            ) for _ in range(self.n_mfa_layers)
        ])

        
        feat_dim = adapter_dim * self.n_mfa_layers
        
        self.pooling = getattr(pooling_v2, pooling_layer)(
            feat_dim, 
            adapter_dim,
            )
        
        self.bottleneck = nn.Linear(
            feat_dim * self.pooling.expansion, 
            embd_dim,
            )

    def forward(self, x):
        x = self.front(x)
        if self.n_mfa_layers == 1:
            x = x.last_hidden_state
        else:
            layer_outputs = []
            x.hidden_states = x.hidden_states[-self.n_mfa_layers:]
            for i in range(self.n_mfa_layers):
                layer_outputs.append(self.adapter_layers[i](x.hidden_states[i]))
            x = torch.cat(layer_outputs, dim=-1)
            
        x = self.pooling(x)    
        if self.drop: 
            x = self.drop(x)
        x = self.bottleneck(x)

        return x


class Audio2Vec_based_Weighted_ECAPATDNN(nn.Module):
    def __init__(
        self,
        model_name='openai/whisper-tiny', 
        frozen_encoder=True,
        bnb_config=None,
        peft_config=None,
        encoder_config=None,
        n_mfa_layers=-1,
        channels=[512,512,512,512,1536], 
        embd_dim=256,
        dropout=0,
        ):
        super(Audio2Vec_based_Weighted_ECAPATDNN, self).__init__()

        self.front = AudioEncoder(
            model_name, 
            frozen_encoder,
            bnb_config,
            peft_config,
            encoder_config,
            )
        self.drop = nn.Dropout(dropout) if dropout else None

        if n_mfa_layers == -1:
            self.n_mfa_layers = self.front.n_hidden_states
        else:
            self.n_mfa_layers = n_mfa_layers
        assert 1 <= self.n_mfa_layers <= self.front.n_hidden_states, \
            'Invalid Input: n_mfa_layers'   
        
        
        self.layer_weights = nn.Parameter(torch.ones(self.n_mfa_layers))
        feat_dim = self.front.d_model

        self.ecapa_tdnn = ECAPA_TDNN(input_size=feat_dim, lin_neurons=embd_dim, channels=channels, n_mels=feat_dim)

    def forward(self, x):
        x = self.front(x)
        if self.n_mfa_layers == 1:
            x = x.last_hidden_state
        else:
            hs = torch.stack(  # shape: (L, B, T, D)
                x.hidden_states[-self.n_mfa_layers:], dim=0
            )
            norm_w = torch.softmax(self.layer_weights, dim=0)  # (L,)

            x = (norm_w[:, None, None, None] * hs).sum(dim=0)  # (B, T, D)
        x = x.transpose(1,2)
        x = self.ecapa_tdnn(x)

        return x


class Audio2Vec_based_Prune(nn.Module):
    def __init__(
        self,
        model_name='openai/whisper-tiny', 
        bnb_config=None,
        peft_config=None,
        model_config_teacher=None,
        model_config_student=None,
        distillation_layers=[],
        pretrain_encoder=None
        ):
        super(Audio2Vec_based_Prune, self).__init__()
        
        self.distillation_layers = distillation_layers
        self.teacher_front = AudioEncoder(
            model_name, 
            True,
            bnb_config,
            peft_config,
            model_config_teacher,
            )
    
        self.student_front = AudioEncoder(
            model_name, 
            False,
            bnb_config,
            peft_config,
            model_config_student,
            )
        if pretrain_encoder:
            ckpt_state_dict = torch.load(pretrain_encoder, map_location='cpu')['modules']['spk_model']
            t_cur_state_dict = self.teacher_front.state_dict()
            s_cur_state_dict = self.student_front.state_dict()
            for k in t_cur_state_dict.keys():
                if 'hard_concrete' in k:
                    continue
                c_k = 'front.' + k
                if c_k in ckpt_state_dict and t_cur_state_dict[k].shape == ckpt_state_dict[c_k].shape:
                    t_cur_state_dict[k] = ckpt_state_dict[c_k]
                    s_cur_state_dict[k] = ckpt_state_dict[c_k]
                else:
                    print(f'{k}_is_mismatch')
            self.teacher_front.load_state_dict(t_cur_state_dict)
            self.student_front.load_state_dict(s_cur_state_dict)
        
    def forward(self, x):
        x_teacher = self.teacher_front(x).hidden_states
        x_student = self.student_front(x).hidden_states
        x_t_out = [x_teacher[idx] for idx in self.distillation_layers]
        x_s_out = [x_student[idx] for idx in self.distillation_layers]
        return torch.stack(x_t_out), torch.stack(x_s_out)

class Lambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.tensor(0.0))
        self.lambda2 = nn.Parameter(torch.tensor(0.0))
    


if __name__ == "__main__":
    from calflops import calculate_flops
    torch.set_default_device("cuda:0")

    from deeplab.pretrained.audio2vector.api import create_lora_config
    peft_config = create_lora_config(
        model_type='w2v-bert',
        r=64,
        lora_alpha=128,
        target_modules=["linear_q", "linear_v"],
        lora_dropout=0.0,
        bias='none')
    # peft_config = None

    model = Audio2Vec_based_Adapter(
        model_name='facebook/w2v-bert-2.0',
        peft_config=peft_config,
        encoder_config='config_prune_tea.json',
        frozen_encoder=False,
        n_mfa_layers=-1,
        pooling_layer='ASP', 
        )
    model.eval()

    test_input = torch.randn(1, 16000)
    input_shape = (1, 16000)
    
    calculate_flops(
        model=model, 
        input_shape=input_shape,
        print_detailed=False,
        print_results=True,
        )