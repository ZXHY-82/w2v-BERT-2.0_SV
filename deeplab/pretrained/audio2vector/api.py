import os,sys,json
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('./module/transformers/src')
from deeplab.pretrained.audio2vector.module.transformers.src.transformers import Wav2Vec2BertModel, Wav2Vec2BertConfig, BitsAndBytesConfig, AutoFeatureExtractor, AutoModel
# transformers import AutoFeatureExtractor, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
try:
    from .forward_impl import forward_whisper, forward_w2v_bert
except ImportError:
    from forward_impl import forward_whisper, forward_w2v_bert


def create_bnb_config(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="llm_int8",
    bnb_8bit_compute_dtype="bfloat16",
    ):
    """创建 BitsAndBytes 量化配置"""
    return BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        bnb_8bit_use_double_quant=bnb_8bit_use_double_quant,
        bnb_8bit_quant_type=bnb_8bit_quant_type,
        bnb_8bit_compute_dtype=getattr(torch, bnb_8bit_compute_dtype),
        )


def create_lora_config(
    model_type,
    r=16,
    lora_alpha=32,
    target_modules=None,
    lora_dropout=0.1,
    bias="none",
    ):
    """创建 LoRA 配置
    
    Args:
        model_type: 模型类型，用于自动确定target_modules和task_type
        r: LoRA rank
        lora_alpha: LoRA alpha参数
        target_modules: 目标模块，如果为None则根据model_type自动确定
        lora_dropout: LoRA dropout率
        bias: bias设置
    """
    # 根据模型类型自动确定配置
    if model_type == "whisper":
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        task_type = "SEQ_CLS"
    elif model_type == "w2v-bert":
        if target_modules is None:
            target_modules = ["linear_q", "linear_v"]
        task_type = "FEATURE_EXTRACTION"
    else:
        supported = ", ".join(AudioEncoder.SUPPORTED_MODEL_TYPES)
        raise ValueError(f"不支持的模型类型: {model_type}. 当前支持的模型类型: {supported}")
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        )


def get_model_type(model_name):
    """
    根据model_name自动检测模型类型
    
    Args:
        model_name (str): 模型名称，如 'openai/whisper-tiny' 或 'facebook/w2v-bert-2.0'
        
    Returns:
        str: 模型类型，支持 'whisper' 和 'w2v-bert'
    """
    model_name_lower = model_name.lower()
    if "whisper" in model_name_lower:
        return "whisper"
    elif "w2v-bert" in model_name_lower:
        return "w2v-bert"
    else:
        supported = ", ".join(AudioEncoder.SUPPORTED_MODEL_TYPES)
        raise ValueError(f"不支持的模型类型: {model_name}. 当前支持的模型类型: {supported}")


class AudioEncoder(nn.Module):
    """
    音频编码器类，用于微调
    """
    
    # PEFT参数识别常量
    PEFT_INDICATORS = [
        'lora_',      # LoRA 参数
        'adapter',    # Adapter 参数
        'prefix',     # Prefix Tuning 参数
        'prompt',     # Prompt Tuning 参数
    ]
    
    # 支持的模型类型
    SUPPORTED_MODEL_TYPES = ["whisper", "w2v-bert"]
    
    def __init__(
        self,
        model_name="openai/whisper-tiny", 
        frozen_encoder=True,
        bnb_config=None,
        peft_config=None,
        model_config=None,
        ):
        super(AudioEncoder, self).__init__()        
        # 检测模型类型
        self.model_type = get_model_type(model_name)
        self.model_config = model_config
        
        # 量化和全量微调不能同时使用
        if not frozen_encoder and bnb_config:
            raise ValueError("不支持全量微调(frozen_encoder=False)和量化(bnb_config)同时使用")
        
        # 获取模型路径
        local_model_path = os.path.join(
            os.path.dirname(__file__), 'ckpts', model_name)
        
        # 加载特征提取器
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            local_model_path,
            local_files_only=True,
            )
        
        # 根据模型类型进行特定处理
        self._setup_model(local_model_path, bnb_config)
        
        # 应用 PEFT 配置
        if peft_config is not None:
            self.encoder = get_peft_model(self.encoder, peft_config)
        
        # 根据参数决定是否冻结编码器
        if frozen_encoder:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()
        
    def forward(self, aud_inputs):
        """
        根据模型类型调用对应的forward函数
        """
        if self.model_type == "whisper":
            return forward_whisper(self, aud_inputs)
        elif self.model_type == "w2v-bert":
            return forward_w2v_bert(self, aud_inputs)
        else:
            supported = ", ".join(self.SUPPORTED_MODEL_TYPES)
            raise ValueError(f"不支持的模型类型: {self.model_type}. 当前支持的模型类型: {supported}")
    
    def _setup_model(self, local_model_path, bnb_config):
        if self.model_config is not None:
            with open(os.path.join(local_model_path,self.model_config), 'r') as f:
                config_dict = json.load(f)
            config = Wav2Vec2BertConfig(**config_dict)
            full_model = Wav2Vec2BertModel(config)
            if 'prune' not in self.model_config:
                from safetensors.torch import load_file
                ckpt_state_dict = load_file(os.path.join(local_model_path,'model.safetensors'), device='cpu')
                cur_state_dict = full_model.state_dict()
                for k in cur_state_dict.keys():
                    if 'hard_concrete' in k:
                        continue
                    if k in ckpt_state_dict and cur_state_dict[k].shape == ckpt_state_dict[k].shape:
                        cur_state_dict[k] = ckpt_state_dict[k]
                    else:
                        print(f'{k}_is_mismatch')
                full_model.load_state_dict(cur_state_dict)
        else:
            full_model = AutoModel.from_pretrained(
                local_model_path,
                local_files_only=True,
                quantization_config=bnb_config
                )
        if self.model_type == "whisper":
            self.encoder = full_model.encoder
            self.d_model = self.encoder.config.d_model
            self.n_hidden_states = self.encoder.config.encoder_layers + 1
        elif self.model_type == "w2v-bert":
            self.encoder = full_model
            self.d_model = self.encoder.config.hidden_size
            self.n_hidden_states = self.encoder.config.num_hidden_layers + 1
            # 删除微调时不需要的masked_spec_embed参数，确保DDP兼容性
            delattr(self.encoder, 'masked_spec_embed') 
        else:
            supported = ", ".join(self.SUPPORTED_MODEL_TYPES)
            raise ValueError(f"不支持的模型类型: {self.model_type}. 当前支持的模型类型: {supported}")
        
    def _is_peft_parameter(self, param_name):
        """判断是否为 PEFT 参数"""
        return any(indicator in param_name for indicator in self.PEFT_INDICATORS)
    
    def _module_has_peft_parameter(self, module):
        """判断一个模块是否包含 PEFT 参数"""
        for param_name, param in module.named_parameters():
            if self._is_peft_parameter(param_name):
                return True
            
        return False
    
    def freeze_encoder(self):
        """冻结原始编码器参数，保持PEFT参数可训练"""
        self.frozen_encoder = True
        for name, param in self.encoder.named_parameters():
            if not self._is_peft_parameter(name):
                param.requires_grad = False
    
    def unfreeze_encoder(self):
        """解冻原始编码器参数，PEFT参数保持可训练"""
        self.frozen_encoder = False
        for name, param in self.encoder.named_parameters():
            if not self._is_peft_parameter(name):
                param.requires_grad = True

    def train(self, mode=True):
        """重写train方法，处理冻结编码器时的训练模式"""
        super().train(mode)
        if self.frozen_encoder:
            # 冻结模式：只让包含PEFT参数的模块进入训练模式
            for name, module in self.encoder.named_modules():
                if self._module_has_peft_parameter(module):
                    module.train(mode)
                else:
                    module.eval()
        else:
            # 非冻结模式：正常训练整个编码器
            self.encoder.train(mode)
        return self
    