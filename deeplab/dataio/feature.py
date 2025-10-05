import torch
import torch.nn as nn
import random
from torchaudio import transforms
from scipy.signal import fftconvolve
from python_speech_features import sigproc
from deeplab.dataio.audio import norm_audio


def signal2tensor(signal, norm_mode):
    """
    将信号转换为张量格式，支持多通道音频输入
    
    Args:
        signal: numpy.ndarray, 输入信号
        norm_mode: str, 归一化模式
        
    Returns:
        tensor: torch.Tensor, 转换后的张量
        对于单通道信号,返回shape为[sig_len]的张量
        对于多通道信号,返回shape为[channels, sig_len]的张量
    """
    if norm_mode is not None:
        signal = norm_audio(signal, norm_mode)

    if len(signal.shape) == 1:
        tensor = torch.from_numpy(sigproc.preemphasis(signal, 0.97)).float()

    if len(signal.shape) == 2:
        tensor_list = []
        for i in range(signal.shape[1]):
            tensor_list.append(torch.from_numpy(sigproc.preemphasis(signal[:,i], 0.97)).float())
        tensor = torch.stack(tensor_list)
        
    return tensor


class logFbankCal(nn.Module):
    """
    对数梅尔频谱特征计算，支持多通道输入

    Args:
        sample_rate: int, 采样率
        n_fft: int, 傅里叶变换窗口大小
        win_length: int, 窗口长度
        hop_length: int, 帧移
        n_mels: int, 梅尔滤波器数量
    """
    def __init__(
        self, 
        sample_rate, 
        n_fft, 
        win_length, 
        hop_length, 
        n_mels,
        ):
        super(logFbankCal, self).__init__()
        self.fbankCal = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            )

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, is_aug=[]):
        """
        对数梅尔频谱特征计算
        """
        out = self.fbankCal(x)[..., :-1] # x.shape = [batch, sig_len]，舍弃最后一个0.01s，对齐长度
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2) # out.shape = [batch, freq, time]

        # 频谱增强
        for i in range(len(is_aug)):
            assert len(out.shape) == 3 # 以下代码只针对单通道输入
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start  = random.randrange(0, out.shape[1] - offset)
                out[i][start:start+offset] = out[i][start:start+offset]  * random.random() / 2

        return out

    @torch.amp.autocast('cuda', enabled=False)
    def forward_sample(self, x, is_aug=False):
        """
        对数梅尔频谱特征计算，单个样本输入（不包括batch维）
        """
        x = self.forward(x.unsqueeze(0), [is_aug])
        x = x.squeeze(0)

        return x


