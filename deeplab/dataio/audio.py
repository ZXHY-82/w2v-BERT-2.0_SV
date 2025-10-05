import os
import numpy as np
import struct
import audioop
import torch, torchaudio
import random
import io
from pydub import AudioSegment
from copy import deepcopy
from scipy.signal import fftconvolve
from deeplab.utils.fileio import load_audio
from deeplab.utils.misc import trim_time_interval
from scipy.io import wavfile
import scipy

def norm_audio(signal, mode='std'):
    """
    对音频信号进行归一化处理
    
    Args:
        signal: 输入音频信号，可以是numpy数组
        mode: 归一化模式
            - 'std': 使用标准差归一化，均值为0，标准差为1
            - 'max': 使用最大值归一化，范围在[-1,1]之间
            
    Returns:
        归一化后的音频信号，float类型
    """
    signal = signal.astype('float')

    if mode == 'std':
        std = np.std(signal)
        if std != 0: 
            signal = (signal - np.mean(signal)) / std

    elif mode == 'max': 
        signal = signal / (np.abs(signal).max()+ 1e-4)

    else:
        raise NotImplementedError('Unknown normalization mode.')
        
    return signal


def norm_audio_to_int16(signal):
    """
    将音频信号归一化到int16范围
    
    Args:
        signal: 输入音频信号，float类型
        
    Returns:
        归一化到int16范围的音频信号，int16类型
        范围: [-32768, 32767]
    """
    signal = signal.astype('float')
    signal = signal -  np.min(signal)
    maxval = np.max(signal)
    if maxval != 0:
        signal = signal /  maxval
        signal = signal * 32767*2 - 32767

    signal = np.clip(signal, a_min=-32768, a_max=32767).astype('int16')
    
    return signal


def pcm2signal(pcm_data):
    """
    将PCM格式的音频数据转换为归一化的音频信号
    
    Args:
        pcm_data: PCM格式的音频数据，bytes类型
            通常为16位有符号整数格式，每个采样点占2字节
        
    Returns:
        归一化后的音频信号，float类型
    """
    signal = struct.unpack("%ih" % (len(pcm_data) / 2), pcm_data)
    signal = norm_audio(np.array([float(val) for val in signal]))

    return signal


def signal2pcm(signal):
    """
    将归一化的音频信号转换为PCM格式的音频数据
    
    Args:
        signal: 归一化后的音频信号，float类型
        
    Returns:
        PCM格式的音频数据，bytes类型
    """
    factor = np.max(np.abs(signal))
    if factor > 0:
        signal = (signal - np.mean(signal)) / factor

    ints = (signal * int(32767//16)).astype('int16')
    pcm_data = ints.astype('<u2').tobytes()

    return pcm_data


def truncate_audio(signal, tlen, head_first=True):
    """
    截取音频信号，如果信号长度小于目标长度，则进行零填充
    支持多通道音频处理
    
    Args:
        signal: 输入音频信号，可以是numpy数组
        tlen: 目标长度
        head_first: 对于长度不够的在尾部补0，优先取前面的值；否则在头部补0，优先取后面的值；            

    Returns:
        截取后的音频信号，float类型
    """
    if signal.shape[0] < tlen: 
        if len(signal.shape) == 2:
            padding = np.zeros((tlen-len(signal), signal.shape[1]), dtype=signal.dtype)
        else:
            padding = np.zeros(tlen-len(signal), dtype=signal.dtype)
        
        if head_first:
            signal = np.concatenate([signal, padding], axis=0) # pad zeros
        else:
            signal = np.concatenate([padding, signal], axis=0) # pad zeros

    if head_first:
        return signal[:tlen]
    else:
        return signal[-tlen:]


def truncate_audio_random(signal, tlen, crossfade=0):
    """
    随机截取音频信号片段，如果信号长度小于目标长度，则通过交叉淡入淡出连接自身进行扩展
    支持多通道音频处理

    Args:
        signal: 输入音频信号，可以是numpy数组
        tlen: 目标长度
        crossfade: 交叉淡入淡出的长度，默认为0表示直接拼接
            
    Returns:
        随机截取的音频信号片段，float类型
    """
    while len(signal) <= tlen:
        signal = cat_audio_with_crossfade(signal, deepcopy(signal), crossfade)
        
    offset = np.random.randint(0, signal.shape[0] - tlen)
    signal = signal[offset:offset+tlen]

    return signal


def cat_audio_with_crossfade(sig1, sig2, crossfade):
    """
    将两个音频信号进行交叉淡入淡出拼接
    支持多通道音频处理

    Args:
        sig1: 第一个音频信号，可以是numpy数组
        sig2: 第二个音频信号，可以是numpy数组
        crossfade: 交叉淡入淡出长度 
        
    Returns:
        拼接后的音频信号，float类型
    """
    if len(sig1)<(crossfade*2) or len(sig2)<(crossfade*2) or crossfade==0:
         
        return np.concatenate((sig1, sig2))
    
    raw_dtype = sig1.dtype
    
    _fade_out = -np.linspace(0, 1, crossfade) + 1
    _fade_out = np.concatenate((np.ones(len(sig1)-crossfade), _fade_out)).clip(0, 1)
    if len(sig1.shape) == 2:
        _fade_out =  np.tile(np.expand_dims(_fade_out,axis=1), (1,sig1.shape[1]))
    sig1 = sig1.astype('float') * _fade_out
    
    _fade_in = np.linspace(0, 1, crossfade)
    _fade_in = np.concatenate((_fade_in, np.ones(len(sig2)-crossfade))).clip(0, 1)
    if len(sig2.shape) == 2:
        _fade_in = np.tile(np.expand_dims(_fade_in,axis=1), (1,sig2.shape[1]))
    sig2 = sig2.astype('float') * _fade_in

    padding = np.zeros(len(sig2)-crossfade)
    if len(sig2.shape) == 2:
        padding = np.tile(np.expand_dims(padding,axis=1), (1,sig2.shape[1]))

    sig1 = np.concatenate((sig1, padding))
    sig1[-len(sig2):] += sig2
    
    return sig1.astype(raw_dtype)


def resample_audio(signal, sr, resample):
    """
    重采样音频信号，支持多通道音频处理
    
    Args:
        signal: 输入音频信号，可以是numpy数组
        sr: 原始采样率
        resample: 目标采样率
        
    Returns:
        重采样后的音频信号，float类型
    """
    effects = [
        ['remix', '1'],
        ['lowpass', f'{resample//2}'],
        ['rate', f'{resample}'],
    ]
  
    signal = np.expand_dims(signal, axis=-1) if len(signal.shape)<2 else signal
    signal = torch.from_numpy(signal).float()
    
    factor = signal.abs().max().item()
    signal = signal/factor if factor>0 else signal

    signal_sox = torchaudio.sox_effects.apply_effects_tensor(signal, sr, effects, channels_first=False)[0]
    signal_sox = signal_sox.flatten().numpy() * factor

    return signal_sox, resample
    

def add_reverberation(signal, sr, path_list, prob):
    """
    添加混响效果，支持多通道音频处理
    
    Args:
        signal: 输入音频信号，可以是numpy数组
        sr: 原始采样率
        path_list: 混响文件路径列表
        prob: 添加混响的概率
        
    Returns:
        添加混响后的音频信号，float类型
    """
    if np.random.random() < prob: 
        reverb = load_audio(random.sample(path_list,k=1)[0], sr)[0]
        reverb = norm_audio(reverb)
        signal = norm_audio(signal)

        if len(signal.shape) == 1:
            signal = truncate_audio(fftconvolve(reverb, signal), len(signal))

        if len(signal.shape) == 2:
            for i in range(signal.shape[1]):
                signal[:,i] = truncate_audio(fftconvolve(reverb, signal[:,i]), len(signal))

        signal = norm_audio(signal)
        
    return signal


def add_noise(signal, sr, path_list, prob, snr=[5,20], max_num=1, segmental_mixing=False):
    """
    添加噪声，支持多通道音频处理
    
    Args:
        signal: 输入音频信号，可以是numpy数组
        sr: 原始采样率
        path_list: 噪声文件路径列表
        prob: 添加噪声的概率
        snr: 信噪比范围，单位为dB
        max_num: 最大噪声数量
        segmental_mixing: 是否使用分段混合，如果为True，则噪声信号会被随机截取一段并与原始信号混合
        
    Returns:
        添加噪声后的音频信号，float类型
    """
    noise_signals = []
    noise_num = np.random.randint(1, max_num+1)

    for _ in range(noise_num):
        if segmental_mixing:
            length = int(np.random.uniform(0.2, 1.0) * len(signal))
            offset = np.random.randint(len(signal) - length)
        else:
            length = signal.shape[0]
            offset = 0
        if np.random.random() < prob:
            noise = load_audio(random.sample(path_list,k=1)[0], sr)[0]
            noise = truncate_audio_random(noise, length)
            noise_signal = np.zeros(signal.shape[0], dtype='float32')
            noise_signal[offset:offset+length] = norm_audio(noise)
            noise_signals.append(noise_signal)
            
    if len(noise_signals) > 0:
        mixed_noise_signal = np.array(noise_signals).sum(axis=0)
        if len(signal.shape) == 2:
            mixed_noise_signal = np.tile(np.expand_dims(mixed_noise_signal,axis=1), (1,signal.shape[1]))

        snr = np.random.uniform(snr[0], snr[1])
        sigma_n = np.sqrt(10 ** (- snr / 10))
        signal = norm_audio(signal) + norm_audio(mixed_noise_signal)*sigma_n
        signal = norm_audio(signal)
        
    return signal


def add_noise_from_musan_dict(signal, sr, path_dict, prob, snr=[5,20]):
    """
    添加噪声，支持多通道音频处理；
    噪声文件路径字典，key为噪声类型，value为噪声文件路径列表，和标准版加噪比一次可能从固定组合中加载多个噪声的组合

    Args:
        signal: 输入音频信号，可以是numpy数组
        sr: 原始采样率
        path_dict: 噪声文件路径字典
        prob: 添加噪声的概率
        snr: 信噪比范围，单位为dB
        
    Returns:
        添加噪声后的音频信号，float类型
    """
    if np.random.random() < prob: 
        noise_signal = np.zeros(signal.shape[0], dtype='float32')
        noise_types = random.choice([['noise'], ['music'], ['babb','music'], ['babb']*random.randint(3,8)])
        
        for noise_type in noise_types:
            noise = load_audio(random.sample(path_dict[noise_type],k=1)[0], sr)[0]
            noise_signal += truncate_audio_random(noise, signal.shape[0])
        if len(signal.shape) == 2:
            noise_signal = np.tile(np.expand_dims(noise_signal,axis=1), (1,signal.shape[1])) 

        if 'babb' in noise_types:
            snr = np.random.uniform(snr[0]+10, snr[1]+5)
        else:
            snr = np.random.uniform(snr[0], snr[1])
        sigma_n = np.sqrt(10 ** (-snr/10))
        signal = norm_audio(signal) + norm_audio(noise_signal)*sigma_n
        signal = norm_audio(signal)
    return signal


def lossy_codec_augmentation(signal, sr):
    """
    对音频信号进行有损压缩编解码增强
    该函数通过随机选择一种有损压缩编解码器(a-law、mu-law、mp3或aac)对输入音频进行处理,
    模拟实际场景中音频传输和存储过程中的压缩失真。
    不支持输入多通道音频
    
    Args:
        signal: 输入音频信号，numpy数组格式
        sr: 采样率
        
    Returns:
        经过有损压缩编解码处理后的音频信号，保持原始长度
    """
    # 随机选择一种有损压缩的编解码器，并记录原始音频长度
    codec = np.random.choice(['a-law', 'mu-law', 'mp3', 'aac'])
    raw_len = signal.shape[0]

    if codec == 'a-law':
        pcm_data = signal2pcm(signal)
        pcm_data = audioop.alaw2lin(audioop.lin2alaw(pcm_data,2), 2)
        signal = pcm2signal(pcm_data)
       
    elif codec == 'mu-law':
        pcm_data = signal2pcm(signal)
        pcm_data = audioop.ulaw2lin(audioop.lin2ulaw(pcm_data,2), 2)
        signal = pcm2signal(pcm_data)
      
    elif codec == 'mp3':
        pcm_data = signal2pcm(signal)
        audio_segment = AudioSegment(data=pcm_data, sample_width=2, frame_rate=sr, channels=1)
        mp3_data = audio_segment.export(format="mp3").read()
        
        audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format='mp3')
        signal = np.array(audio_segment.get_array_of_samples())
        
    elif codec == 'aac':
        pcm_data = signal2pcm(signal)
        audio_segment = AudioSegment(data=pcm_data, sample_width=2, frame_rate=sr, channels=1)
        aac_data = audio_segment.export(format="adts").read()
        
        audio_segment = AudioSegment.from_file(io.BytesIO(aac_data), format='aac')
        signal = np.array(audio_segment.get_array_of_samples())
        
    else:
        raise NotImplementedError('Invalid Corruption Codec: {}'.format(codec))

    signal = truncate_audio(signal, raw_len)
            
    return signal


def vol_augmentation(signal, sr, gain=[5,20]):
    """
    对音频信号进行音量增强处理
    
    Args:
        signal: 输入音频信号，一维numpy数组
        sr: 采样率
        gain: 音量增益范围，默认为[5,20]dB
        
    Returns:
        经过音量增强处理后的音频信号，保持原始长度
    """
    factor = np.max(np.abs(signal))
    if factor == 0:
        return signal

    signal = torch.from_numpy(signal).float() / factor
    effects = [["vol",str(np.random.uniform(*gain))], ['rate',str(sr)]]
    
    signal_sox = torchaudio.sox_effects.apply_effects_tensor(signal.unsqueeze(0), sr, effects, channels_first=True)[0] * factor
    signal_sox = truncate_audio_random(signal_sox.flatten().numpy(), len(signal))

    return signal_sox

    
def tempo_augmentation(signal, sr, tempo=[0.9, 1.1]):
    """
    对音频信号进行节奏增强处理
    
    Args:
        signal: 输入音频信号，一维numpy数组
        sr: 采样率
        tempo: 节奏增益范围，默认为[0.9,1.1]                        
        
    Returns:
        经过节奏增强处理后的音频信号，保持原始长度
    """
    factor = np.max(np.abs(signal))
    if factor == 0:
        return signal

    signal = torch.from_numpy(signal).float() / factor
    effects = [["tempo",str(np.random.choice(tempo))], ['rate',str(sr)]]
    
    signal_sox = torchaudio.sox_effects.apply_effects_tensor(signal.unsqueeze(0), sr, effects, channels_first=True)[0] * factor
    signal_sox = truncate_audio_random(signal_sox.flatten().numpy(), len(signal))

    return signal_sox


def speed_augmentation(signal, sr, speed_value):
    """
    对音频信号进行速度增强处理
    
    Args:
        signal: 输入音频信号，一维numpy数组
        sr: 采样率
        speed_value: 速度增益值                                                                                                                                         
        
    Returns:
        经过速度增强处理后的音频信号，保持原始长度
    """
    factor = np.max(np.abs(signal))
    if factor == 0:
        return signal

    signal = torch.from_numpy(signal).float() / factor
    effects = [['speed',str(speed_value)], ['rate',str(sr)]]
    
    signal_sox = torchaudio.sox_effects.apply_effects_tensor(signal.unsqueeze(0), sr, effects, channels_first=True)[0] * factor
    signal_sox = truncate_audio_random(signal_sox.flatten().numpy(), len(signal))

    return signal_sox


def mask_audio(signal, prob, scale):
    """
    对音频信号进行掩蔽处理
    
    Args:
        signal: 输入音频信号，一维numpy数组
        prob: 掩蔽概率
        scale: 掩蔽比例                                             
        
    Returns:
        经过掩蔽处理后的音频信号，保持原始长度
    """
    curr_len = len(signal)
    mask_len = int(curr_len * np.random.uniform(0, scale))
    if mask_len==0 or mask_len>curr_len:
        return signal
    
    ori_dtype = signal.dtype
    if np.random.random() < prob:
        start = np.random.randint(0, curr_len-mask_len)
        indices = list(range(start, start+mask_len))
        signal[indices] = np.mean(signal)
    
    return signal.astype(ori_dtype)

    