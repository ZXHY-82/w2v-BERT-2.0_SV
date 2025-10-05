---
license: mit
language:
- af
- am
- ar
- as
- az
- be
- bn
- bs
- bg
- ca
- cs
- zh
- cy
- da
- de
- el
- en
- et
- fi
- fr
- or
- om
- ga
- gl
- gu
- ha
- he
- hi
- hr
- hu
- hy
- ig
- id
- is
- it
- jv
- ja
- kn
- ka
- kk
- mn
- km
- ky
- ko
- lo
- ln
- lt
- lb
- lg
- lv
- ml
- mr
- mk
- mt
- mi
- my
- nl
- nb
- ne
- ny
- oc
- pa
- ps
- fa
- pl
- pt
- ro
- ru
- sk
- sl
- sn
- sd
- so
- es
- sr
- sv
- sw
- ta
- te
- tg
- tl
- th
- tr
- uk
- ur
- uz
- vi
- wo
- xh
- yo
- ms
- zu
- ary
- arz
- yue
- kea
inference: false
---
# W2v-BERT 2.0 speech encoder

We are open-sourcing our Conformer-based [W2v-BERT 2.0 speech encoder](#w2v-bert-20-speech-encoder) as described in Section 3.2.1 of the [paper](https://arxiv.org/pdf/2312.05187.pdf), which is at the core of our Seamless models.

This model was pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages. It requires finetuning to be used for downstream tasks such as Automatic Speech Recognition (ASR), or Audio Classification.

| Model Name        | #params | checkpoint                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W2v-BERT 2.0 | 600M    | [checkpoint](https://huggingface.co/reach-vb/conformer-shaw/resolve/main/conformer_shaw.pt)

**This model and its training are supported by 🤗 Transformers, more on it in the [docs](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2-bert).**


# 🤗 Transformers usage

This is a bare checkpoint without any modeling head, and thus requires finetuning to be used for downstream tasks such as ASR. You can however use it to extract audio embeddings from the top layer with this code snippet:

```python
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/w2v-bert-2.0")
model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

To learn more about the model use, refer to the following resources:
- [its docs](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2-bert)
- [a blog post showing how to fine-tune it on Mongolian ASR](https://huggingface.co/blog/fine-tune-w2v2-bert)
- [a training script example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py)


# Seamless Communication usage

This model can be used in [Seamless Communication](https://github.com/facebookresearch/seamless_communication), where it was released.
 
Here's how to make a forward pass through the voice encoder, after having completed the [installation steps](https://github.com/facebookresearch/seamless_communication?tab=readme-ov-file#installation):

```python
import torch

from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import get_seqs_and_padding_mask
from pathlib import Path
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model


audio_wav_path, device, dtype = ...
audio_decoder = AudioDecoder(dtype=torch.float32, device=device)
fbank_converter = WaveformToFbankConverter(
    num_mel_bins=80,
    waveform_scale=2**15,
    channel_last=True,
    standardize=True,
    device=device,
    dtype=dtype,
)
collater = Collater(pad_value=1)

model = load_conformer_shaw_model("conformer_shaw", device=device, dtype=dtype)
model.eval()

with Path(audio_wav_path).open("rb") as fb:
    block = MemoryBlock(fb.read())

decoded_audio = audio_decoder(block)
src = collater(fbank_converter(decoded_audio))["fbank"]
seqs, padding_mask = get_seqs_and_padding_mask(src)

with torch.inference_mode():
  seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
  seqs, padding_mask = model.encoder(seqs, padding_mask)
```