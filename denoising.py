"""
from google.colab import drive
drive.mount('/content/drive/')
!pip install -U denoiser
!pip install audio2numpy
import os
"""
data_path = " "
wav_files = os.listdir('data_path')
from IPython import display as disp
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
import audio2numpy as a2n
from scipy.io.wavfile import write
model = pretrained.dns64().cuda()
path = " "
output_path = " "

for wav_file in wav_files:
  wav, sr = a2n.audio_from_file('path' + wav_file)
  wav = torch.reshape(torch.from_numpy(wav), (1, wav.shape[0]))
  wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
  with torch.no_grad():
    denoised = model(wav[None])[0]
  write("output_path" + wav_file, sr, denoised.data.cpu().numpy().T)
