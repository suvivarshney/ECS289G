import warnings
warnings.filterwarnings('ignore')
from huggingsound import SpeechRecognitionModel
import torch
import os
import pandas as pd
import tqdm
import torchmetrics
import numpy as np

import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tqdm
from huggingsound.speech_recognition.decoder import Decoder, GreedyDecoder
from huggingsound.utils import get_chunks, get_waveforms

model = os.listdir('finetuned/checkpoints/dir/best_model')[-1]
processor= = Wav2Vec2Processor.from_pretrained(model)
model = Wav2Vec2ForCTC.from_pretrained(model)

#UA Speech
#Dys
uas_dys = 'Data/UASpeech/Dys/'
gt_uasd=[]
prompts_uasd=[]
flag=0
for i in tqdm.tqdm(os.listdir(uas_dys+'prompts')):
    with open(uas_dys+'prompts/'+i, "r") as f:
        lines = f.readlines()
        for j in lines[1:]:
            if(not flag):
                if('.lab' in j):
                    prompts_uasd.append(j[3:].split('.')[0])
                    #print('prompt:',j[3:].split('.')[0])
                    flag=1
            if(flag):
                if(j.isupper()):
                    gt_uasd.append(j.rstrip('\n').lower().replace('-',' '))
                    #print('line:',j)
                    flag=0

rnd_idx = np.random.uniform(low=0, high=len(prompts_uasd), size=(2750)).astype(int)
prompts_uasd = [prompts_uasd[i] for i in rnd_idx]
gt_uasd = [gt_uasd[i] for i in rnd_idx]

predicted_sentences_uasd=[]
fails_uasd=[]
c=0
for i in tqdm.tqdm(prompts_uasd):
    try:
        waveforms = get_waveforms([uas_dys+'16kbitMono/'+i+'.wav'], 16_000)
        inputs = processor(waveforms, sampling_rate=16_000, return_tensors="pt", padding=True, do_normalize=True)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_id=torch.topk(logits,k=5,dim=-1).indices
        predicted_sentences_uasd.append(processor.batch_decode(predicted_id[:,:,0]))
    except:
        fails_uasd.append(c)
    c+=1

predicted_sentences_uasd=[i[0] for i in predicted_sentences_uasd]

print(torchmetrics.functional.char_error_rate(predicted_sentences_uasd,gt_uasd))
##WER without a  LM
metric = torchmetrics.WordErrorRate()
print(metric(predicted_sentences_uasd,gt_uasd))


#TORGO
#Dys
prompt_td=[]
gt_td=[]
predicted_sentences_td=[]
for i in tqdm.tqdm(os.listdir(t+'16bitMono_Dys/')):
    try:
        with open(t+'Prompts_Dys/'+i.split('.')[0]+'.txt', "r") as f:
            lines = f.readline()
        unwanted_chars = ".,_\?)(;\"$:\n"
        tmp = lines.lower()
        tmp = tmp.strip(unwanted_chars)
        if('[' not in tmp):
            try:
                waveforms = get_waveforms([t+'16bitMono_Dys/'+i], 16_000)
                inputs = processor(waveforms, sampling_rate=16_000, return_tensors="pt", padding=True, do_normalize=True)
                with torch.no_grad():
                    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
                predicted_id=torch.topk(logits,k=5,dim=-1).indices
                torch.save(predicted_id, 'Tensors/TORGO/Dys/'+i+'.pt')
                predicted_sentences_td.append(processor.batch_decode(predicted_id[:,:,0]))
                prompt_td.append(i.split('.')[0])
                gt_td.append(tmp)
            except:
                print('Error Wav',i)
    except:
        print('No Prompt',i)

predicted_sentences_td=[i[0] for i in predicted_sentences_td]

print(torchmetrics.functional.char_error_rate(predicted_sentences_td,gt_td))

##WER without a  LM
metric = torchmetrics.WordErrorRate()
print(metric(predicted_sentences_td,gt_td))

n = 'Data/Nemorous/'
gt_n=[]
prompt_n=[]
for i in os.listdir(n+'Prompts/'):
    if('.txt' in i):
        prompt_n.append(i.split('.')[0])
        with open(n+'Prompts/'+i, "r") as f:
            lines = f.readline()
        unwanted_chars = ".,_\?)(;\"$:\n"
        tmp = lines.lower()
        tmp = tmp.strip(unwanted_chars)
        gt_n.append(tmp)

predicted_sentences_n=[]
for i in tqdm.tqdm(prompt_n):
    waveforms = get_waveforms([n+'16bitmono/'+i+'.WAV'], 16_000)
    inputs = processor(waveforms, sampling_rate=16_000, return_tensors="pt", padding=True, do_normalize=True)
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_id=torch.topk(logits,k=5,dim=-1).indices
    torch.save(predicted_id, 'Tensors/Nemorous/'+i+'.pt')
    predicted_sentences_n.append(processor.batch_decode(predicted_id[:,:,0]))

predicted_sentences_n=[i[0] for i in predicted_sentences_n]
print(torchmetrics.functional.char_error_rate(predicted_sentences_n,gt_n))
##WER without a  LM
metric = torchmetrics.WordErrorRate()
print(metric(predicted_sentences_n,gt_n))


##Confusion Matrix
from Levenshtein import editops
from Levenshtein import opcodes
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

s=[]
d=[]

s.extend(predicted_sentences_n)
s.extend(predicted_sentences_td)
s.extend(predicted_sentences_uasd)
d.extend(gt_n)
d.extend(gt_td)
d.extend(gt_uasd)

matrix=[[0 for i in range(27)]for j in range(27)]
for k in tqdm.tqdm(range(len(d))):
    a=editops(s[k],d[k])
    #print(a)
    b=[]
    for i in range(len(a)):
        if(a[i][0]!='insert'):
            b.append(list(a[i]))
    ops_idx=[]
    for i in range(len(b)):
        try:
            b[i][1]=s[k][b[i][1]] 
            b[i][2]=d[k][b[i][2]]
            ops_idx.append(a[i][1])
        except:
            continue
    for i in b:
        try:
            if(ord(i[1])<97 or ord(i[1])>122 or ord(i[2])<97 or ord(i[2])>122):
                continue
            if i[0]=='replace':
                matrix[ord(i[1])-97][ord(i[2])-97]+=1
            elif i[0]=='insert' or i[0]=='delete':
                matrix[ord(i[1])-97][-1]+=1
        except:
            #print(i)
            continue
        r = [i for i in range(len(s[k])) if i not in ops_idx]
        for i in r:
            if(ord(s[k][i])>=97 and ord(s[k][i])<=122):
                matrix[ord(s[k][i])-97][ord(s[k][i])-97]+=1
matrix=np.array(matrix).astype(np.float32)
for i in range(len(matrix[0])-1):
    matrix[:-1,i]=matrix[:-1,i]/max(max(matrix[:-1,i]),1e-5)
matrix[-1,:] = matrix[-1,:]/max(max(matrix[-1,:]),1e-5)
matrix[:,-1] = matrix[:,-1]/max(max(matrix[:,-1]),1e-5)

l='abcdefghijklmnopqrstuvwxyz '
df_cm = pd.DataFrame(matrix[:27,:27], index = [i for i in l],
                  columns = [i for i in l])
plt.figure(figsize = (29,25))
#plt.set(xticklabels=[])
#plt.tick_params(axis='both', which='major', labelsize=24)
#plt.tick_params(axis='both', which='minor', labelsize=24)
sn.set(font_scale=7)
#sn.set_style("white",  {'figure.facecolor': 'black'})
#sn.reset_orig()
ax=sn.heatmap(df_cm, annot=False,  cmap="YlGnBu", xticklabels=l,yticklabels=l)
ax.set(xlabel=None)
ax.set(ylabel=None)
ax.set(xlabel='Ground Truth Phonemes')
ax.set(ylabel='Predicted Phonemes')
#ax.set(xticklabels=[])
#ax.set(yticklabels=[])
ax.tick_params(labelsize=50)
ax.hlines([44],colors='white', *ax.get_xlim())
ax.vlines([44],colors='white', *ax.get_ylim())
#ax.hlines([43],colors='white', *ax.get_xlim())
#ax.vlines([43],colors='white', *ax.get_ylim())
plt.xticks(rotation=0) 
plt.savefig('289-deepspeech.png')

