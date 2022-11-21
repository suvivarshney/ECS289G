import os
from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet

model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53")
output_dir = "finetuned/checkpoints/dir/"

tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "]
token_set = TokenSet(tokens)

#defining_training_data
#Nemours
n = 'Data/Nemorous/'
gt=[]
audio_path=[]
prompt_n=[]
for i in os.listdir(n+'Prompts/'):
    if('.txt' in i):
        prompt_n.append(i.split('.')[0])
        with open(n+'Prompts/'+i, "r") as f:
            lines = f.readline()
        unwanted_chars = ".,_\?)(;\"$:\n"
        tmp = lines.lower()
        tmp = tmp.strip(unwanted_chars)
        gt.append(tmp)
        audio_path.append(n+'16bitmono/'+i.split('.')[0]+'.WAV')
for i in prompt_n:
    audio_path.append(n+'16bitmono/'+i+'.WAV')

#TORGO
t='Data/TORGO_Filtered/'
for i in os.listdir(t+'16bitMono_Dys/'):
    try:
        with open(t+'Prompts_Dys/'+i.split('.')[0]+'.txt', "r") as f:
            lines = f.readline()
        unwanted_chars = ".,_\?)(;\"$:\n"
        tmp = lines.lower()
        tmp = tmp.strip(unwanted_chars)
        if('[' not in tmp):
            audio_path.append(t+'16bitMono_Control/'+i)
            gt.append(tmp)
    except:
        print('No Prompt',i)

#UA Speech
#Dys
uas_dys = 'Data/UASpeech/Dys/'

tmp=[]
flag=0
for i in os.listdir(uas_dys+'prompts'):
    with open(uas_dys+'prompts/'+i, "r") as f:
        lines = f.readlines()
        for j in lines[1:]:
            if(not flag):
                if('.lab' in j):
                    tmp.append(j[3:].split('.')[0])
                    #print('prompt:',j[3:].split('.')[0])
                    flag=1
            if(flag):
                if(j.isupper()):
                    gt.append(j.rstrip('\n').lower().replace('-',' '))
                    #print('line:',j)
                    flag=0

for i in tmp:
    audio_path.append(uas_dys+'16kbitMono/'+i+'.wav')
print(len(audio_path))
print(len(gt))
data=[]
for i,j in zip(gt,audio_path):
    data.append({'path':j,'transcription':i})

#60% for training
#10% for eval
#30% for test
train_data=data[:len(data)//60]
eval_data=data[len(data)//60:len(data)//70]

from huggingsound.trainer import TrainingArguments, ModelArguments

train_args=TrainingArguments()
print(train_args)
model_args=ModelArguments()
print(model_args)

train_args.num_train_epochs=1000
train_args._n_gpu=1

#loss is logged in the logging file
model.finetune(
    output_dir, 
    train_data=train_data,
    token_set=token_set,
    model_args=model_args,
    training_args=train_args,
    eval_data=eval_data,
    num_workers=2
)