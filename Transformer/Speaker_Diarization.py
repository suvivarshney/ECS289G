#An access token has to be added, which can be obtained from hf.co/settings/tokens to create an access token (only if you had to go through
#I have removed mine, for security purposes
from pyannote.audio import Pipeline
import os
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token=Access_Token)

#Removing the data from Datasets which have multiple speakers

#TORGO
torgo = 'Data/TORGO_Filtered/'
#Removing both the prompts and the audio if there are multiple speakers
for i in tqdm.tqdm(os.listdir(torgo+'16bitMono_Dys/')):
    try:
        diarization = pipeline(torgo+'16bitMono_Dys/'+i)
        _,_,speaker =diarization.itertracks(yield_label=True)
        if(len(speakers)>1):
            os.remove(torgo+'16bitMono_Dys/'+i)
            os.remove(torgo+'Prompts_Dys/'+i.split('.')[0]+'.txt')
    except:
        print('Diarization Failed on:',i)


#UA Speech
#Dys
uas_dys = 'Data/UASpeech/Dys/'
#Removing both the prompts and the audio if there are multiple speakers
for i in tqdm.tqdm(os.listdir(uas_dys+'16kbitMono/')):
    try:
        diarization = pipeline(uas_dys+'16kbitMono/'+i)
        _,_,speaker =diarization.itertracks(yield_label=True)
        if(len(speakers)>1):
            os.remove(uas_dys+'16kbitMono/'+i)
            os.remove(uas_dys+'prompts/'+i.split('.')[0]+'.lab')
    except:
        print('Diarization Failed on:',i)

#Nemours
n = 'Data/Nemorous/'
for i in tqdm.tqdm(os.listdir(n+'16bitmono/')):
    try:
        diarization = pipeline(n+'16bitmono/'+i)
        _,_,speaker =diarization.itertracks(yield_label=True)
        if(len(speakers)>1):
            os.remove(n+'16bitmono/'+i)
            os.remove(n+'Prompts/'+i.split('.')[0]+'.txt')
    except:
        print('Diarization Failed on:',i)