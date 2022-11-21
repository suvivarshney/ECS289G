import warnings
warnings.filterwarnings('ignore')
import os
import ffmpeg
import tqdm
import shutil
import tqdm

#Nemours
#Nemours files are with the same name as audio, so no shifting required
n_data = 'Nemorous/Wav/'
n_out = 'Nemorous/16bitmono/'

for file in tqdm.tqdm(os.listdir(n_data)):
    if '.WAV' in file:
        os.system('ffmpeg -i '+n_data+file+' -acodec pcm_s16le -ac 1 -ar 16000 '+n_out+file)

#TORGO
t_data = 'Old/TORGO_Filtered/'
out_Folder = 'Old/TORGO_Filtered/'

##Dys
for folder in os.listdir(t_data):
    if ('F'==folder or 'M'==folder):
        for sub_folder in os.listdir(t_data+folder):
            for session in os.listdir(t_data+folder+'/'+sub_folder):
                if 'Session' in session:
                    for wavFolder in os.listdir(t_data+folder+'/'+sub_folder+'/'+session):
                        if('phn' in wavFolder):
                            targetFolder = wavFolder.split('_')[1]
                            #shutil.copyfile(t_data+folder+'/'+sub_folder+'/'+session+'/'+wavFolder, out_Folder+'Phn_Dys')
                            #shutil.copyfile(t_data+folder+'/'+sub_folder+'/'+session+'/'+wavFolder, out_Folder+'Prompts_Dys')
                            #os.system('ffmpeg -i '+t5_data+file+' -acodec pcm_s16le -ac 1 -ar 16000 '+t5_out+file)
                            for file in os.listdir(t_data+folder+'/'+sub_folder+'/'+session+'/'+wavFolder):
                                shutil.copyfile(t_data+folder+'/'+sub_folder+'/'+session+'/'+wavFolder+'/'+file, out_Folder+'Phn_Dys/'+sub_folder+'_'+session+'_'+file)
                            for file in os.listdir(t_data+folder+'/'+sub_folder+'/'+session+'/prompts/'):
                                shutil.copyfile(t_data+folder+'/'+sub_folder+'/'+session+'/prompts/'+file,out_Folder+'Prompts_Dys/'+sub_folder+'_'+session+'_'+file)
                            #print(t_data+folder+'/'+sub_folder+'/'+session+'/'+wavFolder)
                            #print(t_data+folder+'/'+sub_folder+'/'+session+'/prompts/')
                    for wavFolder in os.listdir(t_data+folder+'/'+sub_folder+'/'+session):
                        if(targetFolder in wavFolder and 'wav' in wavFolder):
                            for file in os.listdir(t_data+folder+'/'+sub_folder+'/'+session+'/'+wavFolder):
                                os.system('ffmpeg -i '+t_data+folder+'/'+sub_folder+'/'+session+'/'+wavFolder+'/'+file+' -acodec pcm_s16le -ac 1 -ar 16000 '+out_Folder+'16bitMono_Dys/'+sub_folder+'_'+session+'_'+file)
                            #print(t_data+folder+'/'+sub_folder+'/'+session+'/'+wavFolder)
                    


#UA Speech
uas_wav = 'UASpeech/audio/noisereduce/'
uas_mlf = 'UASpeech/mlf/'
uas_control = 'UASpeech/Control/'
uas_dys = 'UASpeech/Dys/'

#Dys
for folder in tqdm.tqdm(os.listdir(uas_wav)):
    if '.' not in folder and 'C' not in folder:
        for sub_folder in os.listdir(uas_wav+folder):
            if '.wav' in sub_folder:
                os.system('ffmpeg -i '+uas_wav+folder+'/'+sub_folder+' -acodec pcm_s16le -ac 1 -ar 16000 '+uas_dys+'16kbitMono'+'/'+sub_folder)
                print(sub_folder)

#Dys prompts
for folder in tqdm.tqdm(os.listdir(uas_mlf)):
    if '.' not in folder and 'C' not in folder:
        for sub_folder in os.listdir(uas_mlf+folder):
            if '.mlf' in sub_folder:
                shutil.copyfile(uas_mlf+folder+'/'+sub_folder, uas_dys+prompts)
                #print(sub_folder)
