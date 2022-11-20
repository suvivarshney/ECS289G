import os
import csv
import pandas as pd
path = r"/content/mydrive/MyDrive/DeepSpeechv/DeepSpeech/data/Nemorous/Wav"
promptPath = r"/content/mydrive/MyDrive/DeepSpeechv/DeepSpeech/data/Nemorous/Prompts"
fun = lambda x : os.path.isfile(os.path.join(path,x))
files_list = filter(fun, os.listdir(path))
files = [[f,os.stat(os.path.join(path, f)).st_size] for f in files_list]
os.chdir(promptPath)
start = 0
while (start != len(files)):
    for file in os.listdir():
        if file.endswith((files[start][0].split('.')[0])+'.txt'):
            file_path = f"{promptPath}/{file}"
            with open(file_path, 'r') as f:
                text = f.read().strip()
                if ("(" in text or ")" in text):
                    files[start].append('')
                else:
                    files[start].append(text.lower())
            break
    start = start+1
os.chdir(r"/content/mydrive/MyDrive/DeepSpeechv/DeepSpeech/data/Nemorous")
with open('train_old.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['wav_filename', 'wav_filesize', 'transcript'])
    for f in files:
        if f[2] != '':
            filewriter.writerow(['/content/mydrive/MyDrive/DeepSpeechv/DeepSpeech/data/Nemorous/Wav/'+f[0], int(f[1]), str(f[2].strip())])
df = pd.read_csv('train_old.csv')
df.to_csv('train.csv', index=False)
os.remove('train_old.csv')
