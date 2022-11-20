import os
import csv
import string
import pandas as pd
os.chdir(r"/content/mydrive/MyDrive/DeepSpeechv/DeepSpeech/data/TORGO")
with open('train1_old.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['wav_filename', 'wav_filesize', 'transcript'])
path = r"/content/mydrive/MyDrive/DeepSpeechv/DeepSpeech/data/TORGO/TORGO"
for folder1 in os.listdir(path):
    files =[]
    for folder2 in os.listdir(path + '/' + str(folder1)):
        if folder2 == 'wav_headMic':
            fun = lambda x : os.path.isfile(os.path.join((path + '/'+ str(folder1) + '/' + str(folder2)),x))
            files_list = filter(fun, os.listdir(path + '/'+ str(folder1) + '/' + str(folder2)))
            files = [[f,os.stat(os.path.join((path + '/'+ str(folder1) + '/' + str(folder2)), f)).st_size] for f in files_list]
            folder2 = 'prompts'
            promptPath = path + '/' + str(folder1) + '/' + str(folder2)
            os.chdir(promptPath)
            start = 0
            while (start != len(files)):
                for file in os.listdir():
                    if file.endswith((files[start][0].split('.')[0])+'.txt'):
                        file_path = f"{promptPath}/{file}"
                        with open(file_path, 'r') as f:
                            text = f.read().strip()
                            if ("," in text or ")" in text or "[" in text or "]" in text or "?" in text or "'" in text or '\\' in text or '/' in text or ':' in text or ';' in text):
                                files[start].append('')
                            elif text.startswith("x"):
                                files[start].append('')
                            else:
                                if text.endswith("."):
                                    text = text.translate(str.maketrans('', '', string.punctuation))
                                files[start].append(text.lower())
                        break
                if len(files[start]) == 2:
                    files[start].append('')
                start = start+1
    os.chdir(r"/content/mydrive/MyDrive/DeepSpeechv/DeepSpeech/data/TORGO")
    with open('train1_old.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for f in files:
            if f[2] != '':
                filewriter.writerow(['/content/mydrive/MyDrive/DeepSpeechv/DeepSpeech/data/TORGO/TORGO/'+str(folder1)+'/wav_headMic/'+f[0], int(f[1]), str(f[2].strip())])
df = pd.read_csv('train1_old.csv')
df.to_csv('train1.csv', index=False)
os.remove('train1_old.csv')