import sys
import os
import csv
import pandas as pd

def prepare_data(datasetpath):
    path = datasetpath + r"/Wav"
    promptPath = datasetpath + r"/Prompts"
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
    os.chdir(datasetpath)
    count = 0
    with open('train_old.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['wav_filename', 'wav_filesize', 'transcript'])
        for f in files:
            if f[2] != '':
                filewriter.writerow([datasetpath + '/Wav/' + f[0], int(f[1]), str(f[2].strip())])
                count = count + 1
    df = pd.read_csv('train_old.csv')
    df.to_csv('train.csv', index=False)
    os.remove('train_old.csv')

    dev = int(0.20*count)
    test = int(0.10*count)

    added = -1
    with open('test_old.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['wav_filename', 'wav_filesize', 'transcript'])
        with open("train.csv", 'r') as csvfile1:
            rows = csv.reader(csvfile1, delimiter='\t')
            for row, data in enumerate(rows):
                if added == -1:
                    added = 0
                elif added <= test or added == 0:
                    filewriter.writerow([data[0].split(',')[0], data[0].split(',')[1], data[0].split(',')[2]])
                    added = added + 1
    df = pd.read_csv('test_old.csv')
    df.to_csv('test.csv', index=False)
    os.remove('test_old.csv')

    lines = list()
    remove= [i for i in range(2,test+3)]

    with open('train.csv', 'r') as read_file:
        reader = csv.reader(read_file)
        for row_number, row in enumerate(reader, start=1):
            if(row_number not in remove):
                lines.append(row)
    with open('train.csv', 'w') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(lines)
    df = pd.read_csv('train.csv')
    df.to_csv('train.csv', index=False)

    added = -1
    with open('dev_old.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['wav_filename', 'wav_filesize', 'transcript'])
        with open("train.csv", 'r') as csvfile1:
            rows = csv.reader(csvfile1, delimiter='\t')
            for row, data in enumerate(rows):
                if added == -1:
                    added = 0
                elif added <= dev or added == 0:
                    filewriter.writerow([data[0].split(',')[0], data[0].split(',')[1], data[0].split(',')[2]])
                    added = added + 1
    df = pd.read_csv('dev_old.csv')
    df.to_csv('dev.csv', index=False)
    os.remove('dev_old.csv')

    lines = list()
    remove= [i for i in range(2,dev+4)]

    with open('train.csv', 'r') as read_file:
        reader = csv.reader(read_file)
        for row_number, row in enumerate(reader, start=1):
            if(row_number not in remove):
                lines.append(row)
    with open('train.csv', 'w') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(lines)
    df = pd.read_csv('train.csv')
    df.to_csv('train.csv', index=False)

def main():
    if len(sys.argv) < 1:
        print('Missing argument. Needed 1. Path for the data set')
    datasetpath = str(sys.argv[1])
    prepare_data(datasetpath)

if __name__ == '__main__':
    main()