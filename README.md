# ECS289G
Training ASR Algorithms on Dysarthric Audio

To denoise data, use and modify denoising.py with the following settings:

- Uncomment the initial code block to use in Google Colab as an interactive notebook
- Imports imply requiring the package to perform denoising
- Modify `data_path`, `path`, `output_path` for your own paths for reading/writing data. Line 9 uses `data_path`, line 21 uses `path`, and line 26 uses `output_path`
- denoiser requires the use of a GPU instance / CUDA cores, so use in Google Colab if you do not have the ability to use a GPU instance locally

To run the DeepSpeech model, follow the code blocks in the Colab Notebook attached as DeepSpeech.ipynb.

- Uncomment lines in the notebook follow all the steps to install all required dependency.
- The trained models can be found under model/Trained_Model folder which can be used to run the inference on testing data.
- Due to very large size of checkpoints and models produced, different colab accounts were used to train the models.
- Modify the `data_path` and run data<dataset_name>.py models to produce .csv files to be used for training.
- For inference, run the code in Colab with comment "Running inference on trained model by using unseen testing data". This requires checkpoints to be available at `checkpoint_dir`
- Modify the paths for location of data and checkpoints while running training and during inferences.

The code for preprocessing_bitrate_channnel_conversion, Speaker_diarization, XLSR_Training, XLSR_Inference_Confusion_Matrix, Making_Graphs is present in the Tensformer folder.
- First, use the Pre_processing_bitrate_and_channel_Conversion file to preprocess and convert channel. The directory structure in which the data is already accounted for.
- Then used the denoise data process as describe above. Again, it is set to use the directory structure of the dataset automatically.
- Then use the Speaker_Diarization.py file to remove the audio files with multiple speakers. Again, it is set to use the directory structure of the dataset automatically.
- Then use the train.py file to train. The pretrained model downloads automatically. It is also set to use the directory structure of the dataset automatically.
- Then use the inference_and_confusion_matrix.py file to generate the CER, WER and Comfusion matrix. It is also set to use the directory structure of the dataset automatically.
- Manually put the CER and WER values in Graphs-289G.ipynb to make graphs. The previous graph images are preloaded.
