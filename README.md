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
- Modify the 'data_path' and run data<dataset_name>.py models to produce .csv files to be used for training.
- For inference, run the code in Colab with comment "Running inference on trained model by using unseen testing data". This requires checkpoints to be available at 'checkpoint_dir'
