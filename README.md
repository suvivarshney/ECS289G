# ECS289G
Training ASR Algorithms on Dysarthric Audio

To denoise data, use and modify denoising.py with the following settings:

- Uncomment the initial code block to use in Google Colab as an interactive notebook
- Imports imply requiring the package to perform denoising
- Modify `data_path`, `path`, `output_path` for your own paths for reading/writing data. Line 9 uses `data_path`, line 21 uses `path`, and line 26 uses `output_path`
- denoiser requires the use of a GPU instance / CUDA cores, so use in Google Colab if you do not have the ability to use a GPU instance locally
