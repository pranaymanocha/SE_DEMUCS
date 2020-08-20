# Pytorch implementation of the paper "[Real Time Speech Enhancement in the Waveform Domain](https://arxiv.org/pdf/2006.12847.pdf)"

## Installation

Use pytorch version 1.4.0

Create .npy files for the dataset OR load the dataset using a dataloader from a directory.
VCTK Dataset (28spk) train/valset .npy present [here](https://percepaudio.cs.princeton.edu/SE_DEMUCS_saved_models/) along with a few pretrained models.

## To be implemented:
1) Normalising the input by standard deviation
2) Upsampling the waveform by a factor before feeding to a network. 

## License
[MIT](https://choosealicense.com/licenses/mit/)
