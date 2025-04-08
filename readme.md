# JSCC for Semantic Text Transmission: Comm with Built-in Translation

Semantic communication such as deep JSCC has been proven to work well in low SNR enviornment. 
Compared to traditional source coding and channel coding, JSCC does not suffer from the water
fall effect. However, in deep JSCC for text communication, the sentences recieved often suffer 
from semantic distortion, which result in the sentence recieved not exactly matching with the 
sentences transmitted.

In realistic communication senarios, most users want to receive exact text data transmitted. 
We decided to embrace this semantic distortion by utilizing deep JSCC for translation, since
semantic distortion is expected and tolerable in translation applications.

Our deep JSCC is modular, allowing fast re-training for switching languages and channel models.
This modularity can also be used for simultanous translation to multiple languages  

The model we implemented is a version of DeepSC with the following citation

@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing}, 
  title={Deep Learning Enabled Semantic Communication Systems}, 
  year={2021},
  volume={Early Access}}

Authors of this repo: Joshua Ning, Jason Ning

## Usage

### Environment files
Use anaconda to import the environment files.
For MacOS, load: `env/macEnv`
For Windows with Nvidia GPU, load: `env/winNvEnv`

### Obtain the Dataset:
The dataset is Europarl from hugging face [https://huggingface.co/datasets/Helsinki-NLP/europarl]
Get it by using the following command:
```
    git lfs install  
    git clone https://huggingface.co/datasets/Helsinki-NLP/europarl  
```

### Preprocessing
To preprocess the data, run `preprocess.py` with `--input-data-dir` set to the dataset directory.
In the `--input-data-dir` there must be a folder named `--lang1`-`--lang2` for example if
`--lang1 == 'da'` and `--lang2 == 'en'`, in `--input-data-dir` there must be a folder named
`da-en`.

### Training
To train the model, run `train.py`. Training loop will save model weight for the lowest evaluation
loss and every 10 epoch in `weights/` folder in a time stamped folder. In addition, the script will
output telemetry data (training loss and evaluation loss). Typically the model will converge within
30 epochs with each epoch taking around 10 minutes on RTX-4070 (12GB VRAM) for the default parameters. 




