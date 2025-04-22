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

Authors of this repo: Joshua Ning, Jason Ning

## Usage

### Environment files
Use anaconda to import the environment files.

For MacOS, load: `env/macEnv` (not tested for training/evaluation)

For Windows with Nvidia GPU, load: `env/winNvEnv`

### Obtain the Dataset:
The dataset Europarl from hugging face [https://huggingface.co/datasets/Helsinki-NLP/europarl]
Get it by using the following command:
```
    git lfs install  
    git clone https://huggingface.co/datasets/Helsinki-NLP/europarl  
```

### Preprocessing
To preprocess the data, run `preprocess.py` with `--input-data-dir` set to the europarl dataset directory.
The input argument `--lang-pairs` should be set to the folder you want to process in the `--input-data-dir`
you can process multiple language pairs at the same time by seperating the folder names with a `_`.
For example, you can process `da-en`,`en-fr`, and `en-es` at the same time by setting `--lang-pairs`
to `da-en_en-fr_en-es`. Additionally if multiple datasets are processed at the same time, the dictionary
of the common lanugage will be combined. In the above example, all the english vocabulary will be identical.

### Training
To train the model, run `train_multiDecoder.py`. Training loop will save model weight for the lowest evaluation
loss and every few epoch in `weights/` folder in a time stamped folder. In addition, the script will
output telemetry data (training loss and evaluation loss). Typically the model will converge within
30 epochs with each epoch taking around 10 minutes on RTX-4070 (12GB VRAM) for the default parameters.
Set input argument `--lang-pairs` with the same syntax as `preprocessing.py`. The script will automatically
determin which language to use as encoder based on the common language across the language pairs.

### Fine tuning
To fine tune the model, run `finetune.py`. The user can select which encoder to use by specifying the 
input argument `--encoder-pth`. The user should also specify the `--src-lang` and `--trg-lang` argument
by providing where `--src-lang` must be the same language used to train the encoder at `--encoder-pth`.

### Inference
To inference the models run `inference.py`. The script will store the inference result in `--out-label-f`
and `--out_pred_f`. The user should specify the encoder and decoder to use with arguments `--enc-name` and
`--dec-name`


## citation
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing}, 
  title={Deep Learning Enabled Semantic Communication Systems}, 
  year={2021},
  volume={Early Access}}