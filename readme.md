# JSCC for Semantic Text Transmission: Comm with Built-in Translation

Semantic communication, such as deep JSCC, has been proven to work well in low SNR environments. Compared to traditional source coding and channel coding, JSCC does not suffer from the waterfall effect. However, in deep JSCC for text communication, the sentences received often suffer from semantic distortion, which results in the sentence received not exactly matching the sentence transmitted.

In realistic communication scenarios, most users want to receive the exact text data transmitted. We decided to embrace this semantic distortion by utilizing deep JSCC for translation since semantic distortion is expected and tolerable in translation applications. 

Our deep JSCC (DeepSC-T) is modular, allowing fast re-training for switching languages and channel models. This modularity can also be used for simultaneous translation to multiple languages. The model we implemented is a version of DeepSC with the following citation.

Authors of this repo: Joshua Ning, Jason Ning

## Usage

A printout of the directory structure can be found at `dir.txt`

Pre-trained weights can be obtained at this [Google Drive folder](https://drive.google.com/drive/folders/1VkmZyJaN91ZZwJc2PMEDYqqw8QepcC28?usp=drive_link)

### Environment files
Use Anaconda to import the environment files.

For MacOS, load: `env/macEnv` (not tested for training/evaluation)

For Windows with Nvidia GPU, load: `env/winNvEnv`

### Obtain the Dataset:
The dataset Europarl from [Hugging Face](https://huggingface.co/datasets/Helsinki-NLP/europarl)

Get the dataset by using the following command in git bash:
```
    git lfs install  
    git clone https://huggingface.co/datasets/Helsinki-NLP/europarl  
```

### Preprocessing
To preprocess the data, run `src/preprocess.py` with `--input-data-dir` set to the Europarl dataset directory. The input argument `--lang-pairs` should be set to the folder you want to process in the `--input-data-dir`. You can process multiple language pairs simultaneously by separating the folder names with a `_`. For example, you can process `da-en`,`en-fr`, and `en-es` simultaneously by setting `--lang-pairs` to `da-en_en-fr_en-es`. Additionally, if multiple datasets are processed at the same time, the dictionary of the common language will be combined. In the above example, all the English vocabulary will be identical.

### Training
To train the model, run `src/train_multiDecoder.py`. The training loop will save model weights for the lowest evaluation loss and every few epochs in the `weights/` folder in a time-stamped folder. In addition, the script will output telemetry data (training loss and evaluation loss). Typically, the model will converge within 30 epochs with each epoch taking around 10 minutes on RTX-4070 (12GB VRAM) for the default parameters. Set input argument `--lang-pairs` with the same syntax as `src/preprocessing.py`. The script will automatically determine which language to use as an encoder based on the common language across the language pairs.

### Fine-tuning
To fine-tune the model, run `finetune.py`. The user can select which encoder to use by specifying the input argument `--encoder-pth`. The user should also specify the `--src-lang` and `--trg-lang` arguments by providing where `--src-lang` must be the same language used to train the encoder at `--encoder-pth`.

### Inference
To infer the models, run `inference.py`. The script will store the inference result in `--out-label-f` and `--out_pred_f`. The user should specify the encoder and decoder to use with arguments `--enc-name` and `--dec-name`


## citation
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing}, 
  title={Deep Learning Enabled Semantic Communication Systems}, 
  year={2021},
  volume={Early Access}}
