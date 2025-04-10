import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import json

class EuroparlDataset(Dataset):
    '''
    requires running preprocess.py first to generate the cleaned dataset at
    datadir/dataset/[lang]/[split]_data.pkl
    '''

    def __init__(self, data_dir = '', split = 'train', src_lang = 'en', trg_lang = 'en'):

        src_dir = os.path.join(data_dir, src_lang, '{}_data.pkl'.format(split))
        trg_dir = os.path.join(data_dir, trg_lang, '{}_data.pkl'.format(split))

        with open(src_dir, 'rb') as f:
            self.data_src = pickle.load(f)
        
        with open(trg_dir, 'rb') as f:
            self.data_trg = pickle.load(f)

        assert(len(self.data_src) == len(self.data_trg))

    def __getitem__(self, index):
        src = self.data_src[index]
        trg = self.data_trg[index]
        return src, trg
    
    def __len__(self):
        return len(self.data_src)
    
def collate(batch, maxNumToken = 27, numlang = 2):
    batch_size = len(batch)
    #create padding (assume padding token has index = 0)
    padded = np.zeros((batch_size, numlang, maxNumToken), dtype=np.int64)

    for i, sentences in enumerate(batch):
        for lang in range(numlang):
            sentence_length = len(sentences[lang])
            padded[i, lang, :sentence_length] = sentences[lang]

    return torch.from_numpy(padded)


class TextTokenConverter():
    def __init__(self, data_dir = '', lang = 'en'):
        with open(data_dir +'dataset/{}/vocab.json'.format(lang)) as f:
            self.data = json.load(f)
            self.token_to_idx = self.data['token_to_idx']
            self.idx_to_token = {value:key for key, value in self.token_to_idx.items()}
    
    def idx2text(self, input_idxs):
        output_text = []
        for idx in input_idxs:
            try:
                output_text.append(self.idx_to_token[idx.item()])
            except:
                output_text.append("<UNK>")
        return ' '.join(output_text)
    
    def text2idx(self, input_token):
        output_idx = []
        for token in input_token:
            output_idx.append(self.token_to_idx[token])
        return output_idx
    
    def get_pad_idx(self):
        return self.token_to_idx["<PAD>"]
    
    def get_vocab_size(self):
        return len(self.token_to_idx.keys())


