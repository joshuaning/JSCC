import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, split='train', src_lang = 'en', trg_lang = 'en'):
        data_dir = ''
        with open(data_dir + 'dataset/{}/{}_data.pkl'.format(src_lang, split), 'rb') as f:
            self.data_src = pickle.load(f)
        
        with open(data_dir + 'dataset/{}/{}_data.pkl'.format(trg_lang, split), 'rb') as f:
            self.data_trg = pickle.load(f)