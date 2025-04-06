import argparse
from dataset import EuroparlDataset, TextTokenConverter, collate
from torch.utils.data import DataLoader
from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime
from DeepSC_model import *
import csv
from train import *


parser = argparse.ArgumentParser()
parser.add_argument('--MAX-LENGTH', default=27, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-lang', default=2, type=int)
parser.add_argument('--num-epoch', default=80, type=int)
parser.add_argument('--model-out-dir', default='weights', type=str)
parser.add_argument('--src-lang', default='en', type=str)
parser.add_argument('--trg-lang', default='da', type=str)
parser.add_argument('--num-sent', default=10, type=int)


def inference(model, loader, pad_idx, device, num_to_print, ttc1, ttc2):
    '''
    Evalutate iteration for DeepSC_translate
    returns the avg per patch validation loss of current epoch
    '''

    model.eval()
    sentences_ctr = 0
    with torch.no_grad():
        for data in loader:

            inputs, labels = data[:, 0, :], data[:, 1, :]
            inputs = inputs.contiguous().to(device = device, dtype=torch.long)
            labels = labels.contiguous().to(device = device, dtype=torch.long)
            src_mask, combined_mask = generate_mask(inputs, labels, pad_idx, device)

            pred = model(inputs, src_mask, labels, combined_mask)
            pred = torch.argmax(pred, dim=-1) #get most probable word

            sentences_ctr = print_pred(sentences_ctr, num_to_print, 
                                       inputs, labels, pred, ttc1, ttc2)

            


if __name__ == "__main__":
    print("PyTorch Version: ",torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only.")

    args = parser.parse_args()
    collate_fn = partial(collate, maxNumToken=args.MAX_LENGTH, numlang=args.num_lang) 
    val_set = EuroparlDataset(split="test", src_lang=args.src_lang, trg_lang=args.trg_lang)
    val_loader = DataLoader(val_set, num_workers=2, batch_size=args.batch_size, collate_fn = collate_fn)
    ttc_src = TextTokenConverter(lang = args.src_lang)
    ttc_trg = TextTokenConverter(lang = args.trg_lang)
    pad_idx = ttc_trg.get_pad_idx()


    src_vocab_size = ttc_src.get_vocab_size()
    trg_vocab_size = ttc_trg.get_vocab_size()
    model = Build_DeepSC(src_vocab_size = src_vocab_size, tgt_vocab_size=trg_vocab_size, device=device,
                     src_seq_len = args.MAX_LENGTH, tgt_seq_len = args.MAX_LENGTH).to(device)
    
    load_path = 'C:/Users/Joshua Ning/Desktop/JSCC/weights/04_04_2025__16_14_00'
    load_file = 'epoch19_best.pth'
    f_path = os.path.join(load_path, load_file)

    check_point = {'model_state_dict':torch.load(f_path, map_location=torch.device(device))}
    model.load_state_dict(check_point['model_state_dict'])

    inference(model, val_loader, pad_idx, device, args.num_sent, ttc_src, ttc_trg)
