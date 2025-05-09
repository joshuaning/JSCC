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
from train_multiDecoder import *
import re



parser = argparse.ArgumentParser()
parser.add_argument('--MAX-LENGTH', default=27, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--num-lang', default=2, type=int)
parser.add_argument('--num-epoch', default=80, type=int)
parser.add_argument('--model-out-dir', default='weights', type=str)
parser.add_argument('--src-lang', default='en', type=str)
parser.add_argument('--trg-lang', default='it', type=str)
parser.add_argument('--num-sent', default=10, type=int)
parser.add_argument('--data-dir', default='dataset', type=str)

parser.add_argument('--load-path', default='weights/04_16_2025__14_05_12', type=str)
parser.add_argument('--enc-name', default='epoch30_encoder.pth', type=str)
parser.add_argument('--dec-name', default='epoch30_decoder_it.pth', type=str)
parser.add_argument('--out-label-f', default='inference_results/single_lang_enc_it_dec_gt.csv', type=str)
parser.add_argument('--out_pred_f', default='inference_results/single_lang_enc_it_dec_pred.csv', type=str)

parser.add_argument('--device', default='cuda:0', type=str)
# parser.add_argument('--device', default='cpu', type=str)



def output_pred(labels, pred, ttc1):
    label_sents = []
    pred_sents = []
    for i, sentences in enumerate(pred):
            label_sent = re.search(r"<START>\s*(.*?)\s*<END>", ttc1.idx2text(labels[i]))
            pred_sent = re.search(r"<START>\s*(.*?)\s*<END>", ttc1.idx2text(sentences))
            label_sents.append(label_sent.group(1))
            pred_sents.append(pred_sent.group(1))
    return  label_sents, pred_sents

def inference(encoder, decoder, loader, pad_idx, device, src_lang, trg_lang, args):
    '''
    Evalutate iteration for DeepSC_translate
    returns the avg per patch validation loss of current epoch
    '''
    
    encoder.eval()
    decoder.eval()
    decoder.to(device)
    lang_data_dir = os.path.join(args.data_dir, src_lang+'-'+trg_lang)
    ttc1 = TextTokenConverter(data_dir = lang_data_dir, lang = src_lang)
    ttc2 = TextTokenConverter(data_dir = lang_data_dir, lang = trg_lang)
    label_sents = []
    pred_sents = []
    encoder_time = []
    decoder_time = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):

            inputs, labels = data[:, 0, :], data[:, 1, :]
            inputs = inputs.contiguous().to(device = device, dtype=torch.long)
            labels = labels.contiguous().to(device = device, dtype=torch.long)

            src_mask, combined_mask = generate_mask(inputs, labels, pad_idx, device)

            encoder_start_time = datetime.now()
            channel_decoder_output = encoder(inputs, src_mask)
            encoder_end_time = datetime.now()

            decoder_start_time = datetime.now()
            pred = decoder(channel_decoder_output, src_mask, labels, combined_mask)
            decoder_end_time = datetime.now()

            delta_enc = encoder_end_time - encoder_start_time
            delta_dec = decoder_end_time - decoder_start_time
            # delta_enc = delta_enc.total_seconds()

            encoder_time.append(delta_enc.total_seconds())
            decoder_time.append(delta_dec.total_seconds())

            pred = torch.argmax(pred, dim=-1) #get most probable word
            sents_out = output_pred(labels, pred, ttc2)
            label_sents += sents_out[0]
            pred_sents += sents_out[1]

            # if i == 10000:
            #     break
    
    print("mean encoding time in seconds: ", np.mean(np.array(encoder_time)))
    print("mean decoding time in seconds: ", np.mean(np.array(decoder_time)))
    
    return label_sents, pred_sents


            


if __name__ == "__main__":
    args = parser.parse_args()
    print("PyTorch Version: ",torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and args.device == 'cuda:0':
        print("Using the GPU!")
    else:
        device = args.device
        print("Using CPU only.")

    lang_data_dir = os.path.join(args.data_dir, args.src_lang+'-'+args.trg_lang)

    collate_fn = partial(collate, maxNumToken=args.MAX_LENGTH, numlang=args.num_lang) 
    val_set = EuroparlDataset(data_dir=lang_data_dir, split="test", src_lang=args.src_lang, trg_lang=args.trg_lang)
    val_loader = DataLoader(val_set, num_workers=2, batch_size=args.batch_size, collate_fn = collate_fn)
    ttc_src = TextTokenConverter(data_dir=lang_data_dir, lang = args.src_lang)
    ttc_trg = TextTokenConverter(data_dir=lang_data_dir, lang = args.trg_lang)
    pad_idx = ttc_trg.get_pad_idx()

    src_vocab_size = ttc_src.get_vocab_size()
    trg_vocab_size = ttc_trg.get_vocab_size()

    deepsc_encoder_and_channel, transformer_decoder_blocks = Build_MultiDecoder_DeepSC(
                                                num_decoders = 1, 
                                                src_vocab_size = src_vocab_size, 
                                                tgt_vocab_size=[trg_vocab_size], 
                                                device=device, 
                                                src_seq_len = args.MAX_LENGTH, 
                                                tgt_seq_len = args.MAX_LENGTH)

    load_path = args.load_path
    enc_name = args.enc_name
    dec_name = args.dec_name
    out_label_f = args.out_label_f
    out_pred_f = args.out_pred_f

    f_path_enc = os.path.join(load_path, enc_name)
    f_path_dec = os.path.join(load_path, dec_name)

    check_point_enc = {'model_state_dict':torch.load(f_path_enc, map_location=torch.device(device))}
    deepsc_encoder_and_channel.load_state_dict(check_point_enc['model_state_dict'])

    check_point_dec = {'model_state_dict':torch.load(f_path_dec, map_location=torch.device(device))}
    transformer_decoder_blocks[0].load_state_dict(check_point_dec['model_state_dict'])

    out = inference(deepsc_encoder_and_channel, transformer_decoder_blocks[0], val_loader,
                     pad_idx, device, args.src_lang, args.trg_lang, args)

    label_sents = out[0]
    pred_sents = out[1]

    with open(out_label_f, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for item in label_sents:
            writer.writerow([item])
    
    with open(out_pred_f, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for item in pred_sents:
            writer.writerow([item])