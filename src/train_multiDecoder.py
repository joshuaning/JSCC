import argparse
from dataset import *
from torch.utils.data import DataLoader
from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime
from DeepSC_model import *
import csv
from preprocess import find_src_lang
import itertools



parser = argparse.ArgumentParser()
parser.add_argument('--MAX-LENGTH', default=27, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--num-lang', default=2, type=int)
parser.add_argument('--num-epoch', default=80, type=int)
parser.add_argument('--model-out-dir', default='weights', type=str)
# parser.add_argument('--src-lang', default='en', type=str)
# parser.add_argument('--trg-lang', default='da', type=str)
parser.add_argument('--lang-pairs', default='da-en_en-fr_en-es', type=str)
parser.add_argument('--data-dir', default='dataset', type=str)




def loss_fn(pred, label, pad_idx, criterion):
    # print(pred.shape)
    # print(label.shape)
    loss = criterion(pred, label) * (label != pad_idx).to(torch.float32)
    return loss.mean()

def generate_mask(inputs, labels, pad_idx, device):
    '''
    inputs of shape [batch_size, seq_len]
    labels of shape [batch_size, labels]

    src_mask of shape [batch_size, 1, seq_len]
    combined_mask of shape [batch_size, seq_len, seq_len]

    in the masks, 1 represent masked, 0 represent transparent 
    '''

    src_mask = (inputs != pad_idx).unsqueeze(-2).to(dtype=torch.float32, device=device)
    lab_mask = (labels != pad_idx).unsqueeze(-2).to(dtype=torch.float32, device=device)
    attn_size = (1, inputs.size()[-1], inputs.size()[-1])
    # attn_size = (inputs.size(0), 1, inputs.size(1))

    #lower triangle is masked for causality
    causual_mask = torch.tril(torch.ones(attn_size))
    causual_mask = causual_mask.to(dtype=torch.float32, device=device)
    combined_mask = torch.min(lab_mask, causual_mask)

    return src_mask.contiguous(), combined_mask.contiguous()

def train_batch(data, opt, device, pad_idx, deepsc_encoder_and_channel, decoder, loss_fn, criterion):
        decoder.train()
        
        opt.zero_grad()

        inputs, labels = data[:, 0, :], data[:, 1, :]

        # print("input shape = ", inputs.shape)
        # print("label shape = ", labels.shape)
        inputs = inputs.contiguous().to(device = device, dtype=torch.long)
        labels = labels.contiguous().to(device = device, dtype=torch.long)

        labels_ = labels.contiguous().to(device = device).view(-1)
        src_mask, combined_mask = generate_mask(inputs, labels, pad_idx, device)


        channel_decoder_output = deepsc_encoder_and_channel(inputs, src_mask)
        pred = decoder(channel_decoder_output, src_mask, labels, combined_mask)
        pred = pred.view(-1, pred.size(-1))
        loss = loss_fn(pred, labels_, pad_idx, criterion)

        # TODO: train MI Net if needed
        
        loss.backward()
        opt.step()
        
        decoder.eval()

        return loss.item()


# Array parameters: transformer_decoder_blocks, loader, opt
def train_iter(deepsc_encoder_and_channel, transformer_decoder_blocks, 
               loader, pad_idx, device, opt, loss_fn, criterion):
    '''
    Train iteration for DeepSC_translate
    returns the avg per patch training loss of current epoch
    '''

    deepsc_encoder_and_channel.train()
    total_loss = np.zeros(len(loader))
    for i, data in tqdm(enumerate(loader[0])):
        #train decoder for first language
        total_loss[0] += train_batch(data, opt[0], device, pad_idx, deepsc_encoder_and_channel,
                                      transformer_decoder_blocks[0], loss_fn, criterion)
        

        #train decoder for the rest of the language
        for j in range(1, len(loader)):
            data = next(loader[j])
            total_loss[j] += train_batch(data, opt[j], device, pad_idx, deepsc_encoder_and_channel,
                                          transformer_decoder_blocks[j], loss_fn, criterion)

    return total_loss / i

def print_pred(sentences_ctr, num_to_print, inputs, labels, pred, ttc1, ttc2):
    for i, sentences in enumerate(pred):
        if sentences_ctr < num_to_print:
            print("src lang =       ", ttc1.idx2text(inputs[i]))
            print("trg lang =       ", ttc2.idx2text(sentences))
            print("trg lang gt =    ", ttc2.idx2text(labels[i]))
            print("\n")
            sentences_ctr += 1
        else:
            return sentences_ctr
    return sentences_ctr

def val_batch(data, device, pad_idx, deepsc_encoder_and_channel, 
              decoder, loss_fn, criterion, cur_lang_idx, ttc):
    #This is bad, but fixing it will be worse. Keep num_to_print <= args.batch-size
    num_to_print = 5

    inputs, labels = data[:, 0, :], data[:, 1, :]
    inputs = inputs.contiguous().to(device = device, dtype=torch.long)
    labels = labels.contiguous().to(device = device, dtype=torch.long)

    labels_ = labels.contiguous().to(device = device).view(-1)
    src_mask, combined_mask = generate_mask(inputs, labels, pad_idx, device)

    channel_decoder_output = deepsc_encoder_and_channel(inputs, src_mask)
    pred = decoder(channel_decoder_output, src_mask, labels, combined_mask)
    pred = pred.view(-1, pred.size(-1))
    loss = loss_fn(pred, labels_, pad_idx, criterion)

    # print out sample sentences
    pred = torch.argmax(pred, dim=-1) #get most probable word
    ttc1 = ttc[0]
    ttc2 = ttc[cur_lang_idx]
    sentences_ctr = print_pred(sentences_ctr, num_to_print, inputs, labels, pred, ttc1, ttc2)
    return loss.item()

# Array parameters: transformer_decoder_blocks, loader, langs
def val_iter(deepsc_encoder_and_channel, transformer_decoder_blocks, 
             loader, pad_idx, device, loss_fn, criterion, langs):
    '''
    Evalutate iteration for DeepSC_translate
    returns the avg per patch validation loss of current epoch
    langs[0] = src lang, all others are destination languages
    '''

    deepsc_encoder_and_channel.eval()
    transformer_decoder_blocks.eval()

    total_loss = np.zeros(len(loader))
    ttc = []
    for lang in langs:
        ttc.append(TextTokenConverter(lang = lang))

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):
                
            total_loss[0] += val_batch(data, device, pad_idx, deepsc_encoder_and_channel, 
                                       transformer_decoder_blocks[0], loss_fn, criterion, 1, ttc)
            
            #train decoder for the rest of the language
            for j in range(1, len(loader)):
                data = next(loader[j])
                total_loss[j] += val_batch(data, device, pad_idx, deepsc_encoder_and_channel, 
                                           transformer_decoder_blocks[j], loss_fn, criterion, 
                                           j+1, ttc)
            
    return total_loss / i

# Array parameters: transformer_decoder_blocks, train_loader, val_loader, opt
def train_loop(deepsc_encoder_and_channel, transformer_decoder_blocks, train_loader, 
               val_loader, pad_idx, device, opt, loss_fn, criterion, args, langs, cur_dir):
    min_loss = [999999999999999]*len(train_loader)

    per_epoch_train_loss = np.array([])
    per_epoch_validation_loss = np.array([])
    train_telemetry_headers = []
    val_telemetry_headers = []
    for trg_lang in langs[1:]:
        train_telemetry_headers.append("{} train loss".format(trg_lang))
        val_telemetry_headers.append("{} val loss".format(trg_lang))
    telemetry_headers = train_telemetry_headers + val_telemetry_headers
        


    for epoch in range(args.num_epoch):
        print("----------------- starting epoch {} -----------------".format(epoch))
        epoch_start_time = datetime.now()

        # added SNR to std calculation, train each epoch with a random SNR
        noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))
        noise_std = noise_std[0]
        deepsc_encoder_and_channel.change_channel(noise_std, device)

        train_loss = train_iter(deepsc_encoder_and_channel, transformer_decoder_blocks,
                                train_loader, pad_idx, device, opt, loss_fn, criterion)
        mean_train_loss = np.mean(train_loss)
        # TODO: train MI NET if needed
        deepsc_encoder_and_channel.change_channel(0.1, device)
        val_loss = val_iter(deepsc_encoder_and_channel, transformer_decoder_blocks, 
                            val_loader, pad_idx, device, loss_fn, criterion, args, langs)
        mean_val_loss = np.mean(val_loss)

        #save model during training
        if(min_loss > mean_val_loss): #save best performing
            #save encoder
            fname_encoder = 'best_encoder.pth'
            fname_encoder =  os.path.join(cur_dir, fname_encoder)
            torch.save(deepsc_encoder_and_channel.state_dict(), fname_encoder)
            print("saved best encoder weights to {}".format(fname_encoder))

            #save decoder
            for i, trg_lang in enumerate(langs[1:]):
                fname_decoder = 'best_decoder_{}.pth'.format(trg_lang)
                fname_decoder =  os.path.join(cur_dir, fname_decoder)
                torch.save(transformer_decoder_blocks[i].state_dict(), fname_decoder)
                print("saved best decoder weights to {}".format(fname_decoder))
            min_loss = mean_val_loss
        if(epoch % 10 == 0): #save every 10 epoch just in case
            #save encoder
            fname_encoder = 'epoch{}_encoder.pth'.format(epoch)
            fname_encoder =  os.path.join(cur_dir, fname_encoder)
            torch.save(deepsc_encoder_and_channel.state_dict(), fname_encoder)
            print("saved weights of encoder weights to {}".format(fname_encoder))

            #save decoder
            for i, trg_lang in enumerate(langs[1:]):
                fname_decoder = 'epoch{}_encoder_{}.pth'.format(epoch, trg_lang)
                fname_decoder =  os.path.join(cur_dir, fname_decoder)
                torch.save(transformer_decoder_blocks[i].state_dict(), fname_decoder)
                print("saved weights of decoder weights to {}".format(fname_decoder))

        epoch_end_time = datetime.now()
        del_time = epoch_end_time - epoch_start_time
        # print some telemetries 
        print("epoch {} took {} minutes to train".format(epoch, del_time.total_seconds()/60))
        print("mean training loss = {}".format(mean_train_loss))
        print("mean validation loss = {}".format(mean_val_loss))

        #save telemetries to file
        output_csv_path = os.path.join(cur_dir, 'telemetry.csv')
        per_epoch_train_loss= np.append(per_epoch_train_loss, train_loss, axis=0)
        per_epoch_validation_loss = np.append(per_epoch_validation_loss, val_loss, axis=0)

        all_loss = np.append(per_epoch_train_loss, per_epoch_validation_loss, axis=1)
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(telemetry_headers)
            writer.writerows(all_loss)



if __name__ == '__main__':

    # basic checks and use GPU
    print("PyTorch Version: ",torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only.")

    # initialize some parameters
    args = parser.parse_args()
    collate_fn = partial(collate, maxNumToken=args.MAX_LENGTH, numlang=args.num_lang) 

    #process the desired languages from args
    now = datetime.now()
    #create dir for saving weights

    dt_string = now.strftime("%m_%d_%Y__%H_%M_%S")
    cur_dir = os.path.join(args.model_out_dir, dt_string)
    os.makedirs(cur_dir)

    #prepare dataloaders for each languages
    lang_pairs = args.lang_pairs.split('_')
    split_languages = [lang_pair.split('-') for lang_pair in lang_pairs]
    split_languages = [item for sublist in split_languages for item in sublist]
    split_languages = set(split_languages)
    src_lang = find_src_lang(lang_pairs)
    split_languages.remove(src_lang)
    trg_langs = list(split_languages)
    langs = [src_lang] + trg_langs
    num_trg_langs = len(trg_langs)
    
    print('source language = {}'.format(src_lang))
    print('target language = ', trg_langs)
    
    #create objects related with langs
    train_loader = []
    val_loader = []
    trg_vocab_size = []

    for i, trg_lang in enumerate(trg_langs):
        lang_data_dir = os.path.join(args.data_dir, src_lang+'-'+trg_lang)
        train_set = EuroparlDataset(data_dir=lang_data_dir, split="train", 
                                    src_lang=src_lang, trg_lang=trg_lang)
        val_set = EuroparlDataset(data_dir=lang_data_dir, split="test", 
                                  src_lang=src_lang, trg_lang=trg_lang)
        cur_train_loader = DataLoader(train_set, num_workers=2, batch_size=args.batch_size, 
                                    collate_fn = collate_fn, shuffle=True)
            
        cur_val_loader = DataLoader(val_set, num_workers=2, batch_size=args.batch_size, 
                                    collate_fn = collate_fn, shuffle=True)
        
        #make the multi-lang dataloader be able to cycle
        if i > 1:
            cur_train_loader= itertools.cycle(cur_train_loader)
            cur_val_loader = itertools.cycle(cur_val_loader)
            
        train_loader.append(cur_train_loader)
        val_loader.append(cur_val_loader)
        ttc_trg = TextTokenConverter(data_dir=lang_data_dir, lang = trg_lang)
        trg_vocab_size.append(ttc_trg.get_vocab_size())
        ttc_src = TextTokenConverter(data_dir=lang_data_dir, lang = src_lang)
    
    src_vocab_size = ttc_src.get_vocab_size()

    # vector input: tgt_vocab_size, each element must be an int
    deepsc_encoder_and_channel, transformer_decoder_blocks = Build_MultiDecoder_DeepSC(
                                                num_decoders = num_trg_langs, 
                                                src_vocab_size = src_vocab_size, 
                                                tgt_vocab_size=trg_vocab_size, 
                                                device=device, 
                                                src_seq_len = args.MAX_LENGTH, 
                                                tgt_seq_len = args.MAX_LENGTH)
    
    pad_idx = ttc_trg.get_pad_idx()

    criterion = nn.CrossEntropyLoss(reduction='none')

    # have different opt for each decoder
    opt = []
    params = list(deepsc_encoder_and_channel.parameters())
    for decoder in transformer_decoder_blocks:
        params_n = params + list(decoder.parameters())
        opt.append(torch.optim.Adam(params_n,
                        lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4))
    
    train_loop(deepsc_encoder_and_channel, transformer_decoder_blocks, train_loader,
                val_loader, pad_idx, device, opt, loss_fn, criterion, args, langs, cur_dir)

