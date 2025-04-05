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




parser = argparse.ArgumentParser()
parser.add_argument('--MAX-LENGTH', default=27, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-lang', default=2, type=int)
parser.add_argument('--num-epoch', default=80, type=int)
parser.add_argument('--model-out-dir', default='weights', type=str)
parser.add_argument('--src-lang', default='en', type=str)
parser.add_argument('--trg-lang', default='da', type=str)



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

    src_mask = (inputs == pad_idx).unsqueeze(-2).to(dtype=torch.float32, device=device)
    lab_mask = (labels == pad_idx).unsqueeze(-2).to(dtype=torch.float32, device=device)
    attn_size = (1, inputs.size()[-1], inputs.size()[-1])
    # attn_size = (inputs.size(0), 1, inputs.size(1))

    #lower triangle is masked for causality
    causual_mask = torch.triu(torch.ones(attn_size), diagonal=1)
    causual_mask = causual_mask.to(dtype=torch.float32, device=device)

    combined_mask = torch.max(lab_mask, causual_mask)

    return src_mask.contiguous(), combined_mask.contiguous()

def train_iter(model, loader, pad_idx, device, opt, loss_fn, criterion):
    '''
    Train iteration for DeepSC_translate
    returns the avg per patch training loss of current epoch
    '''
    model.train()
    total_loss = 0
    for i, data in tqdm(enumerate(loader)):
        opt.zero_grad()

        inputs, labels = data[:, 0, :], data[:, 1, :]

        # print("input shape = ", inputs.shape)
        # print("label shape = ", labels.shape)
        inputs = inputs.contiguous().to(device = device, dtype=torch.long)
        labels = labels.contiguous().to(device = device, dtype=torch.long)

        labels_ = labels.contiguous().to(device = device).view(-1)
        src_mask, combined_mask = generate_mask(inputs, labels, pad_idx, device)

        pred = model(inputs, src_mask, labels, combined_mask)
        pred = pred.view(-1, pred.size(-1))

        loss = loss_fn(pred, labels_, pad_idx, criterion)

        # TODO: train MI Net if needed

        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / i

def val_iter(model, loader, pad_idx, device, loss_fn, criterion):
    '''
    Evalutate iteration for DeepSC_translate
    returns the avg per patch validation loss of current epoch
    '''

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):

            inputs, labels = data[:, 0, :], data[:, 1, :]
            inputs = inputs.contiguous().to(device = device, dtype=torch.long)
            labels = labels.contiguous().to(device = device, dtype=torch.long)

            labels_ = labels.contiguous().to(device = device).view(-1)
            src_mask, combined_mask = generate_mask(inputs, labels, pad_idx, device)

            pred = model(inputs, src_mask, labels, combined_mask)
            pred = pred.view(-1, pred.size(-1))

            loss = loss_fn(pred, labels_, pad_idx, criterion)
            total_loss += loss.item()

    return total_loss / i

def train_loop(model, train_loader, val_loader, pad_idx, device, opt, loss_fn, criterion, args):
    min_loss = 999999999999999
    now = datetime.now()

    #create dir for saving weights
    dt_string = now.strftime("%m_%d_%Y__%H_%M_%S")
    cur_dir = os.path.join(args.model_out_dir, dt_string)
    os.makedirs(cur_dir)

    per_epoch_train_loss = []
    per_epoch_validation_loss = []

    for epoch in range(args.num_epoch):
        print("----------------- starting epoch {} -----------------".format(epoch))
        epoch_start_time = datetime.now()

        train_loss = train_iter(model, train_loader, pad_idx, device, opt, loss_fn, criterion)
        # TODO: train MI NET if needed
        val_loss = val_iter(model, val_loader, pad_idx, device, loss_fn, criterion)

        #save model during training
        if(min_loss > val_loss): #save best performing
            fname = 'epoch{}_best.pth'.format(epoch)
            fname =  os.path.join(cur_dir, fname)
            torch.save(model.state_dict(), fname)
            min_loss = val_loss
            print("saved weights to {}".format(fname))
        elif(epoch % 10 == 0): #save every 10 epoch just in case
            fname = 'epoch{}.pth'.format(epoch)
            fname =  os.path.join(cur_dir, fname)
            torch.save(model.state_dict(), fname)
            print("saved weights to {}".format(fname))
        
        epoch_end_time = datetime.now()
        del_time = epoch_end_time - epoch_start_time
        # print some telemetries 
        print("epoch {} took {} minutes to train".format(epoch, del_time.total_seconds()/60))
        print("training loss = {}".format(train_loss))
        print("validation loss = {}".format(val_loss))

        #save telemetries to file
        output_csv_path = os.path.join(cur_dir, 'telemetry.csv')
        per_epoch_train_loss.append(train_loss)
        per_epoch_validation_loss.append(val_loss)
        all_loss = [per_epoch_train_loss, per_epoch_validation_loss]
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
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
    train_set = EuroparlDataset(split="train", src_lang=args.src_lang, trg_lang=args.trg_lang)
    val_set = EuroparlDataset(split="test", src_lang=args.src_lang, trg_lang=args.trg_lang)
    train_loader = DataLoader(train_set, num_workers=2, batch_size=args.batch_size, collate_fn = collate_fn)
    val_loader = DataLoader(val_set, num_workers=2, batch_size=args.batch_size, collate_fn = collate_fn)
    ttc_src = TextTokenConverter(lang = args.src_lang)
    ttc_trg = TextTokenConverter(lang = args.trg_lang)
    # model = [] #to be replaced with net initialization

    src_vocab_size = ttc_src.get_vocab_size()
    trg_vocab_size = ttc_trg.get_vocab_size()
    model = Build_DeepSC(src_vocab_size = src_vocab_size, tgt_vocab_size=trg_vocab_size, device=device,
                     src_seq_len = args.MAX_LENGTH, tgt_seq_len = args.MAX_LENGTH).to(device)
    
    pad_idx = ttc_trg.get_pad_idx()

    criterion = nn.CrossEntropyLoss(reduction='none')
    opt = torch.optim.Adam(model.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    
    train_loop(model, train_loader, val_loader, pad_idx, device, opt, loss_fn, criterion, args)


    


    '''
    # for checking the dataloader for 1 batch input and output
    i = 0
    for data in test_loader:
        for sentences in data:
            if i < 1:
                # print(sentences[0][0].cpu().numpy())
                print(ttc_en.idx2text(sentences[0]))
                print(ttc_da.idx2text(sentences[1]))
                # print(len(sentences[0]))
                # print(len(sentences[1]))
            else:
                break
        i += 1
    '''
  
