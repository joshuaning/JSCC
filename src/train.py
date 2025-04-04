import argparse
from dataset import EuroparlDataset, TextTokenConverter, collate
from torch.utils.data import DataLoader
from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime




parser = argparse.ArgumentParser()
parser.add_argument('--MAX-LENGTH', default=27, type=int)
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--num-lang', default=2, type=int)
parser.add_argument('--num-epoch', default=80, type=int)
parser.add_argument('--model-out-dir', default='weights', type=str)

def loss_fn(pred, label, pad_idx, criterion):
    loss = criterion(pred, label) * (label != pad_idx).to(torch.float32)
    return loss.mean()

def generate_mask(inputs, labels, pad_idx):
    '''
    inputs of shape [batch_size, seq_len]
    labels of shape [batch_size, labels]

    src_mask of shape [batch_size, 1, seq_len]
    combined_mask of shape [batch_size, seq_len, seq_len]

    in the masks, 1 represent masked, 0 represent transparent 
    '''

    src_mask = (inputs == pad_idx).unsqueeze(1).to(torch.float32)
    lab_mask = (labels == pad_idx).unsqueeze(1).to(torch.float32)
    attn_size = (1, input.shape()[-1], input.shape()[-1])

    #lower triangle is masked for causality
    causual_mask = torch.triu(torch.ones(attn_size), k=1).to(torch.float32)
    combined_mask = torch.max(lab_mask, causual_mask)

    return src_mask, combined_mask




def train_iter(model, loader, pad_idx, device, opt, loss_fn, criterion):
    '''
    Train iteration for DeepSC_translate
    returns the avg per patch training loss of current epoch
    '''
    model.train()
    total_loss = 0
    for i, data in tqdm(enumerate(loader)):
        opt.zero_grad()

        #TODO: need to mask the transformer here
        inputs, labels = data[:, 0, :], data[:, 1, :]
        inputs.contiguous().to(device)
        labels.contiguous().to(device)
        src_mask, combined_mask = generate_mask(inputs, labels, pad_idx)
        src_mask.contiguous().to(device)
        combined_mask.contiguous().to(device)


        pred = model(inputs)

        loss = loss_fn(pred, labels, pad_idx, criterion)

        # TODO: train MI Net if needed

        loss.backward()
        opt.step
        total_loss += loss.item()
    return loss / i

        
def val_iter(model, loader, pad_idx, device, loss_fn, criterion):
    '''
    Evalutate iteration for DeepSC_translate
    returns the avg per patch validation loss of current epoch
    '''

    model.eval()
    total_loss = 0
    with torch.no_grad:
        for i, data in tqdm(enumerate(loader)):
            #TODO: need to mask the transformer here
            inputs, labels = data[:, 0, :], data[:, 1, :]
            inputs.contiguous().to(device)
            labels.contiguous().to(device)

            pred = model(inputs)

            loss = loss_fn(pred, labels, pad_idx, criterion)
            total_loss += loss.item()

    return total_loss / i

def train_loop(model, train_loader, val_loader, pad_idx, device, opt, loss_fn, criterion, args):
    min_loss = 999999999999999
    now = datetime.now()

    #create dir for saving weights
    dt_string = now.strftime("%m_%d_%Y__%H_%M_%S")
    cur_dir = os.path.join(args.model_out_dir, dt_string)
    os.makedirs(cur_dir)

    for epoch in args.num_epoch:
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
        elif(epoch % 5 == 0): #save every 5 epoch just in case
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
    train_set = EuroparlDataset(split="train", src_lang='en', trg_lang='da')
    val_set = EuroparlDataset(split="test", src_lang='en', trg_lang='da')
    train_loader = DataLoader(train_set, num_workers=2, batch_size=args.batch_size, collate_fn = collate_fn)
    val_loader = DataLoader(val_set, num_workers=2, batch_size=args.batch_size, collate_fn = collate_fn)
    ttc_en = TextTokenConverter(lang = 'en')
    ttc_da = TextTokenConverter(lang = 'da')
    model = [] #to be replaced with net initialization
    #model = DeepSC_translate()
    pad_idx = ttc_en.get_pad_idx()

    print(pad_idx)

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
  
