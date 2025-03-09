import argparse
from dataset import EuroparlDataset, TextTokenConverter, collate
from torch.utils.data import DataLoader
from functools import partial



parser = argparse.ArgumentParser()
parser.add_argument('--MAX-LENGTH', default=52, type=int)
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--num-lang', default=2, type=int)


if __name__ == '__main__':

    args = parser.parse_args()

    # make sure collate with the correct parameters is correct
    collate_fn = partial(collate, maxNumToken=args.MAX_LENGTH, numlang=args.num_lang) 
    train_set = EuroparlDataset(split="train", src_lang='en', trg_lang='da')
    test_loader = DataLoader(train_set, num_workers=2, batch_size=args.batch_size, collate_fn = collate_fn)
    ttc_en = TextTokenConverter(lang = 'en')
    ttc_da = TextTokenConverter(lang = 'da')


    i = 0
    for data in test_loader:
        for sentences in data:
            if i < 1:
                print(sentences[0][0].cpu().numpy())
                print(ttc_en.idx2text(sentences[0]))
                print(ttc_da.idx2text(sentences[1]))
            else:
                break
        i += 1
