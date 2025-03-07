import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pa
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='C:/Users/Joshua Ning/Documents/Dataset/europarl', type=str)
parser.add_argument('--output-dir', default='dataset/', type=str)
parser.add_argument('--train-test-split', default=0.9, type=float)
parser.add_argument('--lang1', default='da', type=str)
parser.add_argument('--lang2', default='en', type=str)



SPECIAL_TOKENS = {
  '<PAD>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}

def clean_string(s):
    '''
    warning: current implementation only works on words sepearted by space
    1. remove XML-tages from string
    2. remove list formating
    3. add white space before and after special characters
    4. add white space before and after each number
    5. make sure only one white space seperates each token
    6. turn all letters to lower case for normalization
    '''
    # remove XML tags
    s = remove_tags(s)

    # remove list  formating (leading number and dot)
    s = re.sub(r'^\d+\.\s*', '', s)

    # remove content within () and any ()
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'[()]', '', s)

    # add white space before and after !@#$%^&*_+:"":;,./?<>-
    s = re.sub(r'([!@#$%^&*_+":;,./?<>`~\'\-])', r' \1 ', s)

    #add white space before each number
    s = re.sub(r'(?<=\d)(?=\d)', r' ', s)

    # make sure only one white space seperate each token
    s = re.sub(r'\s+', r' ', s)

    # change to lower case letter
    s = s.lower()

    return s

def trim_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=50):
    '''
    ensure each sentence in cleaned is between token length 4-50
    '''
    trimed_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            trimed_lines.append(line)
    return trimed_lines

def clean_file(fname):
    '''
    preprocess a files by cleaning and trimming
    '''

    file = open(fname, 'r', encoding='utf8')
    raw_data = file.read()
    sentences = raw_data.strip().split('\n')
    clean_data = [clean_string(data) for data in sentences]
    clean_data = trim_data(clean_data)
    file.close()
    return(clean_data)

def clean_folder(folder_name):
    '''
    return all unique, useable sentences in folder_name as a list
    '''

    unique_sentences = set()
    for file in tqdm(os.listdir(folder_name)):
        if file.endswith('.txt'): 
            process_sentences = clean_file(os.path.join(folder_name, file))
            unique_sentences.update(process_sentences)
    
    return list(unique_sentences)


def clean_parquets(full_file_path, lang1, lang2, unique_lang1, unique_lang2):
    ul1 = unique_lang1
    ul2 = unique_lang2
    clean_lang1_sentences = []
    clean_lang2_sentences = []
    table = pa.read_table(full_file_path)
    df = table.to_pandas()

    for i in tqdm(range(len(df))):

        lang1_clean = clean_string(df.loc[i]['translation'][lang1])
        lang2_clean = clean_string(df.loc[i]['translation'][lang2])

        lang1_clean = ''.join(trim_data([lang1_clean]))
        lang2_clean = ''.join(trim_data([lang2_clean]))

        if lang1_clean in ul1 or lang1_clean == '': continue # check for duplicates
        if lang2_clean in ul2 or lang2_clean == '': continue # check for duplicates

        ul1.update([lang1_clean])
        ul2.update([lang2_clean])

        clean_lang1_sentences.append(lang1_clean)
        clean_lang2_sentences.append(lang2_clean)

    return clean_lang1_sentences, clean_lang2_sentences, ul1, ul2



def clean_parquets_in_folder(folder_name, lang1, lang2):
    unique_lang1 = set()
    unique_lang2 = set()
    clean_lang1_sentences = []
    clean_lang2_sentences = []

    for file in os.listdir(folder_name):
        if file.endswith('.parquet'): 
            fpath = os.path.join(folder_name, file)
            print("cleaning data at {}".format(fpath))
            l1sent, l2sent, unique_lang1, unique_lang2 = clean_parquets(fpath, lang1, lang2, 
                                                                        unique_lang1, unique_lang2)
            clean_lang1_sentences += l1sent
            clean_lang2_sentences += l2sent

            
    print("unique lang1 set has lenth = ", len(unique_lang1))
    print("unique lang2 set has lenth = ", len(unique_lang2))

    return clean_lang1_sentences, clean_lang2_sentences

def tokenize(s, delim=' ',  add_start_token=True, add_end_token=True, punct_to_remove=None):
    '''
    return a list of tokens by splitting s on the specified delim. 
    Option to remove punctuation and add SOS and EOS token
    '''
    
    if punct_to_remove is not None:
        for punc in punct_to_remove:
            s = s.replace(punc, '')
            s = re.sub(r'\s+', r' ', s)
    
    tokens = s.split(delim)
    tokens = list(filter(None, tokens)) #remove empty string
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, token_to_idx = SPECIAL_TOKENS, delim=' ', punct_to_remove=None):
    '''
    returns a dictionary with token as the key and index as the value.
    token with smaller index has higher apparance freqeuency in the dataset.
    '''
    
    token_to_count = {}
    for seq in sequences:
      seq_tokens = tokenize(seq, delim=delim, punct_to_remove=punct_to_remove)
      for token in seq_tokens:
        if token not in token_to_count:
          token_to_count[token] = 0
        token_to_count[token] += 1

    # sort by frequency
    for token, _ in sorted(token_to_count.items(),  key=lambda x: x[1], reverse=True):
        if token not in SPECIAL_TOKENS.keys():
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx

if __name__ == '__main__':
    np.random.seed(1)
    args = parser.parse_args()

    lang_names = [args.lang1,  args.lang2]
    data_folder_name = os.path.join(args.input_data_dir, '-'.join(lang_names))


    print("extracting sentences---------------------------")
    sentence_pairs = clean_parquets_in_folder(data_folder_name, args.lang1, args.lang2)

    print("there are {} unique sentences in {}. \n\n".format(len(sentence_pairs[0]), args.lang1))
    print("there are {} unique sentences in {}. \n\n".format(len(sentence_pairs[1]), args.lang2))

    
    # TODO: Ideally only build vocab on the training set
    # process and save files for each language
    lang_counter = 0
    for sentences in sentence_pairs:
        curr_lang = lang_names[lang_counter]
        curr_out_train_dir = os.path.join(args.output_dir, curr_lang, 'train_data.pkl')
        curr_out_test_dir = os.path.join(args.output_dir, curr_lang, 'test_data.pkl')
        curr_out_vocab_dir = os.path.join(args.output_dir, curr_lang, 'vocab.json')

        # build & save vocabulary:
        print("building vocab for {} ---------------------------".format(curr_lang))
        token_to_idx = build_vocab(sentences)
        vocab = {'token_to_idx': token_to_idx}
        print('Number of unique words in vocab: {}'.format(len(token_to_idx)))
        with open(curr_out_vocab_dir, 'w') as f:
            json.dump(vocab, f)
        print("vocab saved to {}".format(curr_out_vocab_dir))

        # encode
        print("Begin to encode sentences")
        results = []
        for seq in tqdm(sentences):
            words = tokenize(seq)
            tokens = [token_to_idx[word] for word in words]
            results.append(tokens)
        # np.array(results)

        # select and write data for training and testing
        print('Writing Data---------------------------")')
        num_train = round(args.train_test_split * len(results))

        train_data = results[0:num_train]
        test_data = results[num_train:]

        print("amount of sentences in  training set for {} : {}".format(curr_lang, len(train_data)))
        print("amount of sentences in testing set for {} : {}".format(curr_lang, len(test_data)))

        with open(curr_out_train_dir, 'wb') as f:
            pickle.dump(train_data, f)
        with open(curr_out_test_dir, 'wb') as f:
            pickle.dump(test_data, f)

        lang_counter += 1

