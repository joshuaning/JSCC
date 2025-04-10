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
# parser.add_argument('--lang1', default='da', type=str)
# parser.add_argument('--lang2', default='en', type=str)

parser.add_argument('--lang-pairs', default='da-en_en-fr_en-es', type=str)

# parser.add_argument('--multilang', default='eng-swe-dan-spa', type=str)
# parser.add_argument('--multilang-input-data-dir', default='C:/Users/Joshua Ning/Documents/Dataset/flores101_dataset', type=str)
# parser.add_argument('--multilang-out-dir', default='dataset/multilang/', type=str)

parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--MAX-LENGTH', default=30, type=int)




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
    # s = re.sub(r'\uff08.*?\uff09', '', s)

    # add white space before and after !@#$%^&*_+:"":;,./?<>-
    s = re.sub(r'([!@#$%^&*_+":;,./?<>`~\'\-])', r' \1 ', s)

    #add white space before each number
    s = re.sub(r'(?<=\d)(?=\d)', r' ', s)

    # make sure only one white space seperate each token
    s = re.sub(r'\s+', r' ', s)

    # change to lower case letter
    s = s.lower()

    return s

def trim_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=25):
    '''
    ensure each sentence in cleaned is between token length 4-25
    '''
    trimed_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            trimed_lines.append(line)
    return trimed_lines

def clean_file(fname, MIN_LENGTH=4, MAX_LENGTH=25):
    '''
    preprocess a files by cleaning and trimming
    '''

    file = open(fname, 'r', encoding='utf8')
    raw_data = file.read()
    sentences = raw_data.strip().split('\n')
    clean_data = [clean_string(data) for data in sentences]
    clean_data = trim_data(clean_data, MIN_LENGTH=MIN_LENGTH, MAX_LENGTH=MAX_LENGTH)
    file.close()
    return(clean_data)

def clean_folder(folder_name, MIN_LENGTH=4, MAX_LENGTH=25):
    '''
    return all unique, useable sentences in folder_name as a list
    '''

    unique_sentences = set()
    for file in tqdm(os.listdir(folder_name)):
        if file.endswith('.txt'): 
            process_sentences = clean_file(os.path.join(folder_name, file),  
                                           MIN_LENGTH=MIN_LENGTH, MAX_LENGTH=MAX_LENGTH)
            unique_sentences.update(process_sentences)
    
    return list(unique_sentences)

def clean_europarl_parquets(full_file_path, lang1, lang2, unique_lang1, unique_lang2, 
                   MIN_LENGTH=4, MAX_LENGTH=25):
    ul1 = unique_lang1
    ul2 = unique_lang2
    clean_lang1_sentences = []
    clean_lang2_sentences = []
    table = pa.read_table(full_file_path)
    df = table.to_pandas()

    for i in tqdm(range(len(df))):

        lang1_clean = clean_string(df.loc[i]['translation'][lang1])
        lang2_clean = clean_string(df.loc[i]['translation'][lang2])

        lang1_clean = ''.join(trim_data([lang1_clean], 
                                        MIN_LENGTH=MIN_LENGTH, MAX_LENGTH=MAX_LENGTH))
        lang2_clean = ''.join(trim_data([lang2_clean], 
                                        MIN_LENGTH=MIN_LENGTH, MAX_LENGTH=MAX_LENGTH))

        if lang1_clean in ul1 or lang1_clean == '': continue # check for duplicates
        if lang2_clean in ul2 or lang2_clean == '': continue # check for duplicates

        ul1.update([lang1_clean])
        ul2.update([lang2_clean])

        clean_lang1_sentences.append(lang1_clean)
        clean_lang2_sentences.append(lang2_clean)

    return clean_lang1_sentences, clean_lang2_sentences, ul1, ul2

def clean_europarl_parquets_in_folder(folder_name, lang1, lang2, MIN_LENGTH=4, MAX_LENGTH=25):
    unique_lang1 = set()
    unique_lang2 = set()
    clean_lang1_sentences = []
    clean_lang2_sentences = []

    for file in os.listdir(folder_name):
        if file.endswith('.parquet'): 
            fpath = os.path.join(folder_name, file)
            print("cleaning data at {}".format(fpath))

            l1sent, l2sent, unique_lang1, unique_lang2 = \
            clean_europarl_parquets(fpath, lang1, lang2, unique_lang1, unique_lang2, 
                           MIN_LENGTH=MIN_LENGTH, MAX_LENGTH=MAX_LENGTH)

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

def build_vocab(sequences, token_to_idx = {}, delim=' ', punct_to_remove=None):
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

def find_src_lang(languages):
    '''
    requires languages to be a list of language pairs in the following format:
    example: ['a-b', 'a-c', 'd-a', ... ]. where a, b, c, d, are languages 
    returns the common language in the list of strings
    '''

    split_languages = [lang_pair.split('-') for lang_pair in languages]

    # Find the intersection of the lists (common language across all pairs)
    common_languages = set(split_languages[0])
    for lang_pair in split_languages[1:]:
        common_languages.intersection_update(lang_pair)

    # The common language will be in the intersection
    src_lang = common_languages.pop() if common_languages else None

    return src_lang

def fix_src_lang_vocab(curr_out_vocab_dir):
    with open(curr_out_vocab_dir[0]) as f:
        src_data = json.load(f)
        src_token_to_idx = src_data['token_to_idx']

    for i in range(1, len(curr_out_vocab_dir)):
        with open(curr_out_vocab_dir[i]) as f:
            trg_data = json.load(f)
            trg_token_to_idx = trg_data['token_to_idx']
        for token in trg_token_to_idx.keys():
            if token not in src_token_to_idx.keys():
                src_token_to_idx[token] = len(src_token_to_idx)
        
    print('Number of unique words in src vocab: {}'.format(len(src_token_to_idx)))

    vocab = {'token_to_idx': src_token_to_idx}

    for fname in curr_out_vocab_dir:
        with open(fname, 'w') as f:
            json.dump(vocab, f)


def build_europarl(args):

    lang_pairs = args.lang_pairs.split('_')
    src_lang = find_src_lang(lang_pairs)
    src_lang_vocab_pths = []

    for lang_pair in lang_pairs:
    
        lang1, lang2 = lang_pair.split('-')

        lang_names = [lang1, lang2]


        print("extracting sentences---------------------------")
        input_folder = os.path.join(args.input_data_dir, lang_pair)
        sentence_pairs = clean_europarl_parquets_in_folder(input_folder, lang1, lang2,
                                                args.MIN_LENGTH, args.MAX_LENGTH)

        print("there are {} unique sentences in {}. \n\n".format(len(sentence_pairs[0]), lang1))
        print("there are {} unique sentences in {}. \n\n".format(len(sentence_pairs[1]), lang2))

        
        # TODO: Ideally only build vocab on the training set
        # process and save files for each language
        lang_counter = 0
        for sentences in sentence_pairs:
            curr_lang = lang_names[lang_counter]
            cur_out_dir = os.path.join(args.output_dir, lang_pair, curr_lang)
            if not os.path.exists(cur_out_dir):
                os.makedirs(cur_out_dir)
            curr_out_train_dir = os.path.join(cur_out_dir, 'train_data.pkl')
            curr_out_test_dir = os.path.join(cur_out_dir, 'test_data.pkl')
            curr_out_vocab_dir = os.path.join(cur_out_dir, 'vocab.json')

            if curr_lang == src_lang:
                src_lang_vocab_pths.append(curr_out_vocab_dir)

            # build & save vocabulary:
            print("building vocab for {} ---------------------------".format(curr_lang))
            tkns = SPECIAL_TOKENS.copy()
            token_to_idx = {}
            vocab = {}
            token_to_idx = build_vocab(sentences, token_to_idx=tkns)
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

            print("# of sentences in  training set for {} : {}".format(curr_lang, len(train_data)))
            print("# of sentences in testing set for {} : {}".format(curr_lang, len(test_data)))

            with open(curr_out_train_dir, 'wb') as f:
                pickle.dump(train_data, f)
            with open(curr_out_test_dir, 'wb') as f:
                pickle.dump(test_data, f)

            lang_counter += 1
    
    #fix the src lang vocab
    if len(curr_out_vocab_dir) <= 1: return
    print("start fixing src language vocab files")
    fix_src_lang_vocab(src_lang_vocab_pths)


        



def build_flores101(args):
    multilang = args.multilang.split('-')
    dev_path = os.path.join(args.multilang_input_data_dir, 'dev')
    dev_test_path = os.path.join(args.multilang_input_data_dir, 'devtest')

    # df = pd.DataFrame(columns=multilang)
    all_lang_lines = []

    for lang in multilang:
        fname_dev =  os.path.join(dev_path, lang + '.dev')
        fname_devtest = os.path.join(dev_test_path, lang + '.devtest')

        lines = []
        with open(fname_dev, 'r', encoding='utf-8') as f:
            lines += [line.strip() for line in f]
        with open(fname_devtest, 'r', encoding='utf-8') as f:
            lines += [line.strip() for line in f]
        
        # print(len(lines))
        all_lang_lines.append(lines)

    all_lang_lines = list(map(list, zip(*all_lang_lines))) #transpose
    df = pd.DataFrame(all_lang_lines, columns=multilang)
    print("all data")
    print(df)

    def check_length(s):
        word_count = len(s.split())
        return args.MIN_LENGTH <= word_count <= args.MAX_LENGTH


    #clean rows
    df_cleaned = df[multilang].applymap(clean_string)
    valid_rows = df_cleaned.applymap(check_length)
    valid_rows = valid_rows.all(axis=1)  # Check if all length condition are met in the row

    # Return the dataframe with valid rows
    df = df[valid_rows]

    print("clean data")
    print(df)

    # test = {"key" : df['zho_simpl'].iloc[2]}

    # with open("test.json", 'w') as f:
    #         json.dump(test, f)



if __name__ == '__main__':
    np.random.seed(1)
    args = parser.parse_args()
    build_europarl(args)
    # build_flores101(args)
    # src_lang_vocab_pths = ['dataset\\da-en\\en\\vocab.json',
    #                        'dataset\\en-es\\en\\vocab.json',
    #                        'dataset\\en-fr\\en\\vocab.json']
    # fix_src_lang_vocab(src_lang_vocab_pths)


    

