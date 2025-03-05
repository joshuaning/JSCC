import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='C:/Users/Joshua Ning/Documents/Dataset/europarl/txt/da', type=str)
parser.add_argument('--output-train-dir', default='dataset/da/train_data_da.pkl', type=str)
parser.add_argument('--output-test-dir', default='dataset/da/test_data_da.pkl', type=str)
parser.add_argument('--output-vocab', default='dataset/da/vocab_da.json', type=str)



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
    s = re.sub(r'([!@#$%^&*_+"":;,./?<>-])', r' \1 ', s)

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
    preprocess a files
    '''

    file = open(fname, 'r', encoding='utf8')
    raw_data = file.read()
    sentences = raw_data.strip().split('\n')
    clean_data = [clean_string(data) for data in sentences]
    clean_data = trim_data(clean_data)
    file.close()
    return(clean_data)

def clean_folder(folder_name):
    unique_sentences = set()
    for file in tqdm(os.listdir(folder_name)):
        if file.endswith('.txt'): 
            process_sentences = clean_file(os.path.join(folder_name, file))
            unique_sentences.update(process_sentences)
    
    return list(unique_sentences)


if __name__ == '__main__':
    args = parser.parse_args()
    sentences = clean_folder(args.input_data_dir)
    print("there are {} unique sentences. \n\n".format(len(sentences)))

    for i in range(10):
        print(sentences[i])






