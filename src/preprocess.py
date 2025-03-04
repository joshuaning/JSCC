import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()


def clean_string(s):
    '''
    warning: current implementation only works on words sepearted by space
    1. remove XML-tages from string
    2. add white space before and after special characters
    3. add white space before and after each number
    4. make sure only one white space seperates each token
    5. turn all letters to lower case for normalization
    '''
    # remove XML tags
    s = remove_tags(s)

    # add white space before and after !@#$%^&*()_+:"":;,./?<>-
    s = re.sub(r'([!@#$%^&*()_+"":;,./?<>-])', r' \1 ', s)

    #add white space before each number
    s = re.sub(r'(?<=\d)(?=\d)', r' ', s)

    # make sure only one white space seperate each token
    s = re.sub(r'\s+', r' ', s)

    # change to lower case letter
    s = s.lower()

    return s


if __name__ == '__main__':
    file = open("C:\\Users\\Joshua Ning\\Downloads\\DeepSC-master\\europarl\\txt\\da\\ep-11-11-16-001.txt", 'r', encoding='utf8')
    raw_data = file.read()
    sentences = raw_data.strip().split('\n')
    raw_data_input = [clean_string(data) for data in sentences]
    print(raw_data_input)


    file.close()





