# this model follows this tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

'''
Syntax:
# ?? = question to come back to

'''

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random 

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS & EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words + 1
        else:
            self.word2count[word] += 1

### DATA PRE PROCESSING 

# getting the data from unicode to english ?

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

'''
def unicodeToAscii(s):
    print("s is ", s)
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    '''

# getting rid of punctuation & capitalization
def normalizeString(s):
    s = unicodeToAscii(s.lower)
    s = s.strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# creating a function that reads the file
def readLangs(lang1, lang2, reverse=False):
    print("reading lines...")

    # splitting the data into lines
    lines = open('/Users/selenawilliams/Desktop/Tech/Seq2seq2/data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    
    # splitting every line into a pair & normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # reverse the pairs, making Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# trimming the data 
# since there are a lot of diff words & sentences, we trim the data to make compute time reasonable

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs): # ?? 
    return [pair for pair in pairs if filterPair(pair)] # what does this line say?

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("counted words: ")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

