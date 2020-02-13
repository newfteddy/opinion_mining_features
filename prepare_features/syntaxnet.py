import pandas as pd
import numpy as np
from collections import Counter
import codecs
import re
import os
import pymorphy2
from stop_words import get_stop_words
from tqdm import tqdm


def syntaxnet_preprocess(path, filename):
    f = codecs.open(path + filename + '.txt', 'r')
    t = open(path + filename + '_prepared.txt','w')
    for line in f.readlines():
        line = re.sub(r'([.,!?()])', r' \1 ', line)
        line = re.sub('  ',' ',line)
        line = re.sub('«', '', line)
        line = re.sub('»', '', line)
        line = re.sub('"', '', line)
        line = re.sub('-', '', line)
        
        line = line.replace(r'. ', '.\n')
        t.write(line)
        

def run_syntaxnet(textfile, conllfile):
    command = "cat " + textfile + " | docker run --rm -i inemo/syntaxnet_rus > " + conllfile
    os.system(command)
    
    
def get_processed_sentences(conll_file):
    processed_sentences = []
    sentence = []
    for line in codecs.open(conll_file, 'r', 'utf-8'):
        if len(line) == 1:
            processed_sentences.append(sentence)
            sentence = []
        else:
            word = line.split("\t")
            sentence.append(word)
    return processed_sentences



def lemmatize(word):
    lemmatizer = pymorphy2.MorphAnalyzer()
    stop_words = get_stop_words('russian')
    return lemmatizer.parse(word)[0].normal_form.strip()


def get_sn_df(docs, path):
    sn_output = pd.DataFrame()

    for doc_id in tqdm(docs):
        text_input = codecs.open(path+'sn_input.txt','w')
        text_input.write(docs[doc_id])
        text_input.close()

        syntaxnet_preprocess(path, 'sn_input')
        run_syntaxnet(path+'sn_input_prepared.txt', path+'sn_input.conll')
        processed_sentences = get_processed_sentences(path+'sn_input.conll')

        for i, sent in enumerate(processed_sentences):
            for word in sent:
                sn_word = {
                    'word_id': word[0],
                    'word': word[1],
                    'parent_id': word[6],
                    'tag': word[3],
                    'dependency': word[7],
                    'lemmatized': lemmatize(word[1]),
                    'sentence_id': i,
                    'doc_id': doc_id
                }
                sn_output = sn_output.append(sn_word, ignore_index=True)
    return sn_output


def dict_from_raw(path, textfile):
    docs = {}
    data = codecs.open(path+textfile,'r')
    
    for line in data.readlines():
        tmp = line.split('|text')
        if len(tmp) == 1:
            continue
        elif len(tmp) == 2:
            text = line.split('|text')[1]
            num = int(line.split('|text')[0])
            docs[num] = text
        else:
            print('Invalid text file')
            return
    return docs

def get_sn_from_raw(path, textfile, csv_path):
    docs = dict_from_raw(path, textfile)
    sn = get_sn_df(docs, path)
    sn.to_csv(csv_path)
    return sn