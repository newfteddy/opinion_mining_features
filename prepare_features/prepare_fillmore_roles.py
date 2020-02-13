from nltk import DependencyGraph
import codecs
import numpy as np
import pandas as pd
import itertools
import re
import os
import pymorphy2
import math
from collections import Counter
from stop_words import get_stop_words
import time
import codecs
import os.path
from tqdm import tqdm


from isanlp_srl_framebank.pipeline_default import PipelineDefault  

ppl = PipelineDefault(address_morph=('localhost', 3333),
                      address_syntax=('localhost', 3334),
                      address_srl=('localhost', 3335))



def get_roles(text):
    res = ppl(text)
    data = []
    for sent_num, ann_sent in enumerate(res['srl']):
        for i, event in enumerate(ann_sent):
            sr = {'lemma': res['lemma'][sent_num][event.pred[0]], 'role': 'pred'}
            data.append(sr)
            for arg in event.args:
                sr = {'lemma': res['lemma'][sent_num][arg.begin], 'role': arg.tag}
                data.append(sr)
    return data


def preprocess(sentence):
    f = codecs.open(filename + '.txt', 'r')
    t = open(filename + '_prepared.txt','w')
    for line in f.readlines():
        line = re.sub(r'([.,!?()])', r' \1 ', line)
        line = re.sub('  ',' ',line)
        line = re.sub('«', '', line)
        line = re.sub('»', '', line)
        line = re.sub('"', '', line)
        line = re.sub('-', '', line)
        
        line = line.replace(r'. ', '.\n')
        t.write(line)
        
        
        
roles_map = {'pred': 'pred',
         'агенс': 'agent',
         'адресат': 'goal',
         'говорящий': 'speaker',
         'исходный посессор': 'source',
         'каузатор': 'source',
         'конечная точка': 'goal',
         'конечный посессор': 'goal',
         'контрагент': 'instrument',
         'контрагент социального отношения': 'instrument',
         'место': 'locative',
         'начальная точка': 'source',
         'пациенс': 'patient',
         'пациенс перемещения': 'patient',
         'пациенс социального отношения': 'patient',
         'потенциальная угроза': 'locative',
         'потенциальный пациенс': 'patient',
         'предмет высказывания': 'instrument',
         'предмет мысли': 'experiencer',
         'признак': 'theme',
         'признак действия': 'theme',
         'причина': 'source',
         'результат': 'goal',
         'ситуация в фокусе': 'theme',
         'содержание высказывания': 'instrument',
         'содержание действия': 'instrument',
         'содержание мысли': 'instrument',
         'способ': 'instrument',
         'срок': 'locative',
         'статус': 'locative',
         'стимул': 'source',
         'субъект восприятия': 'experiencer',
         'субъект ментального состояния': 'experiencer',
         'субъект перемещения': 'experiencer',
         'субъект поведения': 'experiencer',
         'субъект психологического состояния': 'experiencer',
         'субъект социального отношения': 'experiencer',
         'сфера': 'theme',
         'тема': 'theme',
         'траектория': 'theme',
         'цель': 'goal',
         'эффектор': 'agent',
         'эталон': 'source'}

def map_roles(roles_map, roles):
    for word in roles:
        word['role'] = roles_map[word['role']]
            


def pairs_from_roles(roles):
    tmp = list(map(dict, set(tuple(sorted(d.items())) for d in roles)))
    pairs = [{pair['lemma']+'--'+pair['role']: str(roles.count(pair))} for pair in tmp]
    return pairs


def get_text_pairs(text):
    roles = get_roles(text)
    map_roles(roles)
    
    pairs = pairs_from_roles(roles)
    return pairs


def dict_from_raw(textfile):
    docs = {}
    marks = {}
    data = codecs.open(textfile,'r')
    
    for line in data.readlines():
        tmp = line.split('|text')
        if len(tmp) == 1:
            tmp1 = line.split('|mark')
            if len(tmp1) == 1:
                continue
            elif len(tmp1) == 2:
                mark = line.split('|mark')[1]
                marks[num] = int(mark)
            else:
                print('Invalid text file')
                return
        elif len(tmp) == 2:
            text = line.split('|text')[1]
            num = int(line.split('|text')[0])
            docs[num] = text
        else:
            print('Invalid text file')
            return

    return docs, marks


def prepare_fillmore_vw(path, file, vw_file):
    docs, marks = dict_from_raw(path+file)
    
    f = open(path+vw_file,'w')

    for ind in tqdm(docs):
        f.write(str(ind) + ' |pairs ')
        data = get_roles(docs[ind])
        map_roles(roles_map, data)
        tmp = list(map(dict, set(tuple(sorted(d.items())) for d in data)))
        for pair in tmp:
            f.write(pair['lemma']+'--'+pair['role']+':'+str(data.count(pair))+' ')
        f.write('|mark   {}\n'.format(str(marks[ind])))
        
    f.close()