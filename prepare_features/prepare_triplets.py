from nltk import DependencyGraph
import codecs
import itertools
import numpy as np
import pandas as pd
import re
import os
import pymorphy2
import math
from collections import Counter
from stop_words import get_stop_words
import codecs
import os.path
from tqdm import tqdm

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


def get_raw_sentences(text_file):
    sentences = []
    for line in codecs.open(text_file, 'r', 'utf-8'):
        sentences.append(line)
    return sentences


def get_deps(processed_sentences):
    deps = []
    for sentence in processed_sentences:
        s = u''
        for line in sentence:
            s += u"\t".join(line) + u'\n'
        deps.append(s)
    return deps


    
    
    
    
    
# Transforms conll lines into lists:
def get_lists(sent_dep):
    dependencies = []
    pos = []
    tp = []
    words = []
    for t in sent_dep.split('\n'):
        if len(t) > 1:
            splt = t.split('\t')
            dependencies.append(int(splt[6]) - 1)
            pos.append(splt[3])
            tp.append(splt[7])
            words.append(splt[1])
            
    for i in range(len(tp)):
        # Find 'and' sequences
        if tp[i] == 'conj' and pos[i] == 'VERB':
            ids = [x for x in range(len(tp)) if dependencies[x] == dependencies[i] and tp[x] == 'nsubj'] 
            for j in ids:
                words.append(words[j])
                pos.append(pos[j])
                tp.append(tp[j])
                dependencies.append(i)
        elif tp[i] == 'conj' and pos[i] != 'VERB':
            dep = dependencies[i]
            pos[i] = pos[dep]
            dependencies[i] = dependencies[dep]
            tp[i] = tp[dep]
            
        # Find complex verbs
        if tp[i] in ['xcomp','dep']:
            dep = dependencies[i]
            words[dep] = words[dep] + ' ' + words[i]
            ids = [x for x in range(len(tp)) if dependencies[x] == i]
            for j in ids:
                dependencies[j] = dep
            pos[dep] = u'VERB'
            pos[i] = 'ADD_VERB'
            tp[i] = 'ADD_VERB'
            
        # Adjective triplets
        if tp[i] == 'ADJ' and pos[dependencies[i]] == 'VERB':
            dep = dependencies[i]
            words[dep] = words[dep]+' '+words[i]
        
        # Determine negative verbs
        if tp[i] == u'neg':
            dep = dependencies[i]
            words[dep] = words[i]+' '+words[dep]
        
        # Substitude words with their names if present
        if tp[i] == u'name':
            dep = dependencies[i]
            words[dep] = words[i]

    return words, pos, dependencies, tp
            
                
# Find triplets in conll processed form        
def get_triplets(processed_sentence):
    triplets = []
    sent_dep = u''
    for line in processed_sentence:
        sent_dep += u"\t".join(line) + u'\n'
    words, pos, dependencies, tp = get_lists(sent_dep)
    
    ids = range(len(words))
    
    # regular triplets
    verbs = [x for x in ids if pos[x] == u'VERB' and tp[x] != 'amod']
    for i in verbs:
        verb_subjects = [words[x] for x in ids if tp[x] in ['nsubj','nsubjpass'] and dependencies[x] == i]
        if len(verb_subjects) == 0:
            verb_subjects.append(u'imp')
        verb_objects = [words[x] for x in ids if tp[x] == 'dobj' and dependencies[x] == i]
        if len(verb_objects) == 0:
            verb_objects.append(u'imp')
        for subj, obj in itertools.product(verb_subjects, verb_objects):
            triplets.append([subj, words[i], obj])
       
    # participle triplets
    participles = [x for x in ids if pos[x] == u'VERB' and tp[x] == 'amod']
    for i in participles:
        participle_subjects = [words[x] for x in ids if dependencies[i] == x]
        if len(participle_subjects) == 0:
            participle_subjects.append(u'imp')
        participle_objects = [words[x] for x in ids if tp[x] == 'dobj' and dependencies[x] == i]
        if len(participle_objects) == 0:
            participle_objects.append(u'imp')
        for subj, obj in itertools.product(participle_subjects, participle_objects):
            triplets.append([subj, words[i], obj])
            
    # implicit noun-noun triplets
    appos = [x for x in ids if tp[x] == u'appos']
    for i in appos:
        obj = words[dependencies[i]]
        triplets.append([words[i], u'есть', obj])

                
    #adjectives triplets
    adjectives = [x for x in ids if pos[x] == 'ADJ' and tp[x] == 'amod']
    for adj in adjectives:
        triplets.append([words[dependencies[adj]], u'есть', words[adj]])
    return triplets


# Preprocess raw text for syntaxnet input
def syntaxnet_preprocess(filename):
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
    t.close()
        

def run_syntaxnet(textfile, conllfile):
    command = "cat " + textfile + " | docker run --rm -i inemo/syntaxnet_rus > " + conllfile
    os.system(command)
    
# Get triplets from text doc or conll doc    
def get_doc_triplets(filename, conll = False):
    if conll == False: 
        syntaxnet_preprocess(filename)
        run_syntaxnet(filename + '_prepared.txt', filename + '.conll')
    processed_sentences = get_processed_sentences(filename + '.conll')
    text_triplets = []
    for sent in processed_sentences:
        text_triplets.extend(get_triplets(sent))
    return text_triplets


# Extract all subjects from triplet list
def subjects_from_triplets(triplet_list):
    stop_words = get_stop_words('russian')
    return [x[0] for x in triplet_list if x[0] != u'imp' and x[0] not in stop_words]


# Extract all objects from triplet list
def objects_from_triplets(triplet_list):
    stop_words = get_stop_words('russian')
    return [x[2] for x in triplet_list if x[2] != u'imp' and x[2] not in stop_words]


def get_subjects_from_triplet_lists(triplet_lists):
    subject_lists = []
    for triplets in triplet_lists:
        subject_lists.append(subjects_from_triplets(triplets))
    return subject_lists

# Lemmatize each triplet in triplet list
def lemmatize_triplet_list(triplet_list):
    lemmatizer = pymorphy2.MorphAnalyzer()
    stop_words = get_stop_words('russian')
    for i, triplet in enumerate(triplet_list):
        triplet_list[i] = [lemmatizer.parse(token)[0].normal_form.strip()
                           for token in triplet]

    
        
def prepare_spo(text, path):

    textfile = open(path+'spo_text.txt','w')
    textfile.write(text)
    textfile.close()

    triplets = get_doc_triplets(path+'spo_text', conll = False)
    lemmatize_triplet_list(triplets)
    subjects = subjects_from_triplets(triplets)
    objects = objects_from_triplets(triplets)
            
    subj_res = []
    obj_res = []
    
    for subject in set(subjects):
        if subject == u'—':
            continue
        subject = re.sub(':', '', subject)
        subj_res.append(subject.lower()+':'+str(subjects.count(subject)))
    
    for obj in set(objects):
        if obj == u'—':
            continue
        obj = re.sub(':', '', obj)
        obj_res.append(obj.lower()+':'+str(objects.count(obj)))
     
    return subj_res, obj_res


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
        
    data.close()

    return docs, marks


def prepare_triplet_vw(path, filename, vw_file):
    docs, marks = dict_from_raw(path+filename)
    
    f = open(path+vw_file,'w')

    for ind in tqdm(docs):
        f.write(str(ind))
        subj_res, obj_res = prepare_spo(docs[ind], path)
        print(subj_res)
        
        f.write(' |subjects ')
        for s in subj_res:
            f.write(s+' ')
            
        f.write('|objects ')
        for o in obj_res:
            f.write(o+' ')
            
        f.write('|mark   {}\n'.format(str(marks[ind])))
        
    f.close()