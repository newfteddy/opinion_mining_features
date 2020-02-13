import pandas as pd
import numpy as np
from collections import Counter
import codecs
import re
import os
import pymorphy2
from stop_words import get_stop_words
from tqdm import tqdm

path_to_opinion_vocab = './opinion_words.csv'
opinion_words = pd.read_csv(path_to_opinion_vocab, sep=",",encoding='utf-8', index_col=0)

def polarity(word):
    if word in opinion_words['index'].values:
        return opinion_words[opinion_words['index']==word]['word'].values[0]
    else:
        return 0
    
def assign_polarity(sn_words):
    sn_words['pol'] = [polarity(word) for word in sn_words['lemmatized']]
    return sn_words



def syntaxnet_opinion_without_tw(doc, o_words, to_file=False, vw_file=None, doc_id=None):
    i=0
   
    posw=Counter()
    negw=Counter()
    posw_pair = Counter()
    negw_pair = Counter()
    
    for j in range(doc.shape[0]):
        row = doc[j:j+1]
        word = row['lemmatized'].values[0]
        pol = row['pol'].values[0]
        
                
        if pol!=0:
            s_id=row['sentence_id'].values[0]
            childs = doc[(doc['sentence_id']==s_id) & (doc['parent_id'] == row['word_id'].values[0])]
            #Проверка на наличие отрицаний
            neg = childs[childs['dependency']=='neg']
            if len(neg)!=0:
     #           print(doc_id, ' не ', word)
                pol = -1 if pol==1 else 1
            if pol < 0:
                negw += Counter({word: abs(pol)})
            else:
                posw += Counter({word: pol})
            
            #проверка родителя            
            p_id=row['parent_id'].values[0]
            parent_row = doc[(doc['sentence_id']==s_id) & (doc['word_id']==p_id)]
            if len(parent_row)!=0:
                parent = parent_row['lemmatized'].values[0]
                if pol < 0:
                    negw += Counter({parent: abs(pol)})
                    negw_pair += Counter({parent + '_' + word: abs(pol)})

                else:
                    posw += Counter({parent: pol})
                    posw_pair += Counter({parent + '_' + word: pol})
              
            #глагол (нет проверки на тематичность. нужна ли?)
            if row['tag'].values[0] == 'VERB':
                #ищем obj и subj
                childs = doc[(doc['sentence_id']==s_id) & (doc['parent_id'] == row['word_id'].values[0])]
                obj = childs[childs['dependency'].str.contains('obj')]
                subj = childs[childs['dependency'].str.contains('subj')]
                
                
                if len(obj)!=0 and len(subj)!=0: #есть и объект и субъект
                    if pol < 0:
                        negw_pair += Counter({subj['lemmatized'].values[0] + '_' + word + '_' + obj['lemmatized'].values[0]: abs(pol)})
                    else:
                        posw_pair += Counter({subj['lemmatized'].values[0] + '_' + word + '_' + obj['lemmatized'].values[0]: pol})
    #                print(doc_id,' ',subj['lemmatized'].values[0] + '_' + word + '_' + obj['lemmatized'].values[0])
                if len(subj)!=0:

                    if pol < 0:
                        negw_pair += Counter({subj['lemmatized'].values[0] + '_' + word: abs(pol)})

                    else:
                        posw_pair += Counter({subj['lemmatized'].values[0] + '_' + word: pol})
   #                 print(doc_id,' ', subj['lemmatized'].values[0] + '_' + word)
                if len(obj)!=0: 

                    if pol < 0:
                        negw_pair += Counter({word + '_' + obj['lemmatized'].values[0]: abs(pol)})

                    else:
                        posw_pair += Counter({word + '_' + obj['lemmatized'].values[0]: pol})
  #                  print(doc_id,' ',word + '_' + obj['lemmatized'].values[0])
            #advmod
    if to_file==True:
        doc_info = u''
        doc_info=doc_info+u"|neg_pol "
                          
        for w in negw:
            doc_info=doc_info+u" "+w+u":"+str(negw[w])
        for w in negw_pair:
             doc_info=doc_info+u" "+w+u":"+str(negw_pair[w])
        
        doc_info = doc_info + u" |pos_pol "
        for w in posw:
            doc_info=doc_info+u" "+w+u":"+str(posw[w])        
        for w in posw_pair:
             doc_info=doc_info+u" "+w+u":"+str(posw_pair[w])
        doc_info+="\n"                
                
 
        return doc_info
    
    
    
    
    
def prepare_tonal_vw(sn, path, filename):
    sn = assign_polarity(sn)
    f = open(path+filename, 'w', encoding='utf-8')
    
    len_all_words=0
    list_of_painters={}
    for i in tqdm(set(sn['doc_id'])):

        doc=sn[sn['doc_id']==i]
        f.write(str(int(i))+ ' ')
        tmp = syntaxnet_opinion_without_tw(doc, opinion_words, to_file=True, vw_file=f,doc_id=i)
        f.write(tmp)
    
    f.close()
    