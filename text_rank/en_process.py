#!/usr/bin/env python

import sys
import nltk

from scipy as sp

def sent_split(text):
    '''
        sentence split, use nltk package
    '''
    return nltk.sent_tokenize()


def word_seg(data):
    '''
        word tokenize, use nltk package
    '''
    if type(data) == str:
        return nltk.word_tokenize()
    elif type(data) == list:
        result = [nltk.word_tokenize(s) for s in data]
        return result
    else:
        raise Exception("Not suuport type for word_seg")


def words_to_map(data):
    '''
        hashmap of bag of words 
    '''
    wordmap = {}
    idx = 0
    for sent in data:
        for word in sent:
            if word not in wordmap:
                wordmap[word] = idx
                idx += 1
    return wordmap

# when to filter stopwords
def text_to_mat_win(sents, wordmap, win_size = 1):
    '''
        convert sentences to sparse matrix
    '''
    pairMap = {}
    for sent in sents:
        for i in range(len(sent)):
            word = sent[i].lower()
            if word not in wordmap:
                continue
            w_idx = wordmap[word]
            for j in range(1, win_size+1):
                if i + j >= len(sent):
                    break;
                pair_word = sent[i+j].lower()
                if pair_word not in wordmap:
                    continue
                pw_idx = wordmap[pair_word]
                
                max_id = max(w_idx, pw_idx)
                min_id = min(w_idx, pw_idx)
                key = "%d#%d" %(min_id, max_id)
                cnt = pairMap.get(key, 0) + 1
                pairMap[key] = cnt
                
    rows = []
    cols = []
    data = []
    for (key, value) = pairMap.items():
        w_id1, w_id2 = key.split('#')
        w_id1 = int(w_id1)
        w_id2 = int(w_id2)
        rows.append(w_id1)
        cols.append(w_id2)
        data.append(value)

        rows.append(w_id2)
        cols.append(w_id1)
        data.append(value)
     
    size = len(wordmap)
    coo_mat = sp.sparse.coo_matrix((data, (rows, cols)), shape=(size, size))
    return coo_mat

