import os
import os.path
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import re
import math
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer 

punctuations = ['(', ')', ';', ':', '[', ']', ',']
stop_words = set(stopwords.words('english'));


ps = PorterStemmer() 

def query_preprocessing(input_query):
    query_token = re.findall(r'[a-zA-Z0-9]{2,15}',input_query);
    query_df = pd.DataFrame(query_token,columns=["tokens"])
    query_df['tokens'] = query_df['tokens'].apply(lambda x: x.lower())

    for word in query_df['tokens']:
        if word in stop_words or word in punctuations:
            query_df = query_df[query_df.tokens != word]    
            
    query_df.reset_index(inplace = True, drop = True) 
    query_df['tokens'] = query_df['tokens'].apply(lambda x:ps.stem(x))
            
    query_frequency_dist = nltk.FreqDist(query_df['tokens'])
    queryWord_freq_tuple = {}
    for item in query_frequency_dist.items():
        queryWord_freq_tuple[item[0]] = item[1]
            
    query_frequency_matrix = pd.DataFrame(list(queryWord_freq_tuple.values()),index=queryWord_freq_tuple,columns=["query"]);
    return query_frequency_matrix
    
   


def document_preprocessing(path_to_dataset):
    list_corpus = []
    total_word_corpus = []
    
    if os.path.exists(path_to_dataset):
        path, dirs, files = os.walk(path_to_dataset).__next__()
    
    for i in dirs:
        path_to_category = path_to_dataset+i+"/"
        path_,dirs_sub,files = os.walk(path_to_category).__next__()
        for e in files:
            list_corpus.append(path_to_category+e)    
    
    list_corpus.sort()
    for document in list_corpus:
        with open(document,'rb') as f:
            data = f.read()
            data_ = data.decode(errors='ignore')
            text_token = re.findall(r'\b[a-zA-Z0-9]{2,15}\b', data_)
            dframe = pd.DataFrame(text_token,columns=["tokens"])
            dframe['tokens'] = dframe['tokens'].apply(lambda x: x.lower())
            
            for word in dframe['tokens']:
                if word in stop_words or word in punctuations:
                    dframe = dframe[dframe.tokens != word]    
            
            dframe.reset_index(inplace = True, drop = True) 
            dframe['tokens'] = dframe['tokens'].apply(lambda x:ps.stem(x))
            
            word_frequency_dist = nltk.FreqDist(dframe['tokens'])
            word_freq_tuple = {}
            for item in word_frequency_dist.items():
                word_freq_tuple[item[0]] = item[1]
            
            total_word_corpus.append(word_freq_tuple)
            
         
    frames = []
    for doc,element in zip(list_corpus,total_word_corpus):
        doc = doc.split('/')
        indx = doc[len(doc)-1]
        _df = pd.DataFrame(list(element.values()), index=element,columns=["doc-"+str(indx)])
        frames.append(_df)

    word_frequency_matrix = pd.concat(frames,axis=1)    
    word_frequency_matrix.fillna(0, inplace=True)

    return word_frequency_matrix,total_word_corpus
    
    