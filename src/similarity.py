import os
import os.path
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import re
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances


def compute_similarity(_matrix):
    arr = np.array([])
        
    for doc_id in _matrix.iloc[:,1:]:
        a = _matrix.loc[:, 'query'].as_matrix().reshape(1, _matrix.shape[0])
        b = _matrix.loc[:, doc_id].as_matrix().reshape(1, _matrix.shape[0])
        cosine_theta = (cosine_similarity(a, b))
        arr = (np.append(arr, cosine_theta.round(8)))


    res = {}
    for (col,val) in zip(_matrix.columns[1:],arr):
        res[col] = val
    
    res = sorted(res.items(), key=lambda x: (x[1], x[0]), reverse=True)
    return res
    
