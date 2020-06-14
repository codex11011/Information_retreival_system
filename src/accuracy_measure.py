import math
import numpy as np
import pandas as pd


def compute_precision_recall_at_k(k,relevant_class,result_vector):
    
    count_relevant = 0
    total_relevant = 0
    for indx,(id,val) in enumerate(result_vector):
        if int(id[4]) is relevant_class:
            if indx < k:
                count_relevant+=1
            total_relevant+=1
       
    
    precision = count_relevant/k
    recall = count_relevant/total_relevant
    return precision,recall

def compute_fscore(precision,recall):
    fscore = (2*precision*recall)/(precision+recall)
    return fscore