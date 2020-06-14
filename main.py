import json
import pandas as pd
from src.preProcessing import document_preprocessing,query_preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from src.similarity import compute_similarity
from src.tfidf import compute_tfidf
from multiprocessing import  Pool
from src.parallelProcessing import parallelize_dataframe
from src.accuracy_measure import compute_precision_recall_at_k,compute_fscore
from src.write_to_json import write_to_json_file

def initialize_dataframe_dataset(path_to_dataset):
    _document_preprocessed,_tokenized_doc_list = document_preprocessing(path_to_dataset)
    _weighted_matrix = parallelize_dataframe(_document_preprocessed,compute_tfidf)
    return _weighted_matrix
    
    
path_to_dataset = "irws_dataset/"
_weighted_matrix = initialize_dataframe_dataset(path_to_dataset)
k = 20
    
def retreive_documents():
    size_corpus = len(_weighted_matrix.columns)
    _query_preprocessed = pd.DataFrame()
    queries = {}
    
    ground_truth = {}
    with open('./resource/query.json') as f:
        queries = json.load(f)
    
    with open('./resource/relevance_judgement.json') as f:
        ground_truth = json.load(f)
    
    queries = dict(list(queries.items()))
    for queryId,query in queries.items():
        _query_preprocessed = query_preprocessing(query)
        _matrix =  pd.concat([ _query_preprocessed,_weighted_matrix],axis = 1)
        _matrix.fillna(0, inplace=True)
        _similarity_vector = compute_similarity(_matrix)
        relevant_class = ground_truth[queryId]['category']
        precision,recall = compute_precision_recall_at_k(k,relevant_class,_similarity_vector)
        fscore = compute_fscore(precision,recall)
        _result_dict = {}
        for element in _similarity_vector[:k]:
            _result_dict[element[0]] = element[1]
        
            
        json_object = {
            'queryId':queryId,
            'query':query,
            'result':_result_dict,
            'precision':precision,
            'recall':recall,
            'fscore':fscore
        }
        write_to_json_file(json_object,queryId)
        print(queryId + ' processed')
        print('\n')
        
        
    
def random_query():
    input_query = str(input())
    _query_preprocessed = pd.DataFrame()
    _query_preprocessed = query_preprocessing(input_query)
    _matrix =  pd.concat([ _query_preprocessed,_weighted_matrix],axis = 1)
    _matrix.fillna(0, inplace=True)
    _similarity_vector = compute_similarity(_matrix)
    for indx,val in _similarity_vector[:k]:
        print(indx,val)

        

# retreive_documents()
random_query()