import math

def sublinear_term_frequency(freq):
    if freq == 0:
        return 0
    return 1 + math.log(freq)


def inverse_document_frequencies(word_document_df):
    idf_values = {}
    number_of_total_documents = (len(word_document_df.columns))
    for term in word_document_df.index:
        contains_token = (word_document_df.loc[term,:]).astype(bool).sum(axis=0)
        idf_values[term] = (1+math.log((number_of_total_documents)/(1+contains_token)))

    return idf_values

def compute_tfidf(word_document_df):
    idf = inverse_document_frequencies(word_document_df)
    
    for x in word_document_df.columns:
        for y in word_document_df.index:
            tf = sublinear_term_frequency(word_document_df.loc[y][x])    
            word_document_df.loc[y,x] = tf * idf[y] 
    
    return word_document_df    
