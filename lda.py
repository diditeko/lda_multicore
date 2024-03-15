import pandas as pd
import numpy as np
import nltk
from spacy.lang.id import Indonesian
import string
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE     
# from fitsne import FItSNE as TSNE                                                                                                                                                                                                                                                                                                                                         
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import time
from gensim.utils import simple_preprocess


def create_lda_inputs(text):
    # Create a Gensim dictionary
    text = [doc.split() for doc in text]
    dictionary = corpora.Dictionary(text)
    # print(dictionary)

    # Generate the document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text]
    # print(doc_term_matrix)

    return [dictionary, doc_term_matrix]

def perform_lda(doc_term_matrix, total_topics, dictionary, number_words):
    time_s = time.time()
    lda_model = LdaModel(doc_term_matrix, num_topics=total_topics,id2word = dictionary, minimum_probability=0, random_state= 21,alpha= 'asymmetric', eta='symmetric', eval_every=25,minimum_phi_value=0.01)
    topics = lda_model.show_topics(num_topics=total_topics, num_words=number_words,formatted=False)
    formatted_topics = [{"topic_num": str(topic_num), "words": [word for word, prob in words]} for topic_num, words in topics]
    time_e = time.time()
    ffinal = time_e - time_s
    print(f'time_topic : {ffinal}')
    return lda_model, formatted_topics

def perform_tsne(lda_model, doc_term_matrix):
    start_time = time.time()
    # Create a matrix of topic contributions
    hm = np.array([[y for (x, y) in lda_model[doc_term_matrix[i]]] for i in range(len(doc_term_matrix))])
    # Convert to DataFrame and fill NaN values with 0
    arr = pd.DataFrame(hm).fillna(0).values
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(arr)
    # Perform t-SNE dimension reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=21, angle=.7, init='random', perplexity=30, n_jobs=8, n_iter=100, early_exaggeration=12.0, n_iter_without_progress=10, learning_rate=300)
    tsne_lda = tsne_model.fit_transform(scaled_arr)
    # Extract x and y coordinates
    x = tsne_lda[:, 0] * 10
    y = tsne_lda[:, 1] * 10
    # Create a DataFrame to store the coordinates
    coordinates_df = pd.DataFrame({'x': x, 'y': y})
    end_time = time.time()
    final = end_time - start_time
    print(f'final_tsn  :{final})')
    return coordinates_df



