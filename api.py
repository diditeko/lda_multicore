from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
from lda import perform_lda, create_lda_inputs, perform_tsne
from typing import List
from itertools import chain
import os
from openTSNE import TSNE 
import shutil
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")


app = FastAPI()

# Load your LDA model and dictionary here
model_filename = "model\model\lda_model3"
# lda_model =None



lda_model = LdaModel.load(model_filename)
total_topics = 5 # jumlah topik yang akan di extract
number_words = 3

class TweetData(BaseModel):
    tweet: str
    like: int
    comment: int
    retweet: int
    view: int

# Modify the endpoint to accept a list of TweetData objects
@app.post("/topic-modeling_multithread/")
async def perform_topic_modeling(data: List[TweetData]):
    start_time = time.time()
    texts = [tweet.tweet for tweet in data]
    likes = [tweet.like for tweet in data]
    replies = [tweet.comment for tweet in data]  # Assuming "comment" corresponds to "reply"
    views = [tweet.view for tweet in data]
    retweets = [tweet.retweet for tweet in data]

    dictionary, doc_term_matrix = create_lda_inputs(texts)

    # Parallelize the LDA computation
    with ThreadPoolExecutor(max_workers=4) as executor:
        lda_model, topics = next(executor.map(lambda x: perform_lda(*x), [(doc_term_matrix, total_topics, dictionary, number_words)]))

    # Parallelize the t-SNE computation
    with ThreadPoolExecutor(max_workers=4) as executor:
        coordinates_df = next(executor.map(lambda x: perform_tsne(*x), [(lda_model, doc_term_matrix)]))

    results = []
    for i, tweet in enumerate(data):
        text_topics = lda_model[doc_term_matrix[i]]
        dominant_topic = max(text_topics, key=lambda x: float(x[1]))[0]
        topic_perc_contrib = float(max(text_topics, key=lambda x: float(x[1]))[1])

        x_coord = coordinates_df.iloc[i]['x']
        y_coord = coordinates_df.iloc[i]['y']

        result = {
            "x": x_coord,
            "y": y_coord,
            "Dominant_Topic": dominant_topic,
            "Topic_Perc_Contrib": topic_perc_contrib,
            "Text": tweet.tweet,
            "Like": tweet.like,
            "Reply": tweet.comment,
            "Retweets": tweet.retweet,
            "Views": tweet.view
        }
        results.append(result)

    end_time = time.time()
    final = end_time - start_time
    print(f'final_time  :{final})')

    return {"topic": topics, "topic_data": results}


if __name__ == "__main__":
    import uvicorn
    # load_lda_model()

    uvicorn.run(app, host="127.0.0.1", port=1501)
