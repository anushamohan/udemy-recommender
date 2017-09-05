import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats.stats import pearsonr
import graphlab
from sklearn.cross_validation import train_test_split
from math import sqrt
from pattern.vector import stem, PORTER, LEMMA

from mrjob.job import MRJob
from collections import Counter, defaultdict
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
import gensim
import string
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

punctuation = list(".,;:!?()[]{}`'\"@#$^&*+-|=~_")
stop = set(stopwords.words('english'))
stop.add("learn")
stop.add("course")
stop.add("complete")
stop.add("understand")
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    doc = doc.lower().split("-")
    doc = [i for i in doc if i.isalpha()]
    doc = [i for i in doc if i not in stop]
    doc = [ch for ch in doc if ch not in exclude]
    doc = [lemma.lemmatize(word) for word in doc]
    return doc

def category_title(title):
    title_category = {}
    for i, t in enumerate(title):
        topic = max(ldamodel.get_document_topics(dictionary.doc2bow(t)), key = lambda j: j[1])
        title_category[i] = topic[0]
    return title_category


if __name__ == "__main__":

    df = pd.read_csv("processed_data.csv")
    reg = df[df["user_review_count"] > 2]


    reg["title_clean"] = reg["published_title"].apply(clean)
    title = reg.title_clean.tolist()
    dictionary = corpora.Dictionary(title)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in title]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=16, id2word = dictionary, passes=30)
    print ldamodel.print_topics()

    title_category = category_title(title)

    reg["topic"] = title_category.values()
    print reg["topic"].head(10)
    reg.to_csv("reg_lda")
