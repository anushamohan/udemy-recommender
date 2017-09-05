import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import text
from pattern.en import parse
from pattern.en import pprint
from pattern.vector import stem, PORTER, LEMMA
punctuation = list(".,;:!?()[]{}`'\"@#$^&*+-|=~_")
from mrjob.job import MRJob
from collections import Counter, defaultdict
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def get_parts(text):
    nouns = set()
    descriptives=set()

    text = text.lower().split()
    text = [i for i in text if i not in stop]
    text = [i for i in text if i not in punctuation]
    text = [i for i in text if len(i) > 1]
    for word, pos in nltk.pos_tag(text): # remove the call to nltk.pos_tag if `sentence` is a list of tuples as described above

        if pos in ['NN', "NNP"]: # feel free to add any other noun tags
            nouns.add(word)
        elif pos in ["JJ", "JJR"]:
            descriptives.add(word)
    return list(nouns), list(descriptives)


def choose_topic(df):
    noun_dic = {}
    for i in xrange(len(df)):
        noun = df.iloc[i]["review"][0]
        topic = max(ldamodel.get_document_topics(dictionary.doc2bow(df.iloc[i]["review"][0])), key = lambda i: i[1])
        if len(noun) == 0:
            noun_dic[i] = "no topic"

        elif topic[0] == 0:
            noun_dic[i] = "course content and video quality"
        else:
            noun_dic[i] = "instructor and explanations"

    return noun_dic

if __name__ == "__main__":
    df = pd.read_csv("processed_data.csv")
    df = df[df["user_review_count"] > 2]
    df["resp"] = df["rating"] >= df["user_rating_avg"]
    df = df[df["content"].notnull()]
    df["valid_char"] = df["content"].apply(is_ascii)
    df = df[df["valid_char"] == True]

    stop = set(stopwords.words('english'))
    stop.add("course")
    stop.add("course.")
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    df["clean_content"] = df["content"].apply(clean)
    df["review"] = df["clean_content"].apply(get_parts)
    review_parts=df.review.tolist()

    nouns = [e[0] for e in review_parts]
    dictionary = corpora.Dictionary(nouns)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in nouns]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=20)
    print ldamodel.print_topics()

    noun_dic = choose_topic(df)
    topics = noun_dic.values()

    df["topic"] = pd.Series(topics)
    df.to_csv("reg_text.csv")
