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
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pattern.vector import stem, PORTER, LEMMA
punctuation = list(".,;:!?()[]{}`'\"@#$^&*+-|=~_")
from mrjob.job import MRJob
from collections import Counter, defaultdict
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import gensim
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
from sklearn.utils import shuffle


def grid_search(sf_train, k):
    dic = {}
    for reg in [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
        print reg
        for lin_reg in [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
            print lin_reg
            for num_fac in [6,8,10,12,14]:
                print num_fac
                folds = graphlab.cross_validation.KFold(sf_train, k)
                rmse = []
                for train, valid in folds:
                    model= graphlab.recommender.factorization_recommender.create(train, user_id='new_user_id', item_id='new_course_id',target='rating', item_data = itemdata, user_data = userdata, side_data_factorization=True, num_factors = num_fac, regularization=reg, linear_regularization=lin_reg)
                    predictions_valid = model.predict(valid)
                    predictions_valid = np.array(predictions_valid)
                    predictions_valid[predictions_valid > 5] = 5
                    mse = mean_squared_error(valid["rating"], predictions_valid)
                    rmse.append(np.sqrt(mse))
                dic[(reg, lin_reg, num_fac)] = np.mean(rmse)
    return dic

if __name__ == "__main__":
    df = pd.read_csv("reg_text.csv")
    df["bad_review"] = df["rating"] < df["course_rating_avg"]
    df.bad_review = df.bad_review.astype(int)
    dummies_text = pd.get_dummies(df["topic"])
    lda_data = pd.read_csv("reg_lda")
    dummies_lda = pd.get_dummies(lda_data["topic"], prefix="topic_")
    lda_data = pd.concat([lda_data, dummies_lda], axis = 1)
    df = pd.concat([df, dummies_text], axis = 1)
    lda_data = lda_data[["id", 'topic__0', u'topic__1', u'topic__2', u'topic__3',
       u'topic__4', u'topic__5', u'topic__6', u'topic__7', u'topic__8',
       u'topic__9', u'topic__10', u'topic__11', u'topic__12', u'topic__13',
       u'topic__14', u'topic__15']]

    df = df.merge(lda_data, on = "id")
    df = df[["id", "new_course_id", "rating", "new_user_id", "bad_review", "is_paid", "is_practice_test_course",
         "price", "course content and video quality", "instructor and explanations", "no topic",'topic__0', u'topic__1', u'topic__2', u'topic__3',
       u'topic__4', u'topic__5', u'topic__6', u'topic__7', u'topic__8',
       u'topic__9', u'topic__10', u'topic__11', u'topic__12', u'topic__13',
       u'topic__14', u'topic__15'  ]]

    item_data = df[["new_course_id", "is_paid", "is_practice_test_course", "price", 'topic__0', u'topic__1', u'topic__2',
               u'topic__3',  u'topic__4', u'topic__5', u'topic__6', u'topic__7', u'topic__8',
       u'topic__9', u'topic__10', u'topic__11', u'topic__12', u'topic__13', u'topic__14', u'topic__15']]

    item_data.drop_duplicates(inplace=True)

    user_data = df[["id", "new_user_id", "bad_review", "course content and video quality", "instructor and explanations", "no topic"]]

    sfobs = graphlab.SFrame(df)

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    validate_df = pd.read_csv("validation.csv")

    itemdata = graphlab.SFrame(item_data)

    ids = train_df.id.tolist()
    userdata = pd.DataFrame()
    for i in xrange(len(user_data)):
        if user_data.iloc[i]["id"] in ids:
            userdata.append(user_data.iloc[i])

    userdata = graphlab.SFrame(userdata)
    train_df = train_df[["new_course_id", "new_user_id", "rating"]]
    validate_df = validate_df[["new_course_id", "new_user_id", "rating"]]
    test_df = test_df[["new_course_id", "new_user_id", "rating"]]

    train_df = pd.concat([train_df, validate_df])
    train_df = shuffle(train_df)
    sf_train = graphlab.SFrame(train_df)
    sf_test = graphlab.SFrame(test_df)

    dic = grid_search(sf_train, 2)
    print min(dic.items(), key=lambda x: x[1])
    params = min(dic, key=dic.get)
    reg = params[0]
    lin_reg = params[1]
    num_fac = params[2]
    model= graphlab.recommender.factorization_recommender.create(sf_train, user_id='new_user_id', item_id='new_course_id',target='rating', item_data = itemdata, user_data = userdata, side_data_factorization=True, num_factors = num_fac, regularization=reg, linear_regularization=lin_reg)
    predictions_test = model.predict(sf_test)
    predictions_test = np.array(predictions_test)
    predictions_test[predictions_test > 5] = 5 # capping the largest value to be 5
    rmse = np.sqrt(mean_squared_error(sf_test["rating"], predictions_test))
    print rmse
