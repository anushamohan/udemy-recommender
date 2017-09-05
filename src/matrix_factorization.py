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
from collections import Counter, defaultdict
from sklearn.utils import shuffle

def optimal_num_factors(train_df, validate_df):
    predictions_train_dic = {}
    predictions_valid_dic = {}

    for i in xrange(4, 20, 2):
        model = graphlab.recommender.factorization_recommender.create(sf_train, user_id='new_user_id', item_id='new_course_id',target='rating',solver='als',side_data_factorization=False, num_factors = i)
        predictions_train = model.predict(train_df)
        predictions_valid = model.predict(validate_df)
        predictions_train_dic[i] = np.sqrt(mean_squared_error(train_df['rating'], predictions_train))
        predictions_valid_dic[i] = np.sqrt(mean_squared_error(validate_df['rating'], predictions_valid))

    return predictions_train_dic, predictions_valid_dic

def optimal_reg(train_df, validate_df):
    predictions_train_dic = {}
    predictions_valid_dic = {}

    for i in [100, 1, 0.1, 0.01, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0]:
        model = graphlab.recommender.factorization_recommender.create(sf_train, user_id='new_user_id', item_id='new_course_id',target='rating',solver='als',side_data_factorization=False, num_factors = 10, regularization=i)
        predictions_train = model.predict(train_df)
        predictions_valid = model.predict(validate_df)
        predictions_train_dic[i] = np.sqrt(mean_squared_error(train_df['rating'], predictions_train))
        predictions_valid_dic[i] = np.sqrt(mean_squared_error(validate_df['rating'], predictions_valid))

    return predictions_train_dic, predictions_valid_dic


if __name__ == "__main__":
    df = pd.read_csv("processed_data.csv")
    reg = df[df["user_review_count"] > 2]
    n_users = reg.userid.nunique()
    n_items = reg.course_id.nunique()


    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    validate_df = pd.read_csv("validation.csv")

    train_df = train_df[["new_course_id", "new_user_id", "rating"]]
    validate_df = validate_df[["new_course_id", "new_user_id", "rating"]]
    test_df = test_df[["new_course_id", "new_user_id", "rating"]]

    sf_train = graphlab.SFrame(train_df)
    sf_validation = graphlab.SFrame(validate_df)
    sf_test = graphlab.SFrame(test_df)

    predictions_train_dic, predictions_valid_dic = optimal_num_factors(sf_train, sf_validation)

    plt.plot(predictions_train_dic.keys(), predictions_train_dic.values(), color = "b", label = "training error")
    plt.plot(predictions_valid_dic.keys(), predictions_valid_dic.values(), color = "g", label = "validation error")
    plt.xlabel("Number of factors")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

    predictions_train_dic, predictions_valid_dic = optimal_reg(sf_train, sf_validation)
    plt.plot(predictions_train_dic.keys(), predictions_train_dic.values(), color = "b", label = "training error")
    plt.plot(predictions_valid_dic.keys(), predictions_valid_dic.values(), color = "g", label = "validation error")
    plt.xlabel("Regularization")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

    model = graphlab.recommender.factorization_recommender.create(sf_train, user_id='new_user_id', item_id='new_course_id',
    target='rating',solver='als',side_data_factorization=False, num_factors = 10, regularization=0.1)

    predictions_test = model.predict(sf_test)
    rmse = np.sqrt(mean_squared_error(sf_test['rating'], predictions_test))

    print "test set rmse: ", rmse
