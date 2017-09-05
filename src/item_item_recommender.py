import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
import graphlab
from math import sqrt


def get_rmse(true_rat, predictions):
    return np.sqrt(mean_squared_error(true_rat, predictions))


def pred_one_user(userid):
    items_rated_by_user = train_df_matrix[userid-1].nonzero()[0]
    out = np.zeros(n_items)
    for item_to_rate in range(n_items):
        #print item_to_rate
        relevant_items = np.intersect1d(neighborhood[item_to_rate],items_rated_by_user, assume_unique=True)

        out[item_to_rate] = np.mean((train_df_matrix[userid-1, relevant_items] * items_sim[item_to_rate, relevant_items]) / items_sim[item_to_rate, relevant_items].sum())
    cleaned_out = np.nan_to_num(out)
    return np.where(cleaned_out > 0)[0], cleaned_out[cleaned_out > 0]


def predict_rating(df):
    predictions = []
    for i in xrange(len(df)):
        userid = df.iloc[i]["new_user_id"]
        item_not_rated = df.iloc[i]["new_course_id"]
        rel_items = item_item_pred[userid][0]
        if item_not_rated in rel_items:
            idx = np.where(item_item_pred[userid][0] == item_not_rated)
            predictions.append(item_item_pred[userid][1][idx])
        else:
            if item_not_rated in train_df.new_course_id.unique():
                predictions.append(train_df[train_df["new_course_id"] == item_not_rated]["rating"].mean())
            else:
                predictions.append(ybar)
    return predictions


if __name__ == "__main__":
    df = pd.read_csv("processed_data.csv")
    reg = df[df["user_review_count"] > 2]

    high_user = reg.new_user_id.max()
    high_item = reg.new_course_id.max()
    n_items = reg["new_course_id"].nunique()

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    validate_df = pd.read_csv("validation.csv")

    train_df = train_df[["new_course_id", "new_user_id", "rating"]]
    validate_df = validate_df[["new_course_id", "new_user_id", "rating"]]
    test_df = test_df[["new_course_id", "new_user_id", "rating"]]

    ybar = train_df.rating.mean()

    train_df_matrix = np.zeros((high_user, high_item))
    for line in train_df.itertuples():
        train_df_matrix[line[2]-1, line[1]-1] = line[3]


    items_sim = cosine_similarity(train_df_matrix.T)
    least_to_most = np.argsort(items_sim,1)
    neighborhood = least_to_most[:, -15:]

    #items_rated_by_user = train_df_matrix[userid-1].nonzero()
    item_item_pred = {}
    for i in test_df["new_user_id"].unique():
        #print i
        item_item_pred[i] = pred_one_user(i)

    predictions = predict_rating(test_df)
    print get_rmse(test_df["rating"], predictions)
