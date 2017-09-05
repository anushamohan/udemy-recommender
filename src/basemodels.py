import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import pearsonr
import graphlab
from train_test_split import train_test_splitting
from sklearn.cross_validation import train_test_split

def get_rmse(s, s_predict):
    '''takes true values and predicted values as inputs and returns the root mean squared error'''
    mse = mean_squared_error(s, s_predict)
    return np.sqrt(mse)


def baseline(df):
    return ybar*df.shape[0]

def bias_baseline(df):
    '''the function takes user bias and item bias into account'''
    predictions = []
    for i in xrange(len(df)):
        row = df.iloc[i]
        user = row["userid"]
        course = row["course_id"]
        if course not in train_courseids_dic.keys() and user not in train_userids_dic.keys():
            predictions.append(ybar)
        elif course not in train_courseids_dic.keys():
            predictions.append(ybar+train_userids_dic[user])
        elif user not in train_userids_dic.keys():
            predictions.append(ybar+train_courseids_dic[course])
        else:
            predictions.append(ybar+train_userids_dic[user]+train_courseids_dic[course])

    return predictions


if __name__ == "__main__":
    df = pd.read_csv("processed_data.csv")

    reg = df[df["user_review_count"] > 2]
    print reg.shape

    train_df, validate_df, test_df = train_test_splitting(reg)
    ybar = train_df["rating"].mean()
    print ybar

    predictions_train_base = baseline(train_df)
    predictions_test_base = baseline(test_df)
    predictions_validation_base = baseline(validate_df)

    train_base_rmse = get_rmse(train_df["rating"], predictions_train_base)
    test_base_rmse = get_rmse(test_df["rating"], predictions_test_base)
    validation_base_rmse = get_rmse(validate_df["rating"], predictions_validation_base)


    print "base model training rmse: ", train_base_rmse
    print "base model validation rmse: ", validation_base_rmse
    print "base model test rmse: ", test_base_rmse

    train_userids_dic = {}
    for i in train_df["userid"].unique():
        train_userids_dic[i] = (train_df[train_df["userid"]==i].rating.mean())-ybar

    train_courseids_dic = {}
    for i in train_df["course_id"].unique():
        train_courseids_dic[i] = (train_df[train_df["course_id"]==i].rating.mean())-ybar

    train_avgs={'mean':ybar, 'users':train_userids_dic, 'items':train_courseids_dic}

    predictions_train_biasmodel = bias_baseline(train_df)
    predictions_test_biasmodel = bias_baseline(test_df)
    predictions_validation_biasmodel = bias_baseline(validate_df)

    train_bias_rmse = get_rmse(train_df["rating"], predictions_train_biasmodel)
    test_bias_rmse = get_rmse(test_df["rating"], predictions_test_biasmodel)
    validation_bias_rmse = get_rmse(validate_df["rating"], predictions_validation_biasmodel)

    print "bias model training rmse: ", train_bias_rmse
    print "bias model validation rmse: ", validation_bias_rmse
    print "bias model test rmse: ", test_bias_rmse
