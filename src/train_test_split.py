import numpy as np
import pandas as pd


def train_test_splitting(df):

    '''takes dataframe as an input and return training set, validation set
    and test set
    The training set represents all the userid in the dataframe
    '''

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    validate_df = pd.DataFrame()
    for i in df["userid"].unique():
        dataIwant = df[df["userid"] == i]
        if len(dataIwant) == 3:
            test_index = np.random.choice(dataIwant.index, size = 1, replace=False)
            test = dataIwant.loc[test_index]
            dataIwant.drop(test_index, inplace=True)
            validate_index = np.random.choice(dataIwant.index, size = 1, replace=False)
            validate = dataIwant.loc[validate_index]
            dataIwant.drop(validate_index, inplace=True)
            test_df = pd.concat([test_df, test])
            validate_df = pd.concat([validate_df, validate])
            train_df = pd.concat([train_df, dataIwant])

        test_indices = np.random.choice(dataIwant.index, size = len(dataIwant)/4, replace=False)
        test = dataIwant.loc[test_indices]
        dataIwant.drop(test_indices, inplace=True)
        validate_indices = np.random.choice(dataIwant.index, size = len(dataIwant)/4, replace=False)
        validate = dataIwant.loc[validate_indices]
        dataIwant.drop(validate_indices, inplace=True)
        test_df = pd.concat([test_df, test])
        validate_df = pd.concat([validate_df, validate])
        train_df = pd.concat([train_df, dataIwant])

    return train_df, validate_df, test_df
