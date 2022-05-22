import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(data):
    """
    Given a pandas dataframe, encode all categorical features into labels between 0 & n-1
    Note: if a column does not have pandas dtype='category' then it will not be label encoded
    
    Parameters
    ----------
    data: input dataframe, MUST be a pandas dataframe
    
    Returns
    ----------
    data: dataframe with encoded categorical features
    encoder: dictionary where each key, value pair are:
            key: name of label encoded column
            value: sklearn LabelEncoder object for that column in the dataframe
    """
    encoder={}
    for col in data:
        if data[col].dtype=='category':
            encoder[col]=LabelEncoder()
            encoder[col].fit(data[col])
            data[col]=encoder[col].transform(data[col])
    return data, encoder


if __name__ == '__main__':
    print(f"{__file__} called")