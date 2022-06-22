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
            encoder[col]=LabelEncoderExt()
            encoder[col].fit(data[col])
            data[col]=encoder[col].transform(data[col])
    return data, encoder




class LabelEncoderExt(object):
    """
    SOURCE: https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
    
    Really cool workaround!
    
    """
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)


if __name__ == '__main__':
    print(f"{__file__} called")
    
    country_list = ['Argentina', 'Australia', 'Canada', 'France', 'Italy', 'Spain', 'US', 'Canada', 'Argentina, ''US']

    label_encoder = LabelEncoderExt()

    label_encoder.fit(country_list)
    print(label_encoder.classes_) # you can see new class called Unknown
    print(label_encoder.transform(country_list))


    new_country_list = ['Canada', 'France', 'Italy', 'Spain', 'US', 'India', 'Pakistan', 'South Africa']
    print(label_encoder.transform(new_country_list))
