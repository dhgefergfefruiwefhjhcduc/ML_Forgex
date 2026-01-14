import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stopword = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def data_cleaning(df, skew_thres, z_thres,target):
    '''
    Cleans the input DataFrame by handling missing values, removing duplicates,
    and removing outliers based on skewness and z-score thresholds.
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
        skew_thres (float): Skewness threshold to identify skewed numeric columns.
        z_thres (float): Z-score threshold to identify outliers in numeric columns.
        target (str): Name of the target column to exclude from outlier removal.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
        '''

    df.replace(["", "NA", "na", "N/A", "n/a", "?", "--", "-","nan","Nan"], np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            mode_vals = df[col].mode(dropna=True)
            if not mode_vals.empty:
                df[col] = df[col].fillna(mode_vals.iloc[0]) 
            else:
                df[col] = df[col].fillna("")  
        else:
            med = df[col].median()
            if np.isnan(med):
                med = 0
            df[col] = df[col].fillna(med)  
    df.drop_duplicates(inplace=True, ignore_index=True)
    df = remove_outlier(df, skew_thres, z_thres,target)
    return df


def remove_outlier(df, skew_thres, z_thresh,target):
    '''
    Removes outliers from numeric columns in the DataFrame based on skewness and z-score thresholds.
    Args:
        df (pd.DataFrame): Input DataFrame.
        skew_thres (float): Skewness threshold to identify skewed numeric columns.
        z_thresh (float): Z-score threshold to identify outliers in numeric columns.
        target (str): Name of the target column to exclude from outlier removal.
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    '''
    from scipy import stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].nunique(dropna=True) <= 1 or col==target:  
            continue

        if abs(df[col].skew(skipna=True)) > skew_thres:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = ((df[col] >= lower_bound) & (df[col] <= upper_bound)) | df[col].isna()
            df = df[mask]
        else:
            z_score = stats.zscore(df[col], nan_policy='omit')
            mask = (np.abs(z_score) <= z_thresh) | df[col].isna()
            df = df[mask]

    return df.reset_index(drop=True)

def preprocess(text):
    '''
    Preprocesses the input text by removing special characters, converting to lowercase,
    removing stopwords, and lemmatizing the words.
    Args:
        text (str): Input text string to be preprocessed.
    Returns:
        str: Preprocessed text string.
    '''
    text=text.strip()
    text=text.lower()
    text=re.sub('[^a-z A-z 0-9-]+', '',text)
    text=" ".join([y for y in text.split() if y not in stopword])
    text=re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(text))
    text= " ".join(text.split())
    text=" ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def avg_wordtovec(doc,model):
    '''
    Computes the average word vector for a given document using a pre-trained Word2Vec model.
    Args:
        doc (List[str]): Tokenized document (list of words).
        model (gensim.models.Word2Vec): Pre-trained Word2Vec model.
    Returns:
        np.ndarray: Average word vector for the document.
    '''
    vector=[model.wv[word] for word in doc if word in model.wv.index_to_key]
    if not vector:
        return np.zeros(model.vector_size)
    return np.mean(vector,axis=0)
