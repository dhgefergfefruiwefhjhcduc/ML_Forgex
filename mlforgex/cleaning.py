import os
import re
import numpy as np
import time
from mlforgex.logger import logger
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from functools import lru_cache

stopword = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
URL_PATTERN = re.compile(
    r"(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
)

SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s-]+")

@lru_cache(maxsize=5000)
def summarize_text(text: str, max_length: int = 200, ratio: float = 0.2) -> str:
    """
    Lightweight sentence-based summarizer.

    Keeps the first `ratio` of sentences while limiting the output
    to `max_length` characters.

    Parameters
    ----------
    text : str
        Input text.
    max_length : int
        Maximum output length in characters.
    ratio : float
        Fraction of original sentences to retain.

    Returns
    -------
    str
        Summarized text.
    """

    text = str(text).strip()

    if not text:
        return ""

    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents]

    if not sentences:
        return text[:max_length]

    n_sent = max(1, int(len(sentences) * ratio))

    summary = " ".join(sentences[:n_sent])

    if len(summary) > max_length:
        summary = summary[:max_length].rstrip()

    return summary


def data_cleaning(df, skew_thres, z_thres, target):
    """
    Cleans the input DataFrame by handling missing values intelligently, removing duplicates,
    and capping outliers (Winsorization) based on skewness and z-score thresholds.
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
        skew_thres (float): Skewness threshold to identify skewed numeric columns.
        z_thres (float): Z-score threshold to identify outliers in numeric columns.
        target (str): Name of the target column to exclude from outlier removal.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    _start_clean = time.time()
    logger.info("Starting data cleaning process (imputation, duplicates, outliers)...")
    df.replace(
        ["", "NA", "na", "N/A", "n/a", "?", "--", "-", "nan", "Nan", "None", "none"],
        np.nan,
        inplace=True,
    )
    
    for col in df.columns:
        if col == target:
            continue
            
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            # Smart Categorical Imputation
            if df[col].isna().mean() > 0.10:
                # If a large chunk is missing, missingness itself might be a signal
                df[col] = df[col].fillna("Missing")
            else:
                mode_vals = df[col].mode(dropna=True)
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals.iloc[0])
                else:
                    df[col] = df[col].fillna("Missing")
        else:
            # Numeric Imputation
            med = df[col].median()
            if pd.isna(med):
                med = 0
            df[col] = df[col].fillna(med)
            
    logger.info("Imputation completed. Removing duplicates...")
    df.drop_duplicates(inplace=True, ignore_index=True)
    logger.info("Duplicates removed. Capping outliers (Winsorization)...")
    df = remove_outlier(df, skew_thres, z_thres, target)
    _end_clean = time.time()
    logger.info(f"Data cleaning process completed in {_end_clean - _start_clean:.2f}s")
    return df


def remove_outlier(df, skew_thres, z_thresh, target):
    """
    Winsorizes (caps) outliers from numeric columns in the DataFrame based on skewness and z-score thresholds.
    Instead of dropping rows (which loses valuable data), extreme values are capped at their upper/lower bounds.
    Args:
        df (pd.DataFrame): Input DataFrame.
        skew_thres (float): Skewness threshold to identify skewed numeric columns.
        z_thresh (float): Z-score threshold to identify outliers in numeric columns.
        target (str): Name of the target column to exclude from outlier removal.
    Returns:
        pd.DataFrame: DataFrame with outliers capped.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].nunique(dropna=True) <= 1 or col == target:
            continue

        if abs(df[col].skew(skipna=True)) > skew_thres:
            # For highly skewed data, calculate IQR bounds
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap (Winsorize) outliers instead of dropping rows
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        else:
            # For normally distributed data, calculate Z-score bounds
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - (z_thresh * std)
            upper_bound = mean + (z_thresh * std)
            
            # Cap (Winsorize) outliers instead of dropping rows
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df.reset_index(drop=True)


def preprocess(text):
    """
    Preprocesses the input text by removing special characters, converting to lowercase,
    removing stopwords, and lemmatizing the words.
    Args:
        text (str): Input text string to be preprocessed.
    Returns:
        str: Preprocessed text string.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip().lower()

    if not text:
        return ""

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove special characters
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)

    # Normalize whitespace
    text = " ".join(text.split())

    # Remove stopwords
    text = " ".join(
        word for word in text.split()
        if word not in stopword
    )

    # Lemmatize
    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
    )

    # Final whitespace cleanup
    text = " ".join(text.split())

    # Summarize only if text is reasonably long
    if len(text) > 500:
        text = summarize_text(text, max_length=200)

    return text
    return text


def avg_wordtovec(doc, model):
    """
    Computes the average word vector for a given document using a pre-trained Word2Vec model.
    Args:
        doc (List[str]): Tokenized document (list of words).
        model (gensim.models.Word2Vec): Pre-trained Word2Vec model.
    Returns:
        np.ndarray: Average word vector for the document.
    """
    vector = [model.wv[word] for word in doc if word in model.wv.index_to_key]
    if not vector:
        return np.zeros(model.vector_size)
    return np.mean(vector, axis=0)
