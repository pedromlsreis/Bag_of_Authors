import os, string
from sklearn import preprocessing
import numpy as np, pandas as pd
from collections import Counter
import nltk


def label_encoding(df, target_col=str):
    """
    Encodes the target column for the model, i.e.:
        - AlmadaNegreiros     -> 0;
        - CamiloCasteloBranco -> 1;
        ...
    """
    le = preprocessing.LabelEncoder()
    le.fit(df[target_col].unique())
    df[target_col] = le.transform(df[target_col])
    return df, le


def lowercase(df, text_col):
    """
    Lowercases all the text in `df[text_col]`.
    """    
    df[text_col] = df[text_col].str.lower()
    return df


def join_text(text):
    return ' '.join([str(x) for x in text])


def new_features(df, text_col):
    """
    Creates new features in `df`, computed from `df[text_col]`.
    """
    word_count = df[text_col].str.split().str.len()
  
    ellipsis_count = df[text_col].str.count(r"\.\.\.")
    df["ellipsis_per_word"] = ellipsis_count.divide(word_count, axis=0)

    df["avg_word_len"] = df[text_col].str.split(" ").apply(
        lambda x: np.sum([len(i) for i in x])
        ).divide(word_count, axis=0)
    
    selected_punct = "!,.-:;?"
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    df["punct_per_word"] = df[text_col].apply(
        lambda s: count(s, selected_punct)
        ).divide(word_count, axis=0)
    
    specific_punct_count = df[text_col].apply(
        lambda s: {k:v for k, v in Counter(s).items() if k in selected_punct}
        ).apply(pd.Series).fillna(0).divide(word_count, axis=0)
    
    df = pd.concat([df, specific_punct_count], axis=1)
    return df
    

def normalize_features(df, list_of_features):
    for feature in list_of_features:
        df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    return df


def remove_punctuation(text):
    """
        Greedy removal of all the punctuation from a list of text lines.
        However, it also removes the email and website punctuation, making
        them hard to recognise.
        Returns a list of text lines without punctuation.
    """
    no_punct_text = []
    for line in text:
        no_punct = "".join([char for char in line if char not in string.punctuation])
        no_punct_text.append(no_punct)
    return "".join(no_punct_text)


def stemming(text):
    stemmer = nltk.stem.RSLPStemmer() # Portuguese Stemmer ("Removedor de Sufixos da Língua Portuguesa")
    stopwords = nltk.corpus.stopwords.words('portuguese')
    phrase = []
    for word in text:
        if word.lower() not in set(stopwords): # stopwords removal
            if word.lower() not in string.punctuation: # punctuation removal
                phrase.append(stemmer.stem(word.lower()))
    return phrase


def tokenizer(text):
    return nltk.word_tokenize(text.lower(), language='portuguese')
