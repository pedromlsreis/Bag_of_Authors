import os
from sklearn import preprocessing
import numpy as np, pandas as pd
from collections import Counter


def get_dataframe(path_to_train=str, author_list=list, preserve_blank_lines=False,
                  join_every_line=True, separator=""):
    """
    Receives path/to/train/folder which contains a folder for each author with the excerpts inside.
        path_to_train        = str      -> caminho para o diretório de train
        author_list          = list     -> lista de todos os autores
        preserve_blank_lines = bool     -> False para excluir linhas em branco
        join_every_line      = bool     -> True para mostrar o texto como uma única string
        separator            = str      -> separador para juntar texto numa só string (só se `join_every_line` = True)
    """
    list_of_texts, list_of_authors = [], []

    # iterar os autores e os seus diretórios
    for author in author_list:
        if not path_to_train.endswith("/"):
            path_to_train += "/"
        author_folder = path_to_train + author + "/"
        
        # iterar os excertos de cada autor e obter o seu path/to/file.txt
        for excerto in [x for x in os.listdir(author_folder) if x.endswith(".txt")]:
            text_list = []
            text_file = author_folder + excerto
            
            # abrir o ficheiro e fazer juntar cada linha a uma lista
            with open(text_file, encoding="utf-8") as f:
                for line in f:
                    inner_list = [line.strip() for line in line.split("split character")]
                    if preserve_blank_lines:
                        text_list.append(inner_list[0])
                    else:
                        if len(inner_list[0]) > 0:
                            text_list.append(inner_list[0])
            
            # juntar lista numa só string (opcional)
            if join_every_line:
                text_list = separator.join(text_list)
            list_of_texts.append(text_list)
            list_of_authors.append(author)
    
    # devolver pandas DataFrame com dataset de train
    return pd.DataFrame({"text":list_of_texts, "author":list_of_authors})


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


def feature_engineering(df, text_col):
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


# cada vez que uma nova função for criada, introduzi-la em "clean_text()"
def clean_text(df, text_col=str):
    """
        Compiles all the preprocessing functions inside a single function.
    """
    df[text_col] = df[text_col].apply(remove_punctuation)
    return df