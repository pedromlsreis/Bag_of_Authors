import os
import pandas as pd


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


def subset_dataframe(df, chunksize=500):
    """
    Receives dataframe with columns `author` and `tokens`.
    and explodes each text row into multiple rows with text with n=`chunksize` words.
    Returns:
        new_df : new dataframe with columns `author` and `tokens`
        indexes_mapping : dictionary that maps the old row ids (keys)
            to the new ones (values)
    """
    new_df = pd.DataFrame(columns=["author", "tokens"])
    indexes_mapping = {}
    last_index = -1

    for index in df.index:
        tokens = df.loc[index, "tokens"]
        author = df.loc[index, "author"]
        chunks = [tokens[x : (x + chunksize)] for x in range(0, len(tokens), chunksize)]
        if last_index >= 0:
            indexes_mapping[index] = list(range(last_index, last_index+len(chunks)))
        else:
            indexes_mapping[index] = list(range(index, index+len(chunks)))
        last_index = indexes_mapping[index][-1]+1
        new_df = pd.concat([
            new_df,
            pd.DataFrame({
                "author": author,
                "tokens": chunks
                }),
            ])

    new_df.reset_index(drop=True, inplace=True)
    return new_df, indexes_mapping
