import pandas as pd
import sentencepiece as sp
import numpy as np
import matplotlib.pyplot as plt

import sentencepiece as sp


original_data_train = "../en_musik_lytics_500k/cleaned_train_lyrics.csv"
original_data_val = "../en_musik_lytics_500k/cleaned_test_lyrics.csv"

df = pd.concat([pd.read_csv(original_data_train), pd.read_csv(original_data_val)])

print(df.columns)
print(set(df.genre))
df = df[['genre', 'Lyric']].dropna()
# df = df.where((df.genre == 'rock').Lyric.dropna()

lyrics = df.Lyric.to_list()
genres = df.genre.to_list()

assert len(lyrics) == len(genres), "Lyrics and genres lists must be of the same length."

text_corpus = "\n".join(lyrics)

text_file_path = './data/en_musik_500k.txt'
with open(text_file_path, "w") as f:
    f.write(text_corpus)

