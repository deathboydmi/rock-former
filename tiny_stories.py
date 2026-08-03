import csv
import sentencepiece as sp


original_data_train = "../tiny_stories/train.csv"
original_data_val = "../tiny_stories/validation.csv"

data = None
with open(original_data_train, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)[1:]  # Skip header

with open(original_data_val, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data += list(reader)[1:]  # Skip header


data = ['<START>' + s[0] + '<END>' for s in data]

text = "".join(data)
text_path = "data/tiny_stories.txt"
with open(text_path, 'w', encoding='utf-8') as f:
    f.write(text)


sp.SentencePieceTrainer.train(input=[text_path], model_prefix='./tokenizers/tiny_stories', model_type='bpe', vocab_size=1024, accept_language=['en'], num_threads=16,
                              user_defined_symbols=['<END>', '<START>', '\n'])

sp_model = sp.SentencePieceProcessor(model_file='./tokenizers/tiny_stories.model')

input = "Hello there! How are you doing today?"
print(sp_model.encode(input, out_type=str))

