import csv
import sentencepiece as sp


original_data_train = "../tiny_stories/train.csv"
original_data_val = "../tiny_stories/validation.csv"

data = []
with open(original_data_train, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for row in reader:
        data.append(row[0].replace('\n', ' '))

with open(original_data_val, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for row in reader:
        data.append(row[0].replace('\n', ' '))

text = "\n".join(data)
text_path = "data/tiny_stories.txt"
with open(text_path, 'w', encoding='utf-8') as f:
    f.write(text)
