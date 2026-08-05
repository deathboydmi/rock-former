import torch
import random

import pathlib

import sentencepiece as sp

class DataLoader():
    def __init__(self,
                 text_file: str,
                 tokenizer_model: sp.SentencePieceProcessor,
                 context_size: int,
                 val_percentage: float,
                 random_seed:int
                                    ):
        file_dir = pathlib.Path(text_file).parent
        file_name = pathlib.Path(text_file).name

        if file_dir.joinpath(file_name + ".pt").exists():
            data = torch.load(file_dir.joinpath(file_name + ".pt"))
        else:
            text_corpus = None
            with open(text_file, 'r', encoding='utf-8') as f:
                text_corpus = f.read()

            data = torch.tensor(self.__divide_concuer(text_corpus, tokenizer_model), dtype=torch.long)
            torch.save(data, text_file + ".pt")


        num_chunks = len(data) // context_size
        if len(data) % context_size == 0:
            num_chunks -= 1
        chunks = []
        for i in range(num_chunks):
            chunk = data[i : i + context_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
            chunks.append(torch.stack((x, y)))

        random.seed(random_seed)
        random.shuffle(chunks)

        num_val_chunks = int(val_percentage * num_chunks)

        self.val_chunks = chunks[:num_val_chunks]
        self.train_chunks = chunks[num_val_chunks:]

    def __divide_concuer(self, text: str, tokenizer_model: sp.SentencePieceProcessor):
        parts_num = 10
        part_length = len(text) // parts_num
        total_encoded = []
        for i in range(parts_num):
            print(f"\tLoading part {i+1}/{parts_num}...")
            start = i * part_length
            end = (i + 1) * part_length if i < parts_num - 1 else len(text)
            part = text[start:end]
            total_encoded += tokenizer_model.encode(part,
                                                    out_type=int,
                                                    add_bos=True,
                                                    add_eos=True
                                                    )
        return total_encoded

    def train_data(self, batch_size: int):
        train_chunks = [
            torch.stack(self.train_chunks[i:i+batch_size]).unbind(-2) for i in range(0, len(self.train_chunks), batch_size)
            ]
        return train_chunks

    def val_data(self, batch_size: int):
        val_chunks = [
            torch.stack(self.val_chunks[i:i+batch_size]).unbind(-2) for i in range(0, len(self.val_chunks), batch_size)
            ]
        return val_chunks
