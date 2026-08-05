import sentencepiece as sp

input_paths = ["data/tiny_stories.txt", "data/en_musik_500k.txt"]

sp.SentencePieceTrainer.train(
    input=input_paths,
    model_prefix='./tokenizers/tiny_stories_en_musik_mix',
    model_type="bpe",

    vocab_size=16384,

    max_sentence_length=16384,

    input_sentence_size=1000000,
    shuffle_input_sentence=True,

    character_coverage=1.0,
    hard_vocab_limit=False,
    split_digits=True,

    num_threads=24
)
sp_model = sp.SentencePieceProcessor(model_file='./tokenizers/tiny_stories_en_musik_mix.model')
print(sp_model.vocab_size())
input = "Hello there! How are you doing today?"
print(sp_model.encode(input, out_type=str))
print(sp_model.encode(input, out_type=int))