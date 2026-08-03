import torch
import sentencepiece as sp
import random

from modules.rockformer import Rockformer

vocab_size = 2048
embed_size = 1024
num_heads = 16
ff_size = 2048
num_blocks = 8
max_context_size = 1024

def sampling(logits, temperature=0.7, top_p=0.5):
    probs = torch.softmax(logits / temperature, dim=-1).cpu()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1).tolist()
    for i, cumulative_prob in enumerate(cumulative_probs):
        if cumulative_prob > top_p:
            sorted_indices = sorted_indices[:i+1]
            break

    new_token_id = sorted_indices[random.randint(0, len(sorted_indices) - 1)]

    return new_token_id.view(1, 1).to(logits.device)

sp_model = sp.SentencePieceProcessor(model_file='./tokenizers/gutenberg_poetry.model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("./checkpoints/checkpoint_epoch_001.pt", map_location='cuda')

gen_model = Rockformer(
                        vocab_size,
                        max_context_size,
                        embed_size,
                        ff_size,
                        num_heads,
                        num_blocks
                                    ).eval().to(device)


total_params = sum(p.numel() for p in gen_model.parameters() if p.requires_grad)
print(f'Total number of parameters: {total_params}')

gen_model.load_state_dict(checkpoint["model_state_dict"])

initial_str = "How great is Spring"
tokens = torch.tensor(sp_model.encode_as_ids(initial_str)).unsqueeze(0).to(device)

with torch.no_grad():
    for _ in range(200):
        last_token_logits = gen_model(tokens)[0][-1]
        new_token_id = sampling(last_token_logits)
        tokens = torch.cat((tokens, new_token_id), dim=-1)

print(sp_model.decode_ids(tokens.cpu().tolist())[0])