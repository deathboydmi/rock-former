import torch
import sentencepiece as sp
import random
from time import time

from modules.rockformer import Rockformer

vocab_size = 16384
embed_size = 512
num_heads = 16
ff_size = 2048
num_blocks = 8
max_context_size = 2048

def sampling(logits, temperature=1.2, p=0.95):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove = cumulative_probs > p
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    sorted_logits[remove] = -torch.inf

    probs = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, 1)

    return sorted_indices[sampled]


def generate(model: Rockformer, sp_model, initial_str, device, max_len=512, online=False):
    print(f"\tGenerating text with initial string: {initial_str}")

    input_tokens = sp_model.encode(initial_str, out_type=int, add_bos=True, add_eos=False)
    tokens = input_tokens
    input_tokens = torch.tensor(input_tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        start = time()
        for _ in range(max_len - len(initial_str)):
            # TODO: crop input_tokens to (len(input_tokens) max_context_size):
            # TODO: save all tokens
            if input_tokens.numel() > model.max_context_size:
                offset = input_tokens.numel() - model.max_context_size + 1
                input_tokens = input_tokens[..., offset:]

            last_token_logits = model(input_tokens)[0][-1]
            new_token = sampling(last_token_logits)
            tokens.append(new_token.cpu().item())
            if online:
                print(sp_model.decode(tokens), end='', flush=True)

            new_token_id = new_token.unsqueeze(0).to(device)
            input_tokens = torch.cat((input_tokens, new_token_id), dim=-1)
        end = time()

    duration = end - start
    avg_latency = (end - start) / len(tokens)
    throughput = 1 / avg_latency
    print(f"\tGeneration time: {duration}s",
          f"avg latency: {avg_latency}",
          f"throughput: {throughput}", sep="\t| \t")

    if online:
        print()

    generated_text = sp_model.decode_ids(tokens)
    return generated_text


if __name__ == "__main__":
    sp_model = sp.SentencePieceProcessor(model_file='./tokenizers/tiny_stories_en_musik_mix.model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("./checkpoints/checkpoint_step_4124.pt", map_location='cuda')

    gen_model = Rockformer(
                            vocab_size,
                            max_context_size,
                            embed_size,
                            ff_size,
                            num_heads,
                            num_blocks
                                        ).to(device)

    gen_model.generation()

    gen_model.load_state_dict(checkpoint["model_state_dict"])

    generated_text = generate(gen_model, sp_model,
                            "Should I sing a hardcore song?",
                            device, max_len=512, online=False)

    print(generated_text)