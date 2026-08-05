import math
from time import time

import torch
import sentencepiece as sp

import pathlib

from modules.rockformer import Rockformer
from modules.data_loader import DataLoader


vocab_size = 16384
embed_size = 512
num_heads = 16
ff_size = 2048
num_blocks = 8
max_context_size = 2048

context_size = 512
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_model_path = './tokenizers/tiny_stories_en_musik_mix.model'
text_path = './data/tiny_stories.txt'
checkpoint_path_pattern = "./checkpoints/checkpoint_step_{0:03d}.pt"

FINE_TUNING = False
pre_trained_model_path = "./checkpoints/tiny_stories/checkpoint_epoch_010.pt"

epochs = 10
print_every_step = 500
eval_every_step = 10000
generate_every_step = 5000

random_seed = 777
torch.manual_seed(random_seed)

print("Initializing model...")
model = Rockformer(
                    vocab_size,
                    max_context_size,
                    embed_size,
                    ff_size,
                    num_heads,
                    num_blocks
                                ).to(device).train()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\tTotal number of parameters: {total_params}')
model.compile()

print("Initializing optimizer and loss function...")
optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0003,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1
        )
cross_entropy = torch.nn.CrossEntropyLoss()

if FINE_TUNING:
    checkpoint = torch.load(pre_trained_model_path, map_location='cuda')
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

print("Initializing data loader...")
sp_model = sp.SentencePieceProcessor(model_file=tokenizer_model_path)
data_loader = DataLoader(
                            text_path,
                            sp_model,
                            context_size,
                            0.3,
                            random_seed
                                        )

train_data = data_loader.train_data(batch_size)
eval_data = data_loader.val_data(batch_size)

print("Initializing Learning rate scheduler...")
total_steps = len(train_data) * epochs
print(f"\tTotal training steps: {total_steps}")
print(f"\tTotal processed tokens number: {total_steps * batch_size * context_size}")

warmup_steps = 1000
min_lr = 3e-5
max_lr = 3e-4
min_factor = min_lr / max_lr
def lr_schedule(step):
    global warmup_steps, total_steps, min_factor
    if step < warmup_steps:
        return step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    return min_factor + (1 - min_factor) * (
        0.5 * (1 + math.cos(math.pi * progress))
    )
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_schedule
)

checkpoints_dir = pathlib.Path("checkpoints")
checkpoints = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
if checkpoints:
    print("Loading checkpoint...")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}...")
else:
    start_epoch = 0


def evaluate(model, eval_data):
    print("\tEvaluating...")
    model = model.eval()
    with torch.no_grad():
        losses = []
        for x, y in eval_data:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y_pred = model(x.to(device))
                loss = cross_entropy(y_pred.view(-1, vocab_size), y.reshape(-1).to(device=device))
                losses.append(loss.cpu().detach().item())

    # Collecting loss statistic
    val_avg_loss = sum(losses)/len(losses)
    return val_avg_loss

def sampling(logits, temperature=1, p=0.5):
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

def generate(model, sp_model, initial_str, device, max_len=512):
    print(f"\tGenerating text with initial string: {initial_str}")
    model = model.eval()
    initial_ids = sp_model.encode_as_ids(initial_str)
    initial_ids = torch.tensor(initial_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(512 - len(initial_str)):
            last_token_logits = model(initial_ids)[0][-1]
            new_token_id = sampling(last_token_logits).unsqueeze(0).to(device)
            initial_ids = torch.cat((initial_ids, new_token_id), dim=-1)

    generated_ids = initial_ids.cpu().tolist()[0]
    generated_text = sp_model.decode_ids(generated_ids)
    return generated_text


def log_statistics(step, train_avg_loss, val_avg_loss):
    print()
    print("step", "avg eval loss", "avg train loss", sep="\t|\t")
    print(f"{step:6d}\t|\t{val_avg_loss:.4f}\t\t|\t{train_avg_loss:.4f}")
    print()


def save_checkpoint(model,
                    optimizer,
                    scheduler,
                    step,
                    train_avg_loss,
                    val_avg_loss,
                    checkpoint_path_pattern):
    log_statistics(step, train_avg_loss, val_avg_loss)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step
        }

    torch.save(checkpoint, checkpoint_path_pattern.format(step))
    if train_avg_loss < val_avg_loss:
        torch.save(checkpoint, "./checkpoints/best_model.pt")

all_train_avg_losses = []
all_val_avg_losses = []

print("Starting training...")
start_time = time()
train_losses = []
train_epoch_average_losses = []
for i, (x, y) in enumerate(train_data * epochs):
    model = model.train()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_pred = model(x.to(device))

        loss = cross_entropy(y_pred.view(-1, vocab_size), y.reshape(-1).to(device=device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    train_losses.append(loss.cpu().detach().item())

    if i % print_every_step == 0:

        train_average_loss = sum(train_losses) / len(train_losses)
        train_losses.clear()
        train_epoch_average_losses.append(train_average_loss)

        print(f"\ttraining loss {i:5d}/{len(train_data)}: {train_average_loss:.5f} | lr: {scheduler.get_last_lr()[0]:.5f} | avg time latency: {(time() - start_time)/print_every_step:.2f}s")
        start_time = time()

    if i % eval_every_step == 0 and i > 0:
        val_avg_loss = evaluate(model, eval_data)
        all_val_avg_losses.append(val_avg_loss)

        train_avg_loss = sum(train_epoch_average_losses)/len(train_epoch_average_losses)
        train_epoch_average_losses.clear()
        all_train_avg_losses.append(train_avg_loss)

        save_checkpoint(model, optimizer, scheduler, i, train_avg_loss, val_avg_loss, checkpoint_path_pattern)

    if i % (generate_every_step) == 0 and i > 0:
        generated_text = generate(model, sp_model,
                                  "Once upon a time, there was an old man called Slasher",
                                  device, max_len=512)
        print(f"\tGenerated text:\n{generated_text}\n")


print(all_train_avg_losses, all_val_avg_losses, sep="\n")
