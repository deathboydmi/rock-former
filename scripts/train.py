import math
from time import time

import torch
import sentencepiece as sp

import pathlib

from modules.rockformer import Rockformer
from modules.data_loader import DataLoader

from scripts.generate import sampling, generate


vocab_size = 16384
embed_size = 512
num_heads = 16
ff_size = 2048
num_blocks = 8
max_context_size = 2048

context_size = 2048
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_model_path = './tokenizers/tiny_stories_en_musik_mix.model'
text_path = './data/en_musik_500k_rock_metal.txt'
checkpoint_path_pattern = "./checkpoints/checkpoint_step_{0:03d}.pt"

FINE_TUNING = True
pre_trained_model_path = "./checkpoints/en_musik_500k/checkpoint_step_40000.pt"

epochs = 1
print_every_step = 500
eval_every_step = 10000
generate_every_step = 5000

generation_prompt = "Should I sing a hardcore song?"

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
            lr=0.0001,
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

warmup_steps = 0
min_lr = 3e-5
max_lr = 1e-4
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
checkpoints = list(checkpoints_dir.glob("checkpoint_step_*.pt"))
if checkpoints:
    print("Loading checkpoint...")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_step = checkpoint["step"] + 1
    print(f"Resuming from step {start_step}...")
else:
    start_step = 0


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


def log_statistics(step, train_avg_loss, val_avg_loss):
    print()
    print("step\t\t|\tavg eval loss", "avg train loss", sep="\t|\t")
    print(f"{step:7d}\t|\t{val_avg_loss:.4f}\t\t|\t{train_avg_loss:.4f}")
    print()


def save_checkpoint(model,
                    optimizer,
                    scheduler,
                    step,
                    all_train_avg_losses,
                    all_val_avg_losses,
                    checkpoint_path_pattern):
    train_avg_loss, val_avg_loss = all_train_avg_losses[-1], all_val_avg_losses[-1]

    log_statistics(step, train_avg_loss, val_avg_loss)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step
        }

    torch.save(checkpoint, checkpoint_path_pattern.format(step))
    if train_avg_loss < val_avg_loss and val_avg_loss == min(all_val_avg_losses):
        torch.save(checkpoint, "./checkpoints/best_model.pt")


all_train_avg_losses = []
all_val_avg_losses = []

print("Starting training...")
start_time = time()
train_losses = []
train_epoch_average_losses = []
for i, (x, y) in zip(range(start_step, total_steps), train_data * epochs):
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

        print(f"\ttraining loss {i:7d}/{total_steps}: {train_average_loss:.5f} | lr: {scheduler.get_last_lr()[0]:.5f} | avg time latency: {(time() - start_time)/print_every_step:.2f}s")
        start_time = time()

    if i % eval_every_step == 0 and i > 0 or i == (total_steps - 1):
        val_avg_loss = evaluate(model, eval_data)
        all_val_avg_losses.append(val_avg_loss)

        train_avg_loss = sum(train_epoch_average_losses)/len(train_epoch_average_losses)
        train_epoch_average_losses.clear()
        all_train_avg_losses.append(train_avg_loss)

        save_checkpoint(model, optimizer, scheduler, i, all_train_avg_losses, all_val_avg_losses, checkpoint_path_pattern)

    if i % (generate_every_step) == 0 and i > 0 or i == (total_steps - 1):
        generated_text = generate(model.eval(), sp_model,
                                  generation_prompt,
                                  device, max_len=512)
        print(f"\tGenerated text:\n{generated_text}\n")


print(all_train_avg_losses, all_val_avg_losses, sep="\n")
