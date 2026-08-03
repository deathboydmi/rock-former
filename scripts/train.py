import torch
import sentencepiece as sp

import pathlib

from modules.rockformer import Rockformer
from modules.data_loader import DataLoader


vocab_size = 1024
embed_size = 512
num_heads = 16
ff_size = 1024
num_blocks = 8
max_context_size = 1024
context_size = 512
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_model_path = './tokenizers/tiny_stories.model'
text_path = './data/tiny_stories.txt'
checkpoint_path_pattern = "./checkpoints/checkpoint_epoch_{0:03d}.pt"

epochs = 10
print_every_step = 200

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
print(f'Total number of parameters: {total_params}')

model.compile()

print("Initializing optimizer and loss function...")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.1)
cross_entropy = torch.nn.CrossEntropyLoss()

print("Initializing data loader...")
sp_model = sp.SentencePieceProcessor(model_file=tokenizer_model_path)
data_loader = DataLoader(
                            text_path,
                            sp_model,
                            context_size,
                            0.25,
                            random_seed
                                        )

train_data = data_loader.train_data(batch_size)
eval_data = data_loader.val_data(batch_size)

checkpoints_dir = pathlib.Path("checkpoints")
checkpoints = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
if checkpoints:
    print("Loading checkpoint...")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}...")
else:
    start_epoch = 0

all_train_avg_losses = []
all_val_avg_losses = []

print("Starting training...")
for epoch in range(start_epoch, epochs):
    train_losses = []
    train_epoch_average_losses = []
    for i, (x, y) in enumerate(train_data):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_pred = model(x.to(device))
            loss = cross_entropy(y_pred.view(-1, vocab_size), y.reshape(-1).to(device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.cpu().detach().item())

        if i % print_every_step == 0:
            train_average_loss = sum(train_losses) / len(train_losses)
            train_losses.clear()
            train_epoch_average_losses.append(train_average_loss)

            print(f"\ttraining loss {i:4d}/{len(train_data)}: {train_average_loss}")

    # Evaluation
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
    all_val_avg_losses.append(val_avg_loss)

    train_avg_loss = sum(train_epoch_average_losses)/len(train_epoch_average_losses)
    all_train_avg_losses.append(train_avg_loss)

    # Logging
    print()
    print("Epoch", "avg eval loss", "avg train loss", sep="\t|\t")
    print(f"{epoch+1:03d}\t|\t{val_avg_loss:.4f}\t\t|\t{train_avg_loss:.4f}")
    print()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
        }

    torch.save(checkpoint, checkpoint_path_pattern.format(epoch+1))
    model = model.train()

print(all_train_avg_losses, all_val_avg_losses, sep="\n")
