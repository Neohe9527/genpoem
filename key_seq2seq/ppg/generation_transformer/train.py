"""训练 Transformer Poem Generation 模型 (支持 MPS)"""
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from generation_transformer import PoemTransformer
from generation.dataset import PoemDataset, collate_fn


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train():
    device = get_device()
    print(f"Device: {device}")

    train_data = PoemDataset(os.path.join(config.DATA_DIR, "train.txt"))
    val_data = PoemDataset(os.path.join(config.DATA_DIR, "val.txt"))
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE,
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE,
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    model = PoemTransformer().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Params: {param_count / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID, label_smoothing=0.1)

    # Warmup + cosine decay
    warmup_steps = 1000
    total_steps = len(train_loader) * config.MAX_EPOCHS
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(config.DATA_DIR, "poem_transformer.pt")
    global_step = 0

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()

        for kw, text, target in train_loader:
            kw, text, target = kw.to(device), text.to(device), target.to(device)
            logits = model(kw, text, target)
            # logits: (batch, target_len-1, vocab), target[:, 1:]: skip GO
            loss = criterion(logits.reshape(-1, config.VOCAB_SIZE),
                           target[:, 1:logits.size(1)+1].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            global_step += 1

        avg_train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 10 else float('inf')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for kw, text, target in val_loader:
                kw, text, target = kw.to(device), text.to(device), target.to(device)
                logits = model(kw, text, target)
                loss = criterion(logits.reshape(-1, config.VOCAB_SIZE),
                               target[:, 1:logits.size(1)+1].reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 10 else float('inf')

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        saved = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            saved = " *saved*"
        else:
            patience_counter += 1

        print(f"Ep {epoch+1:2d} | TrLoss {avg_train_loss:.3f} PPL {train_ppl:.0f} | "
              f"VlLoss {avg_val_loss:.3f} PPL {val_ppl:.0f} | "
              f"LR {lr:.5f} | {elapsed:.0f}s{saved}")

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"完成! Best Val PPL: {math.exp(best_val_loss):.0f}")


if __name__ == "__main__":
    train()
