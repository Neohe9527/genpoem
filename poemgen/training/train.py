"""训练脚本：config-driven"""
import os
import math
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.transformer import PoemEncoderDecoder
from .dataset import PoemDataset, collate_fn


def get_device(cfg_device="auto"):
    if cfg_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    elif cfg_device != "cpu":
        return torch.device(cfg_device)
    return torch.device("cpu")


def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"Experiment: {exp_name} | Device: {device}")

    # Data
    data_dir = cfg["data"]["processed_dir"]
    train_data = PoemDataset(os.path.join(data_dir, "train.txt"))
    val_data = PoemDataset(os.path.join(data_dir, "val.txt"))
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model
    mcfg = cfg["model"]
    model = PoemEncoderDecoder(
        vocab_size=mcfg["vocab_size"], d_model=mcfg["d_model"],
        nhead=mcfg["nhead"], num_encoder_layers=mcfg["num_encoder_layers"],
        num_decoder_layers=mcfg["num_decoder_layers"], dim_ff=mcfg["dim_ff"],
        dropout=mcfg["dropout"], tie_weights=mcfg["tie_weights"],
        use_segment=mcfg["use_segment"]
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params/1e6:.2f}M")

    # Optimizer + Scheduler
    tcfg = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                                betas=(0.9, 0.98), eps=1e-9)
    warmup = tcfg["warmup_steps"]
    total_steps = len(train_loader) * tcfg["max_epochs"]

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=0,
                                    label_smoothing=tcfg["label_smoothing"])

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(exp_dir, "checkpoints", "best.pt")
    global_step = 0

    for epoch in range(tcfg["max_epochs"]):
        model.train()
        total_loss = 0
        start = time.time()

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_pad_mask = (src == 0)
            logits = model(src, tgt, src_pad_mask)
            # logits: (B, tgt_len-1, V), target: tgt[:, 1:]
            loss = criterion(logits.reshape(-1, mcfg["vocab_size"]),
                           tgt[:, 1:logits.size(1)+1].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            global_step += 1

        train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(train_loss) if train_loss < 10 else float('inf')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                src_pad_mask = (src == 0)
                logits = model(src, tgt, src_pad_mask)
                loss = criterion(logits.reshape(-1, mcfg["vocab_size"]),
                               tgt[:, 1:logits.size(1)+1].reshape(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_ppl = math.exp(val_loss) if val_loss < 10 else float('inf')
        elapsed = time.time() - start

        # Save every epoch independently
        ep_path = os.path.join(exp_dir, "checkpoints", f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ep_path)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            saved = " *saved*"
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:2d} | TrPPL {train_ppl:5.0f} | VlPPL {val_ppl:5.0f} | "
              f"LR {lr:.5f} | {elapsed:.0f}s{saved}")

        if patience_counter >= tcfg["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Done! Best Val PPL: {math.exp(best_val_loss):.0f}")
    # Save config alongside checkpoint
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--config", default="configs/base_enc_dec.yaml")
    args = parser.parse_args()
    train(args.config)
