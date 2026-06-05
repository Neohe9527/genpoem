"""训练 Poem Generation 模型"""
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from generation.model import PoemGenerationModel
from generation.dataset import PoemDataset, collate_fn


def train():
    device = torch.device("cpu")
    print(f"Device: {device}")

    # 数据
    train_data = PoemDataset(os.path.join(config.DATA_DIR, "train.txt"))
    val_data = PoemDataset(os.path.join(config.DATA_DIR, "val.txt"))
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE,
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE,
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"训练集: {len(train_data)} 样本, 验证集: {len(val_data)} 样本")

    # 模型
    model = PoemGenerationModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.LR_DECAY, patience=2)

    best_val_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(config.DATA_DIR, "poem_gen.pt")

    for epoch in range(config.MAX_EPOCHS):
        # Train
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch_idx, (kw, text, target) in enumerate(train_loader):
            kw, text, target = kw.to(device), text.to(device), target.to(device)
            # teacher forcing ratio 随训练递减
            tf_ratio = max(0.5, 1.0 - epoch * 0.02)
            outputs = model(kw, text, target, teacher_forcing_ratio=tf_ratio)
            # outputs: (batch, target_len-1, vocab)
            # target[:, 1:]: 去掉 GO
            loss = criterion(outputs.reshape(-1, config.VOCAB_SIZE),
                           target[:, 1:outputs.size(1)+1].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 10 else float('inf')

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for kw, text, target in val_loader:
                kw, text, target = kw.to(device), text.to(device), target.to(device)
                outputs = model(kw, text, target, teacher_forcing_ratio=0)
                loss = criterion(outputs.reshape(-1, config.VOCAB_SIZE),
                               target[:, 1:outputs.size(1)+1].reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 10 else float('inf')

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.MAX_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} PPL: {train_ppl:.1f} | "
              f"Val Loss: {avg_val_loss:.4f} PPL: {val_ppl:.1f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {elapsed:.1f}s")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> 保存最佳模型 (Val PPL: {val_ppl:.1f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"训练完成! 最佳验证 PPL: {math.exp(best_val_loss):.1f}")


if __name__ == "__main__":
    train()
