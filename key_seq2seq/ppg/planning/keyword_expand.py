"""RNNLM 关键词扩展模型：给定前 k 个关键词，预测下一个关键词"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class KeywordLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=config.PAD_ID)
        self.gru = nn.GRU(emb_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        output, hidden = self.gru(emb, hidden)
        logits = self.fc(output)
        return logits, hidden


def keyword_to_id(kw, vocab):
    """将关键词映射到词表ID（取第一个字的ID）"""
    for ch in kw:
        if ch in vocab:
            return vocab[ch]
    return config.UNK_ID


class KeywordDataset(Dataset):
    def __init__(self, filepath, vocab):
        self.sequences = []
        with open(filepath) as f:
            for line in f:
                kws = line.strip().split()
                ids = [keyword_to_id(kw, vocab) for kw in kws]
                if len(ids) >= 2:
                    self.sequences.append(ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def load_keyword_vocab(vocab_path):
    """加载字级别词表（关键词由字组成，复用同一词表）"""
    vocab = {}
    with open(vocab_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    return x_pad, y_pad


def train_keyword_model():
    vocab = load_keyword_vocab(os.path.join(config.DATA_DIR, "vocab.txt"))
    dataset = KeywordDataset(
        os.path.join(config.DATA_DIR, "keyword_sequences.txt"), vocab)
    dataloader = DataLoader(dataset, batch_size=config.KW_BATCH_SIZE,
                           shuffle=True, collate_fn=collate_fn)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = KeywordLM(config.VOCAB_SIZE, config.KW_EMB_DIM, config.KW_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.KW_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"训练 RNNLM 关键词扩展模型 | device={device} | {len(dataset)} 序列")
    for epoch in range(config.KW_EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, config.VOCAB_SIZE), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config.KW_EPOCHS}, Loss: {avg_loss:.4f}")

    save_path = os.path.join(config.DATA_DIR, "keyword_lm.pt")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")
    return model


def expand_keywords(keywords, target_num, model, vocab, rev_vocab, device):
    """用 RNNLM 将关键词列表扩展到 target_num 个"""
    if len(keywords) >= target_num:
        return keywords[:target_num]

    model.eval()
    current = list(keywords)
    with torch.no_grad():
        while len(current) < target_num:
            ids = [keyword_to_id(kw, vocab) for kw in current]
            x = torch.tensor([ids], dtype=torch.long).to(device)
            logits, _ = model(x)
            last_logits = logits[0, -1]
            # 避免生成已有关键词和特殊 token
            for kw in current:
                kid = keyword_to_id(kw, vocab)
                last_logits[kid] = -float('inf')
            for special_id in range(5):
                last_logits[special_id] = -float('inf')
            next_id = last_logits.argmax().item()
            current.append(rev_vocab.get(next_id, '<UNK>'))
    return current


if __name__ == "__main__":
    train_keyword_model()
