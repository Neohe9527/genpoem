"""PyTorch Dataset for whole-poem generation"""
import os
import torch
from torch.utils.data import Dataset


class PoemDataset(Dataset):
    def __init__(self, filepath):
        self.samples = []
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                q_ids = [int(x) for x in parts[0].split()]
                p_ids = [int(x) for x in parts[1].split()]
                if q_ids and p_ids:
                    self.samples.append((q_ids, p_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_id=0, go_id=1):
    queries, poems = zip(*batch)
    max_q = max(len(q) for q in queries)
    max_p = max(len(p) for p in poems)

    q_tensor = torch.full((len(queries), max_q), pad_id, dtype=torch.long)
    # Target: GO + poem_ids (for teacher forcing)
    tgt_tensor = torch.full((len(poems), max_p + 1), pad_id, dtype=torch.long)

    for i, (q, p) in enumerate(zip(queries, poems)):
        q_tensor[i, :len(q)] = torch.tensor(q)
        tgt_tensor[i, 0] = go_id
        tgt_tensor[i, 1:len(p) + 1] = torch.tensor(p)

    return q_tensor, tgt_tensor
