"""PyTorch Dataset for Poem Generation training triples"""
import os
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PoemDataset(Dataset):
    """加载训练三元组: keyword_ids \t preceding_text_ids \t target_line_ids"""
    def __init__(self, filepath):
        self.samples = []
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                kw_ids = [int(x) for x in parts[0].split()] if parts[0] else []
                text_ids = [int(x) for x in parts[1].split()] if parts[1] else []
                target_ids = [int(x) for x in parts[2].split()] if parts[2] else []
                if kw_ids and target_ids:
                    self.samples.append((kw_ids, text_ids, target_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kw, text, target = self.samples[idx]
        return kw, text, target


def collate_fn(batch):
    kws, texts, targets = zip(*batch)

    # pad keywords to fixed length
    max_kw = 4
    kw_tensor = torch.zeros(len(kws), max_kw, dtype=torch.long)
    for i, k in enumerate(kws):
        length = min(len(k), max_kw)
        kw_tensor[i, :length] = torch.tensor(k[:length])

    # pad preceding text to fixed length
    max_text = 21  # 最多3行前文 * 7字
    text_tensor = torch.zeros(len(texts), max_text, dtype=torch.long)
    for i, t in enumerate(texts):
        if t:
            length = min(len(t), max_text)
            text_tensor[i, :length] = torch.tensor(t[:length])

    # target: prepend GO, fixed length = 7 + GO + EOS = 9
    max_target = 9
    target_tensor = torch.zeros(len(targets), max_target, dtype=torch.long)
    for i, t in enumerate(targets):
        target_tensor[i, 0] = config.GO_ID
        length = min(len(t), 7)
        target_tensor[i, 1:length+1] = torch.tensor(t[:length])
        target_tensor[i, length+1] = config.EOS_ID

    return kw_tensor, text_tensor, target_tensor
