"""数据预处理：从 chinese-poetry 构建整诗生成训练数据"""
import os
import json
import re
import random
from collections import Counter

import jieba
import jieba.analyse


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_poems(raw_dir, formats=("7",)):
    """从 chinese-poetry 加载五言/七言绝句"""
    poems = []
    tang_dir = os.path.join(raw_dir, "全唐诗")

    for dirpath in [tang_dir]:
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.startswith("poet.") or not fname.endswith(".json"):
                continue
            with open(os.path.join(dirpath, fname), encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except:
                    continue
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                paragraphs = item.get("paragraphs", [])
                lines = []
                for p in paragraphs:
                    parts = re.split(r"[，,。！？、；\s]", p)
                    lines.extend([s for s in parts if s])
                if len(lines) == 4:
                    for fmt in formats:
                        n = int(fmt)
                        if all(len(line) == n for line in lines):
                            poems.append((fmt, lines))
    return poems


def build_vocab(poems, max_size=6000):
    """构建字级别词表"""
    counter = Counter()
    for _, lines in poems:
        for line in lines:
            for ch in line:
                counter[ch] += 1
    vocab = {"<PAD>": 0, "<GO>": 1, "<EOS>": 2, "<UNK>": 3, "<SEP>": 4, "<5>": 5, "<7>": 6}
    for ch, _ in counter.most_common(max_size - len(vocab)):
        vocab[ch] = len(vocab)
    return vocab


def extract_queries(lines, num_queries=3):
    """从一首诗构造多个训练 query（快速版，不用 TextRank）"""
    full_text = "".join(lines)
    queries = []

    # Strategy 1: TF-IDF keywords (much faster than TextRank)
    kws = jieba.analyse.extract_tags(full_text, topK=4)
    if kws:
        queries.append("".join(kws[:3]))
        if len(kws) >= 2:
            queries.append("".join(kws[:2]))

    # Strategy 2: Random bigram from poem
    chars = list(full_text)
    if len(chars) >= 4:
        idx = random.randint(0, len(chars) - 2)
        queries.append(chars[idx] + chars[idx + 1])

    return queries[:num_queries] if queries else [full_text[:2]]


def encode_poem(fmt, lines, vocab):
    """编码一首诗为 ID 序列: line1<SEP>line2<SEP>line3<SEP>line4<EOS>"""
    ids = []
    for i, line in enumerate(lines):
        for ch in line:
            ids.append(vocab.get(ch, vocab["<UNK>"]))
        if i < 3:
            ids.append(vocab["<SEP>"])
    ids.append(vocab["<EOS>"])
    return ids


def encode_query(fmt, query, vocab):
    """编码 query: <7>query_chars"""
    fmt_token = f"<{fmt}>"
    ids = [vocab[fmt_token]]
    for ch in query:
        ids.append(vocab.get(ch, vocab["<UNK>"]))
    return ids


def preprocess(config_path):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    raw_dir = data_cfg["raw_poetry_dir"]
    out_dir = data_cfg["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    formats = data_cfg.get("formats", ["7"])
    print(f"加载诗歌 (格式: {formats})...")
    poems = load_poems(raw_dir, formats)
    if data_cfg.get("max_poems"):
        poems = poems[:data_cfg["max_poems"]]
    print(f"  共 {len(poems)} 首绝句")

    # Build vocab
    print("构建词表...")
    vocab = build_vocab(poems, cfg["model"]["vocab_size"])
    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w") as f:
        for ch, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{ch}\t{idx}\n")
    print(f"  词表大小: {len(vocab)}")

    # Construct training samples
    print("构造训练数据...")
    random.seed(42)
    random.shuffle(poems)

    val_size = data_cfg.get("val_size", 2000)
    test_size = data_cfg.get("test_size", 2000)
    train_poems = poems[:-(val_size + test_size)]
    val_poems = poems[-(val_size + test_size):-test_size]
    test_poems = poems[-test_size:]

    for split_name, split_poems in [("train", train_poems), ("val", val_poems), ("test", test_poems)]:
        samples = []
        for fmt, lines in split_poems:
            queries = extract_queries(lines, num_queries=(3 if split_name == "train" else 1))
            poem_ids = encode_poem(fmt, lines, vocab)
            for query in queries:
                query_ids = encode_query(fmt, query, vocab)
                samples.append((query_ids, poem_ids))

        # Write as tab-separated ID sequences
        path = os.path.join(out_dir, f"{split_name}.txt")
        with open(path, "w") as f:
            for q_ids, p_ids in samples:
                q_str = " ".join(str(x) for x in q_ids)
                p_str = " ".join(str(x) for x in p_ids)
                f.write(f"{q_str}\t{p_str}\n")
        print(f"  {split_name}: {len(samples)} 样本")

    print("完成!")


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--config", default="configs/base_enc_dec.yaml")
    args = parser.parse_args()
    preprocess(args.config)
