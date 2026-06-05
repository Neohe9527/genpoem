"""
从 chinese-poetry 仓库提取七言绝句，构建训练数据。
输出:
  - poems_7char.txt: 每首诗4行，诗之间空行分隔
  - keyword_sequences.txt: 每行4个关键词（空格分隔）
  - train_triples.txt: 格式 "关键词\t前文\t当前行"（字级别，空格分隔）
  - vocab.txt: 字表，每行 "字\tID"
"""
import json
import os
import re
import glob
from collections import Counter

import jieba
import jieba.analyse
from opencc import OpenCC

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw_poetry", "全唐诗")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cc = OpenCC('t2s')
VOCAB_SIZE = 6000
MIN_POEMS = 70000

# PLACEHOLDER_STEP2

def split_paragraph_to_lines(paragraphs):
    """将 paragraphs 拆分为单行（按逗号/句号分隔）"""
    lines = []
    for p in paragraphs:
        parts = re.split(r'[，,。！？、；]', p)
        parts = [pt.strip() for pt in parts if pt.strip()]
        lines.extend(parts)
    return lines

def is_valid_7char_quatrain(lines):
    if len(lines) != 4:
        return False
    return all(len(re.findall(r'[一-鿿]', l)) == 7 for l in lines)

def clean_line(line):
    return re.findall(r'[一-鿿]', line)

def extract_poems_from_files(file_pattern, limit=None):
    poems = []
    for filepath in sorted(glob.glob(file_pattern)):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for poem in data:
            paragraphs = poem.get("paragraphs", [])
            # 绝句通常是 2 个 paragraph(每个含2句) 或 4 个 paragraph(每个1句)
            if len(paragraphs) not in (2, 4):
                continue
            lines = split_paragraph_to_lines(paragraphs)
            lines = [cc.convert(l) for l in lines]
            if is_valid_7char_quatrain(lines):
                poems.append([clean_line(l) for l in lines])
        if limit and len(poems) >= limit:
            break
    return poems

def extract_poems():
    poems = extract_poems_from_files(os.path.join(RAW_DIR, "poet.tang.*.json"))
    print(f"唐诗七言绝句: {len(poems)} 首")
    if len(poems) < MIN_POEMS:
        song_poems = extract_poems_from_files(
            os.path.join(RAW_DIR, "poet.song.*.json"),
            limit=MIN_POEMS - len(poems))
        print(f"宋诗七言绝句: {len(song_poems)} 首")
        poems.extend(song_poems)
    print(f"总计提取 {len(poems)} 首七言绝句")
    return poems

def extract_keyword_for_line(chars):
    """用 TextRank 为一行诗提取1个关键词（字级别）"""
    text = ''.join(chars)
    keywords = jieba.analyse.textrank(text, topK=1, withWeight=False)
    if keywords:
        return keywords[0]
    # fallback: 取第一个非常见字
    return text[0]

def build_vocab(poems):
    """构建字表"""
    counter = Counter()
    for poem in poems:
        for line in poem:
            for char in line:
                counter[char] += 1
    # 特殊 token
    vocab = {'<PAD>': 0, '<GO>': 1, '<EOS>': 2, '<UNK>': 3, '<SEP>': 4}
    for char, _ in counter.most_common(VOCAB_SIZE - len(vocab)):
        vocab[char] = len(vocab)
    return vocab

def build_training_data(poems, vocab):
    """构建训练三元组和关键词序列"""
    keyword_sequences = []
    triples = []

    for idx, poem in enumerate(poems):
        if idx % 10000 == 0:
            print(f"处理第 {idx}/{len(poems)} 首...")
        keywords = []
        for line in poem:
            kw = extract_keyword_for_line(line)
            keywords.append(kw)

        keyword_sequences.append(keywords)

        # 构建三元组 (keyword, preceding_text, current_line)
        for i, line in enumerate(poem):
            kw = keywords[i]
            if i == 0:
                preceding = []
            else:
                preceding = []
                for prev_line in poem[:i]:
                    preceding.extend(prev_line)
            triples.append((kw, preceding, line))

    return keyword_sequences, triples

def save_data(poems, keyword_sequences, triples, vocab):
    # 保存诗歌原文
    with open(os.path.join(OUTPUT_DIR, "poems_7char.txt"), 'w') as f:
        for poem in poems:
            for line in poem:
                f.write(''.join(line) + '\n')
            f.write('\n')

    # 保存词表
    with open(os.path.join(OUTPUT_DIR, "vocab.txt"), 'w') as f:
        for char, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{char}\t{idx}\n")

    # 保存关键词序列（用于 RNNLM 训练）
    with open(os.path.join(OUTPUT_DIR, "keyword_sequences.txt"), 'w') as f:
        for kws in keyword_sequences:
            f.write(' '.join(kws) + '\n')

    # 保存训练三元组（字级别 ID）
    def chars_to_ids(chars):
        return ' '.join(str(vocab.get(c, vocab['<UNK>'])) for c in chars)

    def keyword_to_ids(kw):
        return ' '.join(str(vocab.get(c, vocab['<UNK>'])) for c in kw)

    with open(os.path.join(OUTPUT_DIR, "train_triples.txt"), 'w') as f:
        for kw, preceding, line in triples:
            kw_ids = keyword_to_ids(kw)
            pre_ids = chars_to_ids(preceding) if preceding else ''
            line_ids = chars_to_ids(line)
            f.write(f"{kw_ids}\t{pre_ids}\t{line_ids}\n")

    # 划分训练/验证/测试
    import random
    random.seed(42)
    indices = list(range(len(keyword_sequences)))
    random.shuffle(indices)
    test_idx = set(indices[:2000])
    val_idx = set(indices[2000:4000])

    splits = {'train': [], 'val': [], 'test': []}
    for i, (kws, triple_start) in enumerate(zip(keyword_sequences, range(0, len(triples), 4))):
        subset_triples = triples[triple_start:triple_start+4]
        if i in test_idx:
            splits['test'].extend(subset_triples)
        elif i in val_idx:
            splits['val'].extend(subset_triples)
        else:
            splits['train'].extend(subset_triples)

    for split_name, split_triples in splits.items():
        with open(os.path.join(OUTPUT_DIR, f"{split_name}.txt"), 'w') as f:
            for kw, preceding, line in split_triples:
                kw_ids = keyword_to_ids(kw)
                pre_ids = chars_to_ids(preceding) if preceding else ''
                line_ids = chars_to_ids(line)
                f.write(f"{kw_ids}\t{pre_ids}\t{line_ids}\n")

    print(f"训练集: {len(splits['train'])} 三元组")
    print(f"验证集: {len(splits['val'])} 三元组")
    print(f"测试集: {len(splits['test'])} 三元组")

if __name__ == "__main__":
    poems = extract_poems()
    vocab = build_vocab(poems)
    print(f"词表大小: {len(vocab)}")
    keyword_sequences, triples = build_training_data(poems, vocab)
    save_data(poems, keyword_sequences, triples, vocab)
    print("数据预处理完成!")
