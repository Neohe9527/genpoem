"""
端到端诗歌生成推理 Pipeline
输入: 用户文本 → 输出: 四行七言诗
"""
import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from planning.keyword_extract import extract_keywords
from planning.keyword_expand import KeywordLM, expand_keywords, keyword_to_id
from generation.model import PoemGenerationModel


def load_vocab(vocab_path):
    vocab, rev_vocab = {}, {}
    with open(vocab_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
                rev_vocab[int(parts[1])] = parts[0]
    return vocab, rev_vocab


class PoemGenerator:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        vocab_path = os.path.join(config.DATA_DIR, "vocab.txt")
        self.vocab, self.rev_vocab = load_vocab(vocab_path)

        # 加载关键词扩展模型
        kw_model_path = os.path.join(config.DATA_DIR, "keyword_lm.pt")
        self.kw_model = KeywordLM(config.VOCAB_SIZE, config.KW_EMB_DIM, config.KW_HIDDEN)
        self.kw_model.load_state_dict(torch.load(kw_model_path, map_location=self.device))
        self.kw_model.to(self.device).eval()

        # 加载生成模型
        gen_model_path = os.path.join(config.DATA_DIR, "poem_gen.pt")
        self.gen_model = PoemGenerationModel()
        if os.path.exists(gen_model_path):
            self.gen_model.load_state_dict(torch.load(gen_model_path, map_location=self.device))
        self.gen_model.to(self.device).eval()

    def plan_keywords(self, query):
        """Stage 1: 从 query 提取并扩展到 4 个关键词"""
        keywords = extract_keywords(query, num_keywords=4)
        if len(keywords) < 4:
            keywords = expand_keywords(
                keywords, 4, self.kw_model, self.vocab, self.rev_vocab, self.device)
        return keywords[:4]

    def generate_line(self, keyword, preceding_lines):
        """Stage 2: 给定关键词和前文，生成一行诗"""
        # 编码关键词（字级别）
        kw_ids = [self.vocab.get(ch, config.UNK_ID) for ch in keyword]
        if not kw_ids:
            kw_ids = [config.UNK_ID]

        # 编码前文（所有已生成行的字拼接）
        text_ids = []
        for line in preceding_lines:
            for ch in line:
                text_ids.append(self.vocab.get(ch, config.UNK_ID))

        kw_tensor = torch.tensor([kw_ids], dtype=torch.long, device=self.device)
        text_tensor = torch.tensor([text_ids if text_ids else [0]],
                                   dtype=torch.long, device=self.device)

        with torch.no_grad():
            generated = self.gen_model.generate(kw_tensor, text_tensor, max_len=7)

        # 解码
        chars = []
        for token_id in generated[0].tolist():
            if token_id in (config.PAD_ID, config.EOS_ID, config.GO_ID):
                continue
            chars.append(self.rev_vocab.get(token_id, '?'))
        return ''.join(chars[:7])

    def generate_poem(self, query):
        """完整流程：输入 → 四行七言诗"""
        keywords = self.plan_keywords(query)
        print(f"关键词: {keywords}")

        lines = []
        for i, kw in enumerate(keywords):
            line = self.generate_line(kw, lines)
            lines.append(line)
            print(f"  [{kw}] → {line}")

        return '\n'.join(lines)


if __name__ == "__main__":
    generator = PoemGenerator()
    queries = ["春天的桃花开了", "月亮", "思乡", "山水"]
    for query in queries:
        print(f"\n{'='*40}")
        print(f"输入: {query}")
        print(f"{'='*40}")
        poem = generator.generate_poem(query)
        print()
