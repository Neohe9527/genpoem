"""端到端诗歌生成 (Transformer 版本)"""
import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from planning.keyword_extract import extract_keywords
from planning.keyword_expand import KeywordLM, expand_keywords, keyword_to_id
from generation_transformer import PoemTransformer


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
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        vocab_path = os.path.join(config.DATA_DIR, "vocab.txt")
        self.vocab, self.rev_vocab = load_vocab(vocab_path)

        kw_model_path = os.path.join(config.DATA_DIR, "keyword_lm.pt")
        self.kw_model = KeywordLM(config.VOCAB_SIZE, config.KW_EMB_DIM, config.KW_HIDDEN)
        self.kw_model.load_state_dict(torch.load(kw_model_path, map_location=self.device))
        self.kw_model.to(self.device).eval()

        gen_model_path = os.path.join(config.DATA_DIR, "poem_transformer.pt")
        self.gen_model = PoemTransformer(
            d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
            dim_ff=1024, dropout=0.15)
        self.gen_model.load_state_dict(torch.load(gen_model_path, map_location=self.device))
        self.gen_model.to(self.device).eval()

    def plan_keywords(self, query):
        keywords = extract_keywords(query, num_keywords=4)
        if not keywords:
            keywords = [query[:2]] if len(query) >= 2 else [query]
        if len(keywords) < 4:
            keywords = expand_keywords(
                keywords, 4, self.kw_model, self.vocab, self.rev_vocab, self.device)
        return keywords[:4]

    def generate_line(self, keyword, preceding_lines, temperature=0.75, top_p=0.9,
                      repetition_penalty=1.3):
        kw_ids = [self.vocab.get(ch, config.UNK_ID) for ch in keyword]
        if not kw_ids:
            kw_ids = [config.UNK_ID]
        kw_ids = (kw_ids + [0] * 4)[:4]

        text_ids = []
        for line in preceding_lines:
            for ch in line:
                text_ids.append(self.vocab.get(ch, config.UNK_ID))
        text_ids = (text_ids + [0] * 21)[:21]

        kw_tensor = torch.tensor([kw_ids], dtype=torch.long, device=self.device)
        text_tensor = torch.tensor([text_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            generated = self.gen_model.generate(
                kw_tensor, text_tensor, max_len=7,
                temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty)

        chars = []
        for tid in generated[0].tolist():
            if tid in (config.PAD_ID, config.EOS_ID, config.GO_ID):
                continue
            chars.append(self.rev_vocab.get(tid, '?'))
        return ''.join(chars[:7])

    def generate_poem(self, query, temperature=0.75, top_p=0.9, repetition_penalty=1.2):
        keywords = self.plan_keywords(query)
        print(f"关键词: {keywords}")

        lines = []
        for kw in keywords:
            line = self.generate_line(kw, lines, temperature, top_p, repetition_penalty)
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
        generator.generate_poem(query)
        print()
