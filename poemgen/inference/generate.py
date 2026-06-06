"""端到端推理：query → 整首诗"""
import os
import json
import torch
import yaml
import opencc

from ..models.transformer import PoemEncoderDecoder

_t2s = opencc.OpenCC('t2s')


class PoemGenerator:
    def __init__(self, exp_name="base_enc_dec", device=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exp_dir = os.path.join(base_dir, "experiments", exp_name)
        config_path = os.path.join(exp_dir, "config.yaml")

        if not os.path.exists(config_path):
            config_path = os.path.join(base_dir, "configs", f"{exp_name}.yaml")
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        # Load vocab
        vocab_path = os.path.join(base_dir, self.cfg["data"]["processed_dir"], "vocab.txt")
        self.vocab = {}
        self.rev_vocab = {}
        with open(vocab_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.vocab[parts[0]] = int(parts[1])
                    self.rev_vocab[int(parts[1])] = parts[0]

        # Load mapping tables
        mapping_dir = os.path.join(base_dir, self.cfg["data"]["mapping_dir"])
        self.mapping = {}
        for fname in ["common.json", "proper.json"]:
            path = os.path.join(mapping_dir, fname)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for cat, entries in data.items():
                    if cat == "_meta" or not isinstance(entries, dict):
                        continue
                    self.mapping.update(entries)

        # Load model
        mcfg = self.cfg["model"]
        self.model = PoemEncoderDecoder(
            vocab_size=mcfg["vocab_size"], d_model=mcfg["d_model"],
            nhead=mcfg["nhead"], num_encoder_layers=mcfg["num_encoder_layers"],
            num_decoder_layers=mcfg["num_decoder_layers"], dim_ff=mcfg["dim_ff"],
            dropout=mcfg["dropout"], tie_weights=mcfg["tie_weights"],
            use_segment=mcfg["use_segment"]
        ).to(self.device)

        ckpt_path = os.path.join(exp_dir, "checkpoints", "best.pt")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def rewrite_query(self, query):
        """将现代概念改写为古诗语义空间"""
        # Direct mapping match
        if query in self.mapping:
            return "".join(self.mapping[query][:4])
        # Partial match
        for key, vals in self.mapping.items():
            if key in query:
                remaining = query.replace(key, "")
                mapped = "".join(vals[:2])
                return mapped + remaining[:4]
        # No mapping needed (古代概念直接使用)
        return query

    def encode_query(self, query, fmt="7"):
        """编码 query 为 token IDs"""
        fmt_token = f"<{fmt}>"
        ids = [self.vocab.get(fmt_token, self.vocab.get("<7>", 6))]
        for ch in query:
            ids.append(self.vocab.get(ch, self.vocab.get("<UNK>", 3)))
        return ids

    def decode_output(self, token_ids):
        """将 token IDs 解码为诗歌行"""
        sep_id = self.vocab.get("<SEP>", 4)
        eos_id = self.vocab.get("<EOS>", 2)
        lines = []
        current = []
        for tid in token_ids:
            if tid == sep_id:
                lines.append("".join(current))
                current = []
            elif tid == eos_id:
                break
            elif tid > 6:  # Skip special tokens
                current.append(self.rev_vocab.get(tid, "?"))
        if current:
            lines.append("".join(current))
        return [_t2s.convert(line) for line in lines]

    def generate(self, query, fmt="7", temperature=None, top_p=None, rep_pen=None):
        """生成一首诗"""
        gcfg = self.cfg["generation"]
        temperature = temperature or gcfg["temperature"]
        top_p = top_p or gcfg["top_p"]
        rep_pen = rep_pen or gcfg["repetition_penalty"]

        rewritten = self.rewrite_query(query)
        q_ids = self.encode_query(rewritten, fmt)
        src = torch.tensor([q_ids], dtype=torch.long, device=self.device)

        n = int(fmt)
        max_len = n * 4 + 3 + 2  # 4 lines + 3 SEP + EOS + margin

        with torch.no_grad():
            output = self.model.generate(
                src, max_len=max_len, temperature=temperature,
                top_p=top_p, repetition_penalty=rep_pen)

        lines = self.decode_output(output[0].tolist())
        return lines, rewritten


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成古诗")
    parser.add_argument("--exp", default="base_enc_dec", help="实验名称")
    parser.add_argument("--query", default="春天", help="输入 query")
    args = parser.parse_args()

    gen = PoemGenerator(args.exp)
    lines, rewritten = gen.generate(args.query)
    print(f"Query: {args.query} → Rewritten: {rewritten}")
    print(f"诗:")
    for line in lines:
        print(f"  {line}")
