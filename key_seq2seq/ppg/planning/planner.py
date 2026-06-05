"""主题规划模块：用户输入 → 4个古诗意象关键词"""
import os
import json
import torch
import jieba.analyse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from planning.keyword_expand import KeywordLM

# 加载映射表
_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMON_PATH = os.path.join(_DATA_DIR, 'data', 'modern_to_ancient_common.json')
_PROPER_PATH = os.path.join(_DATA_DIR, 'data', 'modern_to_ancient_proper.json')

_common_map = {}
_proper_map = {}

def _load_maps():
    global _common_map, _proper_map
    if _common_map:
        return
    with open(_COMMON_PATH, encoding='utf-8') as f:
        data = json.load(f)
    for cat, entries in data.items():
        if cat == '_meta' or not isinstance(entries, dict):
            continue
        _common_map.update(entries)
    with open(_PROPER_PATH, encoding='utf-8') as f:
        data = json.load(f)
    for cat, entries in data.items():
        if cat == '_meta' or not isinstance(entries, dict):
            continue
        _proper_map.update(entries)


def lookup_modern(word):
    """查找现代词的古诗意象映射"""
    _load_maps()
    if word in _common_map:
        return _common_map[word]
    if word in _proper_map:
        return _proper_map[word]
    # 尝试部分匹配（如"周杰伦的歌"匹配"周杰伦"）
    for key in _proper_map:
        if key in word or word in key:
            return _proper_map[key]
    for key in _common_map:
        if key in word or word in key:
            return _common_map[key]
    return None


STOP_CHARS = set('不一无有是得为在与之乎者也矣焉哉其而则以于又且若虽'
                 '何谁莫未今此更只却已将欲亦复曾须便应但令被把'
                 '的了着过还很都会就要这那些个么上下中大小多少'
                 '三四五六七八九十')

IMAGERY = set('山水风云月日花雪梅竹松柳桃杏兰菊莲荷草木叶枝'
              '春秋冬夏寒暑朝夕天地海江河湖溪泉池潭川岸岩石峰'
              '鸟雁鹤燕莺鸡鹊凤龙虎马牛鱼蝶蝉猿鹿'
              '酒茶琴书剑刀弓旗鼓笛钟楼台亭阁殿宫塔桥船门'
              '金玉珠翠锦丝人客僧将兵王妃女翁'
              '情愁思梦魂泪恨怨悲喜红翠碧青白黄黑苍绿'
              '光影色声香暗明清城关塞漠野林园田'
              '雨露霜雾霞虹冰歌舞诗画曲')


class PoemPlanner:
    def __init__(self, device=None):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device

        vocab_path = os.path.join(config.DATA_DIR, 'vocab.txt')
        self.vocab = {}
        self.rev_vocab = {}
        with open(vocab_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.vocab[parts[0]] = int(parts[1])
                    self.rev_vocab[int(parts[1])] = parts[0]

        # 加载 RNNLM v3
        lm_path = os.path.join(config.DATA_DIR, 'keyword_lm_v3.pt')
        if not os.path.exists(lm_path):
            lm_path = os.path.join(config.DATA_DIR, 'keyword_lm.pt')
        self.kw_model = KeywordLM(config.VOCAB_SIZE, config.KW_EMB_DIM, config.KW_HIDDEN)
        self.kw_model.load_state_dict(torch.load(lm_path, map_location=self.device))
        self.kw_model.to(self.device).eval()

    def plan(self, query, num_keywords=4):
        """主入口：输入任意文本，输出4个古诗意象关键词"""
        # Step 1: 纯映射表匹配（短输入直接命中）
        mapped = lookup_modern(query)
        if mapped and len(query) <= 5:
            return mapped[:num_keywords]

        # Step 2: 分词后逐词查映射 + TextRank 混合
        words = list(jieba.analyse.textrank(query, topK=6, withWeight=False))
        if not words:
            words = list(jieba.analyse.extract_tags(query, topK=6))
        if not words:
            words = [query]

        result_chars = []
        used = set()
        for w in words:
            if len(result_chars) >= num_keywords:
                break
            m = lookup_modern(w)
            if m:
                for ch in m:
                    if ch not in used and len(result_chars) < num_keywords:
                        result_chars.append(ch)
                        used.add(ch)
                        break
            else:
                ch = w[0]
                if ch not in used and ch not in STOP_CHARS and ch in self.vocab:
                    result_chars.append(ch)
                    used.add(ch)

        # Step 3: 不足4个则用 RNNLM 扩展
        if len(result_chars) < num_keywords:
            result_chars = self._expand(result_chars, num_keywords)

        return result_chars[:num_keywords]

    def _expand(self, keywords, target_num, top_k=10):
        """RNNLM + 意象词加分扩展"""
        current = list(keywords)
        with torch.no_grad():
            while len(current) < target_num:
                ids = [self.vocab.get(ch, config.UNK_ID) for ch in current]
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits, _ = self.kw_model(x)
                last = logits[0, -1].clone()

                # 屏蔽已有字、特殊token、虚词
                for ch in current:
                    if ch in self.vocab:
                        last[self.vocab[ch]] = -float('inf')
                for i in range(5):
                    last[i] = -float('inf')
                for ch in STOP_CHARS:
                    if ch in self.vocab:
                        last[self.vocab[ch]] = -float('inf')

                # 意象词加分
                for ch in IMAGERY:
                    if ch in self.vocab:
                        last[self.vocab[ch]] += 2.0

                # top-k 采样
                topk_v, topk_i = last.topk(top_k)
                probs = torch.softmax(topk_v / 0.9, dim=0)
                idx = torch.multinomial(probs, 1).item()
                current.append(self.rev_vocab.get(topk_i[idx].item(), '?'))
        return current
