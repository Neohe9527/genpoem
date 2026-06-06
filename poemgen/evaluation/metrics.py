"""多维度诗歌评估指标（参考 POEMetric 框架适配中国古诗）"""
from pypinyin import pinyin, Style

# ============ 平仄检测 ============

# 七言绝句四种基本句式（中=可平可仄，用 * 表示）
SEVEN_PATTERNS = [
    "**仄仄平平仄",  # 仄起仄收
    "**平平仄仄平",  # 平起平收
    "**平平平仄仄",  # 平起仄收
    "**仄仄仄平平",  # 仄起平收
]


def _get_tone(char):
    """获取单字平仄：1/2声=平，3/4声=仄，未知=None"""
    result = pinyin(char, style=Style.TONE3, heteronym=True)
    if not result or not result[0]:
        return None
    # 取所有多音中最常见的
    for py in result[0]:
        tone_num = py[-1] if py[-1].isdigit() else None
        if tone_num in ('1', '2'):
            return '平'
        elif tone_num in ('3', '4'):
            return '仄'
    return None


def check_tone_pattern(line):
    """检查单句平仄合规度，返回 0-1 分数"""
    if len(line) != 7:
        return 0.0
    tones = [_get_tone(ch) for ch in line]

    best_score = 0.0
    for pattern in SEVEN_PATTERNS:
        match = 0
        total = 0
        for i, (tone, expected) in enumerate(zip(tones, pattern)):
            if expected == '*':
                continue  # 可平可仄位置不计
            total += 1
            if tone == expected:
                match += 1
            elif tone is None:
                match += 0.5  # 未知字给半分
        score = match / max(total, 1)
        best_score = max(best_score, score)
    return best_score


def eval_tone_compliance(lines):
    """整首诗平仄合规率"""
    if len(lines) != 4:
        return 0.0
    return sum(check_tone_pattern(l) for l in lines) / 4


# ============ 押韵检测（增强版）============

RHYME_GROUPS = {
    "一东": "东同铜桐筒童僮瞳中衷忠虫冲终戎融穷弓宫躬雄熊穹风枫丰充隆空公功工攻蒙笼聋通翁",
    "二冬": "冬宗钟龙松冲容从峰封胸浓重慵恐",
    "三江": "江缸降双窗邦庞撞幢桩",
    "四支": "支枝移为垂吹陂碑奇宜仪皮儿离施知池规危师姿期词基疑资雌慈时",
    "五微": "微薇晖辉非飞妃归稀依衣矶机几讥",
    "六鱼": "鱼初书居车渠疏余虚裾",
    "七虞": "虞愚娱隅于夫无芜朱珠株枯湖蒲胡壶乎符图都督呼苏模",
    "八齐": "齐西溪兮迷低堤鸡凄妻题蹄啼",
    "十灰": "灰回杯台来才开哀埃",
    "十一真": "真人邻新身亲津辛频尘春神臣珍陈",
    "十二文": "文云分群勋芬纷焚",
    "十三元": "元源园远原言轩烦魂门村存尊恩痕昏",
    "十四寒": "寒安官端丹欢宽盘残看难",
    "十五删": "删关还闲班山间",
    "一先": "先前天田年边烟然贤弦泉眠连牵悬川船全宣",
    "二萧": "萧条苗腰遥招朝潮桥飘摇",
    "三肴": "肴交郊巢梢高豪劳刀",
    "四豪": "豪高毛曹桃陶遭劳刀",
    "六麻": "麻花茶沙华家霞斜",
    "七阳": "阳光芳香苍黄忘望霜伤长汤郎章昌狂",
    "八庚": "庚生明行英情京平清声城成兵名衡程",
    "十一尤": "尤求游流头楼愁秋洲州舟收忧柔",
    "十二侵": "侵深吟心寻林临琴禽阴今金",
    "十三覃": "覃南含参惭潭谙",
    "十四盐": "盐甜年帘添",
}

_char_to_rhyme = {}
for _group, _chars in RHYME_GROUPS.items():
    for _ch in _chars:
        _char_to_rhyme[_ch] = _group


def check_rhyme(lines):
    """检查押韵：绝句要求 2/4 句押韵，首句可选押韵
    返回 (是否押韵, 韵部, 详情)"""
    if len(lines) != 4:
        return False, "", "行数不足"

    ends = [l[-1] if l else "" for l in lines]
    r = [_char_to_rhyme.get(e, "") for e in ends]

    # 2/4 句必须同韵
    if r[1] and r[1] == r[3]:
        # 检查首句是否也入韵
        first_rhymes = r[0] == r[1] if r[0] else False
        detail = f"{ends[1]}({r[1]})+{ends[3]}({r[3]})"
        if first_rhymes:
            detail = f"{ends[0]}+{detail} (首句入韵)"
        return True, r[1], detail
    return False, "", f"{ends[1]}({r[1]}) vs {ends[3]}({r[3]})"


# ============ MATTR 词汇多样性 ============

def compute_mattr(text, window=5):
    """Moving Average Type-Token Ratio，字级"""
    chars = [ch for ch in text if ch.strip()]
    if len(chars) <= window:
        return len(set(chars)) / max(len(chars), 1)
    ratios = []
    for i in range(len(chars) - window + 1):
        w = chars[i:i + window]
        ratios.append(len(set(w)) / window)
    return sum(ratios) / len(ratios)


# ============ 主题相关度（增强版）============

def eval_topic_relevance(query, rewritten, lines, mapping=None):
    """多层主题相关度检测，返回 0-1 分数
    策略：
    1. rewritten 的字出现在诗中 (权重 0.4)
    2. query 原始字出现在诗中 (权重 0.3)
    3. mapping 回溯：诗中字属于 query 的映射集合 (权重 0.3)
    """
    poem_text = "".join(lines)
    score = 0.0

    # 策略 1：改写后关键字匹配
    rewritten_chars = set(rewritten) - set("的了是在不一")
    if rewritten_chars:
        hit = sum(1 for ch in rewritten_chars if ch in poem_text)
        score += 0.4 * min(hit / max(len(rewritten_chars), 1), 1.0)

    # 策略 2：原始 query 字匹配
    query_chars = set(query) - set("的了是在不一")
    if query_chars:
        hit = sum(1 for ch in query_chars if ch in poem_text)
        score += 0.3 * min(hit / max(len(query_chars), 1), 1.0)

    # 策略 3：mapping 回溯
    if mapping and query in mapping:
        mapped_chars = set(mapping[query][:4]) if isinstance(mapping[query], list) else set()
        if mapped_chars:
            hit = sum(1 for ch in mapped_chars if ch in poem_text)
            score += 0.3 * min(hit / max(len(mapped_chars), 1), 1.0)
    else:
        # 无 mapping 时，策略 1/2 权重提升
        score = score / 0.7 if score > 0 else 0.0

    return min(score, 1.0)


# ============ 行内/行间重复检测 ============

def eval_repetition(lines):
    """检测重复，返回 bigram 重复率和行间重复率"""
    # Bigram 重复
    bigram_total = 0
    bigram_repeat = 0
    for line in lines:
        bgs = [line[i:i+2] for i in range(len(line)-1)]
        bigram_total += len(bgs)
        bigram_repeat += len(bgs) - len(set(bgs))

    # 行间重复（两行完全相同或子串包含）
    line_repeat = 0
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            if lines[i] == lines[j]:
                line_repeat += 1

    return {
        "bigram_rate": bigram_repeat / max(bigram_total, 1),
        "line_repeat": line_repeat,
    }


# ============ 综合评分（映射到 1-5 Likert 量表）============

def compute_all_metrics(lines, query="", rewritten="", mapping=None):
    """计算所有指标，返回字典"""
    n = 7
    fmt_ok = len(lines) == 4 and all(len(l) == n for l in lines)

    rhyme_ok, rhyme_group, rhyme_detail = check_rhyme(lines) if len(lines) == 4 else (False, "", "")
    tone_score = eval_tone_compliance(lines) if fmt_ok else 0.0
    poem_text = "".join(lines)
    mattr = compute_mattr(poem_text)
    topic = eval_topic_relevance(query, rewritten, lines, mapping)
    rep = eval_repetition(lines)

    return {
        "format_ok": fmt_ok,
        "rhyme_ok": rhyme_ok,
        "rhyme_group": rhyme_group,
        "rhyme_detail": rhyme_detail,
        "tone_compliance": round(tone_score, 3),
        "mattr": round(mattr, 3),
        "topic_relevance": round(topic, 3),
        "bigram_repeat_rate": round(rep["bigram_rate"], 4),
        "line_repeat": rep["line_repeat"],
    }


def to_likert(metrics):
    """将指标映射到 1-5 Likert 量表（对标 POEMetric）"""
    return {
        "form_accuracy": 5 if metrics["format_ok"] else 1,
        "rhyme": 5 if metrics["rhyme_ok"] else 2,
        "tone_compliance": round(1 + 4 * metrics["tone_compliance"], 1),
        "lexical_diversity": round(1 + 4 * metrics["mattr"], 1),
        "topic_relevance": round(1 + 4 * metrics["topic_relevance"], 1),
        "repetition_penalty": round(5 - 4 * min(metrics["bigram_repeat_rate"] * 10, 1), 1),
    }
