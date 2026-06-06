"""LLM 古诗生成能力横评：6 模型 × 10 主题，复用 POEMetric 评估"""
import os
import re
import json
import time
import argparse
import requests

from .metrics import compute_all_metrics, to_likert

# ============ 配置 ============

API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "")

MODELS = [
    "Claude Opus 4.6",
    "gpt-5.5",
    "DeepSeek-V4-Pro",
    "GLM-5.1",
    "MiniMax-M2.7",
    "Kimi-K2.5",
]

TOPICS = [
    "春日登高远眺",
    "月下思乡",
    "秋雨中的离别",
    "雪夜独酌",
    "江南水乡晨曦",
    "边塞将士戍守",
    "归隐山林",
    "咏梅",
    "中秋望月怀远",
    "暮春送友远行",
]

SYSTEM_PROMPT = """你是一位精通中国古典诗词的诗人。请严格按照要求创作七言绝句。

要求：
1. 格式：七言绝句（四句，每句恰好七个汉字，共28字）
2. 押韵：偶数句（第2、4句）末字须押韵，首句末字可押可不押
3. 平仄：尽量遵循七言绝句的平仄格律
4. 质量：语言凝练优美，意象生动，有诗意和意境
5. 原创：不得照搬或改写已有名句

输出格式：仅输出四句诗，每句一行，不加标点符号，不加任何解释、标题或注释。"""


# ============ API 调用 ============

def call_claude(model, system, user):
    """Claude 走 /v1/messages"""
    resp = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=120,
    )
    if resp.status_code == 200:
        data = resp.json()
        # Claude thinking 模型返回 [thinking_block, text_block]
        for block in data.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                return block["text"].strip()
    return None


def call_openai_compat(model, system, user, temperature=0.7):
    """其他模型走 /v1/chat/completions"""
    if "Kimi" in model:
        temperature = 1

    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 3000,
            "temperature": temperature,
        },
        timeout=120,
    )
    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        # MiniMax/其他推理模型输出带 <think> 标签
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    return None


def generate_poem(model, topic, retries=2):
    """统一调用入口"""
    user_prompt = f"请以「{topic}」为主题，创作一首七言绝句。"
    for attempt in range(retries + 1):
        try:
            if "Claude" in model:
                raw = call_claude(model, SYSTEM_PROMPT, user_prompt)
            else:
                raw = call_openai_compat(model, SYSTEM_PROMPT, user_prompt)
            if raw:
                return raw
        except Exception as e:
            print(f"    [ERR attempt {attempt+1}] {e}")
            time.sleep(2)
    return None


# ============ 解析 ============

def parse_poem(raw_text):
    """从模型输出中解析四句诗（纯汉字行）"""
    if not raw_text:
        return []
    # 去掉标点
    text = re.sub(r'[，。！？、；：""''「」（）\s,\.!?;:\'\"\(\)]', '\n', raw_text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # 只保留汉字
    chinese_lines = []
    for l in lines:
        l = re.sub(r'^[\d\.、\)\]]+', '', l).strip()
        chars = re.sub(r'[^一-鿿]', '', l)
        if len(chars) >= 5:
            chinese_lines.append(chars)
    # 14字行拆分为两句
    expanded = []
    for l in chinese_lines:
        if len(l) == 14:
            expanded.append(l[:7])
            expanded.append(l[7:])
        else:
            expanded.append(l)
    return expanded[:4]


# ============ 评估 + 保存（对齐项目规范）============

def model_slug(model_name):
    """模型名 → 文件安全 slug"""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', model_name).strip('_')


def evaluate_model(model, poems, exp_dir):
    """对单个模型的生成结果做 POEMetric 评估，保存到 experiments/<exp>/eval_results/"""
    slug = model_slug(model)
    eval_dir = os.path.join(exp_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    agg = {"format": 0, "rhyme": 0, "tone": 0.0, "mattr": 0.0,
           "topic": 0.0, "bigram_rep": 0.0, "total": 0}
    samples = []

    for item in poems:
        lines = item["parsed_lines"]
        topic = item["topic"]
        agg["total"] += 1

        m = compute_all_metrics(lines, topic, topic, None)
        likert = to_likert(m)

        if m["format_ok"]:
            agg["format"] += 1
        if m["rhyme_ok"]:
            agg["rhyme"] += 1
        agg["tone"] += m["tone_compliance"]
        agg["mattr"] += m["mattr"]
        agg["topic"] += m["topic_relevance"]
        agg["bigram_rep"] += m["bigram_repeat_rate"]

        samples.append({
            "topic": topic,
            "lines": lines,
            "raw_output": item["raw_output"],
            "metrics": m,
            "likert": likert,
        })

    n = agg["total"]
    summary = {
        "model": model,
        "num_poems": n,
        "format_accuracy": round(agg["format"] / n, 4),
        "rhyme_rate": round(agg["rhyme"] / n, 4),
        "tone_compliance": round(agg["tone"] / n, 4),
        "mattr": round(agg["mattr"] / n, 4),
        "topic_relevance": round(agg["topic"] / n, 4),
        "bigram_repeat_rate": round(agg["bigram_rep"] / n, 4),
    }

    # 保存到 experiments/<exp>/eval_results/<model_slug>.json
    out_path = os.path.join(eval_dir, f"{slug}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": samples}, f, ensure_ascii=False, indent=2)

    return summary


# ============ 主流程 ============

def run_benchmark(exp_name="llm_benchmark"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(base_dir, "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    all_results = {}

    # Phase 1: 生成
    print("=" * 70)
    print(f"Phase 1: 生成古诗 ({len(MODELS)} 模型 × {len(TOPICS)} 主题)")
    print("=" * 70)

    for model in MODELS:
        print(f"\n  模型: {model}")
        print(f"  {'-'*50}")
        all_results[model] = []

        for i, topic in enumerate(TOPICS):
            print(f"    [{i+1:2d}/10] {topic} ... ", end="", flush=True)
            raw = generate_poem(model, topic)
            lines = parse_poem(raw)

            all_results[model].append({
                "topic": topic,
                "raw_output": raw or "(生成失败)",
                "parsed_lines": lines,
            })

            if lines and len(lines) == 4 and all(len(l) == 7 for l in lines):
                print(f"✓  {'｜'.join(lines)}")
            elif lines:
                print(f"⚠  行数={len(lines)} 字数={'|'.join(str(len(l)) for l in lines)}")
            else:
                print("✗  生成失败")
            time.sleep(0.5)

    # Phase 2: 评估
    print("\n\n" + "=" * 70)
    print("Phase 2: POEMetric 评估")
    print("=" * 70)

    summaries = {}
    for model in MODELS:
        summary = evaluate_model(model, all_results[model], exp_dir)
        summaries[model] = summary
        slug = model_slug(model)
        print(f"\n  {model}:")
        print(f"    格式正确率: {summary['format_accuracy']*100:.0f}%  "
              f"押韵率: {summary['rhyme_rate']*100:.0f}%  "
              f"平仄: {summary['tone_compliance']*100:.0f}%  "
              f"MATTR: {summary['mattr']:.3f}  "
              f"主题: {summary['topic_relevance']*100:.0f}%  "
              f"重复: {summary['bigram_repeat_rate']*100:.1f}%")

    # 保存横向对比汇总
    comparison_path = os.path.join(exp_dir, "eval_results", "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    # Phase 3: 打印排行
    print("\n\n" + "=" * 70)
    print("排行榜 (综合得分 = 格式×0.2 + 押韵×0.2 + 平仄×0.15 + 主题×0.2 + 多样性×0.15 + (1-重复)×0.1)")
    print("=" * 70)

    ranking = []
    for model, s in summaries.items():
        score = (s["format_accuracy"] * 0.20
                 + s["rhyme_rate"] * 0.20
                 + s["tone_compliance"] * 0.15
                 + s["topic_relevance"] * 0.20
                 + s["mattr"] * 0.15
                 + (1 - min(s["bigram_repeat_rate"] * 10, 1)) * 0.10)
        ranking.append((model, score, s))
    ranking.sort(key=lambda x: -x[1])

    print(f"\n  {'排名':<4} {'模型':<20} {'综合':<6} {'格式':<6} {'押韵':<6} {'平仄':<6} {'主题':<6} {'多样':<6} {'重复':<6}")
    print(f"  {'─'*4} {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for rank, (model, score, s) in enumerate(ranking, 1):
        print(f"  {rank:<4} {model:<20} {score:.3f}  "
              f"{s['format_accuracy']*100:4.0f}%  "
              f"{s['rhyme_rate']*100:4.0f}%  "
              f"{s['tone_compliance']*100:4.0f}%  "
              f"{s['topic_relevance']*100:4.0f}%  "
              f"{s['mattr']:5.3f}  "
              f"{s['bigram_repeat_rate']*100:4.1f}%")

    print(f"\n  评估结果目录: {exp_dir}/eval_results/")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 古诗生成横评")
    parser.add_argument("--exp", default="llm_benchmark", help="实验名称")
    args = parser.parse_args()
    run_benchmark(args.exp)
