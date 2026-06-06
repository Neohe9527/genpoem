"""多维度生成质量评估（参考 POEMetric 框架）"""
import os
import json
import argparse

from ..inference.generate import PoemGenerator
from .metrics import compute_all_metrics, to_likert


def evaluate(exp_name="base_enc_dec"):
    gen = PoemGenerator(exp_name)

    test_queries = [
        # 古代主题 (12)
        "春风", "秋月", "离别", "思乡", "登高", "边塞",
        "饮酒", "山水", "梅花", "月夜", "送友", "归隐",
        # 现代概念 (12)
        "啤酒", "手机", "加班", "高铁", "程序员", "咖啡",
        "周杰伦", "流浪地球", "北京", "内卷", "坐地铁", "星巴克",
    ]

    results = []
    agg = {"format": 0, "rhyme": 0, "tone": 0.0, "mattr": 0.0,
           "topic": 0.0, "bigram_rep": 0.0, "total": 0}

    print(f"评估实验: {exp_name}")
    print("=" * 70)

    for query in test_queries:
        lines, rewritten = gen.generate(query)
        agg["total"] += 1

        # 获取 mapping 用于主题相关度计算
        mapping_entry = gen.mapping.get(query)

        m = compute_all_metrics(lines, query, rewritten,
                                {query: mapping_entry} if mapping_entry else None)
        likert = to_likert(m)

        if m["format_ok"]:
            agg["format"] += 1
        if m["rhyme_ok"]:
            agg["rhyme"] += 1
        agg["tone"] += m["tone_compliance"]
        agg["mattr"] += m["mattr"]
        agg["topic"] += m["topic_relevance"]
        agg["bigram_rep"] += m["bigram_repeat_rate"]

        results.append({
            "query": query, "rewritten": rewritten, "lines": lines,
            "metrics": m, "likert": likert,
        })

        # Display
        fmt_mark = "✓" if m["format_ok"] else "✗"
        rhyme_mark = f"韵✓{m['rhyme_group']}" if m["rhyme_ok"] else "韵✗"
        tone_mark = f"仄{m['tone_compliance']:.0%}"
        print(f"\n  [{fmt_mark}][{rhyme_mark}][{tone_mark}] 「{query}」→「{rewritten}」")
        for line in lines:
            print(f"    {line}")
        print(f"    → 主题:{m['topic_relevance']:.2f} 多样性:{m['mattr']:.2f} "
              f"重复:{m['bigram_repeat_rate']:.1%}")

    # Summary
    n = agg["total"]
    print("\n" + "=" * 70)
    print("评估结果汇总 (POEMetric-adapted)")
    print("-" * 70)
    fmt_pct = agg['format'] / n * 100
    rhyme_pct = agg['rhyme'] / n * 100
    tone_pct = agg['tone'] / n * 100
    topic_pct = agg['topic'] / n * 100
    rep_pct = agg['bigram_rep'] / n * 100
    print(f"  格式正确率:   {agg['format']}/{n} = {fmt_pct:.1f}%     (目标 ≥99%)")
    print(f"  押韵率:       {agg['rhyme']}/{n} = {rhyme_pct:.1f}%     (目标 ≥50%)")
    print(f"  平仄合规率:   {tone_pct:.1f}%")
    print(f"  MATTR多样性:  {agg['mattr']/n:.3f}")
    print(f"  主题相关度:   {topic_pct:.1f}%            (目标 ≥70%)")
    print(f"  Bigram重复率: {rep_pct:.2f}%            (目标 ≤2%)")

    # Save results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(base_dir, "experiments", exp_name, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    summary = {
        "format_accuracy": agg["format"] / n,
        "rhyme_rate": agg["rhyme"] / n,
        "tone_compliance": agg["tone"] / n,
        "mattr": agg["mattr"] / n,
        "topic_relevance": agg["topic"] / n,
        "bigram_repeat_rate": agg["bigram_rep"] / n,
    }
    with open(os.path.join(eval_dir, "generation.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": results}, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {eval_dir}/generation.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多维度生成质量评估")
    parser.add_argument("--exp", default="base_enc_dec", help="实验名称")
    args = parser.parse_args()
    evaluate(args.exp)
