"""python3 -m poemgen.inference.generate --exp base_enc_dec --query 春天"""
import argparse
from .generate import PoemGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成古诗")
    parser.add_argument("--exp", default="base_enc_dec", help="实验名称")
    parser.add_argument("--query", default="春天", help="输入 query")
    parser.add_argument("--fmt", default="7", choices=["5", "7"], help="五言/七言")
    args = parser.parse_args()

    gen = PoemGenerator(args.exp)
    lines, rewritten = gen.generate(args.query, fmt=args.fmt)
    print(f"Query: {args.query} → Rewritten: {rewritten}")
    print("诗:")
    for line in lines:
        print(f"  {line}")
