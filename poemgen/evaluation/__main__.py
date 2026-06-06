"""python3 -m poemgen.evaluation.eval_generation --exp base_enc_dec"""
import argparse
from .eval_generation import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估生成质量")
    parser.add_argument("--exp", default="base_enc_dec", help="实验名称")
    args = parser.parse_args()
    evaluate(args.exp)
