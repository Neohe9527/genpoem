"""python3 -m poemgen.data.preprocess --config configs/xxx.yaml"""
import argparse
from .preprocess import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--config", default="configs/base_enc_dec.yaml")
    args = parser.parse_args()
    preprocess(args.config)
