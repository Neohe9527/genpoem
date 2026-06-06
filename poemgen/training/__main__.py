"""python3 -m poemgen.training.train --config configs/xxx.yaml"""
import argparse
from .train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--config", default="configs/base_enc_dec.yaml")
    args = parser.parse_args()
    train(args.config)
