# CLAUDE.md - poemgen 项目规范

## 项目概述
古诗生成系统：输入任意 query → 输出四句绝句（五言/七言）。
整诗一次性生成（非逐行），支持古代+现代概念输入。

## 运行环境
- M3 MacBook, Apple MPS GPU, PyTorch 2.8+
- Python 3.9 (system python3), 无 venv
- 不使用 GRU/LSTM (MPS 不兼容)，只用 Transformer
- 单次训练 < 4 小时

## 项目结构
```
poemgen/
├── configs/          # YAML 实验配置
├── data/             # 数据处理 + 映射表
│   ├── mapping/      # 现代→古代映射 (git tracked)
│   └── processed/    # 处理后数据 (gitignored)
├── models/           # 模型定义
├── planning/         # Query 改写模块
├── training/         # 训练
├── evaluation/       # 评估
├── inference/        # 推理
└── experiments/      # 实验产物 (gitignored)
```

## 命令规范
```bash
# 数据准备
python3 -m poemgen.data.preprocess --config configs/<exp>.yaml

# 训练
python3 -m poemgen.training.train --config configs/<exp>.yaml

# 评估
python3 -m poemgen.evaluation.eval_generation --exp <name>
python3 -m poemgen.evaluation.eval_planning --exp <name>

# 推理
python3 -m poemgen.inference.generate --exp <name> --query "月亮"

# 对比
python3 -m poemgen.evaluation.compare --experiments exp1 exp2
```

## 代码规范
- 超参数全部放 YAML config，代码中不 hardcode
- 每个实验隔离: experiments/<name>/{checkpoints,logs,eval_results}
- 新实验创建新 config，不修改已有实验 config
- 训练数据格式: `<5>` 或 `<7>` + query + `\t` + 整首诗(SEP分隔)

## 收敛目标
| 指标 | 目标 |
|------|------|
| 格式正确率 | ≥ 99% |
| 主题相关度 | ≥ 70% |
| 押韵率 | ≥ 50% |
| 行内重复率 | ≤ 2% |
| 优秀率 | ≥ 40% |
| 现代概念 | 12/12 可用 |

## 数据
- 原始: chinese-poetry repo (已下载在 key_seq2seq/ppg/data/raw_poetry/)
- 映射表: data/mapping/{common.json, proper.json}
- 训练格式: `<7>春风归乡\t春风又绿江南岸<SEP>明月何时照我还<SEP>...<EOS>`

## 模型
- Encoder-Decoder Transformer (主方案)
- d_model/nhead/layers 由 config 控制
- Weight tying + segment embedding + pre-norm
- 解码: top-p + repetition penalty + 韵律 mask
