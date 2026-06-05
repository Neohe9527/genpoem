"""配置文件"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# 词表
VOCAB_SIZE = 6000
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
SEP_ID = 4

# 生成模型
EMB_DIM = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_LINE_LEN = 7
NUM_LINES = 4

# 训练
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LR_DECAY = 0.5
MAX_EPOCHS = 50
PATIENCE = 5
GRAD_CLIP = 5.0

# RNNLM (关键词扩展)
KW_EMB_DIM = 256
KW_HIDDEN = 256
KW_EPOCHS = 30
KW_BATCH_SIZE = 128
KW_LR = 0.001
