# 评估模块升级计划

## 目标
参考 POEMetric 框架，将 `evaluation/eval_generation.py` 从 4 个简单指标升级为多维度评估体系。

## 新增指标

### 规则评估（自动，零 API 成本）
1. **平仄合规率** — 用 pypinyin 获取声调，1/2声=平，3/4声=仄，检查是否符合七言绝句四种基本句式
2. **对仗检测** — 七言绝句不要求对仗，但检查第 1/2 句结构对称性作为加分项
3. **MATTR 词汇多样性** — Moving Average Type-Token Ratio，窗口=5 字
4. **主题相关度升级** — 基于字符 Jaccard 相似度 + mapping 回溯匹配，替代当前的简单 `any(ch in poem)`

### 保留并优化的现有指标
5. **格式正确率** — 保持不变（4 行 × 7 字）
6. **押韵率** — 保持平水韵表，增加首句入韵检测

## 实现方案

- 新建 `evaluation/metrics.py`：所有指标函数集中定义
- 重写 `evaluation/eval_generation.py`：调用 metrics 模块，输出多维度报告
- 输出格式：终端表格 + JSON（与 POEMetric 的 Likert 量表对齐，将规则分数映射到 1-5 分）

## 文件变更
1. 新增 `evaluation/metrics.py` — 指标计算核心
2. 重写 `evaluation/eval_generation.py` — 评估入口

## 不做的事
- 不做 LLM-as-Judge（当前模型阶段不需要）
- 不改变现有的 experiments/ 产物结构
