# LLM 时代诗歌生成技术调研报告

> 调研日期：2026-06-06
> 调研范围：2023-2026 年发表的诗歌生成相关论文
> 关注问题：大语言模型时代，诗歌生成技术的思路和范式有哪些变化？

---

## 一、范式变迁总览

2023 年大语言模型技术爆发后，诗歌生成领域经历了根本性的范式转变：

| 维度 | 传统方法（2016-2022） | LLM 时代新方法（2023-2026） |
|------|----------------------|---------------------------|
| 架构 | 自定义 Seq2Seq/Transformer，从零训练 | 预训练 LLM（GPT-4、Qwen、DeepSeek）微调 |
| 训练策略 | 监督学习，诗歌语料从零训练 | SFT 微调 → RLHF/GRPO 强化学习对齐 |
| 约束控制 | 架构内置规划模块（Planning-based） | 提示工程、受限解码、奖励模型 |
| 多样性 | 隐变量（VAE）、对抗训练 | 多智能体竞争、多样性奖励函数 |
| 格式控制 | 架构层面强制（固定长度输出） | Token-free 模型、逆向提示、GRPO 奖励 |
| 评估方法 | BLEU、Perplexity、人工评分 | LLM-as-judge、多维评估框架（POEMetric） |

核心变化：**从"训练专用小模型"转向"对齐通用大模型"**。

---

## 二、重点论文深度分析

### 2.1 中国古典诗歌生成

#### CharPoet: Token-free LLM for Chinese Classical Poetry
- **作者**: Chengyue Yu, Lei Zang, Jiaotuan Wang, Chenyi Zhuang, Jinjie Gu
- **来源**: arXiv 2024 (arxiv.org/abs/2401.03512)
- **核心贡献**: 从 Qwen-chat-7B 剪枝得到字级（token-free）LLM，专门用于古诗生成
- **方法**: 
  - 移除 Qwen 的 BPE 分词器，改为逐字输入
  - 从预训练 LLM 中剪枝（而非从零训练），继承预训练知识
  - 字级生成天然避免了 subword 分词对格律结构的破坏
- **结果**: 格式准确率 0.96，远超 token-based LLM（0.84）和 GPT-4（0.38）
- **启示**: **分词（tokenization）是 LLM 写格律诗的最大技术瓶颈**。字级模型天然适合有严格字数要求的诗歌形式。

#### PoeTone: Constrained Generation of Chinese Songci with LLMs
- **来源**: arXiv 2025 (arxiv.org/abs/2508.02515)
- **核心贡献**: 面向宋词（词牌）的受限生成框架
- **方法**: 在 LLM 推理阶段施加词牌结构约束（句式、平仄、韵脚）
- **启示**: 对于有复杂格律要求的诗歌形式，推理时约束比训练时约束更灵活。

#### Xunzi Yayun R1: GRPO + RAG for Tang Poetry
- **来源**: Nature npj Heritage Science, 2025
- **核心贡献**: 将 DeepSeek-R1 的 GRPO 强化学习范式应用于唐诗生成
- **方法**:
  - 构建专用古诗语料库
  - 用 DeepSeek-R1-671B 进行数据蒸馏（生成高质量训练对）
  - 设计多维奖励函数：平仄合规率、押韵率、对仗质量、字数准确
  - 用 GRPO（Generative Ranking Policy Optimization）进行强化学习训练
  - 结合 RAG 检索增强，引入典故和意象
- **启示**: **2025 年最前沿的古诗生成方法**。代表了从 SFT 到 RL 驱动的范式转变，奖励信号的设计成为核心技术挑战。

#### BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting
- **作者**: X. Zou
- **来源**: arXiv 2024 (2411.13237)
- **核心贡献**: 零样本受限生成框架，无需微调即可让通用 LLM 生成合格古诗
- **方法**: 在推理时通过"逆向提示"（inverse prompting）逐块约束生成方向
- **启示**: 对于无法微调的闭源模型（如 GPT-4），推理时约束是唯一可行路径。

#### "AI 写的古诗与人类不可区分"
- **来源**: arXiv 2024 (arxiv.org/abs/2401.04952)
- **核心贡献**: 图灵测试式实证研究
- **结论**: 微调的开源 LLM 生成的古诗，人类评审无法与古代诗人作品可靠区分
- **启示**: **基本质量问题已被 LLM "解决"**。研究重心从"生成流畅诗歌"转向"生成有创造力、有独特性的诗歌"。

---

### 2.2 英文及多语言诗歌生成

#### ByGPT5: End-to-End Style-conditioned Poetry Generation
- **作者**: Jonas Belouadi, Steffen Eger
- **来源**: ACL 2023（主会议长文）
- **核心贡献**: Token-free decoder-only 模型，可条件化控制韵律、格律、头韵
- **方法**:
  - 预训练字级语言模型（不使用 subword 分词）
  - 在大规模英德四行诗语料上微调，标注韵律/格律/头韵属性
  - 生成时通过属性条件控制风格
- **结果**: 超越 mT5、ByT5、GPT-2、ChatGPT，且参数量更小
- **启示**: 字级模型 + 风格条件化是兼顾质量和可控性的有效方案。

#### LLM-based Multi-Agent Poetry Generation in Non-cooperative Environments
- **作者**: Ran Zhang, Steffen Eger
- **来源**: arXiv 2024; Journal of Language Modelling 2024/2025
- **核心贡献**: 多个 LLM Agent 在竞争环境中生成诗歌
- **方法**:
  - 多个 Agent 各自生成，互相评判
  - 非合作博弈机制产生竞争压力
  - 竞争促使各 Agent 生成更具差异性的作品
- **启示**: **解决 LLM 生成诗歌同质化问题**的创新思路。单模型生成倾向于"安全"的平均水平，多智能体竞争可以突破这一局限。

#### AI-generated Poetry is Indistinguishable from Human-written Poetry
- **作者**: Brian Porter, Edouard Machery
- **来源**: Scientific Reports (Nature), 2024
- **核心贡献**: ChatGPT 3.5 生成的诗歌不仅不可区分，还被人类评审评为更优
- **争议**: 引发关于评估方法论的广泛讨论——评审者可能偏好"易懂"的诗歌，而人类诗人的作品往往更晦涩

---

### 2.3 评估方法革新

#### POEMetric: The Last Stanza of Humanity
- **来源**: arXiv 2025 (arxiv.org/abs/2604.03695)
- **核心贡献**: 首个全面的诗歌评估框架，10 个指标 3 个维度
- **评估维度**:
  - 基础维度：格式准确率、主题对齐
  - 高级创意维度：创造力、词汇多样性、独特性（idiosyncrasy）、情感共鸣、文学手法/意象
  - 综合评价：整体质量
- **关键发现**: 
  - 顶级模型格式准确率 4.26/5.00，主题对齐 4.99/5.00
  - 但所有模型在**创造力（4.02）、独特性（3.95）、情感共鸣（4.06）、意象（4.49）**上显著落后于人类诗人
- **方法**: LLM-as-judge（Gemini-2.5-Pro）+ 人类专家验证
- **启示**: PPL/BLEU 已不足以评估诗歌质量，需要多维评估框架。

#### Evaluating Diversity in Automatic Poetry Generation
- **作者**: Yanran Chen, Hannes Groner, Sina Zarriess, Steffen Eger
- **来源**: EMNLP 2024（主会议）
- **核心贡献**: 从结构、词汇、语义、风格四个维度评估生成诗歌的多样性
- **关键发现**: AI 生成诗歌在语义和风格多样性上显著弱于人类诗歌语料
- **启示**: 多样性是当前 AI 诗歌的核心短板。

#### LLM 评估偏见研究
- **来源**: arXiv 2025 (2510.15313)
- **核心贡献**: 发现 LLM 作为诗歌评判者存在文化偏见
- **启示**: 对于古诗等文化深度高的内容，纯 LLM 评估不够可靠，需要人机混合评估。

---

### 2.4 综合调查论文

#### Computational Approaches to Automatic Poetry Generation and Evaluation: A Survey
- **来源**: JAIR (Journal of Artificial Intelligence Research), 2025
- **覆盖范围**: 2017-2025 全面综述
- **价值**: 系统性梳理了从前 LLM 时代到 LLM 时代的方法演进

---

## 三、关键技术趋势

### 3.1 Token-free / 字级模型

传统 LLM 使用 BPE/SentencePiece 分词，导致：
- 一个汉字可能被拆成多个 subword token
- 模型无法精确控制输出字数
- 格律诗的"每句七字"约束难以满足

解决方案：CharPoet（剪枝）、ByGPT5（从零训练字级 LM）证明字级模型在格律诗场景显著优于 subword 模型。

### 3.2 强化学习驱动的格律对齐

SFT 微调只能让模型"大致学会"诗歌格式，但无法保证严格合规。GRPO/PPO 通过设计精确的奖励函数（平仄、押韵、对仗），将格律合规作为优化目标。

奖励函数设计示例（Xunzi Yayun R1）：
- 字数奖励：每句字数 = 目标字数 → +1
- 押韵奖励：韵脚在同一韵部 → +1
- 平仄奖励：符合平仄规则的字占比
- 对仗奖励：颔联/颈联词性对仗度

### 3.3 规划思想的延续

传统 Planning-based 方法（如 Wang et al. COLING 2016）的核心思想——将诗歌生成分解为子任务——在 LLM 时代以新形式存活：
- **Chain-of-thought prompting**: "先构思主题 → 确定意象 → 逐句生成"
- **Multi-step generation**: 先生成大纲/关键词，再填充完整诗句
- **Self-refinement**: 生成后自我评估，迭代改进

区别：不再需要训练专门的规划模块，LLM 的 in-context learning 能力可以直接完成规划。

### 3.4 多智能体创作

解决 LLM 生成诗歌的同质化问题：
- 多个 Agent 各自创作
- 竞争机制（非合作博弈）促进差异化
- 评审 Agent 筛选最优作品
- 类似人类诗社的"联句"传统

---

## 四、对我们项目的启示

### 4.1 当前项目定位

我们实现的 PPG（Planning-based Poetry Generation）属于传统范式：
- 关键词规划（TextRank + RNNLM）→ 逐句生成（Transformer）
- 字级模型（vocab=6000），直接操作汉字
- 264K 训练数据，24M 参数，Val PPL 70

### 4.2 与 LLM 方法的对比

| | 我们的 PPG | LLM 方法（如 CharPoet） |
|---|---|---|
| 优势 | 轻量（24M），M3 可训练，完全可控 | 质量更高，知识更丰富 |
| 劣势 | 意象单一，文化知识有限 | 需要 7B+ 模型，算力要求高 |
| 格式控制 | 架构保证 7 字 | CharPoet 0.96，GPT-4 仅 0.38 |

### 4.3 可借鉴的改进方向

**短期（当前可做）**：
1. Planning 模块优化：用 Transformer embedding 相似度替代 RNNLM 扩展（已有方案）
2. 多维评估：参考 POEMetric，增加格律合规率、主题相关度评估

**中期**：
3. 引入 RL 奖励：将平仄合规率、押韵率作为额外 loss/reward
4. 自回归 refinement：生成后用模型自我评估，重新生成不满意的句子

**长期**：
5. 迁移到更大模型：用 Qwen-1.8B 做字级微调
6. GRPO 训练：设计诗歌格律奖励函数，强化学习对齐

---

## 五、参考文献

1. Yu et al. "CharPoet: Token-free LLM for Chinese Classical Poetry." arXiv:2401.03512, 2024.
2. Belouadi & Eger. "ByGPT5: End-to-End Style-conditioned Poetry Generation." ACL 2023.
3. "Xunzi Yayun R1: GRPO + RAG for Tang Poetry." npj Heritage Science, 2025.
4. Zou. "BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting." arXiv:2411.13237, 2024.
5. "Can AI Write Classical Chinese Poetry like Humans?" arXiv:2401.04952, 2024.
6. "PoeTone: Constrained Generation of Chinese Songci with LLMs." arXiv:2508.02515, 2025.
7. Zhang & Eger. "LLM-based Multi-Agent Poetry Generation in Non-cooperative Environments." 2024.
8. Porter & Machery. "AI-generated Poetry is Indistinguishable from Human-written Poetry." Scientific Reports, 2024.
9. "POEMetric: The Last Stanza of Humanity." arXiv:2604.03695, 2025.
10. Chen et al. "Evaluating Diversity in Automatic Poetry Generation." EMNLP 2024.
11. Ma et al. "Capabilities and Evaluation Biases of LLMs in Classical Chinese Poetry Generation." arXiv:2510.15313, 2025.
12. "Computational Approaches to Automatic Poetry Generation and Evaluation: A Survey." JAIR, 2025.
13. "CCL25-Eval Task 5: Chinese Classical Poetry Appreciation." CCL 2025.
14. Chen et al. "Benchmarking LLMs for Translating Classical Chinese Poetry (PoetMT)." EMNLP 2025.
15. Wang et al. "What is the Best Way for ChatGPT to Translate Poetry?" ACL 2024.
