# MemFactory - 强化学习驱动的记忆处理与训练框架

![MemFactory Architecture](https://via.placeholder.com/800x400?text=MemFactory+Architecture+Diagram)

## 项目简介

**MemFactory** 是一个专为记忆系统设计的**强化学习推训一体化框架**，旨在通过模块化、算子化的方式集成并优化各种记忆基本操作。框架原生支持记忆抽取(Extraction)、检索(Retrieval)和更新(Update)等核心功能，并提供完整的强化学习训练管道，使记忆管理策略能够通过与环境的交互不断自我优化。

本框架已成功集成了多种前沿记忆管理方法，包括 **Memory-R1**、**MemAgent** 和 **RMM** 等，同时支持用户自定义数据结构和记忆流程编排，为构建智能记忆系统提供了灵活而强大的基础设施。

## 🏗️ 核心架构

### 模块化记忆算子设计

MemFactory采用**算子化**(Operator-based)设计理念，将记忆处理的各个环节抽象为可插拔的模块：

```
MemFactory/
├── memfactory/             # 核心模块源码
│   ├── agents/             # 智能体实现
│   │   ├── base.py         # 基础智能体类
│   │   ├── memory_agent.py # MemAgent智能体
│   │   └── memory_r1_agent.py # Memory-R1智能体
│   ├── common/             # 公共组件
│   │   ├── __init__.py
│   │   ├── registry.py     # 注册表机制
│   │   └── utils.py        # 工具函数
│   ├── envs/               # 训练环境
│   │   ├── base.py         # 基础环境类
│   │   ├── longcontext_memory.py # 长上下文记忆环境
│   │   ├── memory_bank.py  # 记忆库环境
│   │   └── memory_bank_utils.py # 记忆库工具
│   ├── modules/            # 核心记忆算子
│   │   ├── base.py         # 基础模块类
│   │   ├── memory_extractor.py # 记忆抽取算子
│   │   ├── memory_retriever.py # 记忆检索算子  
│   │   ├── memory_updater.py # 记忆更新算子
│   │   └── placeholders.py # 占位符
│   ├── trainers/           # 训练器
│   │   ├── __init__.py
│   │   └── mem_grpo_trainer.py # GRPO记忆训练器
│   └── __init__.py
├── RL/                     # 强化学习训练模块
│   ├── mem_grpo_trainer.py   # GRPO训练器实现
│   ├── train_grpo.py         # 训练脚本
│   ├── reward.py            # 奖励函数
│   └── mem_utils.py         # 训练工具
├── examples/               # 训练脚本示例
│   ├── run_MemAgent.sh     # 运行记忆智能体脚本
│   ├── run_MemR1.sh        # 运行记忆R1脚本
│   └── train_mem_grpo.py   # GRPO训练脚本
└── scripts/                # 实用工具脚本
    ├── evaluate_mem_model.py # 模型评估脚本
    ├── mem_training_data.py # 训练数据处理
    └── process_locomo.py    # 数据预处理
```

### 强化学习训练管道

框架内置 **GRPO (Generalized Reward Policy Optimization)** 训练算法，支持端到端的记忆策略优化：

```
数据准备 → 经验生成 → 策略优化 → 模型保存
    ↑          ↓
环境交互 ← 奖励计算
```

## 🔧 核心功能特性

### 1. 模块化记忆处理管道

实现标准的记忆处理流水线，每个环节都是可替换的算子：

```
对话输入 → [记忆抽取] → [更新决策] → [统一存储] → [组织构建] → [检索回答]
```

- **记忆抽取算子**：从原始对话中智能识别和提取有价值的记忆信息
- **记忆更新算子**：基于相似度和冲突检测的自动更新决策
- **记忆检索算子**：融合向量检索和图遍历的智能搜索
- **统一存储**：Neo4j(图结构) + Milvus(向量)混合存储架构

### 2. 多种记忆管理策略集成

框架已集成多种先进的记忆管理方法：

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| **Memory-R1** | 基于推理的记忆管理 | 复杂决策场景 |
| **MemAgent** | 模块化智能体架构 | 灵活组合需求 |
| **RMM** | 检索增强记忆模块 | 高精度检索需求 |
| **Naive** | 基础记忆处理 | 快速原型验证 |

### 3. 智能更新决策系统

支持五种更新操作策略，通过强化学习自动优化决策：

| 操作类型 | 判定条件 | 适用场景 |
|---------|---------|---------|
| **ADD** | 无相关记忆 | 新增独特信息 |
| **MERGE** | 相似度0.6-0.85，无冲突 | 信息补充扩展 |
| **OVERWRITE** | 相似度0.6-0.85，事实冲突 | 事实修正更新 |
| **VERSION** | 相似度0.6-0.85，偏好/时间冲突 | 版本管理和偏好追踪 |
| **IGNORE** | 相似度>0.85 | 重复信息过滤 |

### 4. 自定义扩展能力

#### 自定义记忆数据结构
```python
from memfactory.envs.memory_bank_utils import MemoryItem

class CustomMemoryItem(MemoryItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加自定义字段
        self.custom_field = kwargs.get('custom_field', None)
        self.metadata = kwargs.get('metadata', {})
```

#### 自定义记忆流程编排
```python
from memfactory.modules.base import BaseModule

class CustomMemoryPipeline(BaseModule):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        # 组合不同的记忆算子
        self.extractor = CustomExtractor(tokenizer, **kwargs)
        self.updater = CustomUpdater(tokenizer, **kwargs)
        self.retriever = CustomRetriever(tokenizer, **kwargs)
    
    def rollout(self, model, batch_data, **kwargs):
        # 自定义训练流程
        extracted = self.extractor.rollout(model, batch_data, **kwargs)
        updated = self.updater.rollout(model, extracted, **kwargs)
        return updated
```

#### 自定义奖励函数
```python
def custom_reward_function(predictions, ground_truths, **kwargs):
    """自定义奖励计算逻辑"""
    # 实现特定业务场景的奖励计算
    reward = calculate_business_specific_reward(predictions, ground_truths)
    return reward
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd MemFactory

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件配置API密钥和数据库连接
```

### 2. 运行预置训练脚本

```bash
# 训练MemAgent
./examples/run_MemAgent.sh

# 训练Memory-R1
./examples/run_MemR1.sh
```

### 3. 自定义训练

```python
from memfactory.trainers.mem_grpo_trainer import MemGRPOTrainer, MemGRPOArguments
from memfactory.common.registry import AGENT_REGISTRY, ENV_REGISTRY

# 配置训练参数
args = MemGRPOArguments(
    output_dir="./output/custom_model",
    lr=1e-6,
    agent_type="memory_r1_agent",  # 使用Memory-R1智能体
    env_type="memory_bank",        # 使用记忆库环境
    train_extraction=True,         # 训练抽取能力
    train_update=True,             # 训练更新能力
    epoch=5,
    batch_size=2,
    gradient_accumulation_steps=8
)

# 初始化训练器
trainer = MemGRPOTrainer(model, args, tokenizer, ref_model)

# 开始训练
trainer.train(data_path="./datas/train.jsonl")
```

## 📊 示例演示

### 完整Pipeline演示

查看 `example/memory_pipeline_demo.ipynb` 了解：

1. **多轮对话处理**：演示三轮对话中的各种更新场景
2. **更新决策追踪**：实时显示ADD/MERGE/OVERWRITE等操作
3. **记忆检索验证**：验证更新后的记忆检索准确性
4. **强化学习训练**：展示如何通过RL优化记忆策略

### 单模块示例

- `ch10_memory_extraction_01.ipynb`：记忆抽取算法详解
- `ch11_memory_organization_01.ipynb`：记忆组织结构构建  
- `ch13_memory_update_01.ipynb`：更新决策机制演示

## ⚙️ 配置说明

### 环境变量配置 (.env)

```env
# LLM配置
OPENAI_API_KEY=your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-nano

# Embedding配置
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_BASE_URL=your-embedding-api-url
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024

# Neo4j图数据库
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Milvus向量数据库
MILVUS_URI=http://localhost:19530
MILVUS_USER=root
MILVUS_PASSWORD=your-password
```

## 🛠️ 开发指南

### 框架扩展

```python
# 自定义记忆抽取器
class CustomExtractor(MemoryExtractor):
    def extract_from_text(self, text: str) -> List[MemoryItem]:
        # 实现自定义抽取逻辑
        pass

# 自定义更新策略
class CustomUpdater(MemoryUpdater):
    def _auto_select_action(self, candidate: MemoryItem, related: List[Tuple[MemoryItem, float]]) -> str:
        # 实现自定义决策逻辑
        pass
```

### 训练定制

```python
# 自定义训练配置
config = {
    "task": "extraction",
    "model_name": "custom-model",
    "reward_function": custom_reward_function,
    "training_params": {
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 2e-5
    }
}
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！我们特别欢迎以下类型的贡献：

1. **新的记忆算子**：实现创新的记忆处理方法
2. **训练算法改进**：优化现有的GRPO算法或引入新算法
3. **环境扩展**：添加新的训练环境和评估基准
4. **文档完善**：改进文档和示例代码

## 📚 相关资源

- 《Memory Engineering》书籍配套代码
- [Neo4j图数据库](https://neo4j.com/)
- [Milvus向量数据库](https://milvus.io/)
- [OpenAI API文档](https://platform.openai.com/docs)

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情