# MemFactory - 记忆处理与强化学习框架

## 项目简介

MemFactory是一个模块化的记忆处理框架，集成了完整的记忆生命周期管理功能，同时内置强化学习训练支持。该框架提供了记忆抽取、更新决策、存储管理、组织构建、检索回答等核心模块，并支持通过强化学习优化记忆处理能力。

## 🏗️ 框架架构

### 核心模块组件

![logo](https://github.com/Valsure/MemFactory/blob/main/overall.png)

## 🔧 核心功能特性

### 1. 模块化记忆处理管道

实现标准的记忆处理流水线：

```
输入数据 → 记忆抽取 → 更新决策 → 统一存储 → 组织构建 → 检索应用
```

#### 主要模块特性：
- **MemoryExtractor**：智能识别和提取有价值的记忆信息
- **MemoryUpdater**：基于相似度和冲突检测的自动更新决策
- **MemoryStore**：Neo4j(图结构) + Milvus(向量)混合存储
- **MemoryOrganizer**：构建多维度记忆组织结构
- **MemorySearcher**：融合向量检索和图遍历的智能搜索

### 2. 智能更新决策系统

支持五种更新操作策略：

| 操作类型 | 判定条件 | 适用场景 |
|---------|---------|---------|
| **ADD** | 无相关记忆 | 新增独特信息 |
| **MERGE** | 相似度0.6-0.85，无冲突 | 信息补充扩展 |
| **OVERWRITE** | 相似度0.6-0.85，事实冲突 | 事实修正更新 |
| **VERSION** | 相似度0.6-0.85，偏好/时间冲突 | 版本管理和偏好追踪 |
| **IGNORE** | 相似度>0.85 | 重复信息过滤 |

### 3. 多维度记忆组织

构建层次化的记忆网络结构：

- **时间结构**：会话分割、阶段划分、时序关系建模
- **事件结构**：事件要素抽取、因果关系推理
- **语义结构**：基于向量相似性的语义连接
- **抽象层级**：从具体实例中提炼通用模式

### 4. 强化学习训练框架

内置GRPO(Generalized Reward Policy Optimization)训练支持：

- **双任务训练**：分别优化记忆抽取和更新决策能力
- **奖励机制**：基于下游任务表现的综合评估奖励
- **策略优化**：PPO算法驱动的策略改进
- **训练流程**：数据准备 → 经验生成 → 策略优化 → 模型保存

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone git@github.com:Valsure/MemFactory.git
cd MemFactory

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件配置API密钥和数据库连接
```

### 2. 基础使用

```python
from src.common import *
from src.memory_extraction import MemoryExtractor
from src.memory_update import MemoryUpdater
from src.memory_organization import MemoryOrganizer

# 初始化框架组件
extractor = MemoryExtractor(strategy="simple")
updater = MemoryUpdater(strategy="auto")
organizer = MemoryOrganizer()

# 处理对话数据
messages = [
    ConversationMessage(role="user", content="我是Tom，负责AI项目", timestamp="2025-06-26T10:00:00"),
    ConversationMessage(role="assistant", content="你好Tom！", timestamp="2025-06-26T10:01:00")
]

# 运行完整处理管道
result = pipeline.process_conversation(messages)
```

### 3. 强化学习训练

```bash
# 启动GRPO训练
python RL/mem_grpo_trainer.py


## ⚙️ 配置说明

### 环境变量配置 (.env)

```env
# LLM服务配置
OPENAI_API_KEY=your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-nano

# Embedding服务配置
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_BASE_URL=your-embedding-api-url
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024

# Neo4j图数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Milvus向量数据库配置
MILVUS_URI=http://localhost:19530
MILVUS_USER=root
MILVUS_PASSWORD=your-password
```

## 🧪 测试与验证

TODO

## 🛠️ 开发指南

### 框架扩展

```python
# 自定义记忆抽取器
class CustomExtractor(MemoryExtractor):
    def extract_from_text(self, text: str) -> List[MemoryNode]:
        # 实现自定义抽取逻辑
        pass

# 自定义更新策略
class CustomUpdater(MemoryUpdater):
    def _auto_select_action(self, candidate: MemoryNode, related: List[Tuple[MemoryNode, float]]) -> str:
        # 实现自定义决策逻辑
        pass
```

### 训练定制

```python
# 自定义奖励函数
class CustomReward(RewardFunction):
    def calculate_reward(self, extracted_memories: List[MemoryNode], ground_truth: List[str]) -> float:
        # 实现自定义奖励计算
        pass

# 自定义训练配置
config = {
    "task": "extraction",
    "model_name": "custom-model",
    "reward_function": CustomReward(),
    "training_params": {
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 2e-5
    }
}
```

## 🤝 贡献与社区

欢迎参与MemFactory框架的开发和改进！


## 🙏 致谢

感谢开源社区提供的优秀工具和库，以及所有为项目做出贡献的开发者。

