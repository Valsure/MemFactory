# MemFactory - 记忆处理与强化学习框架

## 项目简介

MemFactory是一个模块化的记忆处理框架，集成了完整的记忆生命周期管理功能，同时内置强化学习训练支持。该框架提供了记忆抽取、更新决策、存储管理、组织构建、检索回答等核心模块，并支持通过强化学习优化记忆处理能力。

## 🏗️ 框架架构

### 核心模块组件

![logo](https://github.com/Valsure/MemFactory/blob/main/overall.png)


## 🔧 核心功能特性

### 1. 统一的强化学习训练与评估框架

本项目提供了一套标准化、高内聚的训练与评估流水线（基于 `MemTrainer`）。用户只需指定目标智能体（`Agent`）与交互环境（`Env`），即可一键启动完整的训练流程：

* **环境层 (Envs)**：负责统一定义任务数据集的数据结构，并提供环境交互反馈（即强化学习中的奖励函数 Reward 机制）。
* **执行层 (Agents)**：在指定的 Env 中进行 Rollout（采样），与环境持续交互并产生轨迹（Trajectories）。
* **优化层 (Trainer)**：接收 Agent 产生的轨迹，内置 **GRPO (Group Relative Policy Optimization)** 等策略优化算法，驱动模型能力的迭代进化。

### 2. 积木式的模块化 Agent 架构

本框架采用**“以模块组装 Agent”**的范式。
一个完整的 Agent 是由多种单一职责的核心模块（`Modules`）按需组合搭建而成的。模块类型包括：

* **Extractors (抽取器)**：负责从对话流或观测数据中提取高价值的记忆片段（如 `Naive Extractor`）。
* **Updaters (更新器)**：负责决策如何将新提取的特征与历史记忆进行融合或覆写。
* **Retrievers (检索器)**：负责根据当前任务上下文，从记忆状态中精准召回相关信息（如 `Naive Retriever`, `LRM-Retriever`）
* **Agents (智能体)**：同时具有多个模块功能的复合组件。



### 3. 全链路记忆处理与策略演进

组装完成的 Agent 能够完成记忆处理的全生命周期。

* **记忆处理管线**：`环境输入 → Extractor (记忆抽取) → Updater (状态更新) → Retriever (上下文检索) → 环境输出`。
* **端到端优化**：在 GRPO 训练框架的支持下，Agent 不仅能执行基础的记忆读写，更能通过环境给定的稀疏或稠密奖励信号，联合优化其“抽取、更新、检索”的内部决策策略。

### 4. 面向二次创新的高可定制性

从模块到智能体、环境，本框架为前沿研究和二次创新提供了极大的自由度：

* **同接口多实现**：每种底层模块均允许存在多种技术路线的实现（例如从最简单的 Rule-based 模块到复杂的 Learnable 模块），支持热插拔替换。
* **全层级扩展**：从底层的**基础模块 (Modules)** 的逻辑替换，到中层的**智能体 (Agents)** 的编排创新，再到上层的**任务环境 (Envs)** 与奖励规则设计，均提供清晰的扩展接口，研发人员可以聚焦于核心算法本身而无需重写脚手架。

## 🚀 快速开始

### 1. 安装和环境配置

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

### 2. 启动强化学习训练
我们在 examples/ 目录下提供了强化学习示例。目前支持：

（1）仿造工作 Memory-R1 ，对智能体的记忆抽取和更新策略进行优化。

（2）仿造工作 MemoryAgent，对智能体的长上下文处理能力进行优化。

（3）仿造工作 RMM ，对智能体的记忆检索策略进行优化。

### 3. 二次创新
见“🛠️ 开发指南”

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



## 🛠️ 开发指南

### 框架扩展

```python
# 自定义模块
@MODULE_REGISTRY.register("naive_extractor")
class NaiveExtractor(BaseModule):
    def __init__(self, tokenizer, device="cuda", **kwargs):
#...

@MODULE_REGISTRY.register("your_extractor")
class YourExtractor(BaseModule):
    def __init__(self, tokenizer, device="cuda", **kwargs):


```

### 训练定制

```python
# 自定义奖励函数
@ENV_REGISTRY.register("your_new_env")
class YourNewEnv(MemoryBankEnv):
    def compute_reward(self, predictions: Dict[str, List[str]],     ground_truths: Dict[str, Any], num_generations: int, **kwargs) -> Dict[str, torch.Tensor]:
        # 实现自定义奖励计算
        pass


# 自定义训练配置
# 在训练时自定义环境，Agent，和各种参数。
```

## 🤝 贡献与社区

欢迎参与MemFactory框架的开发和改进！


## 🙏 致谢

感谢开源社区提供的优秀工具和库，以及所有为项目做出贡献的开发者。
