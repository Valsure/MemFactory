import sys
import os
import pdb
import json
# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), 'src'))

print("Python:", sys.version.split()[0])
print("Working directory:", os.getcwd())

# Imports / 依赖集中声明
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# 导入记忆工程模块
from src.common import (
    MemoryItem, ConversationMessage, ExtractionResult, SearchResult, Edge,
    MemoryType, MemoryStatus, UpdateAction, RelationType,
    LLMClient, EmbeddingClient, Neo4jClient, MilvusClient, MemoryStore,
    get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client, get_memory_store,
    generate_id, current_timestamp, format_conversation
)

from src.memory_extraction import MemoryExtractor, ExtractionConfig
from src.memory_search import MemorySearcher, SearchConfig
from src.memory_organization import MemoryOrganizer, OrganizationConfig
from src.memory_update import MemoryUpdater, UpdateConfig

print("所有模块导入成功！")

USER_ID = "demo_user"
SESSION_ID = "demo_session"
VERBOSE = True


JSON_PATH = "./MemBank_example.json"
LOAD_FROM_JSON = False  # 控制是否从 JSON 初始化

print(f"用户ID: {USER_ID}")
print(f"会话ID: {SESSION_ID}")
print(f"JSON路径: {JSON_PATH}")
print(f"从JSON加载: {LOAD_FROM_JSON}")

# 构造示例对话 - 第一轮：建立基础记忆
sample_conversation = [
    ConversationMessage(
        role="user",
        content="我是Tom，最近在负责一个AI项目，团队有5个人。",
        timestamp="2025-06-26T10:00:00"
    ),
    ConversationMessage(
        role="assistant",
        content="你好Tom！听起来是个很有意思的项目。项目进展如何？",
        timestamp="2025-06-26T10:01:00"
    ),
    ConversationMessage(
        role="user",
        content="进展还不错，昨天完成了数据预处理模块。不过我有点担心12月15日的截止日期。",
        timestamp="2025-06-26T10:02:00"
    ),
    ConversationMessage(
        role="assistant",
        content="时间确实比较紧。你们团队的分工是怎样的？",
        timestamp="2025-06-26T10:03:00"
    ),
    ConversationMessage(
        role="user",
        content="我负责整体架构和模型训练，小李负责数据，小王负责前端。对了，我平时喜欢喝美式咖啡，不加糖。",
        timestamp="2025-06-26T10:05:00"
    ),
]

print(f"准备了 {len(sample_conversation)} 条对话消息")

class MemoryPipelineRunner:
    """
    记忆工程完整串行Pipeline (Mock版)
    """
    
    def __init__(self, user_id: str = "default_user", verbose: bool = True):
        self.user_id = user_id
        self.verbose = verbose
        
        # 初始化统一存储管理器
        self.store = get_memory_store()
        
        # 强制检查 Mock 状态
        if not self.store.use_mock:
            print("警告: 当前不是 Mock 模式，example2 旨在 Mock 模式下运行。")
        else:
            print("确认: 运行在 Mock 模式。")
            
        # 尝试从 JSON 加载
        if LOAD_FROM_JSON and os.path.exists(JSON_PATH):
            try:
                print(f"正在从 {JSON_PATH} 加载记忆库...")
                with open(JSON_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.store.from_list(data)
                print(f"成功加载 {len(data)} 条记忆。")
            except Exception as e:
                print(f"加载 JSON 失败: {e}")
        
        # 初始化各模块
        self.extractor = MemoryExtractor(ExtractionConfig(
            strategy="simple",
            auto_save=False,
            auto_embed=False,
            verbose=verbose,
            user_id=user_id
        ))
        
        self.updater = MemoryUpdater(UpdateConfig(
            strategy="auto",
            auto_save=False,
            verbose=verbose
        ))
        
        # 移除 Organizer (图构建)
        
        self.searcher = MemorySearcher(SearchConfig(
            strategy="passive",
            top_k=5,
            similarity_threshold=0.3,
            verbose=verbose,
            user_id=user_id
        ))
        
        # LLM客户端用于最终回答
        self.llm = get_llm_client()
        
        print(f"[MemoryPipelineRunner] 初始化完成，用户: {user_id}")
    
    def _log(self, stage: str, message: str):
        """日志输出"""
        if self.verbose:
            print(f"[{stage}] {message}")
    
    def process_conversation(self, messages: List[ConversationMessage]) -> Dict[str, Any]:
        """
        处理对话的完整流程
        流程: 抽取 → 更新决策 → 存储 (无图构建)
        """
        result = {
            "extracted_memories": [],
            "update_decisions": [],
            "saved_memories": [],
            "errors": []
        }
        
        # ========== Step 1: 记忆抽取 ==========
        self._log("Step1-抽取", f"开始从 {len(messages)} 条消息中抽取记忆...")
        extraction_result = self.extractor.run(messages)
        
        if extraction_result.status != "SUCCESS":
            self._log("Step1-抽取", f"抽取失败: {extraction_result.summary}")
            result["errors"].append(f"抽取失败: {extraction_result.summary}")
            return result
        
        candidate_memories = extraction_result.memory_list
        self._log("Step1-抽取", f"抽取到 {len(candidate_memories)} 条候选记忆")
        result["extracted_memories"] = candidate_memories
        
        if not candidate_memories:
            self._log("Step1-抽取", "没有抽取到记忆，流程结束")
            return result
        
        # ========== Step 2: 更新决策（查询相关记忆） ==========
        print("\n" + "-"*40)
        self._log("Step2-更新决策", "开始对每条候选记忆进行更新决策...")
        print("-"*40 + "\n")
        
        memories_to_save = []
        
        for i, candidate in enumerate(candidate_memories, 1):
            print(f"\n>>> 处理候选记忆 {i}/{len(candidate_memories)}: [{candidate.key}] {candidate.value}")
            
            # 查询相关的已有记忆
            related_memories = self.store.find_related_memories(candidate, top_k=20)
            
            # --- 观察点：寻找相似记忆 ---
            if not related_memories:
                print(f"    (观察) 无相关记忆")
                
                decision = {
                    "candidate": candidate,
                    "action": "ADD",
                    "related": [],
                    "final_memory": candidate,
                    "reason": "无相关记忆，直接新增"
                }
            else:
                print(f"    (观察) 找到 {len(related_memories)} 条相关记忆:")
                for rm, score in related_memories:
                    # --- 观察点：判断记忆相似度 ---
                    print(f"        - 相似度: {score:.4f} | ID: {rm.id[:8]}... | 内容: {rm.value[:60]}...")
                
                # 使用更新模块决定操作
                update_result = self.updater.decide_action(
                    new_memory=candidate,
                    related_memories=[m for m, s in related_memories]
                )
                
                decision = {
                    "candidate": candidate,
                    "action": update_result["action"],
                    "related": related_memories,
                    "final_memory": update_result.get("final_memory", candidate),
                    "deprecated_ids": update_result.get("deprecated_ids", []),
                    "reason": update_result.get("reason", "")
                }
            
            # --- 观察点：决定使用哪种记忆操作 ---
            print(f"    (决策) 操作: [{decision['action']}]")
            print(f"    (决策) 原因: {decision.get('reason', '无详细原因')}")
            
            if decision["action"] == "MERGE" and decision.get("deprecated_ids"):
                 print(f"    (影响) 将废弃旧记忆ID: {decision['deprecated_ids']}")

            result["update_decisions"].append(decision)
            
            # 收集需要保存的记忆
            if decision["action"] in ["ADD", "MERGE", "OVERWRITE", "VERSION"]:
                memories_to_save.append(decision["final_memory"])
        
        print("\n" + "-"*40 + "\n")

        # ========== Step 3: 统一存储（Neo4j + Milvus同步） ==========
        self._log("Step3-存储", f"开始保存 {len(memories_to_save)} 条记忆到内存库...")
        
        for memory in memories_to_save:
            # 确保设置用户ID
            memory.user_id = self.user_id
            
            # 统一保存（自动生成embedding，同步写入Neo4j和Milvus）
            success = self.store.save(memory, generate_embedding=True)
            
            if success:
                self._log("Step3-存储", f"  ✓ 保存成功: {memory.id} - {memory.key}")
                result["saved_memories"].append(memory)
            else:
                self._log("Step3-存储", f"  ✗ 保存失败: {memory.key}")
                result["errors"].append(f"保存失败: {memory.key}")
        
        self._log("完成", "对话处理完成！")
        return result
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        根据用户问题检索记忆并回答
        """
        result = {
            "question": question,
            "retrieved_memories": [],
            "context": "",
            "answer": "",
            "status": "SUCCESS"
        }
        
        # ========== Step 5: 检索相关记忆 ==========
        self._log("Step5-检索", f"检索与问题相关的记忆: {question}")
        
        # 使用统一存储的检索功能
        search_results = self.store.search_similar(
            query=question,
            top_k=5,
            user_id=self.user_id
        )
        
        if not search_results:
            self._log("Step5-检索", "  → 未找到相关记忆")
            result["answer"] = "抱歉，我没有找到相关的记忆信息来回答这个问题。"
            return result
        
        self._log("Step5-检索", f"  → 找到 {len(search_results)} 条相关记忆:")
        
        for memory, score in search_results:
            self._log("Step5-检索", f"     - [{score:.3f}] {memory.key}: {memory.value[:50]}...")
            result["retrieved_memories"].append({
                "memory": memory,
                "score": score
            })
        
        # ========== 构建上下文 ==========
        context_parts = ["以下是与问题相关的记忆信息：\n"]
        for i, (memory, score) in enumerate(search_results, 1):
            context_parts.append(f"{i}. [{memory.key}] {memory.value}")
        
        context = "\n".join(context_parts)
        result["context"] = context
        
        # ========== LLM生成回答 ==========
        self._log("Step5-回答", "调用LLM生成回答...")
        
        system_prompt = "你是一个智能助手，根据提供的记忆信息准确回答用户问题。"
        user_prompt = f"""基于以下记忆信息回答用户的问题。请直接、准确地回答，如果记忆中有明确的信息就使用它。

{context}

用户问题: {question}

请用中文回答："""
        
        answer = self.llm.chat(system_prompt, user_prompt)
        result["answer"] = answer
        self._log("Step5-回答", f"  → 回答: {answer[:100] if answer else '(空)'}...")
        return result
    
    def save_bank(self):
        """保存记忆库到 JSON"""
        try:
            print(f"\n正在保存记忆库到 {JSON_PATH} ...")
            data = self.store.to_list()
            with open(JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"保存成功，共 {len(data)} 条记忆。")
        except Exception as e:
            print(f"保存失败: {e}")

    def run_full_pipeline(self, messages: List[ConversationMessage], 
                          questions: List[str]) -> Dict[str, Any]:
        """
        运行完整的端到端流程
        """
        print("\n" + "=" * 70)
        print("🚀 开始运行完整记忆工程Pipeline (Mock & No-Graph)")
        print("=" * 70)
        
        full_result = {
            "conversation_result": None,
            "question_answers": []
        }
        
        # Part 1: 处理对话
        print("\n📝 Part 1: 处理对话 (抽取 → 更新决策 → 存储)")
        print("-" * 70)
        
        conv_result = self.process_conversation(messages)
        full_result["conversation_result"] = conv_result
        
        # 保存记忆库
        self.save_bank()
        
        # Part 2: 回答问题
        print("\n\n❓ Part 2: 检索记忆并回答问题")
        print("-" * 70)
        
        for q in questions:
            print(f"\n问题: {q}")
            qa_result = self.answer_question(q)
            full_result["question_answers"].append(qa_result)
            print(f"回答: {qa_result['answer']}")
        
        print("\n" + "=" * 70)
        print("✅ Pipeline运行完成！")
        print("=" * 70)
        
        return full_result


if __name__ == "__main__":
    pipeline = MemoryPipelineRunner(user_id=USER_ID, verbose=VERBOSE)
    
    questions = [
        "Tom负责什么工作？",
        "项目的截止日期是什么时候？",
        "用户喜欢喝什么饮料？",
        "团队有多少人？"
    ]
    
    full_result = pipeline.run_full_pipeline(
        messages=sample_conversation,
        questions=questions
    )

    pipeline.save_bank()
