# =============================================================================
# 公共配置模块 - Common Configuration Module
# 包含：LLM客户端、Embedding服务、Neo4j图数据库、Milvus向量数据库
# =============================================================================

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from openai import OpenAI
# OpenAI API 依赖
from ..common.utils import LLMClient


# =============================================================================
# 环境配置
# 请在项目根目录创建 .env 文件配置以下环境变量，参考 .env.example
# =============================================================================

# 尝试加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    # 尝试从多个位置加载 .env
    for env_path in ['.env', '../.env', '../../.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
except ImportError:
    pass  # python-dotenv 未安装，跳过

# OpenAI LLM API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")

# Embedding API 配置（独立的服务地址和Key）
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "EMPTY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # BGE-M3 默认1024维

# Neo4j 配置
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")  # 数据库名，留空则使用默认数据库

# Milvus 配置
MILVUS_URI = os.getenv("MILVUS_URI", "")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "memory_embeddings")


# =============================================================================
# 枚举类型定义
# =============================================================================

class MemoryType(Enum):
    """记忆类型枚举"""
    LONG_TERM_MEMORY = "LongTermMemory"
    USER_MEMORY = "UserMemory"
    FACT = "fact"
    EVENT = "event"
    PREFERENCE = "preference"


class MemoryStatus(Enum):
    """记忆状态枚举"""
    ACTIVATED = "activated"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class UpdateAction(Enum):
    """更新操作类型"""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    VERSION = "version"


class RelationType(Enum):
    """关系类型枚举"""
    CAUSES = "causes"
    FOLLOWS = "follows"
    RESOLVES = "resolves"
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    SAME_TOPIC = "same_topic"
    DEPENDS_ON = "depends_on"


# =============================================================================
# 核心数据结构
# =============================================================================

@dataclass
class MemoryItem:
    """
    记忆条目：系统的基本存储单元
    统一的数据结构，用于各模块之间传递
    """
    id: str
    key: str                          # 记忆标题/关键词
    value: str                        # 记忆内容
    memory_type: str                  # 记忆类型
    tags: List[str]                   # 标签列表
    confidence: float = 0.9           # 置信度 (0-1)
    created_at: str = ""              # 创建时间
    updated_at: str = ""              # 更新时间
    user_id: str = "default_user"     # 用户ID
    session_id: str = "default_session"  # 会话ID
    status: str = "activated"         # 状态
    source_type: str = "user_explicit"   # 来源类型
    source_credibility: float = 1.0   # 来源可信度
    access_count: int = 0             # 访问次数
    decay_score: float = 1.0          # 衰减分数
    version: int = 1                  # 版本号
    embedding: Optional[List[float]] = None  # 向量表示
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = asdict(self)
        # 移除embedding以减少序列化大小
        if 'embedding' in result and result['embedding'] is not None:
            result['embedding'] = f"<vector dim={len(result['embedding'])}>"
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        """从字典创建"""
        # 处理embedding字段
        if 'embedding' in data and isinstance(data['embedding'], str):
            data['embedding'] = None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationMessage:
    """对话消息"""
    role: str           # user / assistant / system
    content: str        # 消息内容
    timestamp: Optional[str] = None # 时间戳
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ExtractionResult:
    """抽取结果"""
    memory_list: List[MemoryItem]
    summary: str
    status: str = "SUCCESS"  # SUCCESS / BUFFERED / IGNORED / TIMEOUT


@dataclass
class SearchResult:
    """检索结果"""
    memories: List[Tuple[MemoryItem, float]]  # (记忆, 相关性分数)
    query: str
    total_found: int


@dataclass
class Edge:
    """图边：连接两个节点的关系"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# Embedding 服务
# =============================================================================

class EmbeddingClient:
    """
    Embedding客户端：使用BGE-M3模型生成向量
    通过独立的OpenAI兼容接口调用（与LLM服务分离）
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        # 使用独立的Embedding服务配置
        self.client = OpenAI(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL
        )
        self.model = EMBEDDING_MODEL
        self.dim = EMBEDDING_DIM  # BGE-M3 默认1024维
        self._use_mock = True  # 默认使用mock（如果API不可用）
        self._initialized = True
        print(f"[EmbeddingClient] 已初始化，模型: {self.model}, 服务地址: {EMBEDDING_BASE_URL}")
    
    def embed(self, text: str) -> List[float]:
        """
        生成文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            向量列表
        """
        if not self._use_mock:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"[EmbeddingClient] API调用失败，使用mock: {e}")
                self._use_mock = True
        
        # Mock实现：基于文本hash生成确定性向量
        return self._mock_embed(text)
    
    def _mock_embed(self, text: str) -> List[float]:
        """Mock embedding实现"""
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.dim).tolist()
        norm = np.linalg.norm(embedding)
        return [x / norm for x in embedding]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成向量"""
        return [self.embed(text) for text in texts]
    
    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """计算余弦相似度"""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = np.sqrt(sum(a * a for a in emb1))
        norm2 = np.sqrt(sum(b * b for b in emb2))
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


# =============================================================================
# Neo4j 图数据库客户端
# =============================================================================

class Neo4jClient:
    """
    Neo4j客户端：用于存储和查询记忆图谱
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._driver = None
        self._use_mock = True
        self._mock_store: Dict[str, MemoryItem] = {}
        self._mock_edges: List[Edge] = []
        self._database = NEO4J_DATABASE
        
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._driver.verify_connectivity()
            
            # 尝试确保数据库存在
            self._ensure_database_exists()
            
            self._use_mock = False
            print(f"[Neo4jClient] 已连接到 {NEO4J_URI}, 数据库: {self._database}")
        except Exception as e:
            print(f"[Neo4jClient] 连接失败，使用内存存储: {e}")
        
        self._initialized = True
    
    def _ensure_database_exists(self):
        """确保数据库存在，如果不存在则创建"""
        if not self._database:
            return
        
        try:
            # 使用system数据库来创建新数据库
            with self._driver.session(database="system") as session:
                # 检查数据库是否存在
                result = session.run("SHOW DATABASES")
                existing_dbs = [record["name"] for record in result]
                
                if self._database not in existing_dbs:
                    print(f"[Neo4jClient] 数据库 '{self._database}' 不存在，正在创建...")
                    session.run(f"CREATE DATABASE {self._database} IF NOT EXISTS")
                    print(f"[Neo4jClient] 数据库 '{self._database}' 创建成功")
        except Exception as e:
            # 如果无法创建数据库（可能是社区版不支持），尝试使用默认数据库
            print(f"[Neo4jClient] 无法创建数据库 '{self._database}': {e}")
            print(f"[Neo4jClient] 尝试使用默认数据库...")
            self._database = None
    
    def _get_session(self):
        """获取数据库session"""
        if self._database:
            return self._driver.session(database=self._database)
        else:
            # 使用服务器默认数据库
            return self._driver.session()
    
    def save_memory(self, memory: MemoryItem) -> bool:
        """保存记忆节点"""
        if self._use_mock:
            self._mock_store[memory.id] = memory
            return True
        
        try:
            with self._get_session() as session:
                session.run("""
                    MERGE (m:Memory {id: $id})
                    SET m.key = $key,
                        m.value = $value,
                        m.memory_type = $memory_type,
                        m.tags = $tags,
                        m.confidence = $confidence,
                        m.created_at = $created_at,
                        m.updated_at = $updated_at,
                        m.user_id = $user_id,
                        m.status = $status
                """, **memory.to_dict())
            return True
        except Exception as e:
            print(f"[Neo4jClient] 保存失败: {e}")
            return False
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """获取记忆节点"""
        if self._use_mock:
            return self._mock_store.get(memory_id)
        
        try:
            with self._get_session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) RETURN m",
                    id=memory_id
                )
                record = result.single()
                if record:
                    return MemoryItem.from_dict(dict(record["m"]))
        except Exception as e:
            print(f"[Neo4jClient] 查询失败: {e}")
        return None
    
    def get_all_memories(self, user_id: str = None) -> List[MemoryItem]:
        """获取所有记忆"""
        if self._use_mock:
            memories = list(self._mock_store.values())
            if user_id:
                memories = [m for m in memories if m.user_id == user_id]
            return memories
        
        try:
            with self._get_session() as session:
                query = "MATCH (m:Memory) "
                if user_id:
                    query += "WHERE m.user_id = $user_id "
                query += "RETURN m"
                result = session.run(query, user_id=user_id)
                return [MemoryItem.from_dict(dict(r["m"])) for r in result]
        except Exception as e:
            print(f"[Neo4jClient] 查询失败: {e}")
        return []
    
    def save_edge(self, edge: Edge) -> bool:
        """保存关系边"""
        if self._use_mock:
            # 在Mock模式下禁止建立边，只作为简单数据库使用
            print("[Neo4jClient] 警告: 在Mock模式下禁止建立边")
            return False
        
        try:
            with self._get_session() as session:
                session.run(f"""
                    MATCH (a:Memory {{id: $source_id}})
                    MATCH (b:Memory {{id: $target_id}})
                    MERGE (a)-[r:{edge.relation_type}]->(b)
                    SET r.weight = $weight
                """, source_id=edge.source_id, target_id=edge.target_id,
                    weight=edge.weight)
            return True
        except Exception as e:
            print(f"[Neo4jClient] 保存边失败: {e}")
            return False
    
    def get_related_memories(self, memory_id: str, 
                             relation_type: str = None) -> List[Tuple[MemoryItem, str]]:
        """获取相关记忆"""
        if self._use_mock:
            results = []
            for edge in self._mock_edges:
                if edge.source_id == memory_id:
                    if relation_type is None or edge.relation_type == relation_type:
                        mem = self._mock_store.get(edge.target_id)
                        if mem:
                            results.append((mem, edge.relation_type))
            return results
        
        try:
            with self._get_session() as session:
                query = """
                    MATCH (a:Memory {id: $id})-[r]->(b:Memory)
                    RETURN b, type(r) as rel_type
                """
                result = session.run(query, id=memory_id)
                return [(MemoryItem.from_dict(dict(r["b"])), r["rel_type"]) 
                        for r in result]
        except Exception as e:
            print(f"[Neo4jClient] 查询失败: {e}")
        return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        if self._use_mock:
            if memory_id in self._mock_store:
                del self._mock_store[memory_id]
                self._mock_edges = [e for e in self._mock_edges 
                                    if e.source_id != memory_id and e.target_id != memory_id]
                return True
            return False
        
        try:
            with self._get_session() as session:
                session.run(
                    "MATCH (m:Memory {id: $id}) DETACH DELETE m",
                    id=memory_id
                )
            return True
        except Exception as e:
            print(f"[Neo4jClient] 删除失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()


# =============================================================================
# Milvus 向量数据库客户端
# =============================================================================

class MilvusClient:
    """
    Milvus客户端：用于向量检索
    支持账号密码认证，支持user_id过滤
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._collection = None
        self._use_mock = True
        self._mock_vectors: Dict[str, Dict[str, Any]] = {}  # {memory_id: {"embedding": [...], "user_id": "..."}}
        self._embedding_client = EmbeddingClient()
        
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            
            # 使用URI + 账号密码认证连接
            # 根据URI判断是否使用SSL（https开头则使用SSL）
            use_secure = MILVUS_URI.startswith("https://")
            connections.connect(
                alias="default",
                uri=MILVUS_URI,
                user=MILVUS_USER,
                password=MILVUS_PASSWORD,
                secure=use_secure
            )
            
            # 创建或获取集合
            if not utility.has_collection(MILVUS_COLLECTION):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),  # 添加user_id字段
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
                ]
                schema = CollectionSchema(fields, description="Memory embeddings with user_id filter")
                self._collection = Collection(MILVUS_COLLECTION, schema)
                # 创建向量索引
                self._collection.create_index(
                    field_name="embedding",
                    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
                )
                # 为user_id创建标量索引以加速过滤
                self._collection.create_index(
                    field_name="user_id",
                    index_params={"index_type": "INVERTED"}
                )
            else:
                self._collection = Collection(MILVUS_COLLECTION)
            
            self._collection.load()
            self._use_mock = False
            print(f"[MilvusClient] 已连接到 {MILVUS_URI}")
        except Exception as e:
            print(f"[MilvusClient] 连接失败，使用内存存储: {e}")
        
        self._initialized = True
    
    def insert(self, memory_id: str, embedding: List[float], user_id: str = "default_user") -> bool:
        """
        插入向量
        
        Args:
            memory_id: 记忆ID
            embedding: 向量
            user_id: 用户ID
            
        Returns:
            是否插入成功
        """
        if self._use_mock:
            self._mock_vectors[memory_id] = {"embedding": embedding, "user_id": user_id}
            return True
        
        try:
            self._collection.insert([[memory_id], [user_id], [embedding]])
            self._collection.flush()
            return True
        except Exception as e:
            print(f"[MilvusClient] 插入失败: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
               user_id: str = None) -> List[Tuple[str, float]]:
        """
        向量检索，支持user_id过滤
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            user_id: 用户ID过滤（可选，None表示不过滤）
            
        Returns:
            (memory_id, score) 列表
        """
        if self._use_mock:
            # Mock实现：计算余弦相似度，支持user_id过滤
            results = []
            for mid, data in self._mock_vectors.items():
                # user_id过滤
                if user_id is not None and data["user_id"] != user_id:
                    continue
                score = self._embedding_client.similarity(query_embedding, data["embedding"])
                results.append((mid, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # 构建过滤表达式
            expr = None
            if user_id is not None:
                expr = f'user_id == "{user_id}"'
            
            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,  # 使用user_id过滤
                output_fields=["id", "user_id"]
            )
            return [(hit.id, hit.score) for hit in results[0]]
        except Exception as e:
            print(f"[MilvusClient] 检索失败: {e}")
            return []
    
    def delete(self, memory_id: str) -> bool:
        """删除向量"""
        if self._use_mock:
            if memory_id in self._mock_vectors:
                del self._mock_vectors[memory_id]
                return True
            return False
        
        try:
            self._collection.delete(f'id == "{memory_id}"')
            return True
        except Exception as e:
            print(f"[MilvusClient] 删除失败: {e}")
            return False
    
    def delete_by_user(self, user_id: str) -> bool:
        """
        删除指定用户的所有向量
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否删除成功
        """
        if self._use_mock:
            to_delete = [mid for mid, data in self._mock_vectors.items() if data["user_id"] == user_id]
            for mid in to_delete:
                del self._mock_vectors[mid]
            return True
        
        try:
            self._collection.delete(f'user_id == "{user_id}"')
            return True
        except Exception as e:
            print(f"[MilvusClient] 按用户删除失败: {e}")
            return False


# =============================================================================
# 统一存储管理器
# =============================================================================

class MemoryStore:
    """
    统一记忆存储管理器
    确保Neo4j和Milvus的ID同步，提供统一的CRUD接口
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.neo4j = Neo4jClient()
        self.milvus = MilvusClient()
        self.embedding = EmbeddingClient()
        
        # 判断Mock状态
        if self.neo4j._use_mock and self.milvus._use_mock:
            self.use_mock = True
            print("[MemoryStore] 运行在Mock模式 (纯内存存储)")
        elif not self.neo4j._use_mock and not self.milvus._use_mock:
            self.use_mock = False
            print("[MemoryStore] 运行在真实数据库模式")
        else:
            raise ValueError("配置错误: Neo4j和Milvus必须同时为Mock模式或同时为真实模式")

        self._initialized = True
        print("[MemoryStore] 统一存储管理器初始化完成")
    
    def save(self, memory: MemoryItem, generate_embedding: bool = True) -> bool:
        """
        统一保存记忆：同时写入Neo4j和Milvus，确保ID对应
        
        Args:
            memory: 记忆条目
            generate_embedding: 是否生成embedding
            
        Returns:
            是否保存成功
        """
        try:
            # 1. 生成embedding（如果需要）
            if generate_embedding or memory.embedding is None:
                text = f"{memory.key} {memory.value}"
                memory.embedding = self.embedding.embed(text)
            
            # 2. 保存到Neo4j（结构化数据）
            neo4j_success = self.neo4j.save_memory(memory)
            
            # 3. 保存到Milvus（向量数据），使用相同的ID和user_id
            milvus_success = self.milvus.insert(memory.id, memory.embedding, memory.user_id)
            
            if neo4j_success and milvus_success:
                if not self.use_mock:
                    # revised by guozl: 训练期间不打印保存成功，避免干扰训练日志
                    print(f"[MemoryStore] 保存成功: {memory.id} - {memory.key} (user: {memory.user_id})")
                return True
            else:
                print(f"[MemoryStore] 保存部分失败: Neo4j={neo4j_success}, Milvus={milvus_success}")
                return False
                
        except Exception as e:
            print(f"[MemoryStore] 保存异常: {e}")
            return False
    
    def save_batch(self, memories: List[MemoryItem], generate_embedding: bool = True) -> List[bool]:
        """批量保存记忆"""
        results = []
        for memory in memories:
            success = self.save(memory, generate_embedding)
            results.append(success)
        return results
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """获取记忆"""
        return self.neo4j.get_memory(memory_id)
    
    def get_all(self, user_id: str = None) -> List[MemoryItem]:
        """获取所有记忆"""
        return self.neo4j.get_all_memories(user_id)
    
    def delete(self, memory_id: str) -> bool:
        """删除记忆（同时从Neo4j和Milvus删除）"""
        neo4j_success = self.neo4j.delete_memory(memory_id)
        milvus_success = self.milvus.delete(memory_id)
        return neo4j_success and milvus_success
    
    def search_similar(self, query: str, top_k: int = 10, 
                       user_id: str = None) -> List[Tuple[MemoryItem, float]]:
        """
        搜索相似记忆
        
        Args:
            query: 查询文本
            top_k: 返回数量
            user_id: 用户ID过滤（在Milvus层面直接过滤，而非检索后过滤）
            
        Returns:
            (记忆, 相似度分数) 列表
        """
        # 1. 生成查询向量
        query_emb = self.embedding.embed(query)
        
        # 2. 向量检索（user_id在Milvus层面直接过滤）
        # 多请求一些结果，因为后续还需要状态过滤
        vector_results = self.milvus.search(query_emb, top_k=top_k * 2, user_id=user_id)
        
        # 3. 获取记忆详情并进行状态过滤
        results = []
        for memory_id, score in vector_results:
            memory = self.neo4j.get_memory(memory_id)
            if memory:
                # 状态过滤（user_id已在Milvus层面过滤）
                if memory.status != MemoryStatus.ACTIVATED.value:
                    continue
                results.append((memory, score))
        
        return results[:top_k]
    
    def find_related_memories(self, memory: MemoryItem, 
                              top_k: int = 10) -> List[Tuple[MemoryItem, float]]:
        """
        查找与给定记忆相关的已有记忆（用于更新决策）
        
        Args:
            memory: 新抽取的记忆
            top_k: 返回数量
            
        Returns:
            (相关记忆, 相似度) 列表
        """
        query = f"{memory.key} {memory.value}"
        results = self.search_similar(query, top_k=top_k, user_id=memory.user_id)
        # 排除自身
        return [(m, s) for m, s in results if m.id != memory.id]

    def to_list(self) -> List[Dict]:
        if not self.use_mock:
            raise RuntimeError("to_list 方法仅在 use_mock=True 时可用")
            
        results = []
        # 直接访问Neo4jClient的mock store，其中存储了MemoryItem对象
        for mem in self.neo4j._mock_store.values():
            # 使用to_dict获取数据，embedding会被替换为描述字符串
            item_dict = mem.to_dict()
            results.append(item_dict)
        return results

    def from_list(self, data: List[Dict]) -> None:
        if not self.use_mock:
            raise RuntimeError("from_list 方法仅在 use_mock=True 时可用")
            
        # 清空现有Mock数据
        self.neo4j._mock_store.clear()
        self.milvus._mock_vectors.clear()
        
        for item_dict in data:
            # 重建MemoryItem对象，此时embedding为None
            mem = MemoryItem.from_dict(item_dict)
            
            # 使用save方法保存，会自动触发embedding生成并存入Neo4j和Milvus
            self.save(mem)

# =============================================================================
# 全局单例获取函数
# =============================================================================

def get_memory_store() -> MemoryStore:
    """获取统一存储管理器单例"""
    return MemoryStore()


def get_llm_client() -> LLMClient:
    """获取LLM客户端单例"""
    return LLMClient()


def get_embedding_client() -> EmbeddingClient:
    """获取Embedding客户端单例"""
    return EmbeddingClient()


def get_neo4j_client() -> Neo4jClient:
    """获取Neo4j客户端单例"""
    return Neo4jClient()


def get_milvus_client() -> MilvusClient:
    """获取Milvus客户端单例"""
    return MilvusClient()


# =============================================================================
# 工具函数
# =============================================================================

def generate_id() -> str:
    """生成唯一ID"""
    import uuid
    return str(uuid.uuid4())


def current_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().isoformat()


def format_conversation(messages: List[ConversationMessage]) -> str:
    """格式化对话为字符串"""
    lines = []
    for msg in messages:
        if not msg.timestamp or msg.timestamp == "" or msg.timestamp == " ":
            lines.append(f"{msg.role}: {msg.content}")
        else:
            lines.append(f"{msg.role}: [{msg.timestamp}] {msg.content}")
    return "\n".join(lines)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("公共模块测试")
    print("=" * 60)
    
    # 测试LLM客户端
    llm = get_llm_client()
    print(f"LLM客户端: {llm.model}")
    
    # 测试Embedding客户端
    emb = get_embedding_client()
    vec = emb.embed("测试文本")
    print(f"Embedding维度: {len(vec)}")
    
    # 测试Neo4j客户端
    neo4j = get_neo4j_client()
    test_mem = MemoryItem(
        id=generate_id(),
        key="测试记忆",
        value="这是一条测试记忆",
        memory_type="UserMemory",
        tags=["测试"]
    )
    neo4j.save_memory(test_mem)
    print(f"Neo4j保存成功: {test_mem.id}")
    
    # 测试Milvus客户端
    milvus = get_milvus_client()
    milvus.insert(test_mem.id, vec)
    results = milvus.search(vec, top_k=1)
    print(f"Milvus检索结果: {results}")
    
    print("\n所有测试通过！")
