
import json
import re
import sys
import os
from typing import List, Dict, Any, Optional
import numpy as np

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.common import LLMClient, EmbeddingClient, MemoryItem, generate_id, get_memory_store, MemoryStore
    from src.common import format_conversation, ConversationMessage
except ImportError:
    # Fallback if running from a different location
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from src.common import LLMClient, EmbeddingClient, MemoryItem, generate_id, get_memory_store, MemoryStore
    from src.common import format_conversation, ConversationMessage

# =============================================================================
# Prompts
# =============================================================================

EXTRACTION_PROMPT_EN = """You are a memory extraction expert.
Your task is to extract memories from the user's perspective based on the conversation between the user and the assistant. This means identifying information the user might remember—including the user's own experiences, thoughts, plans, or statements and actions made by others (such as the assistant) that affect the user or are acknowledged by the user.

Please perform the following operations:
1. Identify information reflecting the user's experiences, beliefs, concerns, decisions, plans, or responses—including meaningful information from the assistant acknowledged or responded to by the user.

2. Clearly parse all references to time, people, and events:
   - If possible, convert relative time expressions (e.g., "yesterday", "next Friday") to absolute dates using message timestamps.
   - Clearly distinguish between event time and message time.
   - If specific locations are mentioned, include them.
   - Resolve all pronouns, aliases, and vague references to full names or clear identities.

3. Always write in the third person perspective, using "User" to refer to the user, rather than the first person.

4. Do not omit any information the user might remember.
   - Include all key experiences, thoughts, emotional reactions, and plans.
   - Prioritize completeness and fidelity over brevity.

Return a valid JSON object with the following structure:

{{
  "memory_list": [
    {{
      "key": "<string, unique and concise memory title>",
      "memory_type": "<string, 'LongTermMemory' or 'UserMemory'>",
      "value": "<detailed, independent, and unambiguous memory statement>",
      "tags": ["<list of relevant topic keywords>"]
    }}
  ],
  "summary": "<paragraph naturally summarizing the above memories from the user's perspective, 120-200 words>"
}}

Conversation:
{conversation}

Your output:"""

UPDATE_MEMORY_PROMPT = """You are a smart memory manager.
You have two lists of memories:
1. **Existing Memories** (from the database).
2. **New Candidate Memories** (extracted from the latest conversation).

Your goal is to decide how to update the memory database.

**Operations Allowed:**

For **Existing Memories**:
- `NONE`: Keep as is.
- `DEL`: Delete this memory (e.g., if it is contradicted by new info, or merged into a new memory).

For **New Candidate Memories**:
- `ADD`: Add this memory to the database.
- `NONE`: Ignore this memory (e.g., if it's redundant or already covered by existing memories).
- `UPDATE`: Modify this memory before adding (e.g., to merge information from an old memory).

**Merging Strategy:**
If a New Candidate (ID: Y) contains updated information for an Existing Memory (ID: X):
1. Mark Existing Memory X as `DEL`.
2. Mark New Candidate Y as `UPDATE` and provide the merged content.

**Output Format:**
Return a JSON object with a list of operations.
You MUST include an operation for **EVERY** memory item (both Existing and Candidate) in the input lists. Do not skip any IDs.

Format:
```json
{{
  "operations": [
    {{ "id": <id>, "op": "NONE" }},
    {{ "id": <id>, "op": "DEL" }},
    {{ "id": <id>, "op": "ADD" }},
    {{ "id": <id>, "op": "UPDATE", "key": "...", "value": "..." }}
  ]
}}
```

**Examples:**

Example 1: Add new info
Existing:
[{{"id": 1, "key": "User Info", "value": "Name is John"}}]
Candidates:
[{{"id": 2, "key": "User Info", "value": "Lives in NY"}}]
Output:
```json
{{
  "operations": [
    {{ "id": 1, "op": "NONE" }},
    {{ "id": 2, "op": "ADD" }}
  ]
}}
```

Example 2: Update/Merge
Existing:
[{{"id": 1, "key": "Pizza", "value": "Likes cheese pizza"}}]
Candidates:
[{{"id": 2, "key": "Pizza", "value": "Loves pepperoni too"}}]
Output:
```json
{{
  "operations": [
    {{ "id": 1, "op": "DEL" }},
    {{ "id": 2, "op": "UPDATE", "key": "Pizza Preference", "value": "Likes cheese pizza and loves pepperoni" }}
  ]
}}
```

Example 3: Redundant info (No Change)
Existing:
[{{"id": 1, "key": "Hobby", "value": "Likes reading"}}]
Candidates:
[{{"id": 2, "key": "Hobby", "value": "Enjoys reading books"}}]
Output:
```json
{{
  "operations": [
    {{ "id": 1, "op": "NONE" }},
    {{ "id": 2, "op": "NONE" }}
  ]
}}
```

**Task:**

Existing Memories:
{context_memory}

New Candidate Memories:
{candidate_memory}

Output:"""

QA_PROMPT = """Based on the following memory information, answer the user's question. Please answer directly and accurately. If the memory contains explicit information, use it.

Context Memories:
{context}

User Question: {question}
Answer:"""

JUDGE_PROMPT = """Please judge whether the predicted answer is correct based on the standard answer.

Question: {question}
Standard Answer: {answer}
Predicted Answer: {prediction}

Is the predicted answer consistent with the standard answer? Please output only "True" or "False".
"""

# =============================================================================
# Helper Functions
# =============================================================================

def parse_json_from_text(response: str) -> Optional[Dict]:
        """解析JSON响应"""
        try:
            # 尝试提取JSON块
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # 清理空白字符
            response = response.strip()
            # 处理思维链
            if response.startswith("<think>"):
                response = response.split("</think>")[-1]
                response = response.strip()
            # 尝试提取JSON对象（处理可能存在的<think>标签或其他前缀）
            if not response.startswith("{"):
                print("extract 结果不是 { 开头无法解析", response[:100])

            return json.loads(response)
        except json.JSONDecodeError as e:
            # print(f"extract 结果 JSON 解析失败: {e}")
            return {}

def prepare_memory_lists(context_memory: List[Dict], extraction_output: str):
    """
    Prepares numbered lists for prompt and mapping for execution.
    Returns:
        context_list_fmt: List[Dict] with 'id', 'key', 'value'
        candidate_list_fmt: List[Dict] with 'id', 'key', 'value'
        id_map: Dict[int, Any] - maps temp ID to (type, original_obj)
            type 'context': original_obj is MemoryItem
            type 'candidate': original_obj is Dict (from extraction)
    """
    context_memory = [ MemoryItem.from_dict(mem) for mem in context_memory ]

    id_counter = 1
    id_map = {}
    
    # 1. Process Context Memory
    context_list_fmt = []
    for mem in context_memory:
        temp_id = id_counter
        id_counter += 1
        id_map[temp_id] = ("context", mem)
        context_list_fmt.append({
            "id": temp_id,
            "key": mem.key,
            "value": mem.value
        })
        
    # 2. Process Extraction Output
    candidate_list_fmt = []
    ext_json = parse_json_from_text(extraction_output)
    if ext_json and "memory_list" in ext_json and isinstance(ext_json["memory_list"], list):
        for item in ext_json["memory_list"]:
            if not isinstance(item, dict): continue
            temp_id = id_counter
            id_counter += 1
            id_map[temp_id] = ("candidate", item)
            candidate_list_fmt.append({
                "id": temp_id,
                "key": item.get("key", "Unknown"),
                "value": item.get("value", "")
            })
    
    return context_list_fmt, candidate_list_fmt, id_map

def construct_extraction_prompt(conversation) -> str:
    conversation_msg_list = []
    for msg in conversation:
        msg_fmt = ConversationMessage(
            role=msg["role"],
            content=msg["content"],
            timestamp="",
        )
        msg_fmt.timestamp = ""
        conversation_msg_list.append(msg_fmt)
    conversation_str = format_conversation(conversation_msg_list)
    return EXTRACTION_PROMPT_EN.format(conversation=conversation_str)

def construct_update_prompt(context_memory: List[Dict], extraction_output: str) -> str:
    ctx_fmt, cand_fmt, _ = prepare_memory_lists(context_memory, extraction_output)
    
    return UPDATE_MEMORY_PROMPT.format(
        context_memory=json.dumps(ctx_fmt, ensure_ascii=False, indent=2),
        candidate_memory=json.dumps(cand_fmt, ensure_ascii=False, indent=2)
    )


# =============================================================================
# Evaluation Logic
# =============================================================================

class MemoryEvaluator:
    def __init__(self):
        self.llm = LLMClient()
        self.embedding = EmbeddingClient()
        self.store = get_memory_store()
        
    def reset_memory(self, memory: List):
        if self.store.use_mock:
            # Convert MemoryItem objects to dicts for from_list
            self.store.from_list(memory)
        else:
            print("Warning: Running evaluation on real database. Skipping memory reset to avoid data loss.")

    def apply_update_plan(self, context_memory: List[Dict], update_plan: Dict, extraction_output: str) -> None:
        """
        Execute update plan on the memory store.
        """
        if not update_plan or "operations" not in update_plan:
            return
            
        # Reconstruct the mapping to interpret sequential IDs
        _, _, id_map = prepare_memory_lists(context_memory, extraction_output)
        
        for op in update_plan["operations"]:
            temp_id = op.get("id")
            action = op.get("op", "NONE").upper()
            
            if temp_id not in id_map:
                continue
                
            origin_type, origin_obj = id_map[temp_id]
            
            if origin_type == "context":
                if action == "DEL":
                    # Delete existing memory
                    self.store.delete(origin_obj.id)
            
            elif origin_type == "candidate":
                if action in ["ADD", "UPDATE"]:
                    # Create new memory item
                    # Use provided key/value in op, or fallback to origin_obj (extracted item)
                    key = op.get("key", origin_obj.get("key"))
                    value = op.get("value", origin_obj.get("value"))
                    
                    new_mem = MemoryItem(
                        id=generate_id(),
                        key=key,
                        value=value,
                        memory_type=origin_obj.get("memory_type", "UserMemory"),
                        tags=origin_obj.get("tags", [])
                    )
                    self.store.save(new_mem)

    def retrieve(self, query: str, top_k: int = 3) -> List[MemoryItem]:
        """
        Retrieve relevant memories from the store.
        """
        # search_similar returns list of (MemoryItem, score)
        results = self.store.search_similar(query, top_k=top_k)
        return [m for m, s in results]

    def evaluate(self, 
                 memory: List[Dict],
                 fact: List[Dict], 
                 query: str, 
                 answer: str, 
                 context_memory: List[Dict], 
                 extraction_output: str, 
                 update_plan_output: str) -> float:
        
        # 0. Format Check & Parsing
        ext_json = parse_json_from_text(extraction_output)
        upd_json = parse_json_from_text(update_plan_output)
        
        is_ext_valid = isinstance(ext_json, dict) and "memory_list" in ext_json
        is_upd_valid = isinstance(upd_json, dict) and "operations" in upd_json
        
        if not is_ext_valid and not is_upd_valid:
            return -0.2
        if not is_ext_valid or not is_upd_valid:
            return -0.15
        
        try:
            # 1. Reset Memory Store
            self.reset_memory(memory)
            
            # 2. Apply Update (Using already parsed json to avoid double parsing issues, though apply_update_plan currently takes json object or relies on helper)
            # Refactoring apply_update_plan to accept parsed objects or ensuring safe usage
            # For now, we will pass the strings as before, but we know they are valid JSON structure-wise
            # Wait, apply_update_plan calls prepare_memory_lists which calls parse_json_from_text again.
            # Since we validated above, it should be fine.
            
            self.apply_update_plan(context_memory, upd_json, extraction_output)
            
            # 5. Retrieval
            retrieved_docs = self.retrieve(query, top_k=15)
            context_str = "\n".join([f"- {m.key}: {m.value}" for m in retrieved_docs])
            
            # 6. QA
            qa_prompt = QA_PROMPT.format(context=context_str, question=query)
            pred_answer = self.llm.chat("You are a helpful assistant.", qa_prompt)
            # 处理思考过程：
            if "<think>" in pred_answer:
                if "</think>" in pred_answer:
                    pred_answer = pred_answer.split("</think>")[-1].strip()
                else:# last 100 chars
                    pred_answer = pred_answer[-100:].strip()

            # 7. Judge
            judge_prompt = JUDGE_PROMPT.format(question=query, answer=answer, prediction=pred_answer)
            judge_result = self.llm.chat("You are an impartial judge.", judge_prompt)
            if "<think>" in judge_result:
                if "</think>" in judge_result:
                    judge_result = judge_result.split("</think>")[-1].strip()
                else:# last 100 chars
                    judge_result = judge_result[-100:].strip()

            # Parse True/False
            if "True" in judge_result:
                return 1.0
            elif "False" in judge_result:
                return 0.0
            else: 
                return 0.0 # Ambiguous
        except Exception as e:
            # Catch all execution errors (DB error, logic error, etc.)
            print(f"Evaluation Error: {e}")
            return 0.0

