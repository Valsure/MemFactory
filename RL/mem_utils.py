
import json
import re
import sys
import os
from typing import List, Dict, Any, Optional
import numpy as np

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.common import LLMClient, EmbeddingClient, MemoryItem, generate_id
except ImportError:
    # Fallback if running from a different location
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from src.common import LLMClient, EmbeddingClient, MemoryItem, generate_id

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

UPDATE_MEMORY_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

There are specific guidelines to select which operation to perform:

1. **Add**: If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "User is a software engineer"
            }
        ]
    - Retrieved facts: ["Name is John"]
    - New Memory:
        {
            "memory" : [
                {
                    "id" : "0",
                    "text" : "User is a software engineer",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Name is John",
                    "event" : "ADD"
                }
            ]
        }

2. **Update**: If the retrieved facts contain information that is already present in the memory but the information is totally different, then you have to update it.
If the retrieved fact contains information that conveys the same thing as the elements present in the memory, then you have to keep the fact which has the most information.
Example (a) -- if the memory contains "User likes to play cricket" and the retrieved fact is "Loves to play cricket with friends", then update the memory with the retrieved facts.
Example (b) -- if the memory contains "Likes cheese pizza" and the retrieved fact is "Loves cheese pizza", then you do not need to update it because they convey the same information.
If the direction is to update the memory, then you have to update it.
Please keep in mind while updating you have to keep the same ID.
Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "I really like cheese pizza"
            },
            {
                "id" : "1",
                "text" : "User is a software engineer"
            },
            {
                "id" : "2",
                "text" : "User likes to play cricket"
            }
        ]
    - Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]
    - New Memory:
        {
        "memory" : [
                {
                    "id" : "0",
                    "text" : "Loves cheese and chicken pizza",
                    "event" : "UPDATE",
                    "old_memory" : "I really like cheese pizza"
                },
                {
                    "id" : "1",
                    "text" : "User is a software engineer",
                    "event" : "NONE"
                },
                {
                    "id" : "2",
                    "text" : "Loves to play cricket with friends",
                    "event" : "UPDATE",
                    "old_memory" : "User likes to play cricket"
                }
            ]
        }

3. **Delete**: If the retrieved facts contain information that contradicts the information present in the memory, then you have to delete it. Or if the direction is to delete the memory, then you have to delete it.
Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "Name is John"
            },
            {
                "id" : "1",
                "text" : "Loves cheese pizza"
            }
        ]
    - Retrieved facts: ["Dislikes cheese pizza"]
    - New Memory:
        {
        "memory" : [
                {
                    "id" : "0",
                    "text" : "Name is John",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Loves cheese pizza",
                    "event" : "DELETE"
                }
        ]
        }

4. **No Change**: If the retrieved facts contain information that is already present in the memory, then you do not need to make any changes.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "Name is John"
            },
            {
                "id" : "1",
                "text" : "Loves cheese pizza"
            }
        ]
    - Retrieved facts: ["Name is John"]
    - New Memory:
        {
        "memory" : [
                {
                    "id" : "0",
                    "text" : "Name is John",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Loves cheese pizza",
                    "event" : "NONE"
                }
            ]
        }

Retrieved facts: {facts}
Old Memory: {old_memory}
New Memory:"""

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

def parse_json_from_text(text: str) -> Optional[Dict]:
    """Extract and parse JSON from text"""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
        return json.loads(text)
    except Exception as e:
        # print(f"JSON Parse Error: {e}")
        return None

def construct_extraction_prompt(conversation: str) -> str:
    return EXTRACTION_PROMPT_EN.format(conversation=conversation)

def construct_update_prompt(facts: List[str], old_memory: List[Dict]) -> str:
    return UPDATE_MEMORY_PROMPT.format(
        facts=json.dumps(facts, ensure_ascii=False),
        old_memory=json.dumps(old_memory, ensure_ascii=False, indent=4)
    )

def memory_to_dict(memories: List[MemoryItem]) -> List[Dict]:
    return [
        {"id": m.id, "text": f"{m.key}: {m.value}"}
        for m in memories
    ]

# =============================================================================
# Evaluation Logic
# =============================================================================

class MemoryEvaluator:
    def __init__(self):
        self.llm = LLMClient()
        self.embedding = EmbeddingClient()
    
    def apply_update_plan(self, context_memory: List[MemoryItem], update_plan: Dict) -> List[MemoryItem]:
        """
        Execute update plan on a temporary memory list
        update_plan structure: {"memory": [{"id":..., "text":..., "event":...}, ...]}
        """
        if not update_plan or "memory" not in update_plan:
            return context_memory
            
        current_mem_map = {m.id: m for m in context_memory}
        updated_mem_map = current_mem_map.copy()
        
        for item in update_plan["memory"]:
            event = item.get("event", "NONE").upper()
            mid = item.get("id")
            text = item.get("text", "")
            
            # Simple parsing of text to key/value if possible, else put all in value
            key = "Updated Memory"
            value = text
            if ":" in text:
                parts = text.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()
            
            if event == "ADD":
                # For ADD, ID might be new or reused from prompt example logic
                # The prompt says "generate a new ID", but in simulation we just accept the ID provided or gen new
                new_mem = MemoryItem(
                    id=mid if mid not in updated_mem_map else generate_id(),
                    key=key,
                    value=value,
                    memory_type="UserMemory",
                    tags=[]
                )
                updated_mem_map[new_mem.id] = new_mem
                
            elif event == "UPDATE":
                if mid in updated_mem_map:
                    # Update existing
                    mem = updated_mem_map[mid]
                    mem.key = key
                    mem.value = value
                    # In a real system, we might update updated_at, etc.
                    
            elif event == "DELETE":
                if mid in updated_mem_map:
                    del updated_mem_map[mid]
                    
            # NONE does nothing
            
        return list(updated_mem_map.values())

    def retrieve(self, memories: List[MemoryItem], query: str, top_k: int = 3) -> List[MemoryItem]:
        """
        Simple retrieval from the in-memory list
        """
        if not memories:
            return []
            
        query_emb = self.embedding.embed(query)
        
        # Calculate scores
        scores = []
        for mem in memories:
            # Re-embed memory if needed (simulation)
            if mem.embedding is None:
                mem_text = f"{mem.key} {mem.value}"
                mem.embedding = self.embedding.embed(mem_text)
            
            # Cosine similarity
            score = self.embedding.similarity(query_emb, mem.embedding)
            scores.append((score, mem))
            
        # Sort
        scores.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scores[:top_k]]

    def evaluate(self, 
                 fact: str, 
                 query: str, 
                 answer: str, 
                 context_memory: List[MemoryItem], 
                 extraction_output: str, 
                 update_plan_output: str) -> float:
        """
        Full evaluation pipeline:
        1. Parse Extraction -> Extracted Facts
        2. Parse Update Plan -> Plan JSON
        3. Execute Plan -> New Memory State
        4. Retrieve(Query) -> Context
        5. QA(Context, Query) -> Predicted Answer
        6. Judge(Predicted, Standard) -> Reward
        """
        
        # 1. Parse Extraction
        # extraction_output is expected to be JSON from the extraction prompt
        ext_json = parse_json_from_text(extraction_output)
        extracted_facts = []
        if ext_json and "memory_list" in ext_json:
            for m in ext_json["memory_list"]:
                extracted_facts.append(f"{m.get('key', '')}: {m.get('value', '')}")
        
        if not extracted_facts:
            # If extraction failed, maybe use raw fact? 
            # Or penalize? For now let's just use the raw fact provided in input if extraction failed
            # But the update prompt depends on "Retrieved facts" which comes from extraction
            # If extraction yields nothing, the update might be NO-OP.
            extracted_facts = [fact] # Fallback to raw fact for robustness in early training
            
        # 2. Parse Update Plan
        update_json = parse_json_from_text(update_plan_output)
        
        # 3. Apply Update
        # Convert context_memory to list if it's not already
        # In simulation, context_memory should be a list of MemoryItem
        new_memory_state = self.apply_update_plan(context_memory, update_json)
        
        # 4. Retrieval
        retrieved_docs = self.retrieve(new_memory_state, query)
        context_str = "\n".join([f"- {m.key}: {m.value}" for m in retrieved_docs])
        
        # 5. QA
        qa_prompt = QA_PROMPT.format(context=context_str, question=query)
        # Using LLM to generate answer
        # Note: In training loop, this might be slow. 
        # But user requested this flow.
        pred_answer = self.llm.chat("You are a helpful assistant.", qa_prompt)
        
        # 6. Judge
        judge_prompt = JUDGE_PROMPT.format(question=query, answer=answer, prediction=pred_answer)
        judge_result = self.llm.chat("You are an impartial judge.", judge_prompt)
        
        # Parse True/False
        if "True" in judge_result:
            return 1.0
        elif "False" in judge_result:
            return 0.0
        else:
            # Fuzzy check
            return 0.5 # Ambiguous
