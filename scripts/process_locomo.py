import sys
import os
import json
import re
import copy
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.common import (
    MemoryItem, ConversationMessage, 
    get_memory_store, get_llm_client, 
    MemoryStore
)
from src.memory_extraction import MemoryExtractor, ExtractionConfig
from src.memory_update import MemoryUpdater, UpdateConfig


class LoCoMoPipeline:
    def __init__(self, data_path: str, output_path: str, verbose: bool = True):
        self.data_path = data_path
        self.output_path = output_path
        self.verbose = verbose
        
        # Initialize Memory Components
        self.store = get_memory_store()
        
        # Force Mock Mode check
        if not self.store.use_mock:
            print("WARNING: Not running in Mock mode! This might overwrite your DB.")
            assert False
            # For safety, you might want to exit or require confirmation
            # sys.exit(1) 
        
        self.extractor = MemoryExtractor(ExtractionConfig(
            strategy="simple",
            auto_save=False, # We handle saving manually
            verbose=False,
            user_id="locomo_user"
        ))
        
        self.updater = MemoryUpdater(UpdateConfig(
            strategy="auto",
            auto_save=False,
            verbose=False
        ))
        
        self.results = []

    def log(self, msg):
        if self.verbose:
            print(f"[LoCoMoPipeline] {msg}")

    def parse_dia_id(self, dia_id: str) -> Tuple[int, int]:
        """Parses 'D1:3' into (1, 3). Returns (0,0) if invalid."""
        match = re.match(r"D(\d+):(\d+)", dia_id)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0

    def load_data(self) -> List[Dict]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_session_content(self, sample: Dict, session_idx: int) -> List[Dict]:
        key = f"session_{session_idx}"
        if 'conversation' in sample and key in sample['conversation']:
            return sample['conversation'][key]
        return []

    def get_session_time(self, sample: Dict, session_idx: int) -> str:
        """Extracts and formats session time (e.g. '2023-05-08')."""
        key = f"session_{session_idx}_date_time"
        if 'conversation' in sample and key in sample['conversation']:
            time_str = sample['conversation'][key]
            # Format: "1:56 pm on 8 May, 2023"
            try:
                dt = datetime.strptime(time_str, "%I:%M %p on %d %B, %Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                return time_str
        return ''
        
    def reset_memory_bank(self):
        """Resets the memory store for a new window."""
        if self.store.use_mock:
            self.store.neo4j._mock_store.clear()
            self.store.neo4j._mock_edges = []
            self.store.milvus._mock_vectors.clear()
        else:
            self.log("WARNING: Cannot reset real DB in this script.")

    def process_sample(self, sample: Dict):
        sample_id = sample.get('sample_id', 'unknown')
        self.log(f"Processing Sample: {sample_id}")
        
        # 1. QA Preprocessing
        qas = sample.get('qa', [])
        valid_qas = []
        
        for qa in qas:
            evidence = qa.get('evidence', [])
            if not evidence:
                continue
                
            parsed_ev = [self.parse_dia_id(e) for e in evidence]
            # Filter invalid IDs
            parsed_ev = [e for e in parsed_ev if e[0] > 0]
            
            if not parsed_ev:
                continue
                
            sessions = [e[0] for e in parsed_ev]
            min_s, max_s = min(sessions), max(sessions)
            
            # Rule: If span > 3 sessions, ignore.
            if (max_s - min_s) > 2:
                continue
            
            # Find trigger (last evidence)
            parsed_ev.sort()
            trigger = parsed_ev[-1] # (session, turn)
            
            # Store processed info
            qa['trigger'] = trigger
            qa['processed'] = False # Flag to ensure we only process once
            valid_qas.append(qa)
            
        # Sort QAs by trigger
        valid_qas.sort(key=lambda x: x['trigger'])
        
        self.log(f"  Found {len(valid_qas)} valid QAs")
        
        # 2. Sliding Window Session Loop
        max_session = 0
        if 'conversation' in sample:
            for k in sample['conversation']:
                if k.startswith('session_'):
                    try:
                        idx = int(k.split('_')[1])
                        max_session = max(max_session, idx)
                    except:
                        pass
        
        for start_s in range(1, max_session + 1):
            window_sessions = [start_s, start_s+1, start_s+2]
            
            self.log(f"  Window: {window_sessions}")
            
            # Initialize Empty Memory Bank
            self.reset_memory_bank()
            
            # Buffer for "f" (sliding window of dialogue)
            dialogue_buffer = [] # List of ConversationMessage
            dia_id_buffer = [] # List of dia_id
            BUFFER_SIZE = 4 # Size of 'f'
            
            # Iterate through sessions in this window
            for s_idx in window_sessions:
                turns = self.get_session_content(sample, s_idx)
                session_time = self.get_session_time(sample, s_idx)
                if not turns:
                    continue
                
                for turn in turns:
                    speaker = turn.get('speaker', 'user')
                    text = turn.get('text', '')
                    dia_id = turn.get('dia_id', '')
                    
                    msg = ConversationMessage(
                        role=speaker, 
                        # 这里的 role 应该怎么改？ 是否需要把 speker 添加到 text 里面？ 
                        # 就这样就行，详情见 common.py 中的 format_conversation 函数
                        content=text,
                        timestamp=turn.get('timestamp', session_time),
                    )
                    
                    # Update dialogue buffer (f)
                    dialogue_buffer.append(msg)
                    dia_id_buffer.append(dia_id)
                    
                    if len(dialogue_buffer) > BUFFER_SIZE:
                        dialogue_buffer.pop(0)
                        dia_id_buffer.pop(0)
                    
                    # Check Triggers
                    current_parsed_id = self.parse_dia_id(dia_id)
                    
                    # Check if this turn triggers any QA
                    for qa in valid_qas:
                        if qa.get('processed'):
                            continue
                            
                        trigger = qa['trigger']
                        if trigger == current_parsed_id:
                            # TRIGGER HIT!
                            
                            # 1. Check if all evidence is within current window (or history of current window)
                            evidence_sessions = [self.parse_dia_id(e)[0] for e in qa['evidence']]
                            min_ev_s = min(evidence_sessions)
                            
                            # Ensure the evidence fits in the window context (>= start_s)
                            if min_ev_s >= start_s:
                                self.log(f"    Trigger Hit: {dia_id} -> QA: {qa['question'][:30]}...")
                                
                                # Construct Data (M, f, q, a)
                                memory_state = self.store.to_list()
                                memory_state = copy.deepcopy(memory_state)
                                
                                f_data = [
                                    {"role": m.role, "content": m.content, "dia_id": dia_id_buffer[i]} 
                                    for i, m in enumerate(dialogue_buffer)
                                ]
                                
                                record = {
                                    "M": memory_state,
                                    "f": f_data,
                                    "q": qa.get('question', ''),
                                    "a": qa.get('answer', ''),                                    
                                    "sample_id": sample_id,
                                    "trigger_id": dia_id,
                                    "evidence": qa['evidence'],
                                }
                                self.results.append(record)
                                
                                qa['processed'] = True
                    
                    # Constantly Update Memory
                    extraction_res = self.extractor.run(dialogue_buffer)
                    if extraction_res.memory_list:
                        candidate_memories = extraction_res.memory_list

                        for candidate in candidate_memories:
                            # 1. Query related existing memories
                            related_memories = self.store.find_related_memories(candidate, top_k=10)
                            
                            if not related_memories:
                                candidate.user_id = "locomo_user"
                                self.store.save(candidate, generate_embedding=True)
                            else:
                                # 2. Decide update action
                                update_result = self.updater.decide_action(
                                    new_memory=candidate,
                                    related_memories=[m for m, s in related_memories]
                                )
                                # return: {"action", "final_memory", "deprecated_ids"}
                                action, final_memory, deprecated_ids = update_result["action"], update_result.get("final_memory", candidate), update_result.get("deprecated_ids", [])
                                if action in ["ADD", "MERGE", "OVERWRITE", "VERSION"]:
                                    final_memory.user_id = "locomo_user"
                                    self.store.save(final_memory, generate_embedding=True)
                                if deprecated_ids:
                                    for old_id in deprecated_ids:
                                        self.store.delete(old_id)
                                    
                            

                    

    def run(self):
        data = self.load_data()
        for sample in data:
            self.process_sample(sample)
        
        self.log(f"Saving {len(self.results)} records to {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Memory-CookBook/datas/locomo10.json")
    parser.add_argument("--output", default="Memory-CookBook/scripts/processed_locomo.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples to process (0 for all)")
    parser.add_argument("--dry-run", action="store_true", help="Use mock LLM for testing")
    args = parser.parse_args()
    
    if args.dry_run:
        print("Running in DRY RUN mode (Mock LLM)")
        from src.common import LLMClient
        LLMClient.chat = mock_llm_chat

    pipeline = LoCoMoPipeline(args.data, args.output)
    
    # Apply limit
    if args.limit > 0:
        original_load = pipeline.load_data
        pipeline.load_data = lambda: original_load()[:args.limit]

    pipeline.run()
