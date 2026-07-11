import pandas as pd 
import json 
from typing import List, Dict, Any
import random 
from pathlib import Path
import os 
from collections import defaultdict
from typing import List, Dict, Any
import pandas as pd
from transformers import PreTrainedTokenizer, AutoTokenizer

# ==========================================
# SCHEMA NORMALIZATION
# ==========================================
def parse_json_col(val) -> List[Dict[str, Any]]:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []
    return val if isinstance(val, list) else []

def unroll_survey_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Unrolls Qualtrics Blocks and Matrix questions into a flat list of individual Q&A dicts.
    Handles both the W1-3 flat structure and the W4 nested Block structure.
    """
    flat_qas = []
    
    for item in raw_data:
        # 1. Unpack "Block" grouping if it exists (Wave 4 style)
        if item.get("ElementType") == "Block":
            questions = item.get("Questions", [])
        else:
            questions = [item] # Already a flat question (Wave 1-3 style)
        #print(questions[0])
        for q in questions:
            qid = q.get("QuestionID", "")
            base_text = str(q.get("QuestionText", "")).strip()
            q_type = q.get("QuestionType", "")
            
            # 2. Unroll Matrix Questions into individual rows
            if q_type == "Matrix":
                rows = q.get("Rows", [])
                options = q.get("Columns", [])
                answers = q.get("Answers", {}).get("SelectedText", [])
                
                # Ensure the lengths match before zipping
                if isinstance(answers, list) and len(rows) == len(answers):
                    for idx, row_text in enumerate(rows):
                        ans = str(answers[idx]).strip()
                        if ans:
                            flat_qas.append({
                                "question_id": f"{qid}_{idx}", 
                                "parent_qid": qid,
                                "question_text": f"{base_text} {row_text}".strip(),
                                "question_type": q_type,
                                "options": options,
                                "answer": ans
                            })
            
            # 3. Handle Standard Multiple Choice Questions
            else:
                ans = ""
                # Check nested format
                answers_dict = q.get("Answers", {})
                if isinstance(answers_dict, dict) and "SelectedText" in answers_dict:
                    ans = str(answers_dict.get("SelectedText", "")).strip()
                # Check flat format
                else:
                    for flat_key in ["Answer", "SelectedOption", "answer"]:
                        if flat_key in q and q[flat_key]:
                            ans = str(q[flat_key]).strip()
                
                if ans:
                    flat_qas.append({
                        "question_id": qid,
                        "parent_qid": qid,
                        "question_text": base_text,
                        "question_type": q_type,
                        "options": q.get("Options", []),
                        "answer": ans
                    })
                    
    return flat_qas

# ==========================================
# DATASET GENERATION
# ==========================================
def save_jsonl(dataset: List[Dict[str, Any]], filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in dataset:
            f.write(json.dumps(row) + '\n')
    print(f"Saved {len(dataset)} rows to {filepath}")

def format_context_string(normalized_qas: List[Dict[str, Any]]) -> str:
    """
    Converts the unrolled, scrubbed Q&A list into a clean text block for the LLM.
    Now includes the options for every historical question inline to save tokens.
    """
    lines = []
    qcnts = 0 
    for qa in normalized_qas:
        q_text = str(qa.get('question_text', '')).strip()
        ans_text = str(qa.get('answer', '')).strip()
        options = qa.get('options', [])
        
        # Format historical options inline to save context window tokens
        if options:
            opt_str = "[" + ", ".join([str(o) for o in options]) + "]"
        else:
            opt_str = "[No options provided]"
            
        if q_text and ans_text:
            lines.append(f"Question: {q_text}\nOptions: {opt_str}\nAnswer: {ans_text}")
        qcnts+=1
    return "\n\n".join(lines),qcnts

def build_train_val_startified_split(
    pids: List[str], 
    df: pd.DataFrame, 
    tokenizer: PreTrainedTokenizer, # <-- Pass the tokenizer here
    max_samples_per_user: int = 50
) -> List[Dict[str, Any]]:
    """
    Builds the Train or Validation dataset using Waves 1-3.
    Applies strict Dynamic Leave-One-Out (LOO), Stratified Sampling, 
    and Token-Based Context Trimming to prevent GPU OOM.
    """
    dataset = []
    df_subset = df[df['pid'].isin(pids)]
    
    for _, row in df_subset.iterrows():
        pid = row['pid']
        w1_3_raw = parse_json_col(row.get('wave1_3_persona_json', []))
        w1_3_qas = unroll_survey_data(w1_3_raw)
        
        # 1. Filter demographics from targets (they act as the static state)
        eligible_targets = [
            qa for qa in w1_3_qas 
            if qa.get('parent_qid') not in DEMOGRAPHIC_QIDS 
            and qa.get('question_id') not in DEMOGRAPHIC_QIDS
        ]
        
        # 2. STRATIFIED SAMPLING BY QUESTION TYPE
        # Group questions by their type (e.g., 'MC', 'Matrix', 'TE')
        categories = defaultdict(list)
        for qa in eligible_targets:
            q_type = qa.get('question_type', 'UNKNOWN') 
            categories[q_type].append(qa)
            
        sampled_targets = []
        if max_samples_per_user and len(eligible_targets) > max_samples_per_user:
            # Distribute the sample count evenly across available categories
            samples_per_category = max_samples_per_user // len(categories)
            
            for q_type, q_list in categories.items():
                if len(q_list) > samples_per_category:
                    sampled_targets.extend(random.sample(q_list, samples_per_category))
                else:
                    sampled_targets.extend(q_list)
                    
            # If rounding left us short, top it off randomly
            remaining = max_samples_per_user - len(sampled_targets)
            if remaining > 0:
                leftovers = [qa for qa in eligible_targets if qa not in sampled_targets]
                sampled_targets.extend(random.sample(leftovers, min(remaining, len(leftovers))))
        else:
            sampled_targets = eligible_targets
            
        # 3. BUILD THE DATASET WITH DYNAMIC TOKEN TRIMMING
        for target_qa in sampled_targets:
            target_qid = target_qa['question_id']
            context_qas = [qa for qa in w1_3_qas if qa['question_id'] != target_qid]
            
            context_text, qcnts = format_context_string(context_qas) 
            
            options = target_qa.get('options', [])
            opt_str = "\n" + "\n".join([f"- {str(o)}" for o in options]) if options else "\n- No options provided"
            
            dataset.append({
                "pid": pid,
                "question_id": target_qid,
                "question_type": target_qa['question_type'],
                "system_context": f"{context_text}",
                "user_question": f"Based on the historical data, how would the user answer this question?\nQuestion: '{target_qa['question_text']}'\nOptions: {opt_str}",
                "target_answer": target_qa['answer'],
                "question_cnts_context": qcnts
            })
            
    return dataset

def build_test_split(pids: List[str], df: pd.DataFrame) -> List[Dict[str, Any]]:
    dataset = []
    df_subset = df[df['pid'].isin(pids)]
    
    total_processed = 0
    total_skipped = 0
    
    for _, row in df_subset.iterrows():
        pid = row['pid']
        
        # Parse and instantly unroll into standard schema
        w1_3_raw = parse_json_col(row.get('wave1_3_persona_json', []))
        w1_3_qas = unroll_survey_data(w1_3_raw)
        
        w4_raw = parse_json_col(row.get('wave4_Q_wave4_A', []))
        w4_qas = unroll_survey_data(w4_raw)
        
        context_text,qcnts = format_context_string(w1_3_qas)
        
        for target_qa in w4_qas:
            # Filter demographics (using parent_qid in case it was a matrix)
            if target_qa['parent_qid'] in DEMOGRAPHIC_QIDS or target_qa['question_id'] in DEMOGRAPHIC_QIDS:
                total_skipped += 1
                continue
                
            options = target_qa['options']
            opt_str = ", ".join([str(o) for o in options]) if options else "No options provided"
            
            dataset_row = {
                "pid": pid,
                "question_id": target_qa['question_id'],
                "question_type": target_qa['question_type'],
                "system_context": f"{context_text}",
                "user_question": f"Based on the historical data, how would the user answer this future question? \nQuestion: '{target_qa['question_text']}'\n Options: [{opt_str}]",
                "target_answer": target_qa['answer'],
                "question_cnts_context":qcnts
            }
            dataset.append(dataset_row)
            total_processed += 1
            
    print(f"[Debug] Wave 4 Stats -> Kept Targets: {total_processed} | Dropped Targets: {total_skipped}")
    return dataset

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right"


    DEMOGRAPHIC_QIDS = {"QID11","QID12","QID13","QID14","QID15","QID16","QID17","QID19","QID18","SAVR","QID20","QID21","QID22","QID24","QID23"}  # Total 14 Demographic Questions

    print("Loading dataset...")
    DATASET_PATH = "../datasets"
   
    df = pd.read_json(os.path.join(DATASET_PATH, "wave_splits/wave_splits.json"), lines=True)
    df = df.sample(10)

    all_pids = df['pid'].unique().tolist()
    random.seed(42)
    random.shuffle(all_pids)
    
    n_total = len(all_pids)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    
    train_pids = all_pids[:n_train]
    val_pids = all_pids[n_train:n_train + n_val]
    test_pids = all_pids[n_train + n_val:]
    print(f"Total PIDs: {n_total} | Train PIDs: {len(train_pids)}")
    print(f"Total PIDs: {n_total} | Val PIDs: {len(val_pids)}")
    print(f"Total PIDs: {n_total} | Test PIDs: {len(test_pids)}")

   # Generate splits
    print("\nBuilding Training Set (W1-3 with Dynamic LOO Masking)...")
    train_dataset = build_train_val_startified_split(train_pids, df, max_samples_per_user=20,tokenizer=tokenizer)
    
    print("Building Validation Set (W1-3 with Dynamic LOO Masking)...")
    val_dataset = build_train_val_startified_split(val_pids, df, max_samples_per_user=20,tokenizer=tokenizer)
    
    print("Building Test Set (W4 Targets | Full W1-3 Context)...")
    test_dataset = build_test_split(test_pids, df)
    #print(test_dataset[0])

    # Save splits
    datasplits_dir = os.path.join(DATASET_PATH, "datasplits")
    save_jsonl(train_dataset, filepath=os.path.join(datasplits_dir, "train_8P_Q.jsonl"))
    save_jsonl(val_dataset, filepath=os.path.join(datasplits_dir, "val_1P_Q.jsonl"))
    save_jsonl(test_dataset, filepath=os.path.join(datasplits_dir, "test_1P_Q.jsonl"))