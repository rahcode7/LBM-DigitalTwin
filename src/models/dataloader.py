import os
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def format_and_tokenize_chatml(example, tokenizer: PreTrainedTokenizer, max_seq_length: int = 2048):
    """
    Formats text into ChatML, applies Middle Truncation to preserve demographics,
    and manually creates the -100 label mask 
    """
    # 1. Protect the System Prefix (The Static State)
    # This prevents the demographic traits from being blindly deleted by standard truncation.
    prefix = "You are a digital twin. Here is the user's historical survey data containing demographic and behavioral questions and corresponding answers:\n\n"
    raw_history = example["system_context"].replace(prefix, "")

    # 2. Tokenize the Target Answer
    answer_text = str(example["target_answer"])
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    # Add EOS if missing. 
    if len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id:
        answer_ids.append(tokenizer.eos_token_id)

    # 3. Calculate remaining budget for the Prompt
    max_prompt_len = max_seq_length - len(answer_ids)
    
    # 4. MIDDLE TRUNCATION: Safely truncate ONLY the history
    history_ids = tokenizer(raw_history, add_special_tokens=False)["input_ids"]
    
    # Estimate safety buffer for the prefix, user question, and ChatML tags (~200 tokens)
    # This guarantees the static state and the final question always fit.
    buffer_tokens = len(tokenizer(prefix + example["user_question"])["input_ids"]) + 30 
    max_history_len = max_prompt_len - buffer_tokens

    if len(history_ids) > max_history_len and max_history_len > 0:
        # Left-truncate the history (Drops Wave 1, keeps Wave 3)
        history_ids = history_ids[-max_history_len:] 

    # Rebuild the safe system context
    safe_history = tokenizer.decode(history_ids)
    safe_system_context = prefix + safe_history

    # 5. Format with ChatML exactly as you did
    prompt_messages = [
        {"role": "system", "content": safe_system_context},
        {"role": "user", "content": example["user_question"]},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,   # Leaves the assistant header ready
        enable_thinking=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # 6. Failsafe Truncation (Just in case tokenizer quirks push it 1-2 tokens over)
    # if len(prompt_ids) > max_prompt_len:
    #     prompt_ids = prompt_ids[-max_prompt_len:]

    # 7. Final Concatenation and Manual Masking
    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)
    
    # Masking the prompt with -100 so loss is only calculated on the answer
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def get_qwen_dataloaders(train_path: str, val_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int = 4096):
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Dataset splits not found. Check your file paths.")

    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    
    # Standard Trainer requires you to strip out raw text strings completely
    processed_dataset = dataset.map(
        lambda x: format_and_tokenize_chatml(x, tokenizer, max_seq_length),
        remove_columns=dataset["train"].column_names,
        num_proc=4,
        desc="Tokenizing data splits for standard Trainer"
    )
    
    return processed_dataset["train"], processed_dataset["validation"]