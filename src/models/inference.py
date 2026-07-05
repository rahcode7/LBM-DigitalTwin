import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Qwen Digital Twin")
    
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B", help="Base model ID")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the saved LoRA adapter")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions")
    
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max tokens to generate per answer")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    
    return parser.parse_args()

def load_data(file_path):
    print(f"Loading test data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def main():
    args = parse_args()
    
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Left padding is strictly required for batched generation

    # 2. Load Base Model in FP16 (V100 compatible)
    print(f"Loading Base Model ({args.model_id})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 3. Load and merge the LoRA adapters
    print(f"Loading LoRA Adapters from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval() # Set model to evaluation mode

    # 4. Load Data
    test_data = load_data(args.test_data_path)
    results = []

    print("Starting Inference...")
    for i in tqdm(range(0, len(test_data), args.batch_size)):
        batch = test_data[i:i + args.batch_size]
        
        # Build prompt messages for the batch
        batch_prompts = []
        for row in batch:
            messages = [
                {"role": "system", "content": row["system_context"]},
                {"role": "user", "content": row["user_question"]}
            ]
            
            # add_generation_prompt=True appends "<|im_start|>assistant\n"
            # This perfectly triggers the exact pattern your model was trained on.
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            batch_prompts.append(prompt)
            
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to("cuda")

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, # Greedy decoding for deterministic multiple-choice answers
                temperature=None, 
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens 
        input_lengths = inputs.input_ids.shape[1]
        generated_ids = outputs[:, input_lengths:]
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Store results
        for row, pred in zip(batch, decoded_responses):
            row_result = {
                "pid": row.get("pid"),
                "question_id": row.get("question_id"),
                "target_answer": row.get("target_answer"),
                "predicted_answer": pred.strip()
            }
            results.append(row_result)

    # 5. Save Results
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
            
    print(f"Inference complete! Saved {len(results)} predictions to {args.output_file}")

if __name__ == "__main__":
    main()