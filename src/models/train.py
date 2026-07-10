import os
import argparse
import torch
import wandb
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,Trainer
from peft import LoraConfig, get_peft_model
#from trl import DataCollatorForCompletionOnlyLM
from dataloader import get_qwen_dataloaders
from transformers import DataCollatorForSeq2Seq,DataCollatorWithPadding

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Argparse paths
    # Model & Data Paths
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str,required=True)
    
    # WandB Configuration
    parser.add_argument("--wandb_project", type=str, default="LBM-digital-twin")
    parser.add_argument("--wandb_run_name", type=str, default="qwen3-0.6b-sft")
    
    # Training Hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train and eval batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    
    # Lora
    parser.add_argument("--lora_r", type=int,default=8)
    parser.add_argument("--lora_alpha", type=int,default=16)
   
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()

    OUTPUT_DIR = args.output_dir
    RUN_NAME = args.wandb_run_name
    WANDB_PROJECT = args.wandb_project
    
    #DO_SAMPLE = args.do_sample
    MAX_INPUT_TOKENS = args.max_seq_length
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum

    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha

    
    # log to WandB
    if accelerator.is_main_process:
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right"

    # Load Training and Validation Datasets
    train_dataset, val_dataset = get_qwen_dataloaders(
        train_path=args.train_data_path, 
        val_path=args.val_data_path, 
        tokenizer=tokenizer,
        max_seq_length=MAX_INPUT_TOKENS
    )

    # 3. Initialize Base Model
    if accelerator.is_main_process:
        print(f"Loading Base Model: {args.model_id}...")
        
    # ACCELERATE FIX: Map the model directly to the specific GPU handling this process
    device_index = accelerator.local_process_index
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={"": device_index},
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    # 4. Configure LoRA (PEFT)
    if accelerator.is_main_process:
        print("Applying LoRA Adapters...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

   

    # 5. Padding collator
    collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding="longest",
    label_pad_token_id=-100,
    return_tensors="pt"
)


    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="steps",
        eval_steps=5,
        logging_steps=1,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        report_to="wandb" if accelerator.is_main_process else "none",                  
        run_name=RUN_NAME
    )

    # 7. Execute Trainer
    if accelerator.is_main_process:
        print("Initializing Trainer")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator = collator
    )

    if accelerator.is_main_process:
        print(f"Starting Training...")
        
    trainer.train()
    
    # Save final model
    if accelerator.is_main_process:
        trainer.save_model(os.path.join(OUTPUT_DIR, "final_sft_adapter"))
        print("Training complete!")
        wandb.finish()


