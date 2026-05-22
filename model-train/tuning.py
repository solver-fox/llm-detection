import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import argparse
import os
from pathlib import Path

# SN32 specific constants
SN32_MAX_LENGTH = 1024
MODEL_NAME = "microsoft/deberta-v3-large"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=SN32_MAX_LENGTH)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for memory efficiency")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    return parser.parse_args()

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize texts with truncation and padding"""
    return tokenizer(
        examples["data"],
        truncation=True,
        padding=False,  # Will pad later with DataCollator
        max_length=max_length,
        return_tensors=None
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=-1)
    preds = np.argmax(predictions, axis=1)
    
    # SN32 rewards use these metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    auc = roc_auc_score(labels, probs[:, 1])
    
    # False Positive Rate (critical for SN32)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fp_score = 1 - (fp / len(labels)) if len(labels) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc": auc,
        "fp_score": fp_score,
    }

def setup_lora_model(model):
    """Apply LoRA to the model for memory-efficient fine-tuning"""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=32,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model
    except ImportError:
        print("PEFT not installed. Install with: pip install peft")
        raise

def main():
    args = parse_args()
    
    print("=" * 50)
    print("SN32 DeBERTa Fine-Tuning")
    print(f"Model: {args.model_name}")
    print(f"Max Length: {args.max_length}")
    print(f"Using LoRA: {args.use_lora}")
    print("=" * 50)
    
    # 1. Load dataset
    print("\n📂 Loading dataset from HuggingFace...")
    dataset = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset")
    
    # Convert label column name from 'generated' to 'label'
    dataset = dataset.rename_column("generated", "label")
    
    print(f"Dataset size: {len(dataset['train'])} examples")
    print(f"Class distribution: {dataset['train'].to_pandas()['label'].value_counts().to_dict()}")
    
    # 2. Split train/validation (95/5 since no validation split)
    print("\n✂️ Splitting into train/validation (95/5)...")
    train_test_split = dataset["train"].train_test_split(test_size=0.05, seed=42, stratify_by_column="label")
    dataset = DatasetDict({
        "train": train_test_split["train"],
        "validation": train_test_split["test"]
    })
    
    # 3. Load tokenizer
    print(f"\n🔧 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 4. Tokenize dataset
    print(f"\n🔄 Tokenizing with max_length={args.max_length}...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    # 5. Load base model
    print(f"\n🤖 Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # 6. Apply LoRA if requested
    if args.use_lora:
        print(f"\n💾 Applying LoRA (rank={args.lora_r})...")
        model = setup_lora_model(model)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 7. Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        report_to="wandb",  # Optional: integrates with SN32's wandb logging
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # 8. Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # 9. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # 10. Train
    print("\n🏋️ Starting training...")
    trainer.train()
    
    # 11. Save final model
    print(f"\n💾 Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 12. Convert to SN32 format (.pth for task weights)
    print("\n🔄 Converting to SN32 format...")
    
    if args.use_lora:
        # Merge LoRA weights with base model
        from peft import PeftModel
        merged_model = model.merge_and_unload()
        torch.save(merged_model.state_dict(), f"{args.output_dir}/sn32_deberta_weights.pth")
    else:
        # Save just the task-specific weights (smaller file)
        torch.save(model.state_dict(), f"{args.output_dir}/sn32_deberta_weights.pth")
    
    print(f"✅ Done! Task weights saved to: {args.output_dir}/sn32_deberta_weights.pth")
    print("\nTo use with SN32 miner:")
    print(f"  cp {args.output_dir}/sn32_deberta_weights.pth models/deberta-large-ls03-ctx1024.pth")
    
    # 13. Final evaluation
    print("\n📊 Final evaluation on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # 14. Save metrics for reference
    import json
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    main()