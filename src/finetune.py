import argparse
import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,)
from dataset import TextBlockDataset
def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on a custom text corpus")
    parser.add_argument("--data", required=True, help="Path to a plain .txt training corpus")
    parser.add_argument("--model", default="gpt2",
                         help="Base model to fine-tune: gpt2 (124M, default), gpt2-medium (355M), etc.")
    parser.add_argument("--output-dir", default="checkpoints/gpt2-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128,
                         help="Sequence length per training example (tokens)")
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Note: fine-tuning on CPU will be slow. A small corpus + a few epochs is recommended for a first run.")
    print(f"Loading base model/tokenizer: {args.model}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.to(device)
    print(f"Preparing dataset from {args.data}")
    train_dataset = TextBlockDataset(tokenizer, args.data, block_size=args.block_size)
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],  # disable wandb/tensorboard auto-logging integrations
        use_cpu=(device == "cpu"),)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,)
    print("Starting fine-tuning...")
    trainer.train()
    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. Generate text from this fine-tuned model with:")
    print(f'  python src/generate.py --model {args.output_dir} --prompt "..."')
if __name__ == "__main__":
    main()
