import torch
from torch.utils.data import Dataset
class TextBlockDataset(Dataset):
    """Tokenizes an entire text file and splits it into non-overlapping
    blocks of `block_size` tokens, for causal language modeling."""
    def __init__(self, tokenizer, file_path, block_size=128):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        # Tokenize the entire corpus at once
        tokenized = tokenizer(text, return_tensors="pt")["input_ids"][0]
        # Split into non-overlapping blocks of block_size tokens.
        # Any leftover tokens that don't fill a full block are dropped.
        num_blocks = len(tokenized) // block_size
        tokenized = tokenized[: num_blocks * block_size]
        self.examples = tokenized.view(num_blocks, block_size)
        print(f"Loaded {file_path}: {len(tokenized)} tokens -> {num_blocks} blocks of {block_size} tokens")
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        # For causal LM, labels = input_ids (the model learns to predict
        # each next token from the ones before it; shifting happens
        # internally in GPT2LMHeadModel's loss computation).
        return {"input_ids": input_ids, "labels": input_ids.clone()}
