import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
def load_model(model_name_or_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from '{model_name_or_path}' on {device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device
def generate(model, tokenizer, device, prompt, strategy="top-p", max_length=100,
             num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.92,
             num_beams=5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,)
    if strategy == "greedy":
        gen_kwargs.update(do_sample=False)
    elif strategy == "beam":
        gen_kwargs.update(do_sample=False, num_beams=num_beams, early_stopping=True)
    elif strategy == "top-k":
        gen_kwargs.update(do_sample=True, top_k=top_k, temperature=temperature)
    elif strategy == "top-p":
        gen_kwargs.update(do_sample=True, top_p=top_p, top_k=0, temperature=temperature)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: greedy, beam, top-k, top-p")
    with torch.no_grad():
        output_sequences = model.generate(input_ids, **gen_kwargs)
    texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output_sequences]
    return texts
def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2")
    parser.add_argument("--prompt", required=True, help="Text prompt to continue")
    parser.add_argument("--model", default="gpt2",
                         help="Model name (e.g. 'gpt2') or path to a fine-tuned checkpoint")
    parser.add_argument("--strategy", choices=["greedy", "beam", "top-k", "top-p"], default="top-p")
    parser.add_argument("--max-length", type=int, default=100, help="Max total tokens (prompt + generated)")
    parser.add_argument("--num-sequences", type=int, default=1, help="How many outputs to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="Higher = more random/creative, lower = more focused (used with top-k/top-p)")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None, help="Set for reproducible output")
    args = parser.parse_args()
    model, tokenizer, device = load_model(args.model)
    texts = generate(
        model, tokenizer, device, args.prompt,
        strategy=args.strategy,
        max_length=args.max_length,
        num_return_sequences=args.num_sequences,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        seed=args.seed,)
    print(f"\n--- Strategy: {args.strategy} ---")
    for i, text in enumerate(texts):
        print(f"\n[{i + 1}] {text}")
if __name__ == "__main__":
    main()
