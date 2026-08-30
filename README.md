# 🤖 Text Generation with GPT-2

> Generate coherent, contextually relevant text from a prompt using **GPT-2**, and fine-tune it on your own text corpus to adapt its style, vocabulary, and structure.

---

## 📌 Overview

This project explores **text generation using GPT-2**, a Transformer-based language model.

It supports two main workflows:

* ✨ Generate text using the pretrained GPT-2 model.
* 🎯 Fine-tune GPT-2 on a custom text corpus and generate text using the fine-tuned model.

The project also demonstrates different **decoding strategies**, allowing you to control the balance between deterministic generation, coherence, and creativity.

---

# 🧠 How GPT-2 Text Generation Works

GPT-2 is a **Transformer-based language model** trained to predict the next token based on the tokens that came before it.

Text generation works by repeatedly predicting and appending the next token:

```text
Prompt
  ↓
Predict Next Token
  ↓
Append Token
  ↓
Predict Next Token
  ↓
Repeat
  ↓
Generated Text
```

For example:

```text
"The old lighthouse keeper"
              ↓
        GPT-2 predicts
              ↓
"The old lighthouse keeper ..."
```

The process continues until the desired number of tokens has been generated.

---

# 🎯 Fine-Tuning GPT-2

Fine-tuning takes an already pretrained GPT-2 model and continues training it on a smaller, domain-specific text corpus.

Instead of learning language from scratch, GPT-2 adapts its existing knowledge toward the patterns found in the new dataset.

```text
Pretrained GPT-2
       ↓
Custom Text Corpus
       ↓
Fine-Tuning
       ↓
Fine-Tuned GPT-2
       ↓
Domain-Specific Text Generation
```

Fine-tuning can help the model adapt to:

* 📝 Writing style
* 📚 Vocabulary
* 🧩 Text structure
* 🎯 Domain-specific patterns

The project supports fine-tuning on any plain `.txt` file, including your own writing, text datasets, or other text for which you have the appropriate rights to use.

---

# 🎛️ Decoding Strategies

After GPT-2 produces a probability distribution over possible next tokens, a **decoding strategy** determines which token is selected.

Different strategies provide different trade-offs between coherence, determinism, and creativity.

| Strategy | How It Selects the Next Token                                                    | Typical Result                              |
| -------- | -------------------------------------------------------------------------------- | ------------------------------------------- |
| `greedy` | Always selects the most likely token                                             | Deterministic but can become repetitive     |
| `beam`   | Tracks multiple likely sequences and keeps the best overall sequence             | Higher-quality deterministic output         |
| `top-k`  | Samples from the `k` most likely tokens                                          | More varied while filtering unlikely tokens |
| `top-p`  | Samples from the smallest group of tokens whose combined probability exceeds `p` | Adaptive and often natural-looking variety  |

---

## 🔹 Greedy Decoding

Greedy decoding always selects the token with the highest probability.

```text
Most Likely Token
       ↓
   Select It
       ↓
Next Token
       ↓
   Repeat
```

### Characteristics

* Deterministic
* Simple
* Fast
* Can become repetitive

---

## 🔹 Beam Search

Beam search keeps several candidate sequences instead of selecting only one token at each step.

```text
                 ┌── Candidate 1
Prompt ──────────┼── Candidate 2
                 └── Candidate 3
                         ↓
                  Best Sequence
```

### Characteristics

* Deterministic
* Considers multiple candidate sequences
* Can produce higher-quality results
* May still repeat on longer generations

---

## 🔹 Top-k Sampling

Top-k sampling restricts the candidate tokens to the `k` most likely options and randomly samples from them.

```text
All Possible Tokens
        ↓
Top-k Candidates
        ↓
Random Sampling
        ↓
Selected Token
```

This provides more variation while avoiding extremely unlikely tokens.

---

## 🔹 Top-p Sampling

Top-p, also called **nucleus sampling**, dynamically selects the smallest group of tokens whose combined probability exceeds a specified value `p`.

```text
All Tokens
    ↓
Probability Distribution
    ↓
Smallest Candidate Set
with cumulative probability ≥ p
    ↓
Random Sampling
```

Unlike top-k, the number of candidate tokens can change at every generation step.

---

# 🏗️ Project Workflow

The complete project workflow is:

```text
                    GPT-2
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
   Pretrained Model       Custom Text Corpus
          │                     │
          │                     ↓
          │                Fine-Tuning
          │                     │
          │                     ↓
          │              Fine-Tuned Model
          │                     │
          └──────────┬──────────┘
                     ↓
                  Prompt
                     ↓
             Decoding Strategy
                     ↓
             Generated Text
```

---

# 📁 Project Structure

```text
gpt2-text-generation/
│
├── src/
│   ├── dataset.py       # Tokenizes and chunks text for fine-tuning
│   ├── finetune.py      # Fine-tunes GPT-2 on a custom corpus
│   └── generate.py      # Generates text using base or fine-tuned GPT-2
│
├── sample_texts/
│   └── sample_corpus.txt
│
├── checkpoints/
│   └── gpt2-finetuned/
│
├── outputs/
│
├── requirements.txt
└── README.md
```

### 📄 File Overview

| File / Directory   | Purpose                                     |
| ------------------ | ------------------------------------------- |
| `dataset.py`       | Tokenizes and prepares text for fine-tuning |
| `finetune.py`      | Fine-tunes GPT-2 on the custom corpus       |
| `generate.py`      | Generates text from prompts                 |
| `sample_texts/`    | Contains sample training data               |
| `checkpoints/`     | Stores fine-tuned model checkpoints         |
| `outputs/`         | Stores generated outputs                    |
| `requirements.txt` | Project dependencies                        |
| `README.md`        | Project documentation                       |

---

# 🚀 Usage

## 1️⃣ Generate Text with the Pretrained GPT-2

No fine-tuning is required to use the pretrained model.

```powershell
python src\generate.py --prompt "The old lighthouse keeper" --strategy top-p
```

The same prompt can be tested with different decoding strategies.

### Greedy

```powershell
python src\generate.py --prompt "The old lighthouse keeper" --strategy greedy
```

### Beam Search

```powershell
python src\generate.py --prompt "The old lighthouse keeper" --strategy beam
```

### Top-k Sampling

```powershell
python src\generate.py --prompt "The old lighthouse keeper" --strategy top-k
```

### Top-p Sampling

```powershell
python src\generate.py --prompt "The old lighthouse keeper" --strategy top-p
```

---

# 🎛️ Generation Parameters

| Flag              | Default | Effect                                             |
| ----------------- | ------: | -------------------------------------------------- |
| `--max-length`    |   `100` | Total tokens including prompt and generated tokens |
| `--num-sequences` |     `1` | Number of outputs to generate                      |
| `--temperature`   |   `1.0` | Controls randomness for top-k/top-p sampling       |
| `--seed`          |  Random | Controls reproducibility                           |

### 🌡️ Temperature

Temperature controls the randomness of token selection.

```text
Lower Temperature
       ↓
More Predictable
       ↓
Less Random

Higher Temperature
       ↓
More Random
       ↓
More Creative
```

---

# 🎯 2️⃣ Fine-Tune GPT-2 on Your Own Text

The model can be fine-tuned using any suitable plain `.txt` corpus.

For example:

```powershell
python src\finetune.py --data sample_texts\sample_corpus.txt --epochs 3
```

The included sample corpus can be used to verify that the training pipeline works correctly.

For larger datasets, training may take significantly longer, especially when running on CPU.

For initial experimentation, start with:

* 📄 A small text corpus
* 🔢 A few epochs
* 🧪 Short test runs

This helps verify the complete pipeline before scaling up.

---

# 🤖 3️⃣ Generate Text with the Fine-Tuned Model

After fine-tuning, generate text using the saved model:

```powershell
python src\generate.py --model checkpoints\gpt2-finetuned --prompt "The old lighthouse keeper" --strategy top-p
```

The generation pipeline becomes:

```text
Custom Corpus
      ↓
Fine-Tuning
      ↓
Fine-Tuned GPT-2
      ↓
Prompt
      ↓
Decoding Strategy
      ↓
Generated Text
```

---

# 💾 Checkpoints

Fine-tuned models are stored inside:

```text
checkpoints/
```

Example:

```text
checkpoints/
└── gpt2-finetuned/
```

These checkpoints allow the fine-tuned model to be loaded later for text generation.

---

# 📤 Outputs

Generated text can be stored inside:

```text
outputs/
```

This makes it easier to compare:

* Different prompts
* Different decoding strategies
* Different model configurations
* Base GPT-2 vs fine-tuned GPT-2

---

# 🔬 What This Project Demonstrates

This project provides practical experience with:

* 🤖 GPT-2
* 🧠 Transformer-based language models
* 📝 Autoregressive text generation
* 🎯 Fine-tuning pretrained language models
* 🔤 Tokenization
* 🎛️ Decoding strategies
* 🔹 Greedy decoding
* 🔹 Beam search
* 🔹 Top-k sampling
* 🔹 Top-p sampling
* 🌡️ Temperature
* 🎲 Reproducible generation
* 📚 Custom text corpora
* 💾 Model checkpoints
* 🧩 NLP
* 🐍 Python
* 🤗 Hugging Face Transformers

---

# ⚠️ Limitations

GPT-2 text generation has several limitations:

### 📌 Model Size

GPT-2 is smaller and less capable than many modern language models.

### 🎯 Fine-Tuning Data

The quality of the fine-tuned output depends heavily on the quality and size of the training corpus.

### 🔁 Repetition

Some decoding strategies may produce repetitive text, especially during longer generations.

### 🧠 Context Limitations

The model can only process a limited context window during generation.

### 💻 Computational Requirements

Fine-tuning can be computationally expensive depending on the dataset size, model configuration, and available hardware.

---

# 🔮 Future Improvements

Possible extensions include:

* [ ] Add support for larger GPT-2 variants
* [ ] Add validation during fine-tuning
* [ ] Add training and validation loss visualization
* [ ] Add TensorBoard monitoring
* [ ] Add configurable generation parameters
* [ ] Add batch generation
* [ ] Add a Streamlit interface
* [ ] Add a Flask/FastAPI inference API
* [ ] Add experiment tracking
* [ ] Compare decoding strategies automatically
* [ ] Compare base and fine-tuned model outputs
* [ ] Add additional evaluation metrics

---

# 📚 References

* [Hugging Face — How to Generate Text](https://huggingface.co/blog/how-to-generate)
* [GPT-2 Fine-Tuning Reference Colab Notebook](https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD?usp=sharing)
* Radford et al. — [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

# ⭐ Conclusion

This project demonstrates **text generation and fine-tuning using GPT-2**, providing a practical introduction to Transformer-based language models.

It supports both pretrained and fine-tuned GPT-2 models and explores multiple decoding strategies:

```text
GPT-2
 ↓
Prompt
 ↓
Decoding Strategy
 ↓
Generated Text
```

It also demonstrates how a pretrained language model can be adapted to a custom text corpus:

```text
Pretrained GPT-2
      ↓
Custom Corpus
      ↓
Fine-Tuning
      ↓
Fine-Tuned GPT-2
      ↓
Domain-Specific Generation
```

Overall, the project provides hands-on experience with **Transformer-based NLP, autoregressive generation, fine-tuning, and modern text decoding techniques**.

---

# 👩‍💻 Author

**Usha.S.Reddy**

If you found this project useful, consider giving the repository a ⭐.
