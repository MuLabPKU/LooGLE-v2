<div align="center">

# LooGLE v2

**The official repository of "LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?"**

*NeurIPS DB Track 2025*

<div>
  <a href="https://huggingface.co/datasets/GraphPKU/LooGLE-v2">
    <img src="https://img.shields.io/badge/🤗-Dataset-blue" alt="Dataset">
  </a>
  <a href="https://mulabpku.github.io/LooGLE-v2/">
    <img src="https://img.shields.io/badge/🌐-Website-green" alt="Website">
  </a>
  <a href="https://arxiv.org/abs/2510.22548">
    <img src="https://img.shields.io/badge/📄-Paper-red" alt="Paper">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
</div>



</div>

---

## 📋 Overview

LooGLE v2 is a comprehensive benchmark designed to evaluate large language models on their ability to understand and process long-context documents with complex dependencies. The benchmark covers diverse domains including **Finance**, **Law**, **Code**, and **Game**.

---

## 🚀 Quick Start

### 📦 Installation

```bash
# Create environment with Python 3.10
conda create -n loogle-v2 python=3.10
conda activate loogle-v2

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention
pip install flash-attn==2.6.3 --no-build-isolation

# Or you can download flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl
```

---

## 📊 Dataset

Download the LooGLE v2 dataset from Hugging Face:

```bash
git clone https://huggingface.co/datasets/MuLabPKU/LooGLE-v2 ./datasets/LooGLE-v2
# Or use the Hugging Face CLI to download:
hf download MuLabPKU/LooGLE-v2   --repo-type dataset  --local-dir ./datasets/LooGLE-v2
```


---

## 🛠️ Usage

### ⚙️ Configuration

**vLLM server (for `predict.py`):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model path/to/your/model \
  --port 8000 \
  --max-model-len 131072
```

**Model entry (`config/models.jsonl`, shared by both scripts):**
```json
{
  "name": "your-model-name",
  "model": "path/to/model",
  "max_len": 131072,
  "base_url": "http://localhost:8000/v1",
  "api_key": "your-api-key"
}
```

Transformers mode (`predict_transformers.py`) does not need a server; it still reuses `name/model/max_len` from this config. Ensure `base_url` matches your vLLM port when using the server route.

### 🔎 Pre-compute RAG Contexts (optional)

If you plan to run `--use_rag`, first generate `context_rag` with the preprocessor:

```bash
python rag_preprocess.py \
  --input_path ./datasets/LooGLE-v2 \
  --split test \
  --output_path ./datasets/LooGLE-v2/test_rag.jsonl \
  --embedding_model THUDM/LongCite-glm4-9b \
  --devices 0,1
```

For multi-turn refinement (using a generator model to iteratively improve retrieval queries):

```bash
python rag_preprocess.py \
  --input_path ./datasets/LooGLE-v2 \
  --split test \
  --output_path ./datasets/LooGLE-v2/test_rag_multi.jsonl \
  --embedding_model THUDM/LongCite-glm4-9b \
  --generator_model meta-llama/Llama-3.1-8B \
  --multi_turn --devices 0,1
```

### 🎯 Running Predictions

#### Option A: vLLM server (`predict.py`)

```bash
python predict.py \
  --model your-model-name \
  --data_dir ./datasets/LooGLE-v2 \
  --save_dir ./results \
  --max_new_tokens 512
```

#### Option B: Transformers local (`predict_transformers.py`)

```bash
python predict_transformers.py \
  --model your-model-name \
  --data_dir ./datasets/LooGLE-v2 \
  --save_dir ./results \
  --max_new_tokens 512
```

Optional prompting flags (both scripts):
- `--use_cot` for Chain-of-Thought
- `--use_rag --rag_topk <k> --rag_context <path>` to inject precomputed `context_rag` (default file: `./datasets/LooGLE-v2/test_rag.jsonl`)

<details>
<summary><b>📝 Core parameters (both options)</b></summary>

| Flag | Purpose |
|------|---------|
| `--model` | Must match `config/models.jsonl` name |
| `--data_dir` | Dataset path (jsonl or HF) |
| `--save_dir` | Output directory |
| `--with_context` | 1/0 to include original context |
| `--n_proc` | Parallel processes |
| `--max_new_tokens` | Generation length |
| `--use_cot` | Enable Chain-of-Thought |
| `--use_rag` | Use retrieved context |
| `--rag_topk` | How many retrieved chunks to keep |
| `--rag_context` | Path to `id + context_rag` jsonl |

</details>

<details>
<summary><b>🖥️ Transformers-only flags</b></summary>

| Flag | Purpose |
|------|---------|
| `--device` | Target device (cuda/cpu, auto by default) |
| `--load_in_8bit` | 8-bit quantization (needs bitsandbytes) |
| `--load_in_4bit` | 4-bit quantization (needs bitsandbytes) |
| `--torch_dtype` | Weight dtype: float16/bfloat16/float32 |

> 💡 Install `bitsandbytes` to enable quantization: `pip install bitsandbytes`

</details>

### 📈 Evaluation

After prediction, evaluate the results:

```bash
python evaluate.py --input_path ./results/your-model-name.jsonl
```

This outputs per-task accuracy for each domain and overall accuracy.

For batch evaluation (e.g., multiple runs with CoT/RAG or no-context variants):

```bash
python evaluate.py --input_path ./results --batch --output_json ./results/summary.json
```

This scans a folder for `.jsonl` files, reports each file’s accuracy, and optionally saves a summary.

---

## 📁 Project Structure

```
LooGLE-v2/
├── src/
│   ├── answer_extractor.py    # Answer extraction logic
│   ├── evaluator.py           # Evaluation metrics
│   ├── llm_client.py          # LLM client implementations
│   ├── data_loader.py         # Data loading utilities
│   └── utils.py               # Common utilities
├── config/
│   └── models.jsonl           # Model configurations
├── predict.py                  # Prediction script (vLLM server)
├── predict_transformers.py     # Prediction script (direct transformers)
├── rag_preprocess.py           # RAG context preprocessing
├── evaluate.py                 # Evaluation script
└── requirements.txt            # Dependencies
```

---


## 📄 Results Format

Prediction outputs are saved in JSONL format:

```json
{
  "id": "sample_id",
  "source": "Finance",
  "task": "Metric Calculation",
  "type": "question_type",
  "correct_answer": "123.45",
  "pred_answer": "123.40",
  "response": "The correct answer is 123.40",
  "judge": true
}
```

---

## 📖 Citation

If you use LooGLE v2 in your research, please cite:

```bibtex
@article{he2025loogle,
  title={LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?},
  author={He, Ziyuan and Wang, Yuxuan and Li, Jiaqi and Liang, Kexin and Zhang, Muhan},
  journal={arXiv preprint arXiv:2510.22548},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
