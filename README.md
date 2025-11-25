<div align="center">

# 🔍 LooGLE v2

**The official repository of "LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?"**

*NeurIPS DB Track 2025*

<div>
  <a href="https://huggingface.co/datasets/GraphPKU/LooGLE-v2">
    <img src="https://img.shields.io/badge/🤗-Dataset-blue" alt="Dataset">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/🌐-Website-green" alt="Website">
  </a>
  <a href="#">
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
conda create -n loogle python=3.10
conda activate loogle

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention
pip install flash-attn==2.6.3 --no-build-isolation

# Or you can download flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl
```

> ⚠️ **Note**: Flash Attention requires CUDA-capable GPU

---

## 📊 Dataset

Download the LooGLE v2 dataset from Hugging Face:

```bash
git clone https://huggingface.co/datasets/MuLabPKU/LooGLE-v2 ./datasets/LooGLE-v2
# Or use the Hugging Face CLI to download:
hf download MuLabPKU/LooGLE-v2 --path ./datasets/LooGLE-v2
```


---

## 🛠️ Usage

### ⚙️ Configuration

#### 1. Start vLLM Server

First, launch a vLLM server with your model:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model path/to/your/model \
  --port 8000 \
  --max-model-len 131072
```

#### 2. Configure Model Settings

Edit `config/models.jsonl` to add your model configuration:

```json
{
  "name": "your-model-name",
  "model": "path/to/model",
  "max_len": 131072,
  "base_url": "http://localhost:8000/v1",
  "api_key": "your-api-key"
}
```

> 💡 **Tip**: Make sure the `base_url` matches your vLLM server port

### 🎯 Running Predictions

We provide two prediction scripts:

#### Option 1: Using vLLM Server (predict.py)

This method requires a running vLLM server and uses OpenAI-compatible API:

```bash
python predict.py \
  --model your-model-name \
  --data_dir ./datasets/LooGLE-v2 \
  --save_dir ./results \
  --max_new_tokens 512
```

<details>
<summary><b>📝 Parameters</b></summary>

| Parameter | Description |
|-----------|-------------|
| `--model` | Model name (must match config) |
| `--data_dir` | Path to dataset |
| `--save_dir` | Output directory |
| `--with_context` | Include context (1) or not (0) |
| `--n_proc` | Number of parallel processes |
| `--max_new_tokens` | Maximum generation length |

</details>

#### Option 2: Using Transformers Directly (predict_transformers.py)

This method directly loads models using the transformers library, without requiring a vLLM server:

```bash
python predict_transformers.py \
  --model your-model-name \
  --data_dir ./datasets/LooGLE-v2 \
  --save_dir ./results \
  --max_new_tokens 512
```

**Key Differences:**

| Feature | predict.py | predict_transformers.py |
|---------|------------|------------------------|
| Requires vLLM Server | ✅ Yes | ❌ No |
| Requires API Key | ✅ Yes | ❌ No |
| Memory Usage | Low (server-side) | High (local) |
| Inference Speed | Fast (vLLM optimized) | Moderate |
| Quantization Support | Server-side config | ✅ Supported |
| Offline Usage | ❌ No | ✅ Yes |

**Additional Parameters for predict_transformers.py:**

| Parameter | Description |
|-----------|-------------|
| `--device` | Device to use (cuda/cpu, default: auto-detect) |
| `--load_in_8bit` | Use 8-bit quantization (saves GPU memory) |
| `--load_in_4bit` | Use 4-bit quantization (saves GPU memory) |
| `--torch_dtype` | Model weight dtype (float16/bfloat16/float32) |

**Examples:**

```bash
# Basic usage with GPU
python predict_transformers.py \
  --model Qwen2.5-7B-Instruct-1M \
  --data_dir ./datasets/LooGLE-v2 \
  --save_dir ./results

# Use 4-bit quantization to save GPU memory
python predict_transformers.py \
  --model Qwen2.5-7B-Instruct-1M \
  --data_dir ./datasets/LooGLE-v2 \
  --save_dir ./results \
  --load_in_4bit

# Use CPU (slower but no GPU required)
python predict_transformers.py \
  --model Qwen2.5-7B-Instruct-1M \
  --data_dir ./datasets/LooGLE-v2 \
  --save_dir ./results \
  --device cpu
```

> 💡 **Note**: For quantization, install `bitsandbytes`: `pip install bitsandbytes`

**Supported Models:**

The script automatically detects model types and applies appropriate prompt formats:
- **Llama series**: Llama-3.1, Llama-3.3, etc.
- **Qwen series**: Qwen2.5, QwQ, etc.
- **GLM series**: GLM-4, etc.
- **Phi series**: Phi-3, etc.
- **Mistral series**: Mistral-7B, etc.
- **Other models**: Automatically uses chat template or default format

**Memory Optimization Tips:**

1. **Use quantization**: `--load_in_4bit` or `--load_in_8bit` significantly reduces GPU memory usage
2. **Use float16**: `--torch_dtype float16` reduces memory and speeds up inference
3. **Single process**: Avoid `--n_proc > 1` as each process loads the model separately

**Troubleshooting:**

- **Out of memory**: Use `--load_in_4bit` or `--device cpu`
- **Model loading fails**: Check model path, ensure internet connection (for first-time download), verify disk space
- **Poor generation quality**: Check prompt format (auto-detected), adjust `--max_new_tokens`, verify model supports chat format

### 📈 Evaluation

After prediction, evaluate the results:

```bash
python evaluate.py --input_path ./results/your-model-name.jsonl
```

This outputs per-task accuracy for each domain and overall accuracy.

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

