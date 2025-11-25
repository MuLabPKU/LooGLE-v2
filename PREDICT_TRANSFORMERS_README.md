# predict_transformers.py 使用说明

`predict_transformers.py` 是 `predict.py` 的替代版本，它直接使用 transformers 库调用模型，而不需要通过 OpenAI 接口向 vLLM 发送请求。

## 主要区别

- **predict.py**: 通过 OpenAI API 接口调用 vLLM 服务器
- **predict_transformers.py**: 直接使用 transformers 库加载和运行模型

## 使用方法

### 基本用法

```bash
python predict_transformers.py \
    --model your-model-name \
    --data_dir ./datasets/LooGLE-v2 \
    --save_dir ./results \
    --max_new_tokens 512
```

### 完整参数说明

```bash
python predict_transformers.py \
    --model <model_name>              # 模型名称（需在 config/models.jsonl 中配置）
    --data_dir <path>                 # 数据集路径（必需）
    --save_dir <path>                 # 结果保存目录（默认: results）
    --with_context <0|1>              # 是否包含上下文（默认: 1）
    --max_new_tokens <int>            # 最大生成token数（默认: 512）
    --device <cuda|cpu>                # 设备选择（默认: 自动检测）
    --load_in_8bit                    # 使用8位量化（节省显存）
    --load_in_4bit                    # 使用4位量化（节省显存）
    --torch_dtype <float16|bfloat16|float32>  # 模型权重数据类型
    --n_proc <int>                    # 并行进程数（注意：每个进程会单独加载模型）
```

## 示例

### 1. 使用 GPU 运行（默认）

```bash
python predict_transformers.py \
    --model Qwen2.5-7B-Instruct-1M \
    --data_dir ./datasets/LooGLE-v2 \
    --save_dir ./results
```

### 2. 使用 CPU 运行

```bash
python predict_transformers.py \
    --model Qwen2.5-7B-Instruct-1M \
    --data_dir ./datasets/LooGLE-v2 \
    --save_dir ./results \
    --device cpu
```

### 3. 使用 4-bit 量化（节省显存）

```bash
python predict_transformers.py \
    --model Qwen2.5-7B-Instruct-1M \
    --data_dir ./datasets/LooGLE-v2 \
    --save_dir ./results \
    --load_in_4bit
```

**注意**: 使用量化需要安装 `bitsandbytes`:
```bash
pip install bitsandbytes
```

### 4. 使用 float16 精度（加速推理）

```bash
python predict_transformers.py \
    --model Qwen2.5-7B-Instruct-1M \
    --data_dir ./datasets/LooGLE-v2 \
    --save_dir ./results \
    --torch_dtype float16
```

### 5. 不使用上下文

```bash
python predict_transformers.py \
    --model Qwen2.5-7B-Instruct-1M \
    --data_dir ./datasets/LooGLE-v2 \
    --save_dir ./results \
    --with_context 0
```

## 支持的模型

脚本会自动检测模型类型并应用相应的 prompt 格式：

- **Llama 系列**: Llama-3.1, Llama-3.3 等
- **Qwen 系列**: Qwen2.5, QwQ 等
- **GLM 系列**: GLM-4 等
- **Phi 系列**: Phi-3 等
- **Mistral 系列**: Mistral-7B 等
- **其他模型**: 自动使用 chat template 或默认格式

## 内存优化建议

1. **使用量化**: `--load_in_4bit` 或 `--load_in_8bit` 可以大幅减少显存占用
2. **使用 float16**: `--torch_dtype float16` 可以减少显存占用并加速推理
3. **单进程运行**: 避免使用 `--n_proc > 1`，因为每个进程都会单独加载模型

## 注意事项

1. **首次运行**: 首次运行时会下载模型，可能需要较长时间
2. **显存要求**: 大模型需要足够的 GPU 显存，建议使用量化选项
3. **多进程**: 使用多进程时，每个进程都会加载完整的模型，显存需求会成倍增加
4. **模型路径**: 确保 `config/models.jsonl` 中的 `model` 字段是有效的 Hugging Face 模型路径或本地路径

## 与 predict.py 的对比

| 特性 | predict.py | predict_transformers.py |
|------|------------|------------------------|
| 需要 vLLM 服务器 | ✅ 是 | ❌ 否 |
| 需要 API Key | ✅ 是 | ❌ 否 |
| 内存占用 | 低（服务器端） | 高（本地） |
| 推理速度 | 快（vLLM优化） | 中等 |
| 量化支持 | 服务器端配置 | ✅ 支持 |
| 离线使用 | ❌ 否 | ✅ 是 |

## 故障排除

### 1. 显存不足

```bash
# 使用量化
--load_in_4bit

# 或使用 CPU（较慢）
--device cpu
```

### 2. 模型加载失败

- 检查模型路径是否正确
- 确保有网络连接（首次下载模型）
- 检查是否有足够的磁盘空间

### 3. 生成结果不理想

- 检查 prompt 格式是否正确（脚本会自动检测）
- 尝试调整 `--max_new_tokens` 参数
- 检查模型是否支持 chat 格式

## 评估结果

生成结果后，使用相同的 `evaluate.py` 脚本进行评估：

```bash
python evaluate.py --input_path ./results/your-model-name.jsonl
```

