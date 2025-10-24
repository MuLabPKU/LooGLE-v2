import json
import tiktoken
from transformers import AutoTokenizer


def load_tokenizer(model):
    if any(x in model for x in ["gpt", "o1", "o3", "o4", "Fin-R1"]):
        return tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def truncate_prompt(prompt, tokenizer, model, max_len):
    use_tiktoken = any(x in model for x in ["gpt", "o1", "o3", "o4", "Fin-R1"])

    if use_tiktoken:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
    else:
        input_ids = tokenizer.encode(prompt)

    if len(input_ids) > max_len:
        half = max_len // 2
        input_ids = input_ids[:half] + input_ids[-half:]

    if use_tiktoken:
        return tokenizer.decode(input_ids)
    else:
        return tokenizer.decode(input_ids, skip_special_tokens=True)


def format_prompt(data, tokenizer, model, max_len, with_context):
    if not with_context:
        data["context"] = "I would not provide you with the context. Please choose the most likely option based on your knowledge and intuition."

    prompt = data["instruction"].format(
        context=data["context"],
        options=data["options"],
        question=data["question"]
    )

    return truncate_prompt(prompt, tokenizer, model, max_len)


def load_model_config(config_path, model_name):
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            config = json.loads(line.strip())
            if config.get("name") == model_name:
                return {
                    "model": config.get("model", ""),
                    "max_len": config.get("max_len", ""),
                    "base_url": config.get("base_url", ""),
                    "api_key": config.get("api_key", "")
                }

    raise ValueError(f"Model '{model_name}' not found in {config_path}")
