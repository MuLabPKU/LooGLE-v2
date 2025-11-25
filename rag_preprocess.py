import argparse
import json
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from src.data_loader import DataLoader


def chunk_text(text, tokenizer, chunk_size=512):
    tokens = tokenizer.encode(text or "", add_special_tokens=False)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    if not chunks:
        return [""]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        pooled = outputs.hidden_states[-1].mean(dim=1)
    return pooled.squeeze(0)


def rank_chunks(query, chunks, tokenizer, model, device, top_k=64):
    q_emb = get_embedding(query, tokenizer, model, device)
    chunk_embs = [get_embedding(chunk, tokenizer, model, device) for chunk in chunks]
    sims = torch.stack([F.cosine_similarity(q_emb, emb, dim=0) for emb in chunk_embs])
    topk = torch.topk(sims, min(top_k, len(sims)))
    return [chunks[i] for i in topk.indices]


def get_model_feedback(query, context_chunks, tokenizer, model, device, max_new_tokens=128):
    prompt = (
        f"Question: {query}\n"
        f"Contexts: {''.join(context_chunks)}\n"
        "What additional information would help you answer the question?"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def multi_turn_rank_chunks(query, chunks, args, embed_tokenizer, embed_model, gen_tokenizer, gen_model, device):
    first = rank_chunks(query, chunks, embed_tokenizer, embed_model, device, top_k=args.top_k_initial)
    feedback1 = get_model_feedback(query, first, gen_tokenizer, gen_model, device, max_new_tokens=args.max_new_tokens)

    second = rank_chunks(feedback1, chunks, embed_tokenizer, embed_model, device, top_k=args.top_k_next)
    feedback2 = get_model_feedback(feedback1, second, gen_tokenizer, gen_model, device, max_new_tokens=args.max_new_tokens)

    final = rank_chunks(feedback2, chunks, embed_tokenizer, embed_model, device, top_k=args.top_k_final)

    merged = first + second + final
    # Preserve order while removing duplicates
    return list(dict.fromkeys(merged))


def load_cached_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    cached = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                cached.add(json.loads(line)["id"])
            except Exception:
                continue
    return cached


def process_partition(data_subset, args, device_id, lock):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)

    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
    embed_model = AutoModel.from_pretrained(args.embedding_model, trust_remote_code=True).eval().to(device)

    gen_model_name = args.generator_model or args.embedding_model
    gen_tokenizer = gen_model = None
    if args.multi_turn:
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
        gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, trust_remote_code=True).eval().to(device)

    with open(args.output_path, "a", encoding="utf-8") as fout:
        for item in tqdm(data_subset, desc=f"worker-{device_id}"):
            context = item.get("context", "")
            chunks = chunk_text(context, embed_tokenizer, chunk_size=args.chunk_size)

            if args.multi_turn:
                ranked_chunks = multi_turn_rank_chunks(
                    item.get("question", ""),
                    chunks,
                    args,
                    embed_tokenizer,
                    embed_model,
                    gen_tokenizer,
                    gen_model,
                    device,
                )
            else:
                ranked_chunks = rank_chunks(
                    item.get("question", ""),
                    chunks,
                    embed_tokenizer,
                    embed_model,
                    device,
                    top_k=args.top_k,
                )

            context_rag = ranked_chunks[: args.top_k]
            output_item = {"id": item["id"], "context_rag": context_rag}
            with lock:
                fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                fout.flush()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute RAG contexts for LooGLE-v2 dataset")
    parser.add_argument("--input_path", required=True, help="Path to dataset (.jsonl), HF dataset id, or local HF clone")
    parser.add_argument("--split", type=str, default="test", help="Dataset split when reading from Hugging Face")
    parser.add_argument("--output_path", required=True, help="Output path for jsonl with context_rag")
    parser.add_argument("--embedding_model", type=str, default="THUDM/LongCite-glm4-9b", help="Model for embeddings")
    parser.add_argument("--generator_model", type=str, default=None, help="(Optional) model for multi-turn feedback")
    parser.add_argument("--chunk_size", type=int, default=512, help="Token chunk size for splitting context")
    parser.add_argument("--top_k", type=int, default=64, help="Top-k chunks to keep (single turn)")
    parser.add_argument("--multi_turn", action="store_true", help="Enable multi-turn refinement")
    parser.add_argument("--top_k_initial", type=int, default=64, help="First round top-k (multi-turn)")
    parser.add_argument("--top_k_next", type=int, default=32, help="Second round top-k (multi-turn)")
    parser.add_argument("--top_k_final", type=int, default=16, help="Final round top-k (multi-turn)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for feedback generation")
    parser.add_argument("--devices", type=str, default="0", help="Comma separated GPU ids, e.g. 0,1,2")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    cached_ids = load_cached_ids(args.output_path)

    dataset = DataLoader.load(args.input_path, split=args.split)
    data = [item for item in dataset if item["id"] not in cached_ids]

    print(f"Loaded {len(dataset)} items, skipping {len(cached_ids)}, processing {len(data)} items.")
    if not data:
        print("No new samples to process.")
        return

    device_list = [int(d.strip()) for d in args.devices.split(",") if d.strip() != ""]
    num_workers = max(1, len(device_list))
    partitions = [data[i::num_workers] for i in range(num_workers)]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    lock = manager.Lock()

    processes = []
    for idx, partition in enumerate(partitions):
        device_id = device_list[idx % len(device_list)]
        p = mp.Process(target=process_partition, args=(partition, args, device_id, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
