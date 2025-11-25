import os
import json
import argparse
from tqdm import tqdm
from collections import OrderedDict
import torch.multiprocessing as mp

from src.answer_extractor import AnswerExtractor
from src.evaluator import Evaluator
from src.llm_client import LLMClient
from src.data_loader import DataLoader, CacheManager
from src.utils import load_tokenizer, format_prompt, load_model_config, format_cot_second_prompt


def load_rag_contexts(path):
    contexts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if "id" in record and "context_rag" in record:
                    contexts[record["id"]] = record["context_rag"]
            except Exception:
                continue
    return contexts


def predict_single_item(data, llm_client, tokenizer, model, max_len,
                        with_context, max_new_tokens, extractor, evaluator,
                        use_cot=False, use_rag=False, rag_topk=5):
    prompt_result = format_prompt(data, tokenizer, model, max_len, with_context, use_cot, use_rag, rag_topk)

    if use_cot:
        prompt, removed_part = prompt_result
        first_response = llm_client.query(prompt, temperature=0.1, max_tokens=4096)
        if not first_response:
            return None

        second_prompt = format_cot_second_prompt(first_response, removed_part)
        response = llm_client.query(second_prompt, temperature=0.1, max_tokens=max_new_tokens)
        if not response:
            return None
    else:
        prompt = prompt_result
        response = llm_client.query(prompt, temperature=0.1, max_tokens=max_new_tokens)
        if not response:
            return None

    pred_answer = extractor.extract(response.strip(), data['task'], data['source'])

    result = OrderedDict()
    result['id'] = data['id']
    result['source'] = data['source']
    result['task'] = data['task']
    result['type'] = data['type']
    result['correct_answer'] = data['answer']
    result['pred_answer'] = pred_answer
    result['response'] = response.strip()

    result['judge'] = evaluator.judge(pred_answer, data['answer'], data['task'], data['source'])

    return result


def process_batch(dataset, args, output_file):
    config = load_model_config("config/models.jsonl", args.model)
    model = config["model"]
    max_len = config["max_len"]

    tokenizer = load_tokenizer(model)
    llm_client = LLMClient(model, config["base_url"], config["api_key"])
    extractor = AnswerExtractor()
    evaluator = Evaluator()

    with open(output_file, 'a', encoding='utf-8') as fout:
        for data in tqdm(dataset):
            result = predict_single_item(
                data, llm_client, tokenizer, model, max_len,
                args.with_context, args.max_new_tokens, extractor, evaluator,
                args.use_cot, args.use_rag, args.rag_topk
            )

            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                fout.flush()


def main():
    parser = argparse.ArgumentParser(description="LooGLE-v2 Prediction Script")
    parser.add_argument("--save_dir", "-s", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Model name (should match config/models.jsonl)")
    parser.add_argument("--with_context", "-c", type=int, default=1,
                       help="Whether to include context (1=yes, 0=no)")
    parser.add_argument("--n_proc", "-n", type=int, default=1,
                       help="Number of parallel processes")
    parser.add_argument("--data_dir", "-d", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Max new tokens for model output")
    parser.add_argument("--use_cot", action="store_true",
                       help="Enable Chain-of-Thought prompting")
    parser.add_argument("--use_rag", action="store_true",
                       help="Enable RAG (use retrieved context)")
    parser.add_argument("--rag_topk", type=int, default=5,
                       help="Number of top-k context chunks for RAG")
    parser.add_argument("--rag_context", type=str, default=None,
                       help="Optional path to context_rag jsonl produced by rag_preprocess.py")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    suffix = "_no_context" if args.with_context == 0 else ""
    if args.use_cot:
        suffix += "_cot"
    if args.use_rag:
        suffix += f"_rag_top{args.rag_topk}"
    output_file = os.path.join(args.save_dir, f"{args.model}{suffix}.jsonl")

    print("Loading dataset...")
    data_list = DataLoader.load(args.data_dir)

    if args.use_rag and args.rag_context:
        rag_contexts = load_rag_contexts(args.rag_context)
        hit = 0
        for item in data_list:
            if item["id"] in rag_contexts:
                item["context_rag"] = rag_contexts[item["id"]]
                hit += 1
        print(f"Attached context_rag for {hit}/{len(data_list)} samples from {args.rag_context}")

    cached_ids = CacheManager.load_cached_ids(output_file)
    data_list = CacheManager.filter_uncached(data_list, cached_ids)

    if not data_list:
        print("All data items have been processed.")
        return

    print(f"Processing {len(data_list)} items...")

    if args.n_proc == 1:
        process_batch(data_list, args, output_file)
    else:
        data_subsets = [data_list[i::args.n_proc] for i in range(args.n_proc)]
        processes = []

        for rank in range(args.n_proc):
            p = mp.Process(target=process_batch, args=(data_subsets[rank], args, output_file))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print(f"Prediction completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
