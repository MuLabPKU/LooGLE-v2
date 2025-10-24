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
from src.utils import load_tokenizer, format_prompt, load_model_config


def predict_single_item(data, llm_client, tokenizer, model, max_len,
                        with_context, max_new_tokens, extractor, evaluator):
    prompt = format_prompt(data, tokenizer, model, max_len, with_context)

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
                args.with_context, args.max_new_tokens, extractor, evaluator
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

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    suffix = "_no_context" if args.with_context == 0 else ""
    output_file = os.path.join(args.save_dir, f"{args.model}{suffix}.jsonl")

    print("Loading dataset...")
    data_list = DataLoader.load(args.data_dir)

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
