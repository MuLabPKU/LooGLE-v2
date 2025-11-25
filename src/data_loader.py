import json
import os
from datasets import load_dataset
from typing import List, Dict


class DataLoader:

    @staticmethod
    def load(source, split='test'):
        if source.endswith('.jsonl'):
            return DataLoader._load_jsonl(source)
        if os.path.isdir(source):
            candidate = os.path.join(source, f"{split}.jsonl")
            if os.path.exists(candidate):
                return DataLoader._load_jsonl(candidate)
        return DataLoader._load_hf_dataset(source, split)

    @staticmethod
    def _load_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        return DataLoader._format_dataset(dataset, file_path)

    @staticmethod
    def _load_hf_dataset(source, split):
        dataset = load_dataset(source, split=split)
        return DataLoader._format_dataset(dataset, source)

    @staticmethod
    def _format_dataset(dataset, source):
        data_all = []
        for item in dataset:
            options = item['options']
            if isinstance(options, list):
                options = "\n".join(options)

            data_all.append({
                "id": item["id"],
                "source": item["source"],
                "task": item["task"],
                "type": item["type"],
                "instruction": item["instruction"],
                "context": item["context"],
                "question": item["question"],
                "options": options,
                "answer": item["answer"]
            })

        print(f"Loaded {len(data_all)} items from {source}")
        return data_all


class CacheManager:

    @staticmethod
    def load_cached_ids(output_file):
        import os
        if not os.path.exists(output_file):
            return set()

        cached_ids = set()
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                cached_ids.add(data["id"])

        return cached_ids

    @staticmethod
    def filter_uncached(data_list, cached_ids):
        return [item for item in data_list if item["id"] not in cached_ids]
