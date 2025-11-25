import os
import json
import argparse
from collections import defaultdict


class ResultAnalyzer:

    def __init__(self, result_file):
        self.result_file = result_file
        self.group_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        self.overall_total = 0
        self.overall_correct = 0

    def analyze(self):
        with open(self.result_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self._process_item(data)

        return self._compute_statistics()

    def _process_item(self, data):
        source = data.get("source", "Unknown")
        task = data.get("task", "Unknown")
        judge = data.get("judge")

        key = (source, task)
        self.group_stats[key]["total"] += 1
        self.overall_total += 1

        if isinstance(judge, bool):
            if judge:
                self.group_stats[key]["correct"] += 1
                self.overall_correct += 1
        elif isinstance(judge, float):
            self.group_stats[key]["correct"] += judge / 100
            self.overall_correct += judge / 100

    def _compute_statistics(self):
        results = []

        for (source, task), stats in sorted(self.group_stats.items()):
            total = stats["total"]
            correct = stats["correct"]
            accuracy = (correct / total * 100) if total > 0 else 0

            results.append({
                "source": source,
                "task": task,
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            })

        overall_accuracy = (self.overall_correct / self.overall_total * 100) if self.overall_total > 0 else 0

        return {
            "group_results": results,
            "overall_accuracy": overall_accuracy,
            "overall_correct": self.overall_correct,
            "overall_total": self.overall_total
        }


def print_results(statistics):
    print("\n" + "="*80)
    print("Evaluation Results by Group")
    print("="*80)

    for result in statistics["group_results"]:
        print(f"Source: {result['source']:<15} | Task: {result['task']:<35} | "
              f"Accuracy: {result['accuracy']:>6.2f}% ({result['correct']:.1f}/{result['total']})")

    print("="*80)
    print(f"Overall Accuracy: {statistics['overall_accuracy']:.2f}% "
          f"({statistics['overall_correct']:.1f}/{statistics['overall_total']})")
    print("="*80 + "\n")


def evaluate_single_file(input_path, output_json=None):
    print(f"Evaluating: {input_path}")

    analyzer = ResultAnalyzer(input_path)
    statistics = analyzer.analyze()

    print_results(statistics)

    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_json}")

    return statistics


def evaluate_batch(folder_path, output_summary=None):
    all_results = {}

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                input_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(input_path, folder_path)

                print(f"\n{'='*80}")
                print(f"Processing: {relative_path}")
                print(f"{'='*80}")

                statistics = evaluate_single_file(input_path)
                all_results[relative_path] = {
                    "overall_accuracy": statistics["overall_accuracy"],
                    "overall_correct": statistics["overall_correct"],
                    "overall_total": statistics["overall_total"]
                }

    print("\n" + "="*80)
    print("Batch Evaluation Summary")
    print("="*80)
    for file_path, result in sorted(all_results.items()):
        print(f"{file_path:<60} | Accuracy: {result['overall_accuracy']:>6.2f}% "
              f"({result['overall_correct']:.1f}/{result['overall_total']})")
    print("="*80 + "\n")

    if output_summary:
        with open(output_summary, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Summary saved to {output_summary}")


def main():
    parser = argparse.ArgumentParser(description="LooGLE-v2 Evaluation Script")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to prediction result file (jsonl) or folder for batch mode")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Optional: save results to JSON file")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch mode to evaluate all .jsonl files in folder")

    args = parser.parse_args()

    if args.batch:
        evaluate_batch(args.input_path, args.output_json)
    else:
        evaluate_single_file(args.input_path, args.output_json)


if __name__ == "__main__":
    main()
