import os
import json
import argparse
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import gc

try:
    import bitsandbytes
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None
    print("Warning: bitsandbytes not available. Quantization options will be disabled.")

from src.answer_extractor import AnswerExtractor
from src.evaluator import Evaluator
from src.data_loader import DataLoader, CacheManager
from src.utils import format_prompt, load_model_config, format_cot_second_prompt


class TransformersLLMClient:
    """Direct model inference using transformers library"""

    def __init__(self, model_path, device=None, load_in_8bit=False, load_in_4bit=False, torch_dtype=None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Determine model type and load accordingly
        print(f"Loading model from {model_path}...")
        
        # Configure quantization if needed
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            if not BITSANDBYTES_AVAILABLE:
                raise ImportError("bitsandbytes is required for quantization. Install it with: pip install bitsandbytes")
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if torch_dtype is None else torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Try to load as causal LM first, fallback to seq2seq if needed
        try:
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            elif torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            elif self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            if self.device == "cpu" and not hasattr(self.model, "device_map"):
                self.model = self.model.to(self.device)
                
        except Exception as e:
            print(f"Failed to load as CausalLM, trying Seq2SeqLM: {e}")
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")
        
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        
        # Check if model uses chat template
        self.use_chat_template = hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None
        
    def _is_chat_model(self):
        """Check if model is a chat model based on model path or tokenizer"""
        chat_indicators = ["chat", "instruct", "alpaca", "vicuna", "llama-3", "qwen", "glm", "phi"]
        model_lower = self.model_path.lower()
        return any(indicator in model_lower for indicator in chat_indicators) or self.use_chat_template
    
    def _format_chat_prompt(self, prompt):
        """Format prompt for chat models"""
        if self.use_chat_template:
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                if formatted:
                    return formatted
            except Exception as e:
                # Fallback to manual format
                pass
        
        # Common chat formats based on model family
        model_lower = self.model_path.lower()
        
        if "llama-3" in model_lower:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "llama" in model_lower or "mistral" in model_lower:
            return f"<s>[INST] {prompt} [/INST]"
        elif "qwen" in model_lower:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "glm" in model_lower:
            return f"[Round 1]\n\n问：{prompt}\n\n答："
        elif "phi" in model_lower:
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            # Default format
            return f"User: {prompt}\nAssistant: "
    
    def query(self, prompt, temperature=0.1, max_tokens=512):
        """Generate response from model"""
        try:
            # Format prompt based on model type
            if self._is_chat_model():
                formatted_prompt = self._format_chat_prompt(prompt)
            else:
                formatted_prompt = prompt
            
            # Tokenize
            # Use model's max length or a reasonable default
            max_input_length = getattr(self.model.config, "max_position_embeddings", None) or 2048
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length
            )
            
            if self.device == "cuda" and not hasattr(self.model, "device_map"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return ""


def predict_single_item(data, llm_client, tokenizer, model_path, max_len,
                        with_context, max_new_tokens, extractor, evaluator,
                        use_cot=False, use_rag=False, rag_topk=5):
    prompt_result = format_prompt(
        data, tokenizer, model_path, max_len, with_context, use_cot, use_rag, rag_topk
    )

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


def process_batch(dataset, args, output_file):
    config = load_model_config("config/models.jsonl", args.model)
    model_path = config["model"]
    max_len = config["max_len"]

    # Load tokenizer for prompt formatting
    from src.utils import load_tokenizer
    tokenizer = load_tokenizer(model_path)
    
    # Initialize transformers client
    llm_client = TransformersLLMClient(
        model_path=model_path,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=torch.float16 if args.torch_dtype == "float16" else (torch.bfloat16 if args.torch_dtype == "bfloat16" else None)
    )
    
    extractor = AnswerExtractor()
    evaluator = Evaluator()

    with open(output_file, 'a', encoding='utf-8') as fout:
        for data in tqdm(dataset):
            result = predict_single_item(
                data, llm_client, tokenizer, model_path, max_len,
                args.with_context, args.max_new_tokens, extractor, evaluator,
                args.use_cot, args.use_rag, args.rag_topk
            )

            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                fout.flush()
    
    # Clean up
    del llm_client.model
    del llm_client.tokenizer
    del llm_client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="LooGLE-v2 Prediction Script (Transformers)")
    parser.add_argument("--save_dir", "-s", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Model name (should match config/models.jsonl)")
    parser.add_argument("--with_context", "-c", type=int, default=1,
                       help="Whether to include context (1=yes, 0=no)")
    parser.add_argument("--n_proc", "-n", type=int, default=1,
                       help="Number of parallel processes (Note: each process loads model separately)")
    parser.add_argument("--data_dir", "-d", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Max new tokens for model output")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu). Auto-detect if not specified")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--torch_dtype", type=str, default=None,
                       choices=["float16", "bfloat16", "float32"],
                       help="Torch dtype for model weights")
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

    # Note: Multi-processing with transformers models is memory-intensive
    # Each process will load the model separately
    if args.n_proc == 1:
        process_batch(data_list, args, output_file)
    else:
        print(f"Warning: Using {args.n_proc} processes. Each process will load the model separately, which may require significant memory.")
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
