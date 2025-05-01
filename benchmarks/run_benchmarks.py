"""Run model benchmarks with and without Unsloth optimizations."""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import numpy as np
from tqdm import tqdm

def load_model(model_id: str, use_unsloth: bool = False) -> tuple:
    """Load model with or without Unsloth optimizations."""
    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    return model, tokenizer

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    num_return_sequences: int = 1,
) -> List[str]:
    """Generate text from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )
    generation_time = time.time() - start_time
    
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]
    
    return generated_texts, generation_time

def run_benchmark(
    model_id: str,
    prompts: List[str],
    use_unsloth: bool = False,
    num_runs: int = 5,
) -> Dict:
    """Run benchmark for a model."""
    print(f"Loading model {model_id} {'with' if use_unsloth else 'without'} Unsloth...")
    model, tokenizer = load_model(model_id, use_unsloth)
    
    results = {
        "model_id": model_id,
        "use_unsloth": use_unsloth,
        "device": str(model.device),
        "runs": [],
    }
    
    for i in tqdm(range(num_runs), desc="Running benchmarks"):
        run_results = []
        for prompt in prompts:
            texts, gen_time = generate_text(model, tokenizer, prompt)
            run_results.append({
                "prompt": prompt,
                "generated_text": texts[0],
                "generation_time": gen_time,
                "tokens_per_second": len(texts[0].split()) / gen_time,
            })
        results["runs"].append(run_results)
    
    # Calculate statistics
    all_times = [r["generation_time"] for run in results["runs"] for r in run]
    all_tps = [r["tokens_per_second"] for run in results["runs"] for r in run]
    
    results["statistics"] = {
        "mean_generation_time": np.mean(all_times),
        "std_generation_time": np.std(all_times),
        "mean_tokens_per_second": np.mean(all_tps),
        "std_tokens_per_second": np.std(all_tps),
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument("--model-type", default="all", help="Type of model to benchmark")
    parser.add_argument("--device", default="cuda", help="Device to run benchmarks on")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()
    
    # Define test prompts
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the key principles of sustainable development?",
    ]
    
    # Define models to benchmark
    models = {
        "llm": [
            "microsoft/phi-2",
            "microsoft/phi-1.5",
            "microsoft/phi-4-mini-reasoning",
        ],
        "embedding": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    all_results = {}
    model_types = [args.model_type] if args.model_type != "all" else models.keys()
    
    for model_type in model_types:
        if model_type not in models:
            continue
            
        all_results[model_type] = {}
        for model_id in models[model_type]:
            print(f"\nBenchmarking {model_id}...")
            
            # Run without Unsloth
            results = run_benchmark(model_id, prompts, use_unsloth=False)
            all_results[model_type][model_id] = {
                "without_unsloth": results
            }
            
            # Run with Unsloth if it's an LLM
            if model_type == "llm":
                results = run_benchmark(model_id, prompts, use_unsloth=True)
                all_results[model_type][model_id]["with_unsloth"] = results
    
    # Save results
    output_file = output_dir / f"benchmark_results_{args.device}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 