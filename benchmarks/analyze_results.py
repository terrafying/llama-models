"""Analyze benchmark results and generate visualizations."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(input_dir: str) -> Dict:
    """Load all benchmark results."""
    results = {}
    for file in Path(input_dir).glob("benchmark_results_*.json"):
        device = file.stem.split("_")[-1]
        with open(file) as f:
            results[device] = json.load(f)
    return results

def create_comparison_dataframe(results: Dict) -> pd.DataFrame:
    """Create a DataFrame for comparing results."""
    data = []
    
    for device, device_results in results.items():
        for model_type, models in device_results.items():
            for model_id, model_results in models.items():
                # Get results without Unsloth
                without_unsloth = model_results["without_unsloth"]
                data.append({
                    "device": device,
                    "model_type": model_type,
                    "model_id": model_id,
                    "optimization": "without_unsloth",
                    "mean_generation_time": without_unsloth["statistics"]["mean_generation_time"],
                    "std_generation_time": without_unsloth["statistics"]["std_generation_time"],
                    "mean_tokens_per_second": without_unsloth["statistics"]["mean_tokens_per_second"],
                    "std_tokens_per_second": without_unsloth["statistics"]["std_tokens_per_second"],
                })
                
                # Get results with Unsloth if available
                if "with_unsloth" in model_results:
                    with_unsloth = model_results["with_unsloth"]
                    data.append({
                        "device": device,
                        "model_type": model_type,
                        "model_id": model_id,
                        "optimization": "with_unsloth",
                        "mean_generation_time": with_unsloth["statistics"]["mean_generation_time"],
                        "std_generation_time": with_unsloth["statistics"]["std_generation_time"],
                        "mean_tokens_per_second": with_unsloth["statistics"]["mean_tokens_per_second"],
                        "std_tokens_per_second": with_unsloth["statistics"]["std_tokens_per_second"],
                    })
    
    return pd.DataFrame(data)

def plot_generation_times(df: pd.DataFrame, output_dir: Path):
    """Plot generation time comparisons."""
    plt.figure(figsize=(12, 6))
    
    # Filter for LLM models
    llm_df = df[df["model_type"] == "llm"]
    
    # Create grouped bar plot
    sns.barplot(
        data=llm_df,
        x="model_id",
        y="mean_generation_time",
        hue="optimization",
        palette="Set2",
    )
    
    plt.title("Generation Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Mean Generation Time (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / "generation_times.png")
    plt.close()

def plot_tokens_per_second(df: pd.DataFrame, output_dir: Path):
    """Plot tokens per second comparisons."""
    plt.figure(figsize=(12, 6))
    
    # Filter for LLM models
    llm_df = df[df["model_type"] == "llm"]
    
    # Create grouped bar plot
    sns.barplot(
        data=llm_df,
        x="model_id",
        y="mean_tokens_per_second",
        hue="optimization",
        palette="Set2",
    )
    
    plt.title("Tokens per Second Comparison")
    plt.xlabel("Model")
    plt.ylabel("Mean Tokens per Second")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / "tokens_per_second.png")
    plt.close()

def plot_speedup_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot speedup comparison between optimized and unoptimized versions."""
    # Calculate speedup
    speedup_data = []
    for model_id in df["model_id"].unique():
        model_data = df[df["model_id"] == model_id]
        if len(model_data) == 2:  # Both with and without Unsloth
            without_unsloth = model_data[model_data["optimization"] == "without_unsloth"]
            with_unsloth = model_data[model_data["optimization"] == "with_unsloth"]
            
            speedup = (
                without_unsloth["mean_tokens_per_second"].values[0] /
                with_unsloth["mean_tokens_per_second"].values[0]
            )
            
            speedup_data.append({
                "model_id": model_id,
                "speedup": speedup,
            })
    
    speedup_df = pd.DataFrame(speedup_data)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=speedup_df, x="model_id", y="speedup", palette="viridis")
    
    plt.title("Speedup with Unsloth Optimization")
    plt.xlabel("Model")
    plt.ylabel("Speedup Factor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / "speedup_comparison.png")
    plt.close()

def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate a summary report of the benchmark results."""
    # Calculate summary statistics
    summary = df.groupby(["model_type", "model_id", "optimization"]).agg({
        "mean_generation_time": ["mean", "std"],
        "mean_tokens_per_second": ["mean", "std"],
    }).round(3)
    
    # Save summary to CSV
    summary.to_csv(output_dir / "summary_report.csv")
    
    # Generate markdown report
    report = ["# Benchmark Results Summary\n"]
    
    for model_type in df["model_type"].unique():
        report.append(f"\n## {model_type.upper()} Models\n")
        
        type_df = df[df["model_type"] == model_type]
        for model_id in type_df["model_id"].unique():
            report.append(f"\n### {model_id}\n")
            
            model_data = type_df[type_df["model_id"] == model_id]
            for _, row in model_data.iterrows():
                report.append(f"\n#### {row['optimization']}\n")
                report.append(f"- Mean Generation Time: {row['mean_generation_time']:.3f}s ± {row['std_generation_time']:.3f}s")
                report.append(f"- Mean Tokens per Second: {row['mean_tokens_per_second']:.3f} ± {row['std_tokens_per_second']:.3f}")
    
    # Save markdown report
    with open(output_dir / "summary_report.md", "w") as f:
        f.write("\n".join(report))

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input-dir", required=True, help="Input directory containing benchmark results")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results(args.input_dir)
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(results)
    
    # Generate visualizations
    plot_generation_times(df, output_dir)
    plot_tokens_per_second(df, output_dir)
    plot_speedup_comparison(df, output_dir)
    
    # Generate summary report
    generate_summary_report(df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 