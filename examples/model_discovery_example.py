"""
Example script demonstrating the model discovery and management system.
"""

from ragtime_llm.utils.discovery import ResourceDiscovery
import argparse
from rich.console import Console
from rich.table import Table
from rich import print as rprint

def display_models(discovery: ResourceDiscovery):
    """Display all discovered models in a nice table format."""
    console = Console()
    
    for model_type in ['llm', 'embedding', 'audio', 'video']:
        models = discovery.resources['models'][model_type]
        if not models:
            continue
            
        table = Table(title=f"{model_type.upper()} Models")
        table.add_column("Name", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Format", style="yellow")
        table.add_column("Size", style="magenta")
        table.add_column("Quantization", style="blue")
        
        for model in models:
            size_str = f"{model.size / (1024*1024):.1f}MB" if model.size else "N/A"
            table.add_row(
                model.name,
                model.source,
                model.format or "N/A",
                size_str,
                model.quantization or "N/A"
            )
        
        console.print(table)
        console.print()

def main():
    parser = argparse.ArgumentParser(description="Model Discovery and Management")
    parser.add_argument("--download", help="Download a specific model from Hugging Face")
    parser.add_argument("--type", default="llm", choices=['llm', 'embedding', 'audio', 'video'],
                      help="Model type for download")
    parser.add_argument("--list", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # Initialize discovery
    discovery = ResourceDiscovery()
    
    if args.download:
        rprint(f"[bold blue]Downloading model {args.download}...[/bold blue]")
        model_info = discovery.download_model(args.download, args.type)
        if model_info:
            rprint(f"[bold green]Successfully downloaded {args.download}[/bold green]")
            rprint(f"Model saved to: {model_info.path}")
        else:
            rprint(f"[bold red]Failed to download {args.download}[/bold red]")
    
    if args.list or not args.download:
        rprint("[bold blue]Discovering available models...[/bold blue]")
        display_models(discovery)

if __name__ == "__main__":
    main() 