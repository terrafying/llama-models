"""
Web interface for RAG-LLM system using Gradio.
"""

import os
import gradio as gr
from typing import Optional, Dict, List
import json
from pathlib import Path

from ragtime_llm.utils.logger import logger
from ragtime_llm.utils.ui_utils import get_emoji
from ragtime_llm.video_processor import process_video, process_playlist
from ragtime_llm.utils.storage_manager import StorageManager

# Initialize storage manager
storage_manager = StorageManager()

# Auto-discover available models and resources
def discover_resources():
    """Discover available models and resources."""
    resources = {
        'models': [],
        'videos': [],
        'playlists': []
    }
    
    # Discover models
    models_dir = Path('models')
    if models_dir.exists():
        resources['models'] = [f.name for f in models_dir.glob('*.pt')]
    
    # Discover processed videos
    videos_dir = Path('output/videos')
    if videos_dir.exists():
        resources['videos'] = [f.name for f in videos_dir.glob('*.mp4')]
    
    # Discover playlists
    playlists_dir = Path('output/playlists')
    if playlists_dir.exists():
        resources['playlists'] = [f.name for f in playlists_dir.glob('*.json')]
    
    return resources

def create_interface():
    """Create Gradio interface."""
    # Discover available resources
    resources = discover_resources()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .gradio-interface {
        max-width: 1200px;
        margin: 0 auto;
    }
    .gradio-input {
        border-radius: 8px;
    }
    .gradio-output {
        border-radius: 8px;
    }
    .gradio-button {
        border-radius: 8px;
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .gradio-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .gradio-button:active {
        transform: translateY(0);
    }
    .gradio-tab {
        border-radius: 8px;
        padding: 20px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .gradio-tab.active {
        background: #f8f9fa;
    }
    .gradio-markdown {
        font-size: 16px;
        line-height: 1.6;
    }
    .gradio-markdown h1 {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #2196F3;
    }
    .gradio-markdown h2 {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #21CBF3;
    }
    .gradio-markdown p {
        margin-bottom: 10px;
    }
    .gradio-markdown code {
        background: #f8f9fa;
        padding: 2px 4px;
        border-radius: 4px;
        font-family: 'Fira Code', monospace;
    }
    .gradio-markdown pre {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
    }
    .gradio-markdown pre code {
        background: none;
        padding: 0;
    }
    .gradio-markdown blockquote {
        border-left: 4px solid #2196F3;
        padding-left: 15px;
        margin-left: 0;
        color: #666;
    }
    .gradio-markdown ul, .gradio-markdown ol {
        margin-left: 20px;
        margin-bottom: 10px;
    }
    .gradio-markdown li {
        margin-bottom: 5px;
    }
    .gradio-markdown a {
        color: #2196F3;
        text-decoration: none;
    }
    .gradio-markdown a:hover {
        text-decoration: underline;
    }
    .gradio-markdown img {
        max-width: 100%;
        border-radius: 8px;
        margin: 10px 0;
    }
    .gradio-markdown table {
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
    }
    .gradio-markdown th, .gradio-markdown td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .gradio-markdown th {
        background: #f8f9fa;
        font-weight: bold;
    }
    .gradio-markdown tr:nth-child(even) {
        background: #f8f9fa;
    }
    .gradio-markdown tr:hover {
        background: #f1f1f1;
    }
    """

    # Create interface
    with gr.Blocks(css=custom_css) as interface:
        gr.Markdown(f"# {get_emoji('ai')} RAG-LLM System")
        
        with gr.Tabs():
            # Video Processing Tab
            with gr.Tab(f"{get_emoji('video')} Process Video"):
                with gr.Row():
                    with gr.Column():
                        video_url = gr.Textbox(
                            label="YouTube Video URL",
                            placeholder="Enter YouTube video URL..."
                        )
                        # Add model selection if available
                        if resources['models']:
                            model_dropdown = gr.Dropdown(
                                choices=resources['models'],
                                label="Select Model",
                                value=resources['models'][0] if resources['models'] else None
                            )
                        max_tokens = gr.Slider(
                            minimum=100,
                            maximum=4000,
                            value=1000,
                            step=100,
                            label="Max Tokens"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        process_btn = gr.Button("Process Video")
                    
                    with gr.Column():
                        output = gr.Markdown(label="Processing Results")
                        # Add processed videos list if available
                        if resources['videos']:
                            gr.Markdown("### Recently Processed Videos")
                            for video in resources['videos'][-5:]:  # Show last 5
                                gr.Markdown(f"- {video}")
            
            # Playlist Processing Tab
            with gr.Tab(f"{get_emoji('video')} Process Playlist"):
                with gr.Row():
                    with gr.Column():
                        playlist_url = gr.Textbox(
                            label="YouTube Playlist URL",
                            placeholder="Enter YouTube playlist URL..."
                        )
                        # Add model selection if available
                        if resources['models']:
                            playlist_model_dropdown = gr.Dropdown(
                                choices=resources['models'],
                                label="Select Model",
                                value=resources['models'][0] if resources['models'] else None
                            )
                        max_videos = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=10,
                            step=1,
                            label="Max Videos"
                        )
                        playlist_max_tokens = gr.Slider(
                            minimum=100,
                            maximum=4000,
                            value=1000,
                            step=100,
                            label="Max Tokens"
                        )
                        playlist_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        process_playlist_btn = gr.Button("Process Playlist")
                    
                    with gr.Column():
                        playlist_output = gr.Markdown(label="Processing Results")
                        # Add processed playlists list if available
                        if resources['playlists']:
                            gr.Markdown("### Recently Processed Playlists")
                            for playlist in resources['playlists'][-5:]:  # Show last 5
                                gr.Markdown(f"- {playlist}")
            
            # Query Tab
            with gr.Tab(f"{get_emoji('search')} Query"):
                with gr.Row():
                    with gr.Column():
                        query_text = gr.Textbox(
                            label="Query",
                            placeholder="Enter your query..."
                        )
                        query_max_tokens = gr.Slider(
                            minimum=100,
                            maximum=4000,
                            value=1000,
                            step=100,
                            label="Max Tokens"
                        )
                        query_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        query_btn = gr.Button("Submit Query")
                    
                    with gr.Column():
                        query_output = gr.Markdown(label="Query Results")
            
            # Generate Video Tab
            with gr.Tab(f"{get_emoji('video')} Generate Video"):
                with gr.Row():
                    with gr.Column():
                        generate_query = gr.Textbox(
                            label="Query",
                            placeholder="Enter query for video generation..."
                        )
                        generate_max_tokens = gr.Slider(
                            minimum=100,
                            maximum=4000,
                            value=1000,
                            step=100,
                            label="Max Tokens"
                        )
                        generate_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        generate_btn = gr.Button("Generate Video")
                    
                    with gr.Column():
                        video_output = gr.Video(label="Generated Video")
                        generation_output = gr.Markdown(label="Generation Details")
            
            # Stats Tab
            with gr.Tab(f"{get_emoji('storage')} Statistics"):
                stats_btn = gr.Button("Refresh Statistics")
                stats_output = gr.Markdown(label="System Statistics")
        
        # Event handlers
        def process_video_handler(url, max_tokens, temperature):
            try:
                result = process_video(url, max_tokens, temperature)
                return f"""
                ### {get_emoji('success')} Video Processed Successfully
                
                **Title:** {result['title']}
                **Duration:** {result['duration']}
                
                #### Storage Information
                {result['storage_info']}
                
                #### Transcript Information
                {result['transcript_info']}
                """
            except Exception as e:
                return f"""
                ### {get_emoji('error')} Error Processing Video
                
                {str(e)}
                """
        
        def process_playlist_handler(url, max_videos, max_tokens, temperature):
            try:
                results = process_playlist(url, max_videos, max_tokens, temperature)
                output = f"""
                ### {get_emoji('success')} Playlist Processed Successfully
                
                **Total Videos:** {len(results)}
                
                #### Processing Results
                """
                
                for result in results:
                    output += f"""
                    - **{result['title']}**
                      - Duration: {result['duration']}
                      - Status: {result['status']}
                    """
                
                return output
            except Exception as e:
                return f"""
                ### {get_emoji('error')} Error Processing Playlist
                
                {str(e)}
                """
        
        def query_handler(query, max_tokens, temperature):
            try:
                result = query(query, max_tokens, temperature)
                return f"""
                ### {get_emoji('success')} Query Results
                
                {result}
                """
            except Exception as e:
                return f"""
                ### {get_emoji('error')} Error Processing Query
                
                {str(e)}
                """
        
        def generate_video_handler(query, max_tokens, temperature):
            try:
                result = generate_video(query, max_tokens, temperature)
                return (
                    result['output_path'],
                    f"""
                    ### {get_emoji('success')} Video Generated Successfully
                    
                    **Output Path:** {result['output_path']}
                    **Duration:** {result['duration']}
                    **Size:** {result['size']}
                    """
                )
            except Exception as e:
                return (
                    None,
                    f"""
                    ### {get_emoji('error')} Error Generating Video
                    
                    {str(e)}
                    """
                )
        
        def stats_handler():
            try:
                stats = storage_manager.get_storage_stats()
                return f"""
                ### {get_emoji('storage')} System Statistics
                
                #### Storage Statistics
                - **IPFS:** {stats['storage']['ipfs']['file_count']} files, {stats['storage']['ipfs']['total_size']} bytes
                - **Local:** {stats['storage']['local']['file_count']} files, {stats['storage']['local']['total_size']} bytes
                - **Cache:** {stats['storage']['cache']['file_count']} files, {stats['storage']['cache']['total_size']} bytes
                
                #### Index Statistics
                - **Total Content:** {stats['index']['total_content']}
                - **Content Types:** {', '.join(f'{k}: {v}' for k, v in stats['index']['content_types'].items())}
                - **Total References:** {stats['index']['total_references']}
                """
            except Exception as e:
                return f"""
                ### {get_emoji('error')} Error Getting Statistics
                
                {str(e)}
                """
        
        # Connect event handlers
        process_btn.click(
            process_video_handler,
            inputs=[video_url, max_tokens, temperature],
            outputs=output
        )
        
        process_playlist_btn.click(
            process_playlist_handler,
            inputs=[playlist_url, max_videos, playlist_max_tokens, playlist_temperature],
            outputs=playlist_output
        )
        
        query_btn.click(
            query_handler,
            inputs=[query_text, query_max_tokens, query_temperature],
            outputs=query_output
        )
        
        generate_btn.click(
            generate_video_handler,
            inputs=[generate_query, generate_max_tokens, generate_temperature],
            outputs=[video_output, generation_output]
        )
        
        stats_btn.click(
            stats_handler,
            inputs=[],
            outputs=stats_output
        )
    
    return interface

def launch_interface(share: bool = False):
    """Launch the web interface."""
    interface = create_interface()
    interface.launch(share=share)

if __name__ == "__main__":
    launch_interface() 