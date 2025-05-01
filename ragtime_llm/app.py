def add_playlist(playlist_url: str) -> str:
    """Add all videos from a YouTube playlist to the knowledge base.
    
    Args:
        playlist_url: URL of the YouTube playlist
        
    Returns:
        Status message
    """
    try:
        # Process playlist
        results = video_processor.process_playlist(playlist_url)
        
        if not results:
            return "No videos were successfully processed from the playlist."
        
        # Add videos to knowledge base
        for result in results:
            metadata = result["metadata"]
            video_id = metadata["video_id"]
            
            # Add to vector store
            vector_store.add_video(video_id, metadata)
            
        return f"Successfully added {len(results)} videos from the playlist to the knowledge base."
        
    except Exception as e:
        logger.error(f"Error adding playlist: {e}")
        return f"Error adding playlist: {str(e)}"

# Update Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAGtime LLM")
    
    with gr.Row():
        with gr.Column():
            video_url = gr.Textbox(label="YouTube Video URL")
            add_video_btn = gr.Button("Add Video")
            video_status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            playlist_url = gr.Textbox(label="YouTube Playlist URL")
            add_playlist_btn = gr.Button("Add Playlist")
            playlist_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Row():
        query = gr.Textbox(label="Ask a question")
        submit_btn = gr.Button("Submit")
    
    response = gr.Textbox(label="Response", interactive=False)
    
    # Add event handlers
    add_video_btn.click(
        fn=add_video,
        inputs=[video_url],
        outputs=[video_status]
    )
    
    add_playlist_btn.click(
        fn=add_playlist,
        inputs=[playlist_url],
        outputs=[playlist_status]
    )
    
    submit_btn.click(
        fn=generate_response,
        inputs=[query],
        outputs=[response]
    ) 