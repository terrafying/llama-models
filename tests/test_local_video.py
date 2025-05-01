from ragtime_llm.core.unified_rag_system import YouTubeRAG
import os
import numpy as np

def main():
    # Initialize RAG system
    rag_system = YouTubeRAG()
    
    # Path to the local video
    video_path = os.path.join("downloads", "I Published a Math Paper! [zVRNcRFaZ94].mp4")
    
    # Create basic metadata since we can't get it from YouTube
    metadata = {
        "id": "zVRNcRFaZ94",
        "title": "I Published a Math Paper!",
        "description": "Local video file",
        "author": "Unknown",
        "url": f"https://www.youtube.com/watch?v=zVRNcRFaZ94"
    }
    
    print("Processing local video...")
    
    # Extract video clips
    print("Extracting video clips...")
    video_chunks = rag_system.extract_video_clips(video_path)
    print(f"Extracted {len(video_chunks)} chunks")
    
    if not video_chunks:
        print("No chunks were extracted. Exiting.")
        return
    
    # Process chunks and add to vector store
    all_chunks = []
    all_embeddings = []
    all_metadata = []
    
    for chunk in video_chunks:
        # Compute embedding for transcript
        embedding = rag_system.model.encode([chunk.transcript], convert_to_tensor=True)
        embedding = embedding.cpu().numpy()
        
        all_chunks.append(chunk.transcript)
        all_embeddings.append(embedding[0])
        all_metadata.append({
            **metadata,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "video_path": chunk.video_path
        })
    
    # Add to vector store
    rag_system.vector_store.add_vectors(
        np.array(all_embeddings),
        all_chunks,
        all_metadata,
        video_chunks
    )
    
    print("Video processed and added to vector store")
    
    # Test a query
    test_query = "What is the main topic of the video?"
    print(f"\nTesting query: {test_query}")
    result = rag_system.generate_response(test_query, llm_provider="openai", model="gpt-3.5-turbo")
    
    print("\nResponse:")
    print(result["response"])
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"{i}. Video: {source['title']} (Score: {source['score']:.2f})")

if __name__ == "__main__":
    main() 