#!/bin/bash

# Core functionality
mv unified_rag_system.py rag_llm_mojo/core/
mv llm_utils.py rag_llm_mojo/core/
mv vector_store.py rag_llm_mojo/core/
mv linguistic_rag.py rag_llm_mojo/core/

# Video processing
mv video_generator.py rag_llm_mojo/video/
mv video_composer.py rag_llm_mojo/video/
mv phonetic_compositor.py rag_llm_mojo/video/
mv phonetic_index.py rag_llm_mojo/video/

# Utils
mv logger.py rag_llm_mojo/utils/
mv youtube_utils.py rag_llm_mojo/utils/

# Move tests to tests directory
mv test_rag.py tests/
mv test_local_video.py tests/
mv test_video_generation.py tests/
mv test_video_generator.py tests/ 