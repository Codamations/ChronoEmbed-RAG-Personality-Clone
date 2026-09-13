# ChronoEmbed-RAG-Personality-Clone
A lightweight local RAG pipeline that cleans and vectorizes Discord conversation logs. It removes metadata noise, extracts key entity anchors, and stores chunked vector representations in a FAISS index for fast semantic memory retrieval and LLM persona synthesis.

Configure API Key:
cp .env.example .env
# Add your OpenAI API Key inside .env
