# 🦙 LangChain + Ollama + Streamlit RAG App

This project is a **local Retrieval-Augmented Generation (RAG) system** that allows you to **chat with your own PDF documents** through a beautiful **Streamlit web interface** — fully offline and privacy-friendly.  
It uses **Ollama** to run open-source Large Language Models (LLMs) such as **Llama 3.1:8B** and **nomic-embed-text** locally, without any API key or cost.

---

## ✨ Features

- **Conversational Q&A:** Ask questions directly about your uploaded or pre-indexed PDFs.  
- **Source References:** Each answer includes references to document names and page numbers.  
- **Conversational Memory:** Follow-up questions maintain context naturally.  
- **Completely Local:** No external API calls — your data stays 100% on your device.  
- **Easy Index Management:** Build, inspect, or append PDFs to a persistent FAISS vector store.  
- **Retriever Comparison:** Compare basic retriever vs. Multi-Query Retriever for deeper search.

---

## ⚙️ Architecture Overview

The system follows a **RAG (Retrieval-Augmented Generation)** pipeline:

1. **Load & Split PDFs:** All PDFs inside `/Dataset/pdfs` are loaded and chunked using `RecursiveCharacterTextSplitter`.  
2. **Embed & Index:** `nomic-embed-text` converts each text chunk into embeddings, stored in a FAISS vector index.  
3. **Retrieve:** When you ask a question, the system retrieves the most semantically relevant chunks.  
4. **Generate:** `llama3.1` uses LangChain’s context-aware chain to produce grounded answers.  
5. **Compare:** Multi-Query retriever expands the question to test retrieval diversity.

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Ollama** | Local LLM runtime for Llama 3.1 and Embedding models |
| **LangChain** | RAG orchestration, document loaders, retrievers |
| **Streamlit** | Web interface for chat, upload, and index management |
| **FAISS** | Vector store for fast semantic search |
| **Docker Compose** | One-command setup for all services |

---
