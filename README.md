# 🎓 Lecture Notes AI Tutor For Education

This project is a powerful, locally-run chatbot that allows you to have conversations with your PDF documents. 
It uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers grounded in the content of your lecture notes, textbooks, or any other PDF files.

The entire system runs 100% offline and for free, using the Ollama platform to serve powerful open-source Large Language Models (LLMs) like Llama 3.

## Features

- **Conversational Q&A:** Ask questions about your documents and get direct answers.
- **Source Citations:** Every answer includes references to the specific file and page number it came from.
- **Conversational Memory:** Ask follow-up questions and the chatbot will understand the context of the conversation.
- **Local & Private:** Your documents and conversations never leave your machine. No API keys or fees required.
- **Persistent Knowledge Base:** Documents are indexed only once for fast startup on subsequent uses.

## How It Works

The application uses a RAG (Retrieval-Augmented Generation) architecture:

1.  **Load & Chunk:** All PDF files in the `my_pdf_lectures` directory are loaded and split into manageable text chunks.
2.  **Embed & Index:** A local embedding model (`nomic-embed-text`) converts these chunks into numerical vectors, which are then stored in a searchable FAISS vector index. This index is saved to disk for persistence.
3.  **Retrieve:** When you ask a question, the same embedding model converts your query into a vector. The FAISS index is searched to find the most semantically similar text chunks from your documents.
4.  **Generate:** A powerful LLM (`llama3`) receives your question, the chat history, and the retrieved text chunks. It then generates a coherent answer based *only* on the provided context.

### Prerequisites

You need to have **Ollama** installed and running on your system. This is the platform that serves the local language models.

1.  **Install Ollama:**
    *   Go to the official website: [ollama.com](https://ollama.com/)
    *   Download and run the installer for your operating system (macOS, Windows, or Linux).
    *   Ollama will run as a background service.

2.  **Download the necessary models via Ollama:**
    *   Open your terminal or command prompt.
    *   Pull the main chat model (Llama 3 8B):
        ```bash
        ollama pull llama3
        ```
    *   Pull the embedding model (used for document retrieval):
        ```bash
        ollama pull nomic-embed-text
        ```

  2.  **Create a Conda environment:**
    It's highly recommended to use a virtual environment to manage dependencies. This ensures that the project's libraries don't conflict with other Python projects on your system.
      ```bash
      conda create -n lecture_bot python=3.10 -y
      conda activate lecture_bot
      ```

4.  **Install Python dependencies:**

    ```bash
    pip install langchain langchain-community pypdf faiss-cpu langchain-text-splitters
    ```
    *(Note: `faiss-cpu` is for CPU-based systems. If you have a compatible NVIDIA GPU and CUDA installed, you can use `faiss-gpu` for faster performance).*

## To-Do / Future Improvements

- [ ] Implement a user-friendly web interface using Streamlit.
- [ ] Add support for more document types (e.g., `.docx`, `.pptx`, `.txt`).
- [ ] Evaluate the performance of LLM
