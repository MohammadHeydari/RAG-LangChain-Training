# RAG-LangChain-Learning

This is a **training and experimental project** demonstrating how to build a simple Retrieval-Augmented Generation (RAG) pipeline using:

---

## Project Overview

The project demonstrates:

1. Loading source files or text documents.
2. Chunking text with `RecursiveCharacterTextSplitter`.
3. Creating embeddings with `intfloat/multilingual-e5-base`.
4. Storing embeddings in **FAISS**.
5. Querying the vector store with a **retriever**.
6. Using a **local Ollama model** for answering questions.
7. Connecting everything with a **LCEL chain**.

This setup works **fully offline**, no OpenAI API key required.

---

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
```
# or
```
source .venv/bin/activate      # macOS/Linux
```

# Install dependencies
```
pip install langchain-community requests faiss-cpu sentence-transformers
```