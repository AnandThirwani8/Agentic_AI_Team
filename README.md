# Multi-Agent RAG System 🚀🤖
This repository contains a multi-agent Retrieval-Augmented Generation (RAG) system built with Python. The system processes PDF documents by creating a vector store and then leverages a team of agents to interact with the content. The UI is powered by Gradio, providing an interactive and user-friendly experience.

### Features ✨
- Document Ingestion: Automatically gathers PDF files from the ./data/ directory.
- Vector Store: Processes PDFs to create a searchable vector store.
- Agentic Team: Initializes a team of agents for retrieval and generation tasks.
- Interactive UI: Uses Gradio to launch an intuitive web interface.

### How to Use ? 🛠
1. Set Up:
```bash
git clone https://github.com/yourusername/multi-agent-rag.git
cd multi-agent-rag
pip install -r requirements.txt
```
2. Add Your PDFs: Place your PDF files in the ./data/ directory.
3. Before running the application, ensure you have a .env file in the project root with your 'GOOGLE_API_KEY'.
4. Run the Application:
```bash
python main.py
```