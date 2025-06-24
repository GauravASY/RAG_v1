# A multimodal RAG 
A local Retrieval-Augmented Generation (RAG) assistant that answers questions based strictly on the content of uploaded PDF documents. Powered by LangChain, Pinecone, Ollama, and Gradio.
----
## Features
- PDF Upload & Parsing: Upload one or more PDF files for context-aware Q&A.
- RAG Pipeline: Uses vector embeddings and Pinecone for document retrieval.
- LLM Integration: Utilizes Ollama's Llama3 model for generating answers.
- Strict Contextuality: Only answers questions if the answer is found in the provided documents.
- Web UI: User-friendly Gradio chat interface.
- Confidentiality: Local models ensure that your data isn't leaked.

## Setup
### Prerequisites
- Python 3.13+
- Ollama running locally with the llama3 and mxbai-embed-large models pulled.
- Pinecone account and API key.

### Installation
1. Clone the repository:
```powershell
git clone <repo-URL>
cd RAG-Local
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
Or use the provided pyproject.toml with your preferred tool.

3. Set up environment variables:
- Create a .env file in the root directory:
```python
PINECONE_API_KEY = your-pinecone-api-key
```

4. Start Ollama:
- Make sure Ollama is running and the required models are available.

5. Run the app:
```python
uv run main.py
```
The app will be available at http://127.0.0.1:7860.

## Usage
- Upload PDF(s) via the Gradio interface.
- Ask questions in the chat box.
- If the answer is in the documents, you'll get a response. Otherwise, you'll see:
`I can not process the request.`

## Notes 
- The .venv/, .env, and __pycache__/ folders are ignored by git.
- Make sure your Pinecone index is set up and your API key is valid.
- Ollama must be running locally for embedding and chat.
