# Documentation: Minimal Retrieval-Augmented Generation (RAG) Web Application

## Introduction
This application is a streamlined Retrieval-Augmented Generation (RAG) system designed to provide accurate answers based strictly on user-uploaded documents. By combining document processing, vector embeddings, and Large Language Model (LLM) generation, the system ensures that responses are grounded in provided data, minimizing hallucinations and providing a reliable tool for document interrogation.

## Technology Stack
The application is built using a modern, lightweight, and efficient stack:

### Backend
- **FastAPI**: A high-performance web framework for building APIs with Python.
- **Cohere API**: Utilized for high-quality text embeddings (`embed-english-v3.0`) and advanced text generation (`command-r-plus`).
- **PyPDF**: Used for extracting textual content from PDF files.
- **LangChain Text Splitters**: Employed to divide large documents into manageable, overlapping chunks.
- **NumPy**: used for mathematical operations, specifically calculating cosine similarity for vector retrieval.
- **Python-Dotenv**: Manages environment variables such as API keys.

### Frontend
- **HTML5/CSS3**: Provides a clean, responsive, and professional user interface.
- **Vanilla JavaScript**: Handles asynchronous API calls (Fetch API) and dynamic UI updates.

## Core Logic and Workflow
The application follows a standard RAG pipeline divided into two main phases:

### Phase 1: Document Processing and Indexing
1. **Upload**: The user uploads a PDF or TXT file through the web interface.
2. **Extraction**: The system reads the file content. For PDFs, it iterates through pages to collect all text.
3. **Chunking**: The text is split into chunks of 500 characters with a 50-character overlap.
4. **Embedding**: Each chunk is sent to the Cohere API to generate a vector representation.
5. **Storage**: The embeddings and original text chunks are stored in an in-memory vector store.

### Phase 2: Retrieval and Generation
1. **Query Input**: The user submits a question via the chat interface.
2. **Query Embedding**: The question is converted into a vector.
3. **Similarity Search**: Calculates cosine similarity between the query and stored chunks.
4. **Retrieval**: The top 10 most relevant chunks are retrieved.
5. **Generation**: The LLM (Cohere Command-R) processes the prompt and generates a response based strictly on the provided context.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- A valid Cohere API Key

### Installation Steps
1. Navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Cohere API key:
   ```text
   COHERE_API_KEY=your_api_key_here
   ```

### Running the Application
1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
2. Open your web browser and navigate to:
   `http://localhost:8000`

## Deployment (Vercel)
This project is configured for easy deployment on Vercel:
1. Connect your repository to Vercel.
2. Add `COHERE_API_KEY` to your Vercel project's **Environment Variables**.
3. Deploy! The `vercel.json` and `requirements.txt` will handle the configuration automatically.

## Troubleshooting: Connection Errors
If you see a "Connection error" on the interface, ensure:
1. The backend server is actually running (`uvicorn main:app`).
2. You are visiting `http://localhost:8000` in your browser.
3. **Do NOT** open the `index.html` file directly from your folder (e.g., `file:///C:/.../index.html`). This will block API calls due to browser security restrictions.

## Key Functions

### Backend (main.py)
- `extract_text(file)`: Identifies file type and extracts raw text securely.
- `get_embeddings(texts)`: Processes text chunks and retrieves embeddings from Cohere with graceful fallback mechanisms.
- `upload_document()`: Primary endpoint for file processing and indexing.
- `ask_question()`: Handles retrieval and LLM response generation with fallback to secondary models if needed.
- `health_check()`: GET `/health` endpoint to verify service status.

### Frontend (script.js)
- `handleFiles()`: Manages file selection and drag-and-drop feedback.
- `askQuestion()`: Manages the chat interaction and provides detailed network error feedback.

