# Documentation: Minimal Retrieval-Augmented Generation (RAG) Web Application

## Introduction
This application is a streamlined Retrieval-Augmented Generation (RAG) system designed to provide accurate answers based strictly on user-uploaded documents. By combining document processing, vector embeddings, and Large Language Model (LLM) generation, the system ensures that responses are grounded in provided data, minimizing hallucinations and providing a reliable tool for document interrogation.

## Technology Stack
The application is built using a modern, lightweight, and efficient stack:

### Backend
- **FastAPI**: A high-performance web framework for building APIs with Python.
- **Cohere API**: Utilized for high-quality text embeddings (`embed-english-v3.0`) and advanced text generation (`command-r`).
- **PyPDF**: Used for extracting textual content from PDF files.
- **LangChain Text Splitters**: Employed to divide large documents into manageable, overlapping chunks.
- **NumPy**: used for mathematical operations, specifically calculating cosine similarity for vector retrieval.
- **Python-Dotenv**: Manages environment variables such as API keys.

### Frontend
- **HTML5/CSS3**: Provides a clean, responsive, and professional user interface.
- **Vanilla JavaScript**: Handles asynchronous API calls (Fetch API) and dynamic UI updates without the overhead of heavy frameworks.

## Core Logic and Workflow
The application follows a standard RAG pipeline divided into two main phases:

### phase 1: Document Processing and Indexing
1. **Upload**: The user uploads a PDF or TXT file through the web interface.
2. **extraction**: The system reads the file content. For PDFs, it iterates through pages to collect all text.
3. **Chunking**: The text is split into chunks of 500 characters with a 50-character overlap. This ensures that context is preserved between consecutive segments while maintaining a high degree of granularity.
4. **Embedding**: Each chunk is sent to the Cohere API to generate a high-dimensional vector representation (embedding).
5. **Storage**: The embeddings and original text chunks are stored in an in-memory vector store for the duration of the session.

### Phase 2: Retrieval and Generation
1. **Query Input**: The user submits a question via the chat interface.
2. **Query Embedding**: The question is converted into a vector using the same embedding model.
3. **Similarity Search**: The system calculates the cosine similarity between the query vector and all stored document vectors.
4. **Retrieval**: The top 10 most relevant chunks (those with the highest similarity score) are retrieved to provide comprehensive context coverage.
5. **Augmentation**: The retrieved context is combined with the original query into a structured prompt.
6. **Generation**: The LLM (Cohere command-r) processes the prompt and generates a response based strictly on the provided context.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- A valid Cohere API Key

### Installation Steps
1. Navigate to the project directory.
2. Install the required dependencies using pip:
   ```bash
   pip install fastapi uvicorn cohere pypdf langchain-text-splitters python-dotenv numpy
   ```
3. Create a `.env` file in the root directory and add your Cohere API key:
   ```text
   COHERE_API_KEY=your_api_key_here
   ```

### Running the Application
1. Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn main:app --reload
   ```
2. Open your web browser and navigate to:
   `http://localhost:8000`

## Key Functions

### Backend (main.py)
- `extract_text(file)`: Identifies the file type and extracts raw text from PDF or TXT files.
- `get_embeddings(texts)`: Processes text chunks in batches and retrieves vector embeddings from Cohere.
- `upload_document()`: The primary endpoint for file processing, chunking, and indexing.
- `ask_question()`: The primary endpoint for handles query embedding, vector search, and LLM response generation.

### Frontend (script.js)
- `handleFiles()`: Manages file selection and drag-and-drop feedback.
- `uploadBtn.onclick`: Handles the asynchronous upload and document processing notification.
- `askQuestion()`: Manages the chat interaction, sending queries to the backend and displaying assistant responses.

## Usage Guide
1. **Upload**: Click the upload area or drag a file (PDF or TXT) into the box.
2. **Process**: Click "Upload & Process". Wait for the success message.
3. **Questioning**: Type your question into the input field at the bottom.
4. **Result**: The assistant will provide an answer derived only from the document. If the answer is not present, the system will explicitly state: "Answer not found in document".
