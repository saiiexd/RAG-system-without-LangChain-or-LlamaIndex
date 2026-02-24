import os
import io
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import cohere
from dotenv import load_dotenv

load_dotenv()

# Configure Cohere
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    print("WARNING: COHERE_API_KEY not found in environment. Please set it in .env file.")
    coh = None
else:
    try:
        coh = cohere.ClientV2(api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize Cohere client: {e}")
        coh = None

app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global store for current document
class DocumentStore:
    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

store = DocumentStore()

def extract_text(file: UploadFile) -> str:
    content = ""
    file_ext = file.filename.split(".")[-1].lower()
    
    try:
        if file_ext == "pdf":
            pdf = PdfReader(io.BytesIO(file.file.read()))
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
        elif file_ext == "txt":
            content = file.file.read().decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or TXT.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    return content

def get_embeddings(texts: List[str]) -> np.ndarray:
    if not coh:
        raise HTTPException(status_code=500, detail="Cohere client not initialized. Check API key.")
    
    try:
        all_embeddings = []
        batch_size = 90 # Cohere limit is 96, using 90 to be safe
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = coh.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document",
                embedding_types=["float"]
            )
            all_embeddings.extend(response.embeddings.float)
        return np.array(all_embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

def get_query_embedding(text: str) -> np.ndarray:
    if not coh:
        raise HTTPException(status_code=500, detail="Cohere client not initialized. Check API key.")
    
    try:
        response = coh.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"]
        )
        return np.array(response.embeddings.float[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query embedding error: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global store
    try:
        text = extract_text(file)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document is empty.")

        # Smaller chunks for better granularity
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        store.chunks = text_splitter.split_text(text)
        
        # Generate Embeddings
        print(f"Processing document: {len(store.chunks)} chunks generated.")
        store.embeddings = get_embeddings(store.chunks)
        
        return {"message": f"Document processed. Split into {len(store.chunks)} chunks."}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/ask")
async def ask_question(data: dict):
    question = data.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    
    if not store.chunks or store.embeddings is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    try:
        # 1. Embed question
        q_emb = get_query_embedding(question)
        
        # 2. Similarity search
        norms = np.linalg.norm(store.embeddings, axis=1)
        q_norm = np.linalg.norm(q_emb)
        
        if q_norm == 0 or np.any(norms == 0):
            return {"answer": "Answer not found in document"}
            
        similarities = np.dot(store.embeddings, q_emb) / (norms * q_norm)
        
        # Get top 10 most relevant chunks for better context coverage
        top_k = min(10, len(store.chunks))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Logging for debugging
        print(f"Query: '{question}' | Top Similarity: {max(similarities):.4f}")
        
        context_chunks = [store.chunks[i] for i in top_indices]
        context_text = "\n---\n".join(context_chunks)
        
        # 3. Construct a more helpful prompt
        prompt = f"""You are an expert document assistant. Use the following context to answer the user's question accurately.

RULES:
- Answer ONLY based on the provided context.
- Be thorough and synthesize the information across different chunks if necessary.
- If the answer is not contained within the context at all, respond with: "Answer not found in document".
- If you can partially answer, do so and mention that other details are missing.

CONTEXT:
{context_text}

USER QUESTION: {question}

DETAILED ANSWER:"""

        # 4. Cohere Generation
        try:
            response = coh.chat(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            print(f"Attempting fallback due to: {e}")
            response = coh.chat(
                model="command-r-08-2024",
                messages=[{"role": "user", "content": prompt}]
            )
            
        answer = ""
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            for item in response.message.content:
                if isinstance(item, dict):
                    answer += item.get("text", "")
                else:
                    answer += getattr(item, "text", "")
        else:
            answer = getattr(response, "text", str(response))

        answer = answer.strip()
        if not answer:
            return {"answer": "Answer not found in document"}
            
        return {"answer": answer}

    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the answer.")

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
