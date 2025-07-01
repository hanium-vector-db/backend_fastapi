import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.llm_handler import LLMHandler
from models.embedding_handler import EmbeddingHandler
from services.rag_service import RAGService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Request models
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 512

class EmbeddingRequest(BaseModel):
    text: str

class RAGRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    message: str

# Initialize handlers (lazy loading)
llm_handler = None
embedding_handler = None
rag_service = None

def get_llm_handler():
    global llm_handler
    if llm_handler is None:
        logger.info("Initializing LLM handler...")
        llm_handler = LLMHandler()
    return llm_handler

def get_embedding_handler():
    global embedding_handler
    if embedding_handler is None:
        logger.info("Initializing embedding handler...")
        embedding_handler = EmbeddingHandler()
    return embedding_handler

def get_rag_service():
    global rag_service
    if rag_service is None:
        logger.info("Initializing RAG service...")
        llm = get_llm_handler()
        emb = get_embedding_handler()
        rag_service = RAGService(llm, emb)
    return rag_service

@router.post("/generate")
async def generate_response(request: GenerateRequest):
    try:
        handler = get_llm_handler()
        response = handler.generate(request.prompt)
        return {"response": response, "prompt": request.prompt}
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_response(request: ChatRequest):
    try:
        handler = get_llm_handler()
        response = handler.chat_generate(request.message)
        return {"response": response, "message": request.message}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed")
async def create_embedding(request: EmbeddingRequest):
    try:
        handler = get_embedding_handler()
        embedding = handler.create_embedding(request.text)
        return {"embedding": embedding, "text": request.text, "dimension": len(embedding)}
    except Exception as e:
        logger.error(f"Error in embed endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag")
async def rag_response(request: RAGRequest):
    try:
        service = get_rag_service()
        response = service.generate_response(request.question)
        relevant_docs = service.get_relevant_documents(request.question)
        return {
            "response": response, 
            "question": request.question,
            "relevant_documents": relevant_docs
        }
    except Exception as e:
        logger.error(f"Error in RAG endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_loaded": llm_handler is not None,
        "embedding_loaded": embedding_handler is not None,
        "rag_loaded": rag_service is not None
    }