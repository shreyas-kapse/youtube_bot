import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from service.RAG_service import RAGService
from service.ingestion_service import IngestionService
from service.qa_service import QaService
from utils.logger import setup_logger
from fastapi.middleware.cors import CORSMiddleware
from langsmith import traceable
from dotenv import load_dotenv
import torch


torch.cuda.empty_cache()
load_dotenv()

logger = logging.getLogger(__name__)

qa_service = None 


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_service

    setup_logger()

    video_id = "hBMoPUAeLnY"
    ingestion_service = IngestionService()
    rag_service = RAGService()

    qa_service = QaService(
        rag_service=rag_service,
        ingestion_service=ingestion_service
    )
    logger.info("App started successfully")
    yield 


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/process")
@traceable
def store_embeddings(video_id:str):
    qa_service.process_video_embeddings(video_id=video_id)
    return {"status":"Processed"}

@app.get("/ask")
@traceable
def ask(query: str, video_id:str):
    return qa_service.answer_question(query=query, video_id=video_id)
