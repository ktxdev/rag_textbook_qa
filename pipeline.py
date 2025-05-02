import os
import re

from huggingface_hub import login
from config.logger import get_logger
from config.settings import Settings
from generation.llm import LLMFactory
from data.downloader import PDFDownloader
from preprocessing.extractor import PDFExtractor
from vectorstore.embedder import EmbeddingFactory
from vectorstore.storage import VectorStoreFactory
from retriever.retriever import SimilarityRetriever

settings = Settings()
logger = get_logger(__name__)

login(token=settings.HF_TOKEN)


class Pipeline:
    def __init__(self):
        if not os.path.exists(settings.TEXTBOOK_PDF_SAVE_PATH):
            PDFDownloader().download(settings.TEXTBOOK_URL, settings.TEXTBOOK_PDF_SAVE_PATH)

        if os.path.exists(settings.VECTOR_STORE_SAVE_PATH):
            logger.info("Loading existing vector store...")
            self.vector_store = VectorStoreFactory.load(settings.VECTOR_STORE_SAVE_PATH)
        else:
            logger.info("Creating new vector store...")
            embedder = EmbeddingFactory.create()
            chunks = PDFExtractor().load(settings.TEXTBOOK_PDF_SAVE_PATH)
            self.vector_store = VectorStoreFactory.create(chunks, embedder)

        self.retriever = SimilarityRetriever()
        self.generator = LLMFactory.get_llm()

    def query(self, query, k=10):
        docs = self.retriever.retrieve(self.vector_store, query, k)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = settings.BASIC_QA_PROMPT.format(context=context.strip(), question=query.strip())
        logger.info(f"Generating answer for query: {query}")

        answer = self.generator.generate(prompt)
        answer = answer.replace(prompt.strip(), "").strip()

        match = re.search(r'(?:Answer:\s*)?(.*?)(?=Question:|$)', answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()

        sources = []
        for doc in docs:
            meta = {"page": doc.metadata.get("page_label")}
            if "chapter" in doc.metadata:
                meta["chapter"] = doc.metadata["chapter"]
            sources.append(meta)

        return {"answer": answer, "sources": sources}
