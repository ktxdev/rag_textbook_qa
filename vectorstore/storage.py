import os
import pickle

from config.logger import get_logger
from langchain_community.vectorstores import FAISS

logger = get_logger(__name__)


class VectorStoreFactory:
    @staticmethod
    def create(chunks, embedder):
        logger.info("Creating vector store...")
        texts = [chunk["text"] for chunk in chunks]
        metadata = [chunk["metadata"] for chunk in chunks]
        return FAISS.from_texts(texts, embedder, metadatas=metadata)

    @staticmethod
    def load(path):
        logger.info("Loading vector store...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at path: {path}")

        with open(path, "rb") as f:
            vector_store = pickle.load(f)

        logger.info("Vector store is loaded from path: {path}")
        return vector_store

    @staticmethod
    def save(path, vector_store):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(vector_store, f)
        logger.info(f"Saved vector store to path: {path}")
