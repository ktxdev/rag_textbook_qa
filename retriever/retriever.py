from config.logger import get_logger
from abc import ABC, abstractmethod

logger = get_logger(__name__)


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, vector_store, query: str):
        pass


class SimilarityRetriever(Retriever):
    def retrieve(self, vector_store, query: str, k: int = 10):
        logger.info(f"Retrieving top {k} similarities...")
        return vector_store.similarity_search(query, k=k)
