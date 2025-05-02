from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from config.logger import get_logger
from config.settings import Settings

settings = Settings()
logger = get_logger(__name__)


class Reranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-base"):
        logger.debug(f"Loading Reranker model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, device_map="auto")
        self.reranker = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def rerank(self, query, docs: List[str], top_n: int = 3):
        logger.debug(f"Reranking {len(docs)} documents for query: '{query}'")

        pairs = [(query, doc) for doc in docs]
        results = self.reranker(pairs, top_k=1, truncation=True, max_length=512, padding=True)
        scores = [results[0]['score'] for results in results]

        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_n]
