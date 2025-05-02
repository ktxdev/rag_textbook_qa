from config.settings import Settings
from config.logger import get_logger
from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)


class EmbeddingFactory:
    @staticmethod
    def create():
        settings = Settings()
        logger.info("Creating embedding factory")
        return HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)
