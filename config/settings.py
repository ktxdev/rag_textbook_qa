import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()


class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 0
        self.EMBED_MODEL = "all-MiniLM-L6-v2"
        self.LLM_MODEL = "google/gemma-3-1b-it"
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
        self.DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
        self.TEXTBOOK_PDF_SAVE_PATH = "./data/Jurafsky_ed3book_Jan25.pdf"
        self.VECTOR_STORE_SAVE_PATH = "./vectorstore/vector_store.index"
        self.TEXTBOOK_URL = "https://web.stanford.edu/~jurafsky/slp3/ed3book_Jan25.pdf"
        self.BASIC_QA_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful assistant. Answer the question strictly using the context below and do not generate additional questions or answers.
            If the answer is not found, say "I could not find the answer in the provided text."
            
            Context:
            {context}

            Question: {question}"""
        )
        self._initialized = True
