from config.logger import get_logger
from config.settings import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

logger = get_logger(__name__)
settings = Settings()


class LLMStrategy:
    def generate(self, prompt: str):
        raise NotImplementedError


class HuggingFaceLLM(LLMStrategy):
    def __init__(self, model_id: str = None):
        self.model_id = model_id
        logger.info(f"Loading model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto"
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt: str) -> str:
        logger.info(f"Generating response..")
        result = self.generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]

class LLMFactory:
    @staticmethod
    def get_llm(engine: str = "huggingface"):
        if engine == "huggingface":
            return HuggingFaceLLM(model_id=settings.LLM_MODEL)
        else:
            raise ValueError(f"Unsupported LLM engine: {engine}")
