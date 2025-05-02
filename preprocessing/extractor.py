from abc import ABC, abstractmethod
from typing import Dict, Union

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.logger import get_logger
from config.settings import Settings

settings = Settings()
logger = get_logger(__name__)


class TextExtractor(ABC):
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " "],
            length_function=len
        )

    @abstractmethod
    def load(self, file_path: str) -> str:
        pass


def detect_structure(page_content: str) -> Union[None, dict[str, str]]:
    import re

    chapter_pattern = re.compile(r'^\d*\s*CHAPTER\s+\d+\s*[•\-]?\s+.+', re.IGNORECASE)
    section_pattern = re.compile(r'^\d+\.\d+ [A-Za-z].+')  # e.g., "2.1 Regular Expressions"
    subsection_pattern = re.compile(r'^\d+\.\d+\.\d+ [A-Za-z].+')  # e.g., "2.1.1 Basic..."

    lines = page_content.split("\n")
    for line in lines:
        line = line.strip()
        if chapter_pattern.match(line):
            chapter_group_pattern = re.compile(r'^\d*\s*(CHAPTER\s+\d+)\s*[•\-]?\s+(.+)')
            match = chapter_group_pattern.match(line)
            if match:
                line = f"{match.group(1).strip()}: {match.group(2).strip()}"
            return {"type": "chapter", "title": line}
        elif section_pattern.match(line):
            return {"type": "section", "title": line}
        elif subsection_pattern.match(line):
            return {"type": "subsection", "title": line}
        else:
            return None


class PDFExtractor(TextExtractor):
    def load(self, file_path: str):
        logger.info(f"Extracting text from {file_path}")
        pages = PyPDFLoader(file_path).load()

        previous_meta = {"chapter": None, "section": None, "subsection": None}

        for page in pages:
            structure = detect_structure(page.page_content)

            if structure:
                previous_meta[structure["type"]] = structure["title"]

            for key, value in previous_meta.items():
                if value:
                    page.metadata[key] = value

        # Split the document with metadata
        chunks = []
        for page in pages:
            page_text = page.page_content
            page_chunks = self.text_splitter.split_text(page_text)

            for chunk in page_chunks:
                chunk_metadata = page.metadata.copy()  # Copy metadata from the page
                chunks.append({"text": chunk, "metadata": chunk_metadata})

        return chunks
