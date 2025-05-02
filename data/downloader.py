import requests

from config.logger import get_logger

logger = get_logger(__name__)

class DownloadStrategy:
    def download(self, url: str, save_path: str):
        raise NotImplementedError

class PDFDownloader(DownloadStrategy):
    def download(self, url: str, save_path: str):
        logger.info(f"Starting download from {url}")
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            logger.error(f"Download failed with status code: {response.status_code}")
            raise RuntimeError(f"Download failed: {response.status_code}")

        with open(save_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Downloaded textbook to {save_path}")