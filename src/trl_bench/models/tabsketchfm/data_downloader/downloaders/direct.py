"""
Direct Downloader - Downloads CSV and Excel files directly from URLs
"""

import os
import requests
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseDownloader, DownloadResult


class DirectDownloader(BaseDownloader):
    """Downloads files directly from URLs"""

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.chunk_size = config.get('chunk_size', 8192)
        self.verify_ssl = config.get('validate_ssl', False)

    def download(self, url: str, classification: Dict[str, Any]) -> DownloadResult:
        """
        Download a file directly from URL

        Args:
            url: The URL to download from
            classification: URL classification metadata

        Returns:
            DownloadResult object
        """
        try:
            domain = classification.get('domain', 'unknown')
            extension = classification.get('extension', '.csv')

            # Use the fixed URL from classification if available
            fixed_url = classification.get('url', url)

            # Generate file paths
            file_path, metadata_path = self._generate_file_path(fixed_url, domain, extension)

            # Check if file already exists
            if self._check_file_exists(file_path):
                self.logger.info(f"File already exists, skipping: {file_path}")
                return DownloadResult(
                    success=True,
                    url=url,
                    file_path=file_path,
                    metadata_path=metadata_path,
                    skipped=True,
                    skip_reason="file_exists"
                )

            # Download the file
            self.logger.info(f"Downloading: {fixed_url}")
            file_size = self._download_file_with_retry(fixed_url, file_path)

            # Create and save metadata
            metadata = self._create_metadata(fixed_url, file_path, {
                'domain': domain,
                'extension': extension,
                'download_method': 'direct',
                'original_url': url if url != fixed_url else None
            })

            self._save_metadata(metadata, metadata_path)

            self.logger.info(f"Successfully downloaded: {file_path} ({file_size} bytes)")

            return DownloadResult(
                success=True,
                url=fixed_url,
                file_path=file_path,
                metadata_path=metadata_path,
                file_size=file_size,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Failed to download {url}: {str(e)}")
            return DownloadResult(
                success=False,
                url=url,
                error=str(e)
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, IOError))
    )
    def _download_file_with_retry(self, url: str, file_path: str) -> int:
        """
        Download file with retry logic

        Args:
            url: URL to download from
            file_path: Path to save file

        Returns:
            File size in bytes

        Raises:
            Exception if download fails after retries
        """
        # Make request with streaming
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(
            url,
            stream=True,
            timeout=self.timeout,
            verify=self.verify_ssl,
            headers=headers
        )
        response.raise_for_status()

        # Check file size before downloading
        content_length = response.headers.get('content-length')
        if content_length:
            file_size = int(content_length)
            if file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")

        # Download file in chunks
        total_size = 0
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    total_size += len(chunk)

                    # Check size limit while downloading
                    if total_size > self.max_file_size:
                        # Remove partial file
                        f.close()
                        os.remove(file_path)
                        raise ValueError(f"File exceeds size limit during download")

        return total_size
