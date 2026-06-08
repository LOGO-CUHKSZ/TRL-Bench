"""
Generic Downloader - Scrapes dataset landing pages to find download links
"""

import os
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from .base import BaseDownloader, DownloadResult


class GenericDownloader(BaseDownloader):
    """Generic web scraper for dataset landing pages"""

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.verify_ssl = config.get('validate_ssl', False)
        self.allowed_extensions = config.get('allowed_extensions', ['.csv', '.xls', '.xlsx', '.tsv'])

    def download(self, url: str, classification: Dict[str, Any]) -> DownloadResult:
        """
        Scrape a landing page and download the dataset

        Args:
            url: The page URL
            classification: URL classification metadata

        Returns:
            DownloadResult object
        """
        try:
            domain = classification.get('domain', 'unknown')

            # Try to find download link
            download_url, file_type = self._find_download_link(url)

            if not download_url:
                self.logger.warning(f"No download link found on page: {url}")
                return DownloadResult(
                    success=False,
                    url=url,
                    error="No download link found"
                )

            # Determine file extension
            extension = file_type or '.csv'

            # Generate file paths
            file_path, metadata_path = self._generate_file_path(url, domain, extension)

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

            self.logger.info(f"Found download link: {download_url}")
            self.logger.info(f"Downloading from landing page: {url}")

            # Download the file
            file_size = self._download_file(download_url, file_path)

            # Create and save metadata
            metadata = self._create_metadata(url, file_path, {
                'domain': domain,
                'download_method': 'scraped',
                'download_url': download_url,
                'extension': extension
            })

            self._save_metadata(metadata, metadata_path)

            self.logger.info(f"Successfully downloaded from landing page: {file_path}")

            return DownloadResult(
                success=True,
                url=url,
                file_path=file_path,
                metadata_path=metadata_path,
                file_size=file_size,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Failed to download from landing page {url}: {str(e)}")
            return DownloadResult(
                success=False,
                url=url,
                error=str(e)
            )

    def _find_download_link(self, url: str) -> tuple:
        """
        Find download link on a page

        Args:
            url: Page URL

        Returns:
            Tuple of (download_url, file_type)
        """
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                verify=self.verify_ssl,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )

            if response.status_code != 200:
                return None, None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Strategy 1: Look for direct links to CSV/Excel files
            download_links = self._find_file_links(soup, url)
            if download_links:
                return download_links[0]  # Return the first match

            # Strategy 2: Look for download buttons/links
            download_url = self._find_download_button(soup, url)
            if download_url:
                return download_url, '.csv'

            # Strategy 3: Look for API endpoints
            api_url = self._find_api_endpoint(soup, url)
            if api_url:
                return api_url, '.csv'

        except Exception as e:
            self.logger.warning(f"Error finding download link: {e}")

        return None, None

    def _find_file_links(self, soup: BeautifulSoup, base_url: str) -> List[tuple]:
        """
        Find direct links to data files

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for relative links

        Returns:
            List of (url, extension) tuples
        """
        links = []

        for link in soup.find_all('a', href=True):
            href = link['href']

            # Check if link points to a data file
            for ext in self.allowed_extensions:
                if ext in href.lower():
                    # Make absolute URL
                    if not href.startswith('http'):
                        href = urljoin(base_url, href)

                    links.append((href, ext))
                    break

        return links

    def _find_download_button(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """
        Find download button or link

        Args:
            soup: BeautifulSoup object
            base_url: Base URL

        Returns:
            Download URL or None
        """
        # Look for common download button patterns
        download_keywords = ['download', 'export', 'csv', 'excel', 'data']

        # Check buttons and links
        for element in soup.find_all(['a', 'button']):
            # Check text content
            text = element.get_text().lower()
            if any(keyword in text for keyword in download_keywords):
                href = element.get('href') or element.get('data-url')
                if href:
                    if not href.startswith('http'):
                        href = urljoin(base_url, href)
                    return href

            # Check class names
            class_names = ' '.join(element.get('class', [])).lower()
            if any(keyword in class_names for keyword in download_keywords):
                href = element.get('href') or element.get('data-url')
                if href:
                    if not href.startswith('http'):
                        href = urljoin(base_url, href)
                    return href

        return None

    def _find_api_endpoint(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """
        Look for API endpoints in page content

        Args:
            soup: BeautifulSoup object
            base_url: Base URL

        Returns:
            API URL or None
        """
        # Look for API URLs in script tags or data attributes
        for script in soup.find_all('script'):
            if script.string:
                # Look for common API patterns
                if 'api' in script.string.lower() and '.csv' in script.string.lower():
                    # Try to extract URL
                    import re
                    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+\.csv', script.string)
                    if urls:
                        return urls[0]

        return None

    def _download_file(self, download_url: str, file_path: str) -> int:
        """
        Download file from URL

        Args:
            download_url: URL to download from
            file_path: Path to save file

        Returns:
            File size in bytes
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(
            download_url,
            stream=True,
            timeout=self.timeout,
            verify=self.verify_ssl,
            headers=headers
        )
        response.raise_for_status()

        # Download file in chunks
        total_size = 0
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

                    if total_size > self.max_file_size:
                        f.close()
                        os.remove(file_path)
                        raise ValueError(f"File exceeds size limit")

        return total_size
