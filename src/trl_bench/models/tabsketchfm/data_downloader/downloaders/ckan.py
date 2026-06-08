"""
CKAN Downloader - Downloads datasets from CKAN portals
"""

import os
import requests
import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse, urljoin

from .base import BaseDownloader, DownloadResult


class CKANDownloader(BaseDownloader):
    """Downloads datasets from CKAN open data portals"""

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.api_key = config.get('api_keys', {}).get('ckan_api_key') or \
                       os.environ.get('CKAN_API_KEY')
        self.verify_ssl = config.get('validate_ssl', False)

    def download(self, url: str, classification: Dict[str, Any]) -> DownloadResult:
        """
        Download a dataset from CKAN

        Args:
            url: The CKAN dataset URL
            classification: URL classification metadata

        Returns:
            DownloadResult object
        """
        try:
            domain = classification.get('domain', 'unknown')
            resource_id = classification.get('resource_id')

            # Generate file paths
            file_path, metadata_path = self._generate_file_path(url, domain, '.csv')

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

            # Get download URL from CKAN
            download_url, ckan_metadata = self._get_ckan_resource_info(url, domain, resource_id)

            if not download_url:
                # Fallback: try to scrape the page
                download_url = self._scrape_download_link(url)

            if not download_url:
                raise ValueError("Could not find download URL for CKAN resource")

            self.logger.info(f"Downloading CKAN resource: {download_url}")

            # Download the file
            file_size = self._download_file(download_url, file_path)

            # Create and save metadata
            metadata = self._create_metadata(url, file_path, {
                'domain': domain,
                'resource_id': resource_id,
                'download_method': 'ckan_api',
                'download_url': download_url,
                'ckan_metadata': ckan_metadata
            })

            self._save_metadata(metadata, metadata_path)

            self.logger.info(f"Successfully downloaded CKAN dataset: {file_path}")

            return DownloadResult(
                success=True,
                url=url,
                file_path=file_path,
                metadata_path=metadata_path,
                file_size=file_size,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Failed to download CKAN dataset {url}: {str(e)}")
            return DownloadResult(
                success=False,
                url=url,
                error=str(e)
            )

    def _get_ckan_resource_info(self, url: str, domain: str, resource_id: Optional[str]) -> tuple:
        """
        Get resource download URL and metadata from CKAN API

        Args:
            url: Original URL
            domain: Domain name
            resource_id: Resource UUID

        Returns:
            Tuple of (download_url, metadata)
        """
        if not resource_id:
            return None, None

        try:
            # Try CKAN API endpoint
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            api_url = urljoin(base_url, f'/api/3/action/resource_show?id={resource_id}')

            headers = {}
            if self.api_key:
                headers['Authorization'] = self.api_key

            response = requests.get(
                api_url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = data.get('result', {})
                    download_url = result.get('url')

                    metadata = {
                        'name': result.get('name'),
                        'description': result.get('description'),
                        'format': result.get('format'),
                        'size': result.get('size'),
                        'created': result.get('created'),
                        'last_modified': result.get('last_modified')
                    }

                    return download_url, metadata

        except Exception as e:
            self.logger.warning(f"Failed to get CKAN resource info via API: {e}")

        return None, None

    def _scrape_download_link(self, url: str) -> Optional[str]:
        """
        Scrape download link from CKAN page

        Args:
            url: CKAN page URL

        Returns:
            Download URL or None
        """
        try:
            from bs4 import BeautifulSoup

            response = requests.get(
                url,
                timeout=self.timeout,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for download links
                # Common patterns in CKAN pages
                download_link = None

                # Try direct CSV/Excel download links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(ext in href.lower() for ext in ['.csv', '.xls', '.xlsx']):
                        download_link = href
                        break

                # Try resource download button
                if not download_link:
                    download_btn = soup.find('a', class_=re.compile(r'resource-url-analytics|btn-primary'))
                    if download_btn and download_btn.get('href'):
                        download_link = download_btn['href']

                # Make absolute URL
                if download_link:
                    parsed = urlparse(url)
                    if not download_link.startswith('http'):
                        base_url = f"{parsed.scheme}://{parsed.netloc}"
                        download_link = urljoin(base_url, download_link)

                    return download_link

        except Exception as e:
            self.logger.warning(f"Failed to scrape CKAN download link: {e}")

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
        if self.api_key:
            headers['Authorization'] = self.api_key

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
