"""
Socrata Downloader - Downloads datasets from Socrata portals
"""

import os
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from .base import BaseDownloader, DownloadResult


class SocrataDownloader(BaseDownloader):
    """Downloads datasets from Socrata open data portals"""

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.app_token = config.get('api_keys', {}).get('socrata_app_token') or \
                         os.environ.get('SOCRATA_APP_TOKEN')
        self.verify_ssl = config.get('validate_ssl', False)

    def download(self, url: str, classification: Dict[str, Any]) -> DownloadResult:
        """
        Download a dataset from Socrata

        Args:
            url: The Socrata dataset URL
            classification: URL classification metadata

        Returns:
            DownloadResult object
        """
        try:
            domain = classification.get('domain', 'unknown')
            dataset_id = classification.get('dataset_id')

            if not dataset_id:
                self.logger.warning(f"No dataset ID found for Socrata URL: {url}")
                # Try to download as direct CSV
                return self._download_as_csv(url, domain)

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

            # Try to get metadata from Socrata API
            api_metadata = self._get_socrata_metadata(domain, dataset_id)

            # Download the dataset as CSV
            csv_url = self._get_csv_export_url(url, domain, dataset_id)
            self.logger.info(f"Downloading Socrata dataset: {csv_url}")

            file_size = self._download_csv(csv_url, file_path)

            # Create and save metadata
            metadata = self._create_metadata(url, file_path, {
                'domain': domain,
                'dataset_id': dataset_id,
                'download_method': 'socrata_api',
                'api_metadata': api_metadata
            })

            self._save_metadata(metadata, metadata_path)

            self.logger.info(f"Successfully downloaded Socrata dataset: {file_path}")

            return DownloadResult(
                success=True,
                url=url,
                file_path=file_path,
                metadata_path=metadata_path,
                file_size=file_size,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Failed to download Socrata dataset {url}: {str(e)}")
            return DownloadResult(
                success=False,
                url=url,
                error=str(e)
            )

    def _get_csv_export_url(self, original_url: str, domain: str, dataset_id: str) -> str:
        """
        Generate CSV export URL for Socrata dataset

        Args:
            original_url: Original dataset URL
            domain: Domain name
            dataset_id: Dataset ID (4x4 format)

        Returns:
            CSV export URL
        """
        # Socrata CSV export format
        parsed = urlparse(original_url)
        scheme = parsed.scheme or 'https'

        # Try multiple URL patterns
        csv_urls = [
            f"{scheme}://{domain}/resource/{dataset_id}.csv",
            f"{scheme}://{domain}/api/views/{dataset_id}/rows.csv",
            original_url.replace('/dataset/', '/resource/').rstrip('/') + '.csv'
        ]

        return csv_urls[0]  # Start with the most common format

    def _get_socrata_metadata(self, domain: str, dataset_id: str) -> Optional[Dict]:
        """
        Get metadata from Socrata API

        Args:
            domain: Domain name
            dataset_id: Dataset ID

        Returns:
            Metadata dictionary or None
        """
        try:
            metadata_url = f"https://{domain}/api/views/{dataset_id}.json"
            headers = {}
            if self.app_token:
                headers['X-App-Token'] = self.app_token

            response = requests.get(
                metadata_url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    'name': data.get('name'),
                    'description': data.get('description'),
                    'category': data.get('category'),
                    'tags': data.get('tags', []),
                    'rows': data.get('rowsUpdatedAt'),
                    'columns': [col.get('name') for col in data.get('columns', [])]
                }
        except Exception as e:
            self.logger.warning(f"Failed to get Socrata metadata: {e}")

        return None

    def _download_csv(self, csv_url: str, file_path: str) -> int:
        """
        Download CSV from Socrata

        Args:
            csv_url: CSV export URL
            file_path: Path to save file

        Returns:
            File size in bytes
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        if self.app_token:
            headers['X-App-Token'] = self.app_token

        response = requests.get(
            csv_url,
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

    def _download_as_csv(self, url: str, domain: str) -> DownloadResult:
        """
        Fallback: try to download Socrata URL as direct CSV

        Args:
            url: Socrata URL
            domain: Domain name

        Returns:
            DownloadResult
        """
        # Try adding .csv extension
        csv_url = url.rstrip('/') + '.csv' if not url.endswith('.csv') else url

        file_path, metadata_path = self._generate_file_path(url, domain, '.csv')

        try:
            file_size = self._download_csv(csv_url, file_path)

            metadata = self._create_metadata(url, file_path, {
                'domain': domain,
                'download_method': 'socrata_direct_csv'
            })
            self._save_metadata(metadata, metadata_path)

            return DownloadResult(
                success=True,
                url=url,
                file_path=file_path,
                metadata_path=metadata_path,
                file_size=file_size,
                metadata=metadata
            )
        except Exception as e:
            return DownloadResult(
                success=False,
                url=url,
                error=str(e)
            )
