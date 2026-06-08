"""
Base Downloader - Abstract base class for all downloaders
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DownloadResult:
    """Result of a download attempt"""
    success: bool
    url: str
    file_path: Optional[str] = None
    metadata_path: Optional[str] = None
    error: Optional[str] = None
    file_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class BaseDownloader(ABC):
    """Abstract base class for all downloaders"""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the downloader

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.output_dir = Path(os.path.expanduser(config.get('output_dir', './data')))
        self.metadata_dir = Path(os.path.expanduser(config.get('metadata_dir', './metadata')))
        self.timeout = config.get('timeout', 60)
        self.max_retries = config.get('max_retries', 3)
        self.max_file_size = config.get('max_file_size_mb', 500) * 1024 * 1024  # Convert to bytes

        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self, url: str, classification: Dict[str, Any]) -> DownloadResult:
        """
        Download data from a URL

        Args:
            url: The URL to download from
            classification: URL classification metadata

        Returns:
            DownloadResult object
        """
        pass

    def _generate_file_path(self, url: str, domain: str, extension: str = '.csv') -> tuple:
        """
        Generate file paths for data and metadata

        Args:
            url: Source URL
            domain: Domain name
            extension: File extension

        Returns:
            Tuple of (data_path, metadata_path)
        """
        # Create subdirectory based on domain
        domain_clean = domain.replace('.', '_').replace(':', '_')
        data_subdir = self.output_dir / domain_clean
        metadata_subdir = self.metadata_dir / domain_clean

        data_subdir.mkdir(parents=True, exist_ok=True)
        metadata_subdir.mkdir(parents=True, exist_ok=True)

        # Generate filename from URL
        filename = self._url_to_filename(url, extension)

        data_path = data_subdir / filename
        metadata_path = metadata_subdir / f"{filename}.meta"

        return str(data_path), str(metadata_path)

    def _url_to_filename(self, url: str, extension: str = '.csv') -> str:
        """
        Convert URL to a safe filename

        Args:
            url: Source URL
            extension: File extension

        Returns:
            Safe filename
        """
        from urllib.parse import urlparse
        import hashlib

        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]

        # Try to use the last part of the path as filename
        if path_parts:
            filename = path_parts[-1]

            # Clean the filename
            filename = filename.split('?')[0]  # Remove query params

            # If it already has an extension, use it
            if '.' in filename:
                return self._sanitize_filename(filename)

        # Otherwise, create a filename from hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

        if path_parts:
            base_name = path_parts[-1].split('?')[0]
            base_name = self._sanitize_filename(base_name)
            if len(base_name) > 50:
                base_name = base_name[:50]
            filename = f"{base_name}_{url_hash}{extension}"
        else:
            filename = f"data_{url_hash}{extension}"

        return filename

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove invalid characters

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove multiple underscores
        filename = re.sub(r'_+', '_', filename)
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        return filename

    def _check_file_exists(self, file_path: str) -> bool:
        """
        Check if file already exists and is valid

        Args:
            file_path: Path to check

        Returns:
            True if file exists and is valid
        """
        if not os.path.exists(file_path):
            return False

        # Check if file is not empty
        if os.path.getsize(file_path) == 0:
            self.logger.warning(f"File exists but is empty: {file_path}")
            return False

        return True

    def _create_metadata(self, url: str, file_path: str, additional_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create metadata dictionary

        Args:
            url: Source URL
            file_path: Path to downloaded file
            additional_metadata: Additional metadata to include

        Returns:
            Metadata dictionary
        """
        import datetime

        metadata = {
            'source_url': url,
            'download_date': datetime.datetime.now().isoformat(),
            'file_path': file_path,
        }

        # Add file size if file exists
        if os.path.exists(file_path):
            metadata['file_size'] = os.path.getsize(file_path)

        # Add additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    def _save_metadata(self, metadata: Dict[str, Any], metadata_path: str) -> None:
        """
        Save metadata to file

        Args:
            metadata: Metadata dictionary
            metadata_path: Path to save metadata
        """
        import json

        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Metadata saved to {metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata to {metadata_path}: {e}")
