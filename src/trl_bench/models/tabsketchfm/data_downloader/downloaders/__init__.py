"""
Downloader modules for different data source types
"""

from .base import BaseDownloader, DownloadResult
from .direct import DirectDownloader
from .socrata import SocrataDownloader
from .ckan import CKANDownloader
from .generic import GenericDownloader

__all__ = [
    'BaseDownloader',
    'DownloadResult',
    'DirectDownloader',
    'SocrataDownloader',
    'CKANDownloader',
    'GenericDownloader'
]
