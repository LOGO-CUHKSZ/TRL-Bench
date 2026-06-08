"""
URL Classifier - Detects the type of data source from a URL
"""

import re
from enum import Enum
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any


class URLType(Enum):
    """Types of data sources"""
    DIRECT_CSV = "direct_csv"
    DIRECT_EXCEL = "direct_excel"
    SOCRATA_DATASET = "socrata_dataset"
    SOCRATA_API = "socrata_api"
    CKAN_DATASET = "ckan_dataset"
    CKAN_API = "ckan_api"
    FTP = "ftp"
    GENERIC_PAGE = "generic_page"
    INVALID = "invalid"


class URLClassifier:
    """Classifies URLs to determine the appropriate download strategy"""

    # File extension patterns
    CSV_EXTENSIONS = ['.csv', '.tsv', '.txt']
    EXCEL_EXTENSIONS = ['.xls', '.xlsx', '.xlsm']

    # Domain patterns
    SOCRATA_DOMAINS = ['data.socrata.com', '.socrata.com']
    CKAN_DOMAINS = ['ckan', 'ckanhosted.com']

    # URL patterns
    SOCRATA_DATASET_PATTERN = r'/dataset/[^/]+/[a-z0-9]{4}-[a-z0-9]{4}'
    SOCRATA_API_PATTERN = r'/resource/[a-z0-9]{4}-[a-z0-9]{4}'
    CKAN_RESOURCE_PATTERN = r'/dataset/[^/]+/resource/[a-f0-9-]+'

    def __init__(self):
        pass

    def classify(self, url: str) -> Dict[str, Any]:
        """
        Classify a URL and return its type and metadata

        Args:
            url: The URL to classify

        Returns:
            Dictionary with 'type', 'url', and additional metadata
        """
        try:
            url = url.strip()

            # Basic validation
            if not url or url.startswith('#'):
                return {'type': URLType.INVALID, 'url': url, 'reason': 'empty_or_comment'}

            # Fix common URL typos
            url = self._fix_common_typos(url)

            # Check if it's a valid URL
            if not self._is_valid_url(url):
                return {'type': URLType.INVALID, 'url': url, 'reason': 'malformed_url'}

            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
        except (ValueError, Exception) as e:
            # Handle any parsing errors (e.g., malformed bracketed hosts)
            return {'type': URLType.INVALID, 'url': url, 'reason': f'parse_error: {str(e)}'}

        # FTP URLs
        if parsed.scheme == 'ftp':
            return {
                'type': URLType.FTP,
                'url': url,
                'domain': domain,
                'path': path
            }

        # Check for direct file downloads by extension
        file_ext = self._get_file_extension(path)

        if file_ext in self.CSV_EXTENSIONS:
            return {
                'type': URLType.DIRECT_CSV,
                'url': url,
                'extension': file_ext,
                'domain': domain
            }

        if file_ext in self.EXCEL_EXTENSIONS:
            return {
                'type': URLType.DIRECT_EXCEL,
                'url': url,
                'extension': file_ext,
                'domain': domain
            }

        # Socrata detection
        if self._is_socrata_domain(domain):
            if re.search(self.SOCRATA_API_PATTERN, path):
                return {
                    'type': URLType.SOCRATA_API,
                    'url': url,
                    'domain': domain,
                    'dataset_id': self._extract_socrata_id(path)
                }
            elif re.search(self.SOCRATA_DATASET_PATTERN, path):
                return {
                    'type': URLType.SOCRATA_DATASET,
                    'url': url,
                    'domain': domain,
                    'dataset_id': self._extract_socrata_id(path)
                }

        # CKAN detection
        if self._is_ckan_domain(domain) or 'ckan' in path:
            if re.search(self.CKAN_RESOURCE_PATTERN, path):
                return {
                    'type': URLType.CKAN_API,
                    'url': url,
                    'domain': domain,
                    'resource_id': self._extract_ckan_resource_id(path)
                }
            elif '/dataset/' in path:
                return {
                    'type': URLType.CKAN_DATASET,
                    'url': url,
                    'domain': domain
                }

        # Generic page (might need scraping)
        return {
            'type': URLType.GENERIC_PAGE,
            'url': url,
            'domain': domain,
            'path': path
        }

    def _fix_common_typos(self, url: str) -> str:
        """Fix common URL typos found in the dataset"""
        # Fix double 'h' in https
        url = re.sub(r'^hhttps?://', 'http://', url)
        # Fix missing 'ht' prefix
        url = re.sub(r'^tps?://', 'https://', url)
        # Fix 'httsp'
        url = re.sub(r'^httsp://', 'https://', url)
        return url

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except (ValueError, Exception):
            return False

    def _get_file_extension(self, path: str) -> Optional[str]:
        """Extract file extension from path"""
        # Handle query parameters
        path = path.split('?')[0]

        # Get extension
        if '.' in path:
            ext = '.' + path.split('.')[-1].lower()
            return ext
        return None

    def _is_socrata_domain(self, domain: str) -> bool:
        """Check if domain is a Socrata portal"""
        return any(pattern in domain for pattern in self.SOCRATA_DOMAINS)

    def _is_ckan_domain(self, domain: str) -> bool:
        """Check if domain is a CKAN portal"""
        return any(pattern in domain for pattern in self.CKAN_DOMAINS)

    def _extract_socrata_id(self, path: str) -> Optional[str]:
        """Extract Socrata dataset ID (4x4 format)"""
        match = re.search(r'([a-z0-9]{4}-[a-z0-9]{4})', path)
        return match.group(1) if match else None

    def _extract_ckan_resource_id(self, path: str) -> Optional[str]:
        """Extract CKAN resource UUID"""
        match = re.search(r'/resource/([a-f0-9-]+)', path)
        return match.group(1) if match else None

    def get_download_strategy(self, url: str) -> str:
        """
        Get the recommended download strategy for a URL

        Returns:
            Strategy name: 'direct', 'socrata', 'ckan', 'scrape', 'ftp', 'skip'
        """
        classification = self.classify(url)
        url_type = classification['type']

        strategy_map = {
            URLType.DIRECT_CSV: 'direct',
            URLType.DIRECT_EXCEL: 'direct',
            URLType.SOCRATA_DATASET: 'socrata',
            URLType.SOCRATA_API: 'socrata',
            URLType.CKAN_DATASET: 'ckan',
            URLType.CKAN_API: 'ckan',
            URLType.FTP: 'ftp',
            URLType.GENERIC_PAGE: 'scrape',
            URLType.INVALID: 'skip'
        }

        return strategy_map.get(url_type, 'skip')


if __name__ == "__main__":
    # Test the classifier
    classifier = URLClassifier()

    test_urls = [
        "https://data.example.com/file.csv",
        "https://albany.data.socrata.com/dataset/Albany-Capital-Budget/wppx-73fm",
        "https://ckan.publishing.service.gov.uk/dataset/test/resource/abc-123",
        "ftp://ftp.example.com/data.csv",
        "hhttps://broken-url.com/data.csv",
        "http://example.com/dataset-page",
    ]

    for url in test_urls:
        result = classifier.classify(url)
        print(f"\nURL: {url}")
        print(f"Type: {result['type']}")
        print(f"Full result: {result}")
