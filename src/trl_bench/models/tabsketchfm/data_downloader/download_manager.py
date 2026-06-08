#!/usr/bin/env python3
"""
Main Download Manager - Orchestrates parallel downloading from multiple sources
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Import downloaders
from url_classifier import URLClassifier, URLType
from progress_tracker import ProgressTracker
from downloaders import (
    DirectDownloader,
    SocrataDownloader,
    CKANDownloader,
    GenericDownloader,
    DownloadResult
)


class DownloadManager:
    """Main download manager that orchestrates the entire download process"""

    def __init__(self, config_path: str):
        """
        Initialize the download manager

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.classifier = URLClassifier()
        self.progress_tracker = ProgressTracker(self.config['progress_file'])

        # Initialize downloaders
        self.downloaders = {
            'direct': DirectDownloader(self.config, self.logger),
            'socrata': SocrataDownloader(self.config, self.logger),
            'ckan': CKANDownloader(self.config, self.logger),
            'scrape': GenericDownloader(self.config, self.logger),
        }

        # Rate limiting setup
        self.rate_limits = self.config.get('rate_limits', {})

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Expand paths
            config['urls_file'] = os.path.expanduser(config.get('urls_file', 'pretraining_tables.txt'))
            config['output_dir'] = os.path.expanduser(config.get('output_dir', './data'))
            config['metadata_dir'] = os.path.expanduser(config.get('metadata_dir', './metadata'))
            config['progress_file'] = config.get('progress_file', './download_progress.json')
            config['error_log'] = config.get('error_log', './download_errors.log')

            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))

        # Create logger
        logger = logging.getLogger('DownloadManager')
        logger.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler for errors
        error_handler = logging.FileHandler(self.config['error_log'])
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(console_formatter)
        logger.addHandler(error_handler)

        return logger

    def load_urls(self) -> List[str]:
        """
        Load URLs from file

        Returns:
            List of URLs
        """
        urls_file = self.config['urls_file']
        self.logger.info(f"Loading URLs from {urls_file}")

        try:
            with open(urls_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            self.logger.info(f"Loaded {len(urls)} URLs")
            return urls

        except Exception as e:
            self.logger.error(f"Failed to load URLs: {e}")
            sys.exit(1)

    def classify_urls(self, urls: List[str]) -> Dict[str, List[tuple]]:
        """
        Classify URLs by download strategy

        Args:
            urls: List of URLs

        Returns:
            Dictionary mapping strategies to (url, classification) tuples
        """
        self.logger.info("Classifying URLs...")

        strategies = {
            'direct': [],
            'socrata': [],
            'ckan': [],
            'scrape': [],
            'ftp': [],
            'skip': []
        }

        for url in tqdm(urls, desc="Classifying URLs", disable=not self.config.get('verbose')):
            classification = self.classifier.classify(url)
            strategy = self.classifier.get_download_strategy(url)
            strategies[strategy].append((url, classification))

        # Print statistics
        print(f"\n{Fore.CYAN}URL Classification Summary:{Style.RESET_ALL}")
        print(f"  Direct downloads: {Fore.GREEN}{len(strategies['direct'])}{Style.RESET_ALL}")
        print(f"  Socrata datasets: {Fore.GREEN}{len(strategies['socrata'])}{Style.RESET_ALL}")
        print(f"  CKAN datasets: {Fore.GREEN}{len(strategies['ckan'])}{Style.RESET_ALL}")
        print(f"  Pages to scrape: {Fore.YELLOW}{len(strategies['scrape'])}{Style.RESET_ALL}")
        print(f"  FTP downloads: {Fore.YELLOW}{len(strategies['ftp'])}{Style.RESET_ALL}")
        print(f"  Invalid/skipped: {Fore.RED}{len(strategies['skip'])}{Style.RESET_ALL}")
        print()

        return strategies

    def download_single(self, url: str, classification: Dict[str, Any], strategy: str) -> DownloadResult:
        """
        Download a single URL

        Args:
            url: URL to download
            classification: URL classification
            strategy: Download strategy

        Returns:
            DownloadResult
        """
        # Check if already completed
        if self.progress_tracker.is_url_completed(url):
            return DownloadResult(
                success=True,
                url=url,
                skipped=True,
                skip_reason="already_completed"
            )

        # Skip invalid URLs
        if strategy == 'skip':
            return DownloadResult(
                success=False,
                url=url,
                error="Invalid URL",
                skipped=True,
                skip_reason="invalid_url"
            )

        # Skip FTP for now (would need separate handler)
        if strategy == 'ftp':
            return DownloadResult(
                success=False,
                url=url,
                error="FTP not yet implemented",
                skipped=True,
                skip_reason="ftp_not_implemented"
            )

        # Get appropriate downloader
        downloader = self.downloaders.get(strategy)
        if not downloader:
            return DownloadResult(
                success=False,
                url=url,
                error=f"No downloader for strategy: {strategy}"
            )

        # Download
        try:
            result = downloader.download(url, classification)
            return result
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {e}")
            return DownloadResult(
                success=False,
                url=url,
                error=str(e)
            )

    def process_batch(self, batch: List[tuple], strategy: str) -> List[DownloadResult]:
        """
        Process a batch of URLs with the same strategy

        Args:
            batch: List of (url, classification) tuples
            strategy: Download strategy

        Returns:
            List of DownloadResults
        """
        if not batch:
            return []

        self.logger.info(f"Processing {len(batch)} URLs with strategy: {strategy}")

        max_workers = self.config.get('max_workers', 10)
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.download_single, url, classification, strategy): url
                for url, classification in batch
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(batch), desc=f"Downloading ({strategy})",
                     disable=not self.config.get('verbose')) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    # Update progress tracker
                    if result.success:
                        if result.skipped:
                            self.progress_tracker.mark_skipped(result.url, result.skip_reason or 'unknown')
                        else:
                            self.progress_tracker.mark_success(
                                result.url,
                                result.file_path,
                                result.file_size
                            )
                    else:
                        self.progress_tracker.mark_failed(result.url, result.error or 'unknown')

                    # Checkpoint if needed
                    if self.progress_tracker.should_checkpoint(self.config.get('checkpoint_interval', 100)):
                        self.progress_tracker.save_state()

                    pbar.update(1)

        return results

    def run(self, start_index: int = 0, limit: Optional[int] = None):
        """
        Run the download process

        Args:
            start_index: Index to start from
            limit: Maximum number of URLs to process
        """
        # Load URLs
        all_urls = self.load_urls()

        # Apply start_index and limit
        if limit:
            urls = all_urls[start_index:start_index + limit]
        else:
            urls = all_urls[start_index:]

        self.logger.info(f"Processing {len(urls)} URLs (starting from index {start_index})")

        # Set total in progress tracker
        self.progress_tracker.set_total_urls(len(urls))

        # Classify URLs
        strategies = self.classify_urls(urls)

        # Process each strategy
        all_results = []

        # Process in order of priority
        strategy_order = ['direct', 'socrata', 'ckan', 'scrape', 'ftp']

        for strategy in strategy_order:
            batch = strategies.get(strategy, [])
            if batch:
                print(f"\n{Fore.CYAN}Processing {len(batch)} {strategy} URLs...{Style.RESET_ALL}")
                results = self.process_batch(batch, strategy)
                all_results.extend(results)

        # Save final progress
        self.progress_tracker.save_state()

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print download summary"""
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(self.progress_tracker.get_summary())
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")

        # Save failed URLs to a file
        failed_urls = self.progress_tracker.get_failed_urls()
        if failed_urls:
            failed_file = 'failed_urls.txt'
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed_urls))
            print(f"\n{Fore.YELLOW}Failed URLs saved to: {failed_file}{Style.RESET_ALL}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Full-Featured Data Downloader for TabSketchFM Pretraining Data'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index in URL list (default: 0)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of URLs to process (default: all)'
    )
    parser.add_argument(
        '--reset-failed',
        action='store_true',
        help='Reset failed URLs and retry them'
    )

    args = parser.parse_args()

    # Create download manager
    manager = DownloadManager(args.config)

    # Reset failed if requested
    if args.reset_failed:
        print("Resetting failed URLs...")
        manager.progress_tracker.reset_failed()

    # Run download process
    manager.run(start_index=args.start, limit=args.limit)


if __name__ == "__main__":
    main()
