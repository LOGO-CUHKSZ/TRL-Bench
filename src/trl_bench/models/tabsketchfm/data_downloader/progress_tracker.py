"""
Progress Tracker - Tracks download progress and enables resume capability
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class ProgressTracker:
    """Tracks download progress and maintains state for resume capability"""

    def __init__(self, progress_file: str):
        """
        Initialize progress tracker

        Args:
            progress_file: Path to progress JSON file
        """
        self.progress_file = progress_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """
        Load progress state from file

        Returns:
            State dictionary
        """
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")

        # Initialize new state
        return {
            'started_at': datetime.now().isoformat(),
            'total_urls': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'completed_urls': {},
            'failed_urls': {},
            'last_checkpoint': None
        }

    def save_state(self):
        """Save current state to file"""
        try:
            self.state['last_checkpoint'] = datetime.now().isoformat()

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.progress_file) or '.', exist_ok=True)

            with open(self.progress_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress file: {e}")

    def set_total_urls(self, total: int):
        """Set total number of URLs to process"""
        self.state['total_urls'] = total

    def is_url_completed(self, url: str) -> bool:
        """
        Check if URL has already been successfully processed

        Args:
            url: URL to check

        Returns:
            True if URL was successfully processed
        """
        return url in self.state['completed_urls']

    def mark_success(self, url: str, file_path: Optional[str] = None, file_size: Optional[int] = None):
        """
        Mark URL as successfully downloaded

        Args:
            url: The URL
            file_path: Path to downloaded file
            file_size: File size in bytes
        """
        self.state['completed_urls'][url] = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'file_size': file_size,
            'status': 'success'
        }
        self.state['successful'] += 1
        self.state['processed'] += 1

    def mark_failed(self, url: str, error: str):
        """
        Mark URL as failed

        Args:
            url: The URL
            error: Error message
        """
        self.state['failed_urls'][url] = {
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'status': 'failed'
        }
        self.state['failed'] += 1
        self.state['processed'] += 1

    def mark_skipped(self, url: str, reason: str):
        """
        Mark URL as skipped

        Args:
            url: The URL
            reason: Skip reason
        """
        self.state['completed_urls'][url] = {
            'timestamp': datetime.now().isoformat(),
            'status': 'skipped',
            'reason': reason
        }
        self.state['skipped'] += 1
        self.state['processed'] += 1

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress statistics

        Returns:
            Progress dictionary
        """
        total = self.state['total_urls']
        processed = self.state['processed']

        progress = {
            'total': total,
            'processed': processed,
            'remaining': total - processed,
            'successful': self.state['successful'],
            'failed': self.state['failed'],
            'skipped': self.state['skipped'],
            'percentage': (processed / total * 100) if total > 0 else 0,
            'started_at': self.state['started_at'],
            'last_checkpoint': self.state['last_checkpoint']
        }

        return progress

    def get_failed_urls(self) -> List[str]:
        """
        Get list of failed URLs

        Returns:
            List of failed URLs
        """
        return list(self.state['failed_urls'].keys())

    def reset_failed(self):
        """Reset failed URLs to retry them"""
        num_failed = len(self.state['failed_urls'])
        self.state['failed_urls'] = {}
        self.state['failed'] = 0
        self.state['processed'] -= num_failed
        self.save_state()

    def get_summary(self) -> str:
        """
        Get a formatted summary of progress

        Returns:
            Summary string
        """
        progress = self.get_progress()

        summary = f"""
Download Progress Summary
========================
Total URLs: {progress['total']}
Processed: {progress['processed']} ({progress['percentage']:.1f}%)
  - Successful: {progress['successful']}
  - Failed: {progress['failed']}
  - Skipped: {progress['skipped']}
Remaining: {progress['remaining']}

Started: {progress['started_at']}
Last checkpoint: {progress['last_checkpoint']}
"""
        return summary

    def should_checkpoint(self, checkpoint_interval: int) -> bool:
        """
        Check if we should save a checkpoint

        Args:
            checkpoint_interval: Interval between checkpoints

        Returns:
            True if checkpoint should be saved
        """
        return self.state['processed'] % checkpoint_interval == 0
