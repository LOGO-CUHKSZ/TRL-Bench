#!/usr/bin/env python3
"""
URL Validator - Checks which URLs are still accessible
"""

import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from tqdm import tqdm
import json
from url_classifier import URLClassifier


def check_url(url: str, timeout: int = 10) -> Dict:
    """
    Check if a URL is accessible

    Args:
        url: URL to check
        timeout: Request timeout

    Returns:
        Dictionary with check results
    """
    result = {
        'url': url,
        'accessible': False,
        'status_code': None,
        'error': None
    }

    try:
        # Fix common typos
        classifier = URLClassifier()
        url = classifier._fix_common_typos(url)

        # Make HEAD request (faster than GET)
        response = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            verify=False,  # Many old URLs have SSL issues
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        result['status_code'] = response.status_code
        result['accessible'] = response.status_code < 400

        # For some servers, HEAD doesn't work, try GET
        if response.status_code == 405 or response.status_code == 403:
            response = requests.get(
                url,
                timeout=timeout,
                stream=True,
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            result['status_code'] = response.status_code
            result['accessible'] = True

    except requests.exceptions.Timeout:
        result['error'] = 'timeout'
    except requests.exceptions.ConnectionError:
        result['error'] = 'connection_error'
    except requests.exceptions.TooManyRedirects:
        result['error'] = 'too_many_redirects'
    except requests.exceptions.RequestException as e:
        result['error'] = str(e)
    except Exception as e:
        result['error'] = f'unexpected_error: {str(e)}'

    return result


def validate_urls(urls_file: str, output_file: str, max_workers: int = 20, sample_size: int = None):
    """
    Validate URLs from file

    Args:
        urls_file: Path to file with URLs
        output_file: Path to save results
        max_workers: Number of parallel workers
        sample_size: Only check first N URLs (for testing)
    """
    # Load URLs
    print(f"Loading URLs from {urls_file}...")
    with open(urls_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if sample_size:
        urls = urls[:sample_size]

    print(f"Checking {len(urls)} URLs...")

    results = []
    accessible = 0
    inaccessible = 0

    # Check URLs in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_url, url): url for url in urls}

        with tqdm(total=len(urls), desc="Validating URLs") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result['accessible']:
                    accessible += 1
                else:
                    inaccessible += 1

                pbar.update(1)
                pbar.set_postfix({
                    'accessible': accessible,
                    'inaccessible': inaccessible
                })

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Validation Summary")
    print(f"{'='*60}")
    print(f"Total URLs: {len(urls)}")
    print(f"Accessible: {accessible} ({accessible/len(urls)*100:.1f}%)")
    print(f"Inaccessible: {inaccessible} ({inaccessible/len(urls)*100:.1f}%)")
    print(f"{'='*60}")

    # Count error types
    errors = {}
    for result in results:
        if not result['accessible'] and result['error']:
            errors[result['error']] = errors.get(result['error'], 0) + 1

    print("\nError breakdown:")
    for error, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error}: {count}")

    # Save accessible URLs to separate file
    accessible_file = output_file.replace('.json', '_accessible.txt')
    with open(accessible_file, 'w') as f:
        for result in results:
            if result['accessible']:
                f.write(result['url'] + '\n')

    print(f"\nAccessible URLs saved to: {accessible_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate URLs from pretraining list')
    parser.add_argument(
        '--input',
        default='../pretraining_tables.txt',
        help='Input file with URLs'
    )
    parser.add_argument(
        '--output',
        default='url_validation_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=20,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Only validate first N URLs (for testing)'
    )

    args = parser.parse_args()

    # Disable SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    validate_urls(args.input, args.output, args.workers, args.sample)


if __name__ == "__main__":
    main()
