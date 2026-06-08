#!/usr/bin/env python3
"""
Source Analyzer - Analyzes the distribution of data sources
"""

import argparse
from collections import defaultdict
from urllib.parse import urlparse
from url_classifier import URLClassifier, URLType
from tqdm import tqdm


def analyze_sources(urls_file: str):
    """
    Analyze source distribution in URL list

    Args:
        urls_file: Path to file with URLs
    """
    # Load URLs
    print(f"Loading URLs from {urls_file}...")
    with open(urls_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Analyzing {len(urls)} URLs...\n")

    # Initialize classifier
    classifier = URLClassifier()

    # Statistics
    domains = defaultdict(int)
    url_types = defaultdict(int)
    strategies = defaultdict(int)
    extensions = defaultdict(int)
    schemes = defaultdict(int)

    # Analyze each URL
    for url in tqdm(urls, desc="Analyzing"):
        try:
            # Classify
            classification = classifier.classify(url)
            url_type = classification['type']
            strategy = classifier.get_download_strategy(url)

            # Count
            url_types[url_type.value] += 1
            strategies[strategy] += 1

            # Parse URL
            parsed = urlparse(url)
            schemes[parsed.scheme] += 1

            # Domain
            if 'domain' in classification:
                domain = classification['domain']
                # Get base domain
                parts = domain.split('.')
                if len(parts) >= 2:
                    base_domain = '.'.join(parts[-2:])
                    domains[base_domain] += 1

            # Extension
            if 'extension' in classification:
                ext = classification['extension']
                extensions[ext] += 1
        except Exception as e:
            # Skip malformed URLs
            url_types['invalid'] += 1
            strategies['skip'] += 1
            continue

    # Print results
    print(f"\n{'='*70}")
    print(f"SOURCE ANALYSIS RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal URLs: {len(urls)}")

    print(f"\n{'-'*70}")
    print(f"URL Schemes:")
    print(f"{'-'*70}")
    for scheme, count in sorted(schemes.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(urls) * 100
        print(f"  {scheme:20s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n{'-'*70}")
    print(f"URL Types:")
    print(f"{'-'*70}")
    for url_type, count in sorted(url_types.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(urls) * 100
        print(f"  {url_type:20s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n{'-'*70}")
    print(f"Download Strategies:")
    print(f"{'-'*70}")
    for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(urls) * 100
        print(f"  {strategy:20s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n{'-'*70}")
    print(f"File Extensions (top 20):")
    print(f"{'-'*70}")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:20]:
        pct = count / len(urls) * 100
        print(f"  {ext:20s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n{'-'*70}")
    print(f"Top 30 Domains:")
    print(f"{'-'*70}")
    for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:30]:
        pct = count / len(urls) * 100
        print(f"  {domain:40s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n{'='*70}")

    # Save domain statistics
    with open('domain_stats.txt', 'w') as f:
        f.write("Domain,Count,Percentage\n")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(urls) * 100
            f.write(f"{domain},{count},{pct:.2f}\n")

    print(f"\nDomain statistics saved to: domain_stats.txt")


def main():
    parser = argparse.ArgumentParser(description='Analyze data source distribution')
    parser.add_argument(
        '--input',
        default='../pretraining_tables.txt',
        help='Input file with URLs'
    )

    args = parser.parse_args()
    analyze_sources(args.input)


if __name__ == "__main__":
    main()
