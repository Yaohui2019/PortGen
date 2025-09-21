"""
SEC EDGAR 10-K Filing Scraper.

This module provides functionality for downloading 10-K filings from the SEC EDGAR database.
It includes rate limiting, error handling, and data organization features.
"""

import ast
import csv
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class SecAPI:
    """
    SEC EDGAR API client with rate limiting and caching.

    This class provides methods to download 10-K filings from the SEC EDGAR database
    with proper rate limiting to avoid hitting SEC's request limits.
    """

    def __init__(self, delay: float = 0.1, cache_dir: str = None):
        """
        Initialize the SEC API client.

        Parameters
        ----------
        delay : float, default 0.1
            Delay between requests in seconds
        cache_dir : str, optional
            Directory to cache downloaded filings
        """
        self.delay = delay
        self.cache_dir = cache_dir or os.path.join(
            os.getcwd(), "data", "sec_cache"
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "PortGen Sentiment Analysis Tool (contact@example.com)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _make_request(
        self, url: str, max_retries: int = 3
    ) -> Optional[str]:
        """
        Make a request with rate limiting and retry logic.

        Parameters
        ----------
        url : str
            URL to request
        max_retries : int, default 3
            Maximum number of retry attempts

        Returns
        -------
        str or None
            Response content or None if failed
        """
        for attempt in range(max_retries):
            try:
                time.sleep(self.delay)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except (HTTPError, URLError, requests.RequestException) as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    print(f"Request failed (attempt {attempt + 1}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(
                        f"Failed to fetch {url} after {max_retries} attempts: {e}"
                    )
                    return None

    def get_company_filings(
        self,
        cik: str,
        form_type: str = "10-K",
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """
        Get company filings from SEC EDGAR.

        Parameters
        ----------
        cik : str
            Company CIK (Central Index Key)
        form_type : str, default "10-K"
            Type of form to retrieve
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format

        Returns
        -------
        List[Dict]
            List of filing information dictionaries
        """
        # Pad CIK with leading zeros
        cik_padded = cik.zfill(10)

        # Build URL for company filings
        base_url = f"https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "CIK": cik_padded,
            "type": form_type,
            "dateb": end_date or "",
            "datea": start_date or "",
            "output": "atom",
            "count": "100",
        }

        url = f"{base_url}?{'&'.join([f'{k}={v}' for k, v in params.items() if v])}"

        # Check cache first
        cache_file = os.path.join(
            self.cache_dir, f"{cik}_{form_type}_filings.xml"
        )
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = self._make_request(url)
            if content and self.cache_dir:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(content)

        if not content:
            return []

        # Parse XML response
        filings = self._parse_filings_xml(content)
        return filings

    def _parse_filings_xml(self, xml_content: str) -> List[Dict]:
        """Parse SEC filings XML response."""
        try:
            soup = BeautifulSoup(xml_content, "xml")
            entries = soup.find_all("entry")

            filings = []
            for entry in entries:
                try:
                    # Extract filing information
                    title = (
                        entry.find("title").text
                        if entry.find("title")
                        else ""
                    )
                    link = entry.find("link", {"type": "text/html"})
                    filing_url = link.get("href") if link else ""

                    # Extract filing date
                    updated = entry.find("updated")
                    filing_date = (
                        updated.text[:10] if updated else ""
                    )  # YYYY-MM-DD

                    # Extract accession number from URL
                    accession_match = re.search(
                        r"/(\d{10}-\d{2}-\d{6})", filing_url
                    )
                    accession = (
                        accession_match.group(1) if accession_match else ""
                    )

                    filings.append(
                        {
                            "title": title,
                            "filing_url": filing_url,
                            "filing_date": filing_date,
                            "accession": accession,
                        }
                    )
                except Exception as e:
                    print(f"Error parsing filing entry: {e}")
                    continue

            return filings
        except Exception as e:
            print(f"Error parsing XML content: {e}")
            return []

    def download_filing(
        self, filing_url: str, cik: str, accession: str
    ) -> Optional[str]:
        """
        Download a specific filing.

        Parameters
        ----------
        filing_url : str
            URL to the filing index page
        cik : str
            Company CIK
        accession : str
            Filing accession number

        Returns
        -------
        str or None
            Filing content or None if failed
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{cik}_{accession}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        # Get the filing index page
        index_content = self._make_request(filing_url)
        if not index_content:
            return None

        # Find the 10-K document URL
        doc_url = self._find_10k_document_url(index_content, filing_url)
        if not doc_url:
            print(f"Could not find 10-K document in filing {accession}")
            return None

        # Download the actual filing
        filing_content = self._make_request(doc_url)
        if filing_content and self.cache_dir:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(filing_content)

        return filing_content

    def _find_10k_document_url(
        self, index_content: str, base_url: str
    ) -> Optional[str]:
        """Find the 10-K document URL from the filing index page."""
        try:
            soup = BeautifulSoup(index_content, "html.parser")

            # Look for 10-K document links
            links = soup.find_all("a", href=True)
            for link in links:
                href = link.get("href", "")
                text = link.get_text().strip().lower()

                # Check if this is a 10-K document
                if ("10-k" in text or "10k" in text) and href.endswith(
                    ".htm"
                ):
                    # Convert relative URL to absolute
                    if href.startswith("/"):
                        return urljoin("https://www.sec.gov", href)
                    else:
                        return urljoin(base_url, href)

            return None
        except Exception as e:
            print(f"Error finding 10-K document URL: {e}")
            return None


class SecScraper:
    """
    High-level SEC scraper for downloading 10-K filings.

    This class provides a convenient interface for downloading 10-K filings
    for multiple companies and organizing them for sentiment analysis.
    """

    def __init__(self, delay: float = 0.1, cache_dir: str = None):
        """
        Initialize the SEC scraper.

        Parameters
        ----------
        delay : float, default 0.1
            Delay between requests in seconds
        cache_dir : str, optional
            Directory to cache downloaded filings
        """
        self.api = SecAPI(delay=delay, cache_dir=cache_dir)
        self.cik_lookup = self._load_cik_lookup()

    def _load_cik_lookup(self) -> Dict[str, str]:
        """Load CIK lookup dictionary for common tickers."""
        # Common CIK mappings (can be extended)
        cik_lookup = {
            "AMZN": "0001018724",
            "BMY": "0000014272",
            "CNP": "0001130310",
            "CVX": "0000093410",
            "FL": "0000850209",
            "FRT": "0000034903",
            "HON": "0000773840",
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "GOOGL": "0001652044",
            "TSLA": "0001318605",
            "JPM": "0000019617",
            "JNJ": "0000200406",
            "V": "0001403161",
            "PG": "0000080424",
            "UNH": "0000731766",
            "HD": "0000354950",
            "MA": "0001141391",
            "DIS": "0001001039",
            "PYPL": "0001633917",
        }
        return cik_lookup

    def get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK for a given ticker symbol.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol

        Returns
        -------
        str or None
            CIK if found, None otherwise
        """
        return self.cik_lookup.get(ticker.upper())

    def add_ticker_cik(self, ticker: str, cik: str):
        """
        Add a ticker-CIK mapping.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        cik : str
            Company CIK
        """
        self.cik_lookup[ticker.upper()] = cik.zfill(10)

    def scrape_company_10ks(
        self,
        ticker: str,
        years_back: int = 5,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, str]:
        """
        Scrape 10-K filings for a single company.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        years_back : int, default 5
            Number of years back to scrape
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format

        Returns
        -------
        Dict[str, str]
            Dictionary mapping filing dates to filing content
        """
        cik = self.get_cik_for_ticker(ticker)
        if not cik:
            print(f"No CIK found for ticker {ticker}")
            return {}

        # Set default date range if not provided
        if not start_date:
            start_date = (
                datetime.now() - timedelta(days=years_back * 365)
            ).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(
            f"Scraping 10-K filings for {ticker} (CIK: {cik}) from {start_date} to {end_date}"
        )

        # Get filing list
        filings = self.api.get_company_filings(
            cik, "10-K", start_date, end_date
        )
        if not filings:
            print(f"No 10-K filings found for {ticker}")
            return {}

        # Download filings
        filing_contents = {}
        for filing in tqdm(filings, desc=f"Downloading {ticker} filings"):
            filing_date = filing["filing_date"]
            accession = filing["accession"]
            filing_url = filing["filing_url"]

            content = self.api.download_filing(filing_url, cik, accession)
            if content:
                filing_contents[filing_date] = content
                print(f"Downloaded {ticker} 10-K for {filing_date}")
            else:
                print(
                    f"Failed to download {ticker} 10-K for {filing_date}"
                )

        return filing_contents

    def scrape_multiple_companies(
        self,
        tickers: List[str],
        years_back: int = 5,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Scrape 10-K filings for multiple companies.

        Parameters
        ----------
        tickers : List[str]
            List of stock ticker symbols
        years_back : int, default 5
            Number of years back to scrape
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format

        Returns
        -------
        Dict[str, Dict[str, str]]
            Nested dictionary: ticker -> filing_date -> filing_content
        """
        all_filings = {}

        for ticker in tickers:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}")
            print(f"{'='*50}")

            filings = self.scrape_company_10ks(
                ticker, years_back, start_date, end_date
            )
            if filings:
                all_filings[ticker] = filings
            else:
                print(f"No filings retrieved for {ticker}")

        return all_filings

    def save_filings_to_csv(
        self, filings: Dict[str, Dict[str, str]], output_path: str
    ):
        """
        Save filings to CSV format compatible with sentiment processor.

        Parameters
        ----------
        filings : Dict[str, Dict[str, str]]
            Nested dictionary: ticker -> filing_date -> filing_content
        output_path : str
            Path to save the CSV file
        """
        print(f"Saving filings to {output_path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(
            output_path, "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.writer(csvfile)

            for ticker, ticker_filings in filings.items():
                for filing_date, content in ticker_filings.items():
                    # Escape the content for CSV
                    content_escaped = content.replace('"', '""')
                    writer.writerow(
                        [ticker, filing_date, f'"{content_escaped}"']
                    )

        print(
            f"Saved {sum(len(tf) for tf in filings.values())} filings to {output_path}"
        )

    def load_filings_from_csv(
        self, csv_path: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Load filings from CSV format.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file

        Returns
        -------
        Dict[str, Dict[str, str]]
            Nested dictionary: ticker -> filing_date -> filing_content
        """
        filings = {}

        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                if len(row) >= 3:
                    ticker, filing_date, content = row[0], row[1], row[2]

                    # Remove quotes and unescape
                    content = content.strip('"').replace('""', '"')

                    if ticker not in filings:
                        filings[ticker] = {}
                    filings[ticker][filing_date] = content

        return filings


def scrape_10k_data(
    tickers: List[str],
    output_path: str,
    years_back: int = 5,
    delay: float = 0.1,
    cache_dir: str = None,
) -> Dict[str, Dict[str, str]]:
    """
    Convenience function to scrape 10-K data for multiple companies.

    Parameters
    ----------
    tickers : List[str]
        List of stock ticker symbols
    output_path : str
        Path to save the scraped data
    years_back : int, default 5
        Number of years back to scrape
    delay : float, default 0.1
        Delay between requests in seconds
    cache_dir : str, optional
        Directory to cache downloaded filings

    Returns
    -------
    Dict[str, Dict[str, str]]
        Nested dictionary: ticker -> filing_date -> filing_content
    """
    scraper = SecScraper(delay=delay, cache_dir=cache_dir)

    # Scrape filings
    filings = scraper.scrape_multiple_companies(tickers, years_back)

    # Save to CSV
    if filings:
        scraper.save_filings_to_csv(filings, output_path)

    return filings
