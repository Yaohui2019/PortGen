"""
10K Sentiment Data Processing Module.

This module provides functionality for processing 10K filings to extract
sentiment-based alpha factors, integrating the analysis from the 10K notebook.
"""

import ast
import csv
import os
import pickle
import re
from collections import Counter, defaultdict

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .sec_scraper import SecScraper, scrape_10k_data


class SentimentProcessor:
    """
    Process 10K filings to extract sentiment-based alpha factors.

    This class integrates the sentiment analysis logic from the 10K notebook
    to create standardized sentiment factors for the portfolio pipeline.
    """

    def __init__(self, sentiment_data_path=None, cache_dir=None):
        """
        Initialize the sentiment processor.

        Parameters
        ----------
        sentiment_data_path : str, optional
            Path to the sentiment data directory
        cache_dir : str, optional
            Directory to cache scraped filings
        """
        self.sentiment_data_path = sentiment_data_path
        self.cache_dir = cache_dir
        self.sentiment_df = None
        self.processed_data = {}
        self.scraper = None

        # Download required NLTK data
        self._download_nltk_data()

        # Load sentiment word lists
        self._load_sentiment_words()

    def _download_nltk_data(self):
        """Download required NLTK corpora."""
        try:
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
        except:
            print(
                "Warning: Could not download NLTK data. Some features may not work."
            )

    def _load_sentiment_words(self):
        """Load Loughran-McDonald sentiment word lists."""
        sentiments = [
            "negative",
            "positive",
            "uncertainty",
            "litigious",
            "constraining",
            "interesting",
        ]

        # Try to load from the expected location
        sentiment_file = os.path.join(
            os.getcwd(),
            "data",
            "project_5_loughran_mcdonald",
            "loughran_mcdonald_master_dic_2016.csv",
        )

        if os.path.exists(sentiment_file):
            self.sentiment_df = pd.read_csv(sentiment_file)
            self.sentiment_df.columns = [
                column.lower() for column in self.sentiment_df.columns
            ]
            self.sentiment_df = self.sentiment_df[sentiments + ["word"]]
            self.sentiment_df[sentiments] = self.sentiment_df[
                sentiments
            ].astype(bool)
            self.sentiment_df = self.sentiment_df[
                (self.sentiment_df[sentiments]).any(1)
            ]
            self.sentiment_df["word"] = self._lemmatize_words(
                self.sentiment_df["word"].str.lower()
            )
            self.sentiment_df = self.sentiment_df.drop_duplicates("word")
        else:
            print(
                f"Warning: Sentiment word list not found at {sentiment_file}"
            )
            # Create a minimal sentiment dataframe
            self.sentiment_df = pd.DataFrame(columns=sentiments + ["word"])

    def _lemmatize_words(self, words):
        """Lemmatize words using NLTK."""
        try:
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(w, wordnet.VERB) for w in words]
        except:
            return words.tolist()

    def _clean_text(self, text):
        """Clean text by removing HTML and converting to lowercase."""
        text = BeautifulSoup(text, "html.parser").get_text()
        text = text.lower()
        return text

    def _get_documents(self, text):
        """Extract documents from SEC filing text."""
        regex = re.findall("<DOCUMENT>[\s\S]*?</DOCUMENT>", text)
        return [
            re.sub("(<DOCUMENT>)|(</DOCUMENT>)", "", string)
            for string in regex
        ]

    def _get_document_type(self, doc):
        """Get document type from SEC document."""
        type_lst = re.findall("<TYPE>\S*", doc)
        if type_lst:
            return re.sub("<TYPE>", "", type_lst[0]).lower()
        return ""

    def _process_10k_documents(self, raw_fillings_by_ticker):
        """Process 10K documents from raw filings."""
        ten_ks_by_ticker = {}

        for ticker, raw_fillings in raw_fillings_by_ticker.items():
            ten_ks_by_ticker[ticker] = []

            for file_date, filling in tqdm(
                raw_fillings.items(),
                desc=f"Processing {ticker} filings",
                unit="filing",
            ):
                # Extract documents
                documents = self._get_documents(filling)

                # Find 10-K documents
                for document in documents:
                    if self._get_document_type(document) == "10-k":
                        # Clean and process document
                        clean_text = self._clean_text(document)
                        words = re.findall("\w+", clean_text)
                        lemmatized_words = self._lemmatize_words(words)

                        # Remove stopwords
                        lemma_english_stopwords = self._lemmatize_words(
                            stopwords.words("english")
                        )
                        filtered_words = [
                            word
                            for word in lemmatized_words
                            if word not in lemma_english_stopwords
                        ]

                        ten_ks_by_ticker[ticker].append(
                            {
                                "file_date": file_date,
                                "words": filtered_words,
                                "clean_text": clean_text,
                            }
                        )

        return ten_ks_by_ticker

    def _calculate_sentiment_metrics(self, ten_ks_by_ticker):
        """Calculate sentiment metrics from processed 10K documents."""
        sentiments = [
            "negative",
            "positive",
            "uncertainty",
            "litigious",
            "constraining",
            "interesting",
        ]
        sentiment_metrics = {}

        for ticker, ten_ks in ten_ks_by_ticker.items():
            if not ten_ks:
                continue

            # Sort by date
            ten_ks.sort(key=lambda x: x["file_date"])

            # Create document strings
            lemma_docs = [" ".join(ten_k["words"]) for ten_k in ten_ks]

            # Calculate TF-IDF for each sentiment
            tfidf_metrics = {}
            for sentiment in sentiments:
                if sentiment in self.sentiment_df.columns:
                    sentiment_words = self.sentiment_df[
                        self.sentiment_df[sentiment]
                    ]["word"]
                    if len(sentiment_words) > 0:
                        vectorizer = TfidfVectorizer(
                            vocabulary=sentiment_words.values
                        )
                        tfidf_matrix = vectorizer.fit_transform(
                            lemma_docs
                        ).toarray()
                        tfidf_metrics[sentiment] = tfidf_matrix

            # Calculate cosine similarities
            cosine_similarities = {}
            for sentiment, tfidf_matrix in tfidf_metrics.items():
                if len(tfidf_matrix) > 1:
                    csim_matrix = cosine_similarity(
                        tfidf_matrix, tfidf_matrix
                    )
                    cosine_similarities[sentiment] = np.diag(
                        csim_matrix, k=1
                    ).tolist()
                else:
                    cosine_similarities[sentiment] = []

            sentiment_metrics[ticker] = {
                "dates": [ten_k["file_date"] for ten_k in ten_ks],
                "cosine_similarities": cosine_similarities,
            }

        return sentiment_metrics

    def _create_sentiment_dataframe(self, sentiment_metrics):
        """Create standardized sentiment dataframe."""
        sentiment_data = []

        for ticker, metrics in sentiment_metrics.items():
            dates = metrics["dates"]
            cosine_similarities = metrics["cosine_similarities"]

            # Create data for each sentiment type
            for (
                sentiment_type,
                similarities,
            ) in cosine_similarities.items():
                for i, similarity_value in enumerate(similarities):
                    if i + 1 < len(dates):  # Ensure we have a next date
                        sentiment_data.append(
                            {
                                "date": dates[i + 1],  # Use the later date
                                "ticker": ticker,
                                "sentiment_type": sentiment_type,
                                "cosine_similarity": similarity_value,
                            }
                        )

        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            df["date"] = pd.to_datetime(df["date"])
            return df
        else:
            return pd.DataFrame()

    def process_sentiment_data(
        self, raw_fillings_path=None, output_path=None
    ):
        """
        Process sentiment data from 10K filings.

        Parameters
        ----------
        raw_fillings_path : str, optional
            Path to raw 10K filings data
        output_path : str, optional
            Path to save processed sentiment data

        Returns
        -------
        pd.DataFrame
            Processed sentiment data
        """
        if raw_fillings_path is None:
            raw_fillings_path = self.sentiment_data_path

        if not raw_fillings_path or not os.path.exists(raw_fillings_path):
            print("No valid raw filings path provided.")
            return pd.DataFrame()

        print("Loading raw 10K filings...")

        # Load raw filings data
        raw_fillings_by_ticker = defaultdict(dict)

        # Try to load from CSV file (as used in the notebook)
        csv_file = os.path.join(
            raw_fillings_path, "raw_fillings_by_ticker.csv"
        )
        if os.path.exists(csv_file):
            pd_fillings = pd.read_csv(
                csv_file,
                header=None,
                names=["ticker", "file_date", "10-k"],
            )
            for _, row in pd_fillings.iterrows():
                raw_fillings_by_ticker[row.ticker][row.file_date] = row[
                    "10-k"
                ]
        else:
            print(f"Raw filings CSV not found at {csv_file}")
            return pd.DataFrame()

        print("Processing 10K documents...")
        ten_ks_by_ticker = self._process_10k_documents(
            raw_fillings_by_ticker
        )

        print("Calculating sentiment metrics...")
        sentiment_metrics = self._calculate_sentiment_metrics(
            ten_ks_by_ticker
        )

        print("Creating sentiment dataframe...")
        sentiment_df = self._create_sentiment_dataframe(sentiment_metrics)

        if not sentiment_df.empty and output_path:
            # Create pivot table with dates as index and tickers as columns
            sentiment_pivot = sentiment_df.pivot_table(
                index=["date", "sentiment_type"],
                columns="ticker",
                values="cosine_similarity",
            )
            sentiment_pivot.to_csv(output_path)
            print(f"Sentiment data saved to {output_path}")

        return sentiment_df

    def scrape_10k_filings(
        self, tickers, years_back=5, delay=0.1, output_path=None
    ):
        """
        Scrape 10-K filings for the given tickers.

        Parameters
        ----------
        tickers : list
            List of stock ticker symbols
        years_back : int, default 5
            Number of years back to scrape
        delay : float, default 0.1
            Delay between requests in seconds
        output_path : str, optional
            Path to save the scraped filings

        Returns
        -------
        dict
            Nested dictionary: ticker -> filing_date -> filing_content
        """
        if not self.scraper:
            self.scraper = SecScraper(
                delay=delay, cache_dir=self.cache_dir
            )

        print(f"Scraping 10-K filings for {len(tickers)} companies...")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Years back: {years_back}")
        print(f"Delay between requests: {delay}s")
        print()

        # Scrape filings
        filings = self.scraper.scrape_multiple_companies(
            tickers, years_back
        )

        # Save to CSV if output path provided
        if filings and output_path:
            self.scraper.save_filings_to_csv(filings, output_path)
            print(f"\nFilings saved to: {output_path}")

        return filings

    def scrape_and_process_sentiment(
        self,
        tickers,
        years_back=5,
        delay=0.1,
        raw_filings_path=None,
        output_path=None,
    ):
        """
        Scrape 10-K filings and process them for sentiment analysis.

        Parameters
        ----------
        tickers : list
            List of stock ticker symbols
        years_back : int, default 5
            Number of years back to scrape
        delay : float, default 0.1
            Delay between requests in seconds
        raw_filings_path : str, optional
            Path to save raw filings (if None, uses cache)
        output_path : str, optional
            Path to save processed sentiment data

        Returns
        -------
        pd.DataFrame
            Processed sentiment data
        """
        # Scrape filings
        filings = self.scrape_10k_filings(
            tickers, years_back, delay, raw_filings_path
        )

        if not filings:
            print("No filings scraped. Cannot process sentiment data.")
            return pd.DataFrame()

        # Process sentiment data
        if raw_filings_path and os.path.exists(raw_filings_path):
            return self.process_sentiment_data(
                raw_filings_path, output_path
            )
        else:
            # Process directly from scraped data
            return self._process_scraped_filings(filings, output_path)

    def _process_scraped_filings(self, filings, output_path=None):
        """
        Process scraped filings directly without saving to CSV first.

        Parameters
        ----------
        filings : dict
            Nested dictionary: ticker -> filing_date -> filing_content
        output_path : str, optional
            Path to save processed sentiment data

        Returns
        -------
        pd.DataFrame
            Processed sentiment data
        """
        print("Processing scraped 10-K documents...")

        # Convert to the format expected by _process_10k_documents
        raw_fillings_by_ticker = {}
        for ticker, ticker_filings in filings.items():
            raw_fillings_by_ticker[ticker] = {}
            for filing_date, content in ticker_filings.items():
                raw_fillings_by_ticker[ticker][filing_date] = content

        # Process documents
        ten_ks_by_ticker = self._process_10k_documents(
            raw_fillings_by_ticker
        )

        print("Calculating sentiment metrics...")
        sentiment_metrics = self._calculate_sentiment_metrics(
            ten_ks_by_ticker
        )

        print("Creating sentiment dataframe...")
        sentiment_df = self._create_sentiment_dataframe(sentiment_metrics)

        if not sentiment_df.empty and output_path:
            # Create pivot table with dates as index and tickers as columns
            sentiment_pivot = sentiment_df.pivot_table(
                index=["date", "sentiment_type"],
                columns="ticker",
                values="cosine_similarity",
            )
            sentiment_pivot.to_csv(output_path)
            print(f"Sentiment data saved to {output_path}")

        return sentiment_df

    def load_processed_sentiment_data(self, file_path):
        """
        Load processed sentiment data from file.

        Parameters
        ----------
        file_path : str
            Path to the processed sentiment data file

        Returns
        -------
        pd.DataFrame
            Loaded sentiment data
        """
        if os.path.exists(file_path):
            return pd.read_csv(
                file_path, index_col=[0, 1], parse_dates=True
            )
        else:
            print(f"Processed sentiment data not found at {file_path}")
            return pd.DataFrame()
