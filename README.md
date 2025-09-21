# Quantitative Portfolio Generation Pipeline

A comprehensive quantitative finance pipeline that generates optimized portfolios using multiple alpha factors, risk models, and machine learning techniques. This system integrates traditional quantitative methods with modern sentiment analysis from 10-K filings and SEC data scraping capabilities.

## Features

### Core Portfolio Generation
- **Multi-Factor Alpha Generation**: Implements proven alpha factors from academic research
- **Risk Modeling**: PCA-based risk model construction for portfolio optimization
- **Machine Learning Integration**: Ensemble methods for factor combination and signal enhancement
- **Advanced Optimization**: CVXPY-based portfolio optimization with multiple constraint types

### Sentiment Analysis Integration
- **10-K Sentiment Processing**: Extracts sentiment signals from corporate filings using Loughran-McDonald word lists
- **SEC Data Scraping**: Automated downloading of 10-K filings from SEC EDGAR database
- **Sentiment Factors**: Six sentiment-based alpha factors (negative, positive, uncertainty, litigious, constraining, interesting)
- **Cosine Similarity Analysis**: Measures sentiment changes between consecutive filings

### Technical Capabilities
- **Zipline Pipeline Integration**: Seamless integration with Zipline's factor pipeline
- **Alphalens Analysis**: Comprehensive factor performance evaluation
- **Rate-Limited Scraping**: Respectful SEC EDGAR API usage with caching
- **Modular Architecture**: Clean, extensible codebase for easy customization

## Installation

### Prerequisites
- Python 3.7+

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PortGen
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Sample Usage Examples

### Example 1: Traditional Alpha Factors Only

```python
from src.main import PortfolioGenerator

# Initialize without sentiment data
generator = PortfolioGenerator(data_path="data/project_4_eod")

# Build risk model
generator.build_risk_model()

# Generate alpha factors
generator.generate_alpha_factors()

# Train ML model
generator.train_ml_model()

# Optimize portfolio
portfolio = generator.optimize_portfolio()
```

### Example 2: Complete Pipeline with Sentiment

```python
from src.main import PortfolioGenerator

# Initialize with all options
generator = PortfolioGenerator(
    data_path="data/project_4_eod",
    sentiment_data_path="data/sentiment_data",
    cache_dir="data/cache"
)

# Scrape recent 10-K data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
generator.scrape_10k_data(
    tickers=tickers,
    years_back=3,
    delay=0.1
)

# Run complete pipeline
results = generator.run_full_pipeline()
print(f"Portfolio Sharpe Ratio: {results['sharpe_ratio']:.3f}")
```


## Architecture Overview

The pipeline consists of several key modules:

- **`src/main.py`**: Main orchestrator class (`PortfolioGenerator`)
- **`src/factors/`**: Alpha factor generation and sentiment processing
- **`src/risk/`**: Risk model construction using PCA
- **`src/ml/`**: Machine learning models for factor combination
- **`src/optimization/`**: Portfolio optimization algorithms
- **`src/data/`**: Data loading and pipeline setup
- **`src/utils/`**: Utility functions and helpers

## Data Requirements

### Market Data
- End-of-day pricing data in Zipline bundle format
- Sector classification data

### Sentiment Data (Optional)
- 10-K filings in CSV format with columns: `ticker`, `filing_date`, `content`
- Or use the built-in SEC scraper to download automatically

## Performance Considerations

- **Scraping**: Use appropriate delays (0.1-0.5s) to respect SEC rate limits
- **Caching**: Enable caching to avoid re-downloading filings
- **Memory**: Large sentiment datasets may require significant RAM
- **Processing**: Sentiment analysis can be computationally intensive

