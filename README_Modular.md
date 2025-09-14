# Modular Quantitative Portfolio Generation

This project has been refactored into a clean, modular structure for better maintainability, reusability, and professional development practices.

## Project Structure

```
PortGen/
├── src/                          # Main source code directory
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main orchestration class
│   ├── data/                    # Data handling module
│   │   ├── __init__.py
│   │   └── data_loader.py       # Data loading and pipeline setup
│   ├── risk/                    # Risk modeling module
│   │   ├── __init__.py
│   │   └── risk_model.py        # PCA-based risk models
│   ├── factors/                 # Alpha factors module
│   │   ├── __init__.py
│   │   └── alpha_factors.py     # Factor creation and implementation
│   ├── ml/                      # Machine learning module
│   │   ├── __init__.py
│   │   └── models.py            # ML models and ensemble methods
│   ├── optimization/            # Portfolio optimization module
│   │   ├── __init__.py
│   │   └── portfolio_optimizer.py # Optimization algorithms
│   └── utils/                   # Utility functions module
│       ├── __init__.py
│       └── helpers.py           # Helper functions and utilities
├── data/                        # Data directory (unchanged)
├── example_usage.py             # Example script demonstrating usage
├── Modular_Portfolio_Generation.ipynb # Updated notebook
├── requirements.txt             # Dependencies (unchanged)
└── README_Modular.md           # This file
```

## Module Descriptions

### 1. Data Module (`src/data/`)
- **Purpose**: Handles data loading, bundle registration, and pipeline engine setup
- **Key Functions**:
  - `register_data_bundle()`: Register EOD data bundle
  - `build_pipeline_engine()`: Create pipeline engine
  - `create_data_portal()`: Setup data portal for historical data
  - `get_pricing()`: Retrieve pricing data

### 2. Risk Module (`src/risk/`)
- **Purpose**: Statistical risk modeling using PCA
- **Key Functions**:
  - `fit_pca()`: Fit PCA model to returns
  - `factor_betas()`: Calculate factor betas
  - `factor_returns()`: Calculate factor returns
  - `factor_cov_matrix()`: Compute factor covariance matrix
  - `idiosyncratic_var_matrix()`: Calculate idiosyncratic variance
  - `predict_portfolio_risk()`: Predict portfolio risk

### 3. Factors Module (`src/factors/`)
- **Purpose**: Alpha factor creation and implementation
- **Key Functions**:
  - `momentum_1yr()`: 1-year momentum factor
  - `mean_reversion_5day_sector_neutral()`: Mean reversion factor
  - `overnight_sentiment()`: Overnight sentiment factor
  - Custom factor classes: `CTO`, `TrailingOvernightReturns`

### 4. ML Module (`src/ml/`)
- **Purpose**: Machine learning models and ensemble methods
- **Key Functions**:
  - `train_valid_test_split()`: Time series data splitting
  - `non_overlapping_samples()`: Handle non-overlapping samples
  - `bagging_classifier()`: Create bagging classifier
  - `NoOverlapVoter`: Custom ensemble for non-overlapping data
  - `sharpe_ratio()`: Calculate Sharpe ratios

### 5. Optimization Module (`src/optimization/`)
- **Purpose**: Portfolio optimization algorithms
- **Key Classes**:
  - `AbstractOptimalHoldings`: Base optimization class
  - `OptimalHoldings`: Basic optimization
  - `OptimalHoldingsRegualization`: Regularized optimization
  - `OptimalHoldingsStrictFactor`: Strict factor constraints

### 6. Utils Module (`src/utils/`)
- **Purpose**: Utility functions and helpers
- **Key Functions**:
  - `plot_tree_classifier()`: Visualize decision trees
  - `rank_features_by_importance()`: Feature importance ranking
  - `get_factor_exposures()`: Calculate factor exposures
  - `show_sample_results()`: Display model results
  - `Sector`: Sector classifier for pipeline

### 7. Main Module (`src/main.py`)
- **Purpose**: High-level orchestration of the entire pipeline
- **Key Class**: `PortfolioGenerator`
  - `setup_data()`: Initialize data components
  - `build_risk_model()`: Create risk model
  - `generate_alpha_factors()`: Generate alpha factors
  - `train_ml_model()`: Train ML model
  - `optimize_portfolio()`: Optimize portfolio weights
  - `run_full_pipeline()`: Execute complete pipeline

## Usage Examples

### Basic Usage
```python
from src.main import PortfolioGenerator
import pandas as pd

# Initialize generator
generator = PortfolioGenerator()

# Set end date
universe_end_date = pd.Timestamp('2016-01-05', tz='UTC')

# Run full pipeline
results = generator.run_full_pipeline(
    universe_end_date=universe_end_date,
    optimization_type='regularized',
    lambda_reg=5.0
)

# Access results
optimal_weights = results['optimal_weights']
risk_model = results['risk_model']
```

### Individual Module Usage
```python
# Use individual modules
from src.data import register_data_bundle, build_pipeline_engine
from src.risk import fit_pca, factor_betas
from src.factors import momentum_1yr
from src.optimization import OptimalHoldings

# Register data
register_data_bundle()

# Build risk model
pca = fit_pca(returns, num_factors=20, svd_solver='full')
betas = factor_betas(pca, asset_names, factor_names)

# Create factors
momentum_factor = momentum_1yr(252, universe, sector)

# Optimize portfolio
optimizer = OptimalHoldings(risk_cap=0.05)
weights = optimizer.find(alpha_vector, betas, cov_matrix, idio_var)
```

## Benefits of Modular Structure

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Reusability**: Components can be used independently in other projects
3. **Testability**: Individual modules can be unit tested in isolation
4. **Maintainability**: Changes to one module don't affect others
5. **Professional Development**: Follows Python packaging best practices
6. **Documentation**: Clear module boundaries make documentation easier
7. **Collaboration**: Multiple developers can work on different modules
8. **Extensibility**: Easy to add new factors, optimization methods, or ML models

## Migration from Original Code

The original notebook code has been preserved in `Quantitative Portfolio Generation Final.ipynb`. The modular version provides the same functionality with improved organization:

- **Before**: All code in a single notebook (5600+ lines)
- **After**: Organized into 7 focused modules with clear interfaces

## Running the Code

1. **Using the Example Script**:
   ```bash
   python example_usage.py
   ```

2. **Using the Jupyter Notebook**:
   ```bash
   jupyter notebook Modular_Portfolio_Generation.ipynb
   ```

3. **Using Individual Modules**:
   ```python
   from src.main import PortfolioGenerator
   # ... (see usage examples above)
   ```

## Dependencies

The modular structure uses the same dependencies as the original project (see `requirements.txt`). No additional packages are required.

## Future Enhancements

The modular structure makes it easy to add:
- New alpha factors in `src/factors/`
- Additional ML models in `src/ml/`
- New optimization methods in `src/optimization/`
- Different data sources in `src/data/`
- Enhanced risk models in `src/risk/`
- Additional utilities in `src/utils/`

This structure provides a solid foundation for professional quantitative finance development.
