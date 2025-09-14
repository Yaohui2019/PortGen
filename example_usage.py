"""
Example usage of the modularized portfolio generation system.

This script demonstrates how to use the new modular structure
to generate optimal portfolios.
"""

import sys
import os
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import PortfolioGenerator


def main():
    """Run the portfolio generation example."""
    
    # Set the end date for the universe
    universe_end_date = pd.Timestamp('2016-01-05', tz='UTC')
    
    # Initialize the portfolio generator
    generator = PortfolioGenerator()
    
    # Run the full pipeline
    results = generator.run_full_pipeline(
        universe_end_date=universe_end_date,
        optimization_type='regularized',
        lambda_reg=5.0,
        risk_cap=0.05,
        factor_max=10.0,
        factor_min=-10.0,
        weights_max=0.55,
        weights_min=-0.55
    )
    
    # Display results
    print("\n" + "="*50)
    print("PORTFOLIO GENERATION RESULTS")
    print("="*50)
    
    print(f"\nOptimal weights shape: {results['optimal_weights'].shape}")
    print(f"Number of assets: {len(results['optimal_weights'])}")
    print(f"Total weight: {results['optimal_weights'].sum().iloc[0]:.4f}")
    print(f"Max weight: {results['optimal_weights'].max().iloc[0]:.4f}")
    print(f"Min weight: {results['optimal_weights'].min().iloc[0]:.4f}")
    
    # Show top 10 holdings
    print("\nTop 10 Holdings:")
    top_holdings = results['optimal_weights'].abs().sort_values(by=0, ascending=False).head(10)
    for i, (asset, weight) in enumerate(top_holdings.iterrows(), 1):
        print(f"{i:2d}. {asset}: {weight.iloc[0]:.4f}")
    
    # Show factor exposures
    print("\nFactor Exposures:")
    factor_exposures = generator.get_factor_exposures(
        results['risk_model']['factor_betas'],
        results['optimal_weights']
    )
    print(factor_exposures.T)
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

