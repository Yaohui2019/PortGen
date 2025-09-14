"""
Data handling module for portfolio generation.

This module provides functionality for:
- Data bundle registration and loading
- Pipeline engine setup
- Data portal creation
- Pricing data retrieval
"""

from .data_loader import (
    build_pipeline_engine,
    create_data_portal,
    get_pricing,
    register_data_bundle,
)

__all__ = [
    "register_data_bundle",
    "build_pipeline_engine",
    "create_data_portal",
    "get_pricing",
]

