# SAS package for fitting and generation tools
"""
SAS Package - Small Angle Scattering Tools

This package contains:
- fitting.py: SasView fitting functionality with CrewAI tool wrapper
- generation.py: Synthetic data generation with CrewAI tool wrapper
"""

try:
    from .fitting import sasview_fit, sasview_fit_tool
except ImportError:
    print("Warning: Could not import fitting tools")
    sasview_fit, sasview_fit_tool = None, None

try:
    from .generation import generate_synthetic_data, synthetic_data_tool
except ImportError:
    print("Warning: Could not import generation tools")
    generate_synthetic_data, synthetic_data_tool = None, None

__all__ = [
    'sasview_fit',
    'sasview_fit_tool',
    'generate_synthetic_data',
    'synthetic_data_tool'
]
