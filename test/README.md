# Test Directory

This directory contains test scripts and test data for the SAS Agent project.

## Structure

```
test/
├── README.md                    # This file
├── test_fitting.py             # Test script for SAS data fitting functionality
├── test_generation.py          # Test script for synthetic SAS data generation
└── test_data/                  # Directory for test data files
    ├── fitting/                # Test data for fitting tests
    └── generation/             # Test data for generation tests
```

## Usage

Run tests from the project root directory:

```bash
# Test data generation
python test/test_generation.py

# Test data fitting
python test/test_fitting.py
```

## Test Data

All test data is automatically generated and stored in the `test_data/` subdirectories:

- **`test_data/fitting/`**: Contains synthetic datasets with known parameters for testing fitting algorithms
- **`test_data/generation/`**: Contains various generated datasets for testing generation capabilities

Test data files are organized by test run and include both CSV data files and corresponding plots.
