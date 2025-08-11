# SAS Agent - Phase 1: Minimum Prototype

This is the Phase 1 implementation of the SAS Agent, demonstrating feasibility with a basic CrewAI agent that uses SasView to fit I(q) scattering data.

## Features

- **Synthetic Data Generation**: Generate I(q) data with known ground truth parameters using SasModels
- **SasView Tool**: CrewAI-compatible tool wrapper for fitting scattering data with SasView/Bumps
- **Basic CrewAI Agent**: Single agent that can fit data based on natural language prompts

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenRouter API key (required for CrewAI):
```bash
export OPENROUTER_API_KEY='your-openrouter-api-key-here'
```

Get your API key from: https://openrouter.ai/keys

## Usage

### Quick Test
Run the complete Phase 1 test suite:
```bash
python test_phase1.py
```

### Individual Components

#### Generate Synthetic Data
```python
from synthetic_data import generate_synthetic_data

csv_path, ground_truth = generate_synthetic_data()
print(f"Data saved to: {csv_path}")
print(f"Ground truth: {ground_truth}")
```

#### Test SasView Fitting Tool
```python
from sasview_tool import sasview_fit

result = sasview_fit("data/synthetic_sphere.csv", "sphere", {"radius": [10, 100]})
print(result['fit_json'])
```

#### Run CrewAI Agent
```python
from crewai_prototype import run_analysis

result = run_analysis("Fit this data to a sphere model with radius 10-100 Å", "data/synthetic_sphere.csv")
print(result)
```

## Files

- `synthetic_data.py` - Generate synthetic I(q) data with known parameters
- `sasview_tool.py` - SasView fitting tool wrapper for CrewAI
- `crewai_prototype.py` - Basic CrewAI agent implementation
- `test_phase1.py` - Complete test suite for Phase 1
- `requirements.txt` - Python dependencies

## Expected Results

When working correctly:

1. **Synthetic Data**: Generates CSV files with q,I columns in `data/` folder
2. **SasView Tool**: Fits sphere model to synthetic data, returns parameters close to ground truth (radius ≈ 50 Å)
3. **CrewAI Agent**: Interprets natural language prompts and calls the fitting tool appropriately

## Example Ground Truth vs Fitted Parameters

For synthetic sphere data with ground truth:
- radius: 50 Å
- sld: 1
- sld_solvent: 0
- background: 0

The fitted parameters should be close to these values (within ~5% due to noise).

## Next Steps

- **Phase 2**: Add RAG tool with scraped SasView documentation
- **Phase 3**: Implement multi-agent system
- **Phase 4**: Build interactive web UI with Gradio

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
- **API Key Errors**: Set `OPENROUTER_API_KEY` environment variable (get from https://openrouter.ai/keys)
- **Fitting Errors**: Check that CSV data has correct format (q,I columns)
- **Plot Generation**: Plots are saved as base64 strings and PNG files for inspection
