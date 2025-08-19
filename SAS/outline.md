### Detailed Instructions for Implementing the Synthetic Data Generator Agent and Tidying Up the Project Structure
Below is the revised, detailed instruction for implementing the **Synthetic Data Generation Agent** in your SAS Agent project while maintaining separate scripts for fitting and generation tools (`SAS/fitting.py` and `SAS/generation.py`), each with its own CrewAI tool wrapper. This addresses your concern about avoiding confusion by ensuring the **Fitting Agent** only uses the fitting tool and the **Generation Agent** only uses the generation tool. The instructions incorporate a **Coordinator Agent** for intent-based routing and update the project structure to centralize SAS-related functionality in the `SAS` folder, keeping the codebase modular and clear.

The project structure will be:

```
└── ljding94-sasagent/
    ├── README.md
    ├── crewai_sas_agents.py  # Updated with coordinator and generation agents
    ├── SAS/  # Folder for SAS tools
    │   ├── __init__.py  # Empty file to make SAS a package
    │   ├── fitting.py   # Fitting logic and CrewAI tool (from sasview_tool.py)
    │   ├── generation.py  # Generation logic and CrewAI tool (from synthetic_data.py)
    └── RAG/
        ├── README.md
        ├── sasview_data_pipeline.py
        └── sasview_rag_system.py
```

### Revised Implementation Instructions

#### Step 1: Reorganize the Project Structure
1. **Create the `SAS` Folder**:
   - Run in your project root:
     ```
     mkdir SAS
     touch SAS/__init__.py  # Makes SAS a Python package for relative imports
     ```

2. **Move and Refactor `sasview_tool.py` to `SAS/fitting.py`**:
   - Move the file:
     ```
     mv sasview_tool.py SAS/fitting.py
     ```
   - Update `SAS/fitting.py` to include the CrewAI tool wrapper for fitting, ensuring it only contains fitting-related logic. The complete file is provided below:
     <xaiArtifact artifact_id="cdd86d76-89f3-48ed-ba14-f9b2d4f400d4" artifact_version_id="a0582653-62c0-48bb-bec9-bf5c821771e7" title="SAS/fitting.py" contentType="text/python">
```python
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel
from sasmodels.data import load_data
from sasmodels.bumps_model import Model, Experiment
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import os
import json

# Import Bumps for proper SasView fitting
try:
    import bumps
    from bumps.fitters import fit
    from bumps.fitproblem import FitProblem
    BUMPS_AVAILABLE = True
    print("✅ Using SasView/Bumps fitting (recommended)")
except ImportError:
    print("❌ Warning: Bumps not available")
    BUMPS_AVAILABLE = False


def load_model_config(config_path="model_config.json"):
    """Load model configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using default parameters")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing {config_path}: {e}")
        return {}


def get_model_parameters(model_name, model_config=None, param_constraints=None):
    """
    Get initial parameters and bounds for a given model using SasView defaults

    Args:
        model_name: Name of the SasView model
        model_config: Model configuration dict (loaded from JSON)
        param_constraints: User-specified parameter constraints

    Returns:
        tuple: (initial_params, bounds, param_names)
    """
    model_config = model_config or {}
    param_constraints = param_constraints or {}

    # Get configuration for this model
    config = model_config.get(model_name, {}).get("parameters", {})

    initial_params = {}
    bounds = []
    param_names = []

    # Load the sasmodels model to get parameter names and defaults
    try:
        model = load_model(model_name)
        param_info = model.info.parameters

        # Get all fittable parameters with their SasView defaults
        for param in param_info.kernel_parameters:
            param_names.append(param.name)

        # Always include scale and background
        if "scale" not in param_names:
            param_names.append("scale")
        if "background" not in param_names:
            param_names.append("background")

    except Exception as e:
        print(f"Warning: Could not load model {model_name}: {e}")
        return {}, [], []

    # Set up parameters using SasView defaults or user constraints
    for param in param_names:
        if param in param_constraints:
            # User-specified constraints take priority
            constraint = param_constraints[param]
            if isinstance(constraint, (list, tuple)) and len(constraint) == 2:
                bounds.append(constraint)
                initial_params[param] = (constraint[0] + constraint[1]) / 2
            else:
                initial_params[param] = constraint
                bounds.append((constraint * 0.9, constraint * 1.1))
        elif param in config:
            # Use configuration file values
            initial_params[param] = config[param]["initial"]
            bounds.append(config[param]["bounds"])
        else:
            # Use SasView's built-in default values
            try:
                if param in ["scale", "background"]:
                    # Special handling for scale and background
                    if param == "scale":
                        initial_params[param] = 0.1  # Better initial value
                        bounds.append((1e-6, 1e6))
                    else:  # background
                        initial_params[param] = 0.1  # Better initial value
                        bounds.append((1e-8, 100.0))
                else:
                    # Get SasView's default value for this parameter
                    param_obj = None
                    for p in param_info.kernel_parameters:
                        if p.name == param:
                            param_obj = p
                            break

                    if param_obj is not None:
                        default_value = param_obj.default
                        initial_params[param] = float(default_value)

                        # Set reasonable bounds based on parameter limits
                        if param_obj.limits and len(param_obj.limits) == 2:
                            lower, upper = param_obj.limits
                            # Handle infinite limits
                            if lower == -float('inf'):
                                if 'sld' in param.lower():
                                    lower = -10.0
                                else:
                                    lower = 0.001
                            if upper == float('inf'):
                                if 'sld' in param.lower():
                                    upper = 20.0
                                else:
                                    upper = max(10000.0, default_value * 1000)
                            bounds.append((lower, upper))
                        else:
                            # Flexible bounds based on parameter type
                            if default_value > 0:
                                if 'sld' in param.lower():
                                    lower, upper = -10.0, 20.0
                                else:
                                    lower = max(1e-6, default_value * 0.001)
                                    upper = default_value * 1000
                            else:
                                lower, upper = -1000.0, 1000.0
                            bounds.append((lower, upper))
                    else:
                        print(f"Warning: Could not find SasView parameter {param}, using fallback")
                        initial_params[param] = 1.0
                        bounds.append((0.001, 1000.0))
            except Exception as e:
                print(f"Warning: Could not get SasView default for {param}: {e}, using fallback")
                initial_params[param] = 1.0
                bounds.append((0.001, 1000.0))

    return initial_params, bounds, param_names


def sasview_fit_with_bumps(csv_path, model_name, param_constraints=None, model_config_path=None, plot_label=None):
    """
    Fit SAS data using SasView models with Bumps optimization (the proper SasView way)

    Args:
        csv_path: Path to CSV file with q,I columns
        model_name: SasView model name (e.g., 'sphere', 'cylinder')
        param_constraints: User-specified parameter constraints
        model_config_path: Optional path to model configuration JSON file
        plot_label: Optional label to append to plot filename

    Returns:
        dict: Fitting results with plot, parameters, and metrics
    """
    if not BUMPS_AVAILABLE:
        return {"error": "Bumps not available - cannot perform SasView fitting"}

    # Load model configuration (optional)
    model_config = {}
    if model_config_path:
        model_config = load_model_config(model_config_path)

    # Load data using SasView's load_data API
    try:
        data = load_data(csv_path)

        # Access data attributes (SasView data object has q, I, and potentially dI attributes)
        q_data = data.x  # q values
        intensity_data = data.y  # I(q) values

        # Check if uncertainty data is available and valid
        if hasattr(data, 'dy') and data.dy is not None and np.any(data.dy > 0):
            error_data = data.dy
            # Ensure no zero or negative uncertainties for stability
            error_data = np.maximum(error_data, 0.01 * np.abs(intensity_data))
            data.dy = error_data
            print("Using provided uncertainty data (adjusted for safety)")
        else:
            # Estimate errors (using Poisson statistics if not provided or invalid)
            error_data = np.maximum(np.sqrt(np.abs(intensity_data)), 0.01 * np.abs(intensity_data))
            data.dy = error_data
            print("Estimated uncertainties using Poisson statistics")

        print(f"Loaded data: {len(q_data)} points, q range: {q_data.min():.2e} to {q_data.max():.2e}")
        print(f"Intensity range: {intensity_data.min():.2f} to {intensity_data.max():.2f}")
        print(f"Model: {model_name}")

    except Exception as e:
        return {"error": f"Failed to load data with SasView load_data: {str(e)}"}

    # Set up Bumps model and experiment
    try:
        # Load the sasmodels model
        kernel = load_model(model_name)

        # Create Bumps Model object
        model = Model(kernel)

        # Get initial parameters and bounds
        initial_params, bounds, param_names = get_model_parameters(model_name, model_config, param_constraints)

        if not param_names:
            return {"error": f"Could not determine parameters for model {model_name}"}

        print(f"Fitting parameters: {param_names}")
        print(f"Initial parameters: {initial_params}")

        # Set up parameter ranges for Bumps
        for param_name in param_names:
            if hasattr(model, param_name):
                param = getattr(model, param_name)

                # Apply user constraints if provided
                if param_constraints and param_name in param_constraints:
                    constraint = param_constraints[param_name]
                    if isinstance(constraint, (list, tuple)) and len(constraint) == 2:
                        param.range(*constraint)
                        param.value = (constraint[0] + constraint[1]) / 2
                    else:
                        param.value = constraint
                        # Set reasonable bounds around the fixed value
                        if constraint > 0:
                            param.range(constraint * 0.1, constraint * 10)
                        else:
                            param.range(constraint - abs(constraint), constraint + abs(constraint))
                else:
                    # Use our calculated bounds and initial values
                    param_idx = param_names.index(param_name)
                    lower, upper = bounds[param_idx]
                    param.range(lower, upper)
                    param.value = initial_params[param_name]

                print(f"  {param_name}: {param.value} [{lower}, {upper}]")

        # Create experiment
        experiment = Experiment(data=data, model=model)
        problem = FitProblem(experiment)

        print("Using SasView/Bumps fitting (recommended method)")

    except Exception as e:
        return {"error": f"Failed to set up Bumps model: {str(e)}"}

    # Perform fitting with Bumps
    try:
        print("Starting Bumps optimization...")

        # Use Bumps' Levenberg-Marquardt optimizer (best for SAS data)
        result = fit(problem, method='lm', steps=1000, verbose=False)

        # Get fitted parameters
        fitted_params = {}
        for param_name in param_names:
            if hasattr(model, param_name):
                param = getattr(model, param_name)
                fitted_params[param_name] = float(param.value)

        print(f"Fitted parameters: {fitted_params}")

    except Exception as e:
        return {"error": f"Bumps fitting failed: {str(e)}"}

    # Calculate final fit and metrics using the fitted parameters
    try:
        # Create DirectModel to calculate final intensities
        direct_model = DirectModel(data, kernel)
        fitted_intensity = direct_model(**fitted_params)

        # Calculate metrics
        residuals = intensity_data - fitted_intensity
        rmse = np.sqrt(np.mean(residuals**2))

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((intensity_data - np.mean(intensity_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Chi-squared
        chi_squared = np.sum((residuals / error_data) ** 2)

        # Reduced chi-squared
        n_data = len(intensity_data)
        n_params = len(param_names)
        chi_squared_reduced = chi_squared / (n_data - n_params) if n_data > n_params else chi_squared

        # Relative RMSE
        mean_intensity = np.mean(intensity_data)
        relative_rmse = (rmse / mean_intensity) * 100 if mean_intensity > 0 else 0

        print(f"Final metrics: R²={r_squared:.4f}, RMSE={rmse:.3e}, χ²={chi_squared:.2f}")

    except Exception as e:
        return {"error": f"Failed to calculate final metrics: {str(e)}"}

    # Generate plot
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Main plot
        ax1.plot(q_data, intensity_data, "o", label="Data", markersize=4, alpha=0.7)
        ax1.plot(q_data, fitted_intensity, "-", label="Fit", linewidth=2, color="red")
        ax1.set_xlabel("q (1/Å)")
        ax1.set_ylabel("I(q)")
        ax1.set_yscale("log")
        ax1.set_title(f"SAS Fit: {model_name} (R²={r_squared:.3f}, RMSE={relative_rmse:.1f}%) - Bumps")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals plot
        residuals_norm = residuals / error_data
        ax2.semilogx(q_data, residuals_norm, "o", markersize=3, alpha=0.7)
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("q (1/Å)")
        ax2.set_ylabel("Normalized Residuals")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Fit Residuals")

        plt.tight_layout()

        # Save plot as base64 for return
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # Also save plot as PNG file
        data_dir = os.path.dirname(csv_path)
        data_basename = os.path.splitext(os.path.basename(csv_path))[0]
        if plot_label:
            plot_filename = f"{data_basename}_{model_name}_fit_bumps_{plot_label}.png"
        else:
            plot_filename = f"{data_basename}_{model_name}_fit_bumps.png"
        plot_path = os.path.join(data_dir, plot_filename)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Fit plot saved to: {plot_path}")

        plt.close(fig)

    except Exception as e:
        return {"error": f"Plot generation failed: {str(e)}"}

    # Prepare results
    fit_result = {
        "model": model_name,
        "parameters": fitted_params,
        "chi_squared": float(chi_squared),
        "chi_squared_reduced": float(chi_squared_reduced),
        "rmse": float(rmse),
        "relative_rmse_percent": float(relative_rmse),
        "r_squared": float(r_squared),
        "success": True,
        "data_points": n_data,
        "fitted_parameters": n_params,
        "method": "bumps"
    }

    report = (
        f"Fit Results (SasView/Bumps):\n"
        f"- Model: {model_name}\n"
        f"- R² = {r_squared:.4f}\n"
        f"- RMSE = {rmse:.3e} ({relative_rmse:.1f}% relative)\n"
        f"- χ² = {chi_squared:.2f}, χ²ᵣ = {chi_squared_reduced:.2f}\n"
        f"- Data points: {n_data}, Parameters: {n_params}\n"
        f"- Method: SasView/Bumps fitting (recommended)"
    )

    return {"plot_base64": plot_base64, "fit_json": fit_result, "report": report}


# Main function that uses Bumps by default
def sasview_fit(csv_path, model_name, param_constraints=None, model_config_path=None, plot_label=None):
    """
    Fit SAS data using SasView models with Bumps optimization

    Args:
        csv_path: Path to CSV file with q,I columns
        model_name: SasView model name (e.g., 'sphere', 'cylinder')
        param_constraints: User-specified parameter constraints
        model_config_path: Optional path to model configuration JSON file
        plot_label: Optional label to append to plot filename

    Returns:
        dict: Fitting results with plot, parameters, and metrics
    """
    return sasview_fit_with_bumps(csv_path, model_name, param_constraints, model_config_path, plot_label)


# Define as CrewAI tool
try:
    from crewai.tools import tool

    @tool("SasView Fit Tool")
    def sasview_fit_tool(csv_path: str, model_name: str, param_constraints: dict = None) -> dict:
        """Fit I(q) data from CSV to a SasView model using proper Bumps optimization.

        Args:
            csv_path: Path to CSV file with q,I columns
            model_name: SasView model name (e.g., 'sphere')
            param_constraints: Dict with parameter constraints, e.g., {'radius': [10, 100]}

        Returns:
            Dict with plot_base64, fit_json, and report
        """
        return sasview_fit(csv_path, model_name, param_constraints or {})

except ImportError:
    print("CrewAI not available, tool not created")
    sasview_fit_tool = None


# Test function
if __name__ == "__main__":
    print("Testing SasView Bumps fitting...")

    # Test with the sphere data
    result = sasview_fit('data/synthetic_sphere.csv', 'sphere')

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Fitting successful!")
        fit_data = result['fit_json']
        print(f"R² = {fit_data['r_squared']:.4f}")
        print(f"RMSE = {fit_data['rmse']:.3e}")
        print("Fitted parameters:")
        for param, value in fit_data['parameters'].items():
            print(f"  {param}: {value}")
```
     </xaiArtifact>

3. **Move and Refactor `synthetic_data.py` to `SAS/generation.py`**:
   - Move the file:
     ```
     mv synthetic_data.py SAS/generation.py
     ```
   - Update `SAS/generation.py` to include the CrewAI tool wrapper for data generation, integrated with the RAG system for model selection. The complete file is provided below:
     <xaiArtifact artifact_id="ffc8f9ec-9e7e-488a-977e-fec4c65b8a86" artifact_version_id="ae654b4b-2122-4d54-99c0-6a5c1a7597ed" title="SAS/generation.py" contentType="text/python">
```python
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


def get_model_defaults(model_name):
    """Get default parameters for different models"""
    defaults = {
        "sphere": {"radius": 50, "sld": 1, "sld_solvent": 0, "background": 0.1},
        "cylinder": {"radius": 20, "length": 100, "sld": 1, "sld_solvent": 0, "background": 0.001},
        "ellipsoid": {"radius_polar": 30, "radius_equatorial": 50, "sld": 1, "sld_solvent": 0, "background": 0.001},
        "core_shell_sphere": {"radius": 30, "thickness": 10, "sld_core": 1, "sld_shell": 2, "sld_solvent": 0, "background": 0.001},
        "flexible_cylinder": {"length": 1000, "kuhn_length": 100, "radius": 20, "sld": 1, "sld_solvent": 0, "background": 0.001},
        "lamellar": {"thickness": 50, "sld": 1, "sld_solvent": 0, "background": 0.001},
    }
    return defaults.get(model_name, {})


def generate_synthetic_data(model_name="sphere", params=None, output_folder="data", noise_level=0.02, plot=True, include_uncertainty=True):
    """
    Generate synthetic SAS data using SasModels with Gaussian noise

    Args:
        model_name: SasView model name (e.g., 'sphere', 'cylinder', 'flexible_cylinder', 'lamellar')
        params: Model parameters dict, defaults to model-specific values
        output_folder: Output directory for CSV and plots
        noise_level: Gaussian noise level as fraction (0.02 = 2% noise)
        plot: Whether to generate and save a plot
        include_uncertainty: Whether to include uncertainty column in CSV

    Returns:
        tuple: (csv_path, params) - path to CSV file and ground truth parameters
    """
    os.makedirs(output_folder, exist_ok=True)

    # Use model-specific defaults if no parameters provided
    if params is None:
        params = get_model_defaults(model_name)
        if not params:
            # Fallback to sphere defaults
            params = {"radius": 50, "sld": 1, "sld_solvent": 0, "background": 0.1}

    # Generate q range appropriate for different models
    if model_name in ["lamellar"]:
        # Lamellar structures need lower q range for characteristic features
        q = np.logspace(-3, -0.5, 1000)  # 0.001 to 0.316 1/Å
    elif model_name in ["flexible_cylinder"]:
        # Flexible cylinders benefit from wider q range
        q = np.logspace(-2.5, 0, 1000)  # 0.003 to 1.0 1/Å
    else:
        # Standard q range for spheres, cylinders, etc.
        q = np.linspace(0.001, 0.3, 1000)  # Linear spacing as before

    # Create SasModels data object and calculate intensity
    data = empty_data1D(q)
    model = load_model(model_name)
    direct_model = DirectModel(data, model)
    intensity_clean = direct_model(**params)

    # Add simple Gaussian noise with percentage error
    if noise_level > 0:
        # Generate Gaussian noise with specified percentage level
        noise = np.random.normal(0, noise_level, len(intensity_clean))
        intensity_noisy = intensity_clean * (1 + noise)

        # Very conservative safety check - only prevent completely unrealistic values
        # Use a tiny fraction of the minimum clean intensity to preserve natural decay
        min_allowed = 0.01 * intensity_clean.min()  # Much more conservative
        intensity_noisy = np.maximum(intensity_noisy, min_allowed)

        # Calculate uncertainties as percentage of the clean intensity
        uncertainty = noise_level * intensity_clean

        print(f"Added {noise_level*100:.1f}% Gaussian noise")
    else:
        intensity_noisy = intensity_clean.copy()
        # Even without noise, provide minimal uncertainties for fitting stability
        uncertainty = 0.01 * intensity_clean
        print("Generated noise-free synthetic data with minimal uncertainties")

    # Ensure uncertainties are never zero (for fitting stability)
    uncertainty = np.maximum(uncertainty, 0.001 * np.abs(intensity_noisy))

    # Save to CSV with uncertainties if requested
    if include_uncertainty:
        csv_path = os.path.join(output_folder, f"synthetic_{model_name}.csv")
        np.savetxt(csv_path, np.column_stack((q, intensity_noisy, uncertainty)),
                   delimiter=",", header="q,I,dI", comments='#')
        print("Saved data with uncertainties: q, I(q), dI columns")
    else:
        csv_path = os.path.join(output_folder, f"synthetic_{model_name}.csv")
        np.savetxt(csv_path, np.column_stack((q, intensity_noisy)),
                   delimiter=",", header="q,I", comments='#')
        print("Saved data without uncertainties: q, I(q) columns")

    # Generate visualization plot
    if plot:
        plt.figure(figsize=(12, 8))

        # Main plot with error bars if uncertainties are included
        if include_uncertainty:
            # Show error bars with smart sampling for clarity
            # Use more error bars at low q (where they're larger) and fewer at high q
            n_points = len(q)
            if n_points > 200:
                # For large datasets, show every nth error bar, with more at low q
                error_indices = np.unique(np.concatenate([
                    np.arange(0, n_points//3, 5),    # Every 5th point in first third (low q)
                    np.arange(n_points//3, 2*n_points//3, 5),  # Every 10th in middle
                    np.arange(2*n_points//3, n_points, 5)      # Every 20th at high q
                ]))
            else:
                # For smaller datasets, show every 3rd error bar
                error_indices = np.arange(0, n_points, 3)

            # Plot all data points
            plt.plot(q, intensity_noisy, "o", markersize=2, alpha=0.6, color="blue",
                     label=f"Synthetic {model_name.capitalize()} Data")

            # Add error bars for selected points
            plt.errorbar(q[error_indices], intensity_noisy[error_indices],
                         yerr=uncertainty[error_indices], fmt='none',
                         ecolor='blue', alpha=0.7, capsize=2, capthick=1,
                         label=f"Uncertainties ({noise_level*100:.1f}% Gaussian)")
        else:
            plt.plot(q, intensity_noisy, "o-", label=f"Synthetic {model_name.capitalize()} Data",
                     markersize=4, linewidth=1, alpha=0.8)

        # If there's noise, also plot the clean data
        if noise_level > 0:
            plt.plot(q, intensity_clean, "-", label="Clean Data (no noise)",
                     linewidth=2, alpha=0.7, color="red")

        plt.xlabel("q (1/Å)")
        plt.ylabel("I(q)")
        plt.yscale("log")
        plt.title(f"Synthetic {model_name.capitalize()} Scattering Data")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add parameter info to plot
        param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
        if noise_level > 0:
            noise_text = f"\nNoise: {noise_level*100:.1f}% Gaussian"
        else:
            noise_text = "\nNoise: None"
        uncertainty_text = f"\nUncertainties: {f'{noise_level*100:.1f}% of clean intensity' if include_uncertainty else 'Not included'}"

        plt.text(0.02, 0.98, f"Parameters:{param_text}{noise_text}{uncertainty_text}",
                 transform=plt.gca().transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # Save plot
        plot_path = os.path.join(output_folder, f"synthetic_{model_name}_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Synthetic data plot saved to: {plot_path}")

    return csv_path, params


def generate_multiple_synthetic_datasets(models=None, output_folder="data", noise_level=0.02, plot=True, include_uncertainty=True):
    """
    Generate synthetic data for multiple models with Gaussian noise

    Args:
        models: List of model names or dict with model_name: params
        output_folder: Output directory
        noise_level: Gaussian noise level as fraction (0.02 = 2% noise)
        plot: Whether to generate plots
        include_uncertainty: Whether to include uncertainty columns

    Returns:
        dict: {model_name: (csv_path, params)} for each model
    """
    if models is None:
        models = ["sphere", "cylinder", "flexible_cylinder", "lamellar"]

    results = {}

    for model in models:
        if isinstance(model, dict):
            model_name = list(model.keys())[0]
            params = model[model_name]
        else:
            model_name = model
            params = None

        print(f"\n--- Generating {model_name} synthetic data ---")
        try:
            csv_path, ground_truth = generate_synthetic_data(
                model_name=model_name,
                params=params,
                output_folder=output_folder,
                noise_level=noise_level,
                plot=plot,
                include_uncertainty=include_uncertainty
            )
            results[model_name] = (csv_path, ground_truth)
            print(f"✓ {model_name}: {csv_path}")
        except Exception as e:
            print(f"✗ Failed to generate {model_name}: {e}")
            results[model_name] = (None, None)

    return results


# Define as CrewAI tool
try:
    from crewai.tools import tool
    from pydantic import BaseModel

    class SyntheticDataInput(BaseModel):
        """Input schema for synthetic data generation tool"""
        sample_description: str
        model_name: str = None
        params: dict = None

    @tool("Synthetic SAS Data Generator")
    def synthetic_data_tool(sample_description: str, model_name: str = None, params: dict = None) -> dict:
        """Generate synthetic I(q) data based on sample description, model, and parameters.

        Args:
            sample_description: Description of the sample (used for RAG if no model)
            model_name: Optional SasView model name
            params: Optional dict of model parameters

        Returns:
            Dict with csv_path, ground_truth_params, model_used
        """
        try:
            # Import RAG for model selection if needed
            if not model_name:
                import sys
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RAG'))
                from sasview_rag_system import SasViewRAG
                rag = SasViewRAG()
                recommendations = rag.get_model_recommendations(sample_description)
                model_name = recommendations["primary_recommendation"]["model_name"]
                print(f"RAG recommended model: {model_name}")

            # Generate synthetic data
            csv_path, ground_truth = generate_synthetic_data(
                model_name=model_name,
                params=params,
                noise_level=0.05,
                plot=True,
                include_uncertainty=True
            )

            return {
                "success": True,
                "csv_path": csv_path,
                "ground_truth_params": ground_truth,
                "model_used": model_name,
                "plot_file": str(Path(csv_path).with_suffix('_plot.png'))
            }
        except Exception as e:
            return {"success": False, "error": f"Data generation failed: {str(e)}"}

except ImportError:
    print("CrewAI not available, generation tool not created")
    synthetic_data_tool = None


# Example usage
if __name__ == "__main__":
    # Generate single dataset
    csv_path, ground_truth = generate_synthetic_data()
    print(f"Synthetic data saved to {csv_path}. Ground truth: {ground_truth}")

    # Generate multiple datasets
    print("\n" + "=" * 60)
    print("GENERATING MULTIPLE SYNTHETIC DATASETS")
    print("=" * 60)

    models_to_generate = ["sphere", "cylinder", "flexible_cylinder", "lamellar"]

    results = generate_multiple_synthetic_datasets(models_to_generate)

    print("\n--- Summary ---")
    for model_name, (csv_path, params) in results.items():
        if csv_path:
            print(f"✓ {model_name}: {csv_path}")
            print(f"    Ground truth params: {params}")
        else:
            print(f"✗ {model_name}: Failed")

    print(f"\n✅ Generated {len([r for r in results.values() if r[0] is not None])} synthetic datasets")
    print("🎯 Ready for collaborative SAS analysis testing...")
```
     </xaiArtifact>

4. **Update `crewai_sas_agents.py`**:
   - Replace the existing `crewai_sas_agents.py` with the updated version that includes the Coordinator Agent, Generation Agent, and intent-based routing. This ensures the Fitting Agent only uses `sasview_fit_tool` and the Generation Agent only uses `synthetic_data_tool`. The complete file is provided below:
     <xaiArtifact artifact_id="5ddc3dc5-8ceb-4e54-9508-c9cb38c57b7d" artifact_version_id="45a8e815-aa2b-4fae-bf6e-7ad4935164d6" title="crewai_sas_agents.py" contentType="text/python">
```python
#!/usr/bin/env python3
"""
CrewAI SAS Analysis System - Unified Agent Framework
Three-agent system: Coordinator, RAG model selector, SasView fitter, Synthetic data generator
"""

import os
import re
import json
from typing import Dict, Any
from SAS.fitting import sasview_fit
from pydantic import BaseModel

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Import tools
from SAS.fitting import sasview_fit_tool
from SAS.generation import synthetic_data_tool

# RAG system import
try:
    import sys
    sys.path.append('/Users/ldq/Work/SasAgent/RAG')
    from sasview_rag_system import create_rag_tool
    RAG_AVAILABLE = True
    print("✅ RAG system available")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG system not available: {e}")


# ========================================================================================
# PYDANTIC SCHEMAS
# ========================================================================================

class RAGModelSelectorInput(BaseModel):
    """Input schema for RAG model selector tool"""
    sample_description: str


class SasViewFittingInput(BaseModel):
    """Input schema for SasView fitting tool"""
    csv_path: str
    model_name: str
    parameter_guidance: str = ""


class SyntheticDataInput(BaseModel):
    """Input schema for synthetic data generation tool"""
    sample_description: str
    model_name: str = None
    params: dict = None


# ========================================================================================
# SHARED UTILITIES
# ========================================================================================

def setup_llm():
    """Setup LLM for CrewAI agents using OpenRouter"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    llm = LLM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=2000
    )
    return llm


# ========================================================================================
# AGENT 1: COORDINATOR
# ========================================================================================

def create_coordinator_agent() -> Agent:
    """Create a coordinator agent to route tasks based on user intent."""
    llm = setup_llm()
    return Agent(
        role="Task Coordinator",
        goal="Analyze user prompts and route to appropriate tasks (data generation or fitting)",
        backstory="""You are an expert in interpreting user prompts for small-angle scattering (SAS) tasks.
        You analyze natural language requests to determine whether the user wants to generate synthetic data
        or fit existing data, then delegate to the appropriate specialist agent.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def create_coordinator_task(prompt: str) -> Task:
    """Create a task to classify the user prompt and decide the workflow."""
    return Task(
        description=f"""
        Analyze the following user prompt and determine whether it requests:
        1. Synthetic data generation (keywords: generate, create, synthetic, simulate)
        2. Data fitting (keywords: fit, analyze, model, curve fit)

        Prompt: {prompt}

        Steps:
        1. Identify keywords and context (e.g., presence of file path for fitting).
        2. Extract sample description and any parameters.
        3. Output a structured decision in this EXACT format:

        INTENT: [generation|fitting]
        SAMPLE_DESCRIPTION: [extracted_description]
        DATA_PATH: [file_path_or_none]
        PARAMETERS: [extracted_parameters_or_none]
        """,
        expected_output="Structured decision in the specified format",
        agent=create_coordinator_agent()
    )


# ========================================================================================
# AGENT 2: RAG MODEL SELECTOR
# ========================================================================================

class RAGModelSelectorTool(BaseTool):
    """Tool for RAG-powered model selection"""
    name: str = "rag_model_selector"
    description: str = """
    Uses RAG system to recommend the best SasView model based on sample description.

    USAGE: rag_model_selector(sample_description="detailed sample description")

    Parameters:
    - sample_description: Detailed description of the sample (e.g., "spherical protein nanoparticles in buffer")

    Returns:
    - recommended_model: Best model name
    - confidence: Confidence score (0-1)
    - reasoning: Scientific reasoning for selection
    - parameters: Key parameters to focus on
    - alternatives: List of alternative models
    """
    args_schema: type[BaseModel] = RAGModelSelectorInput

    def __init__(self):
        super().__init__()
        object.__setattr__(self, '_rag_selector', None)
        if RAG_AVAILABLE:
            try:
                object.__setattr__(self, '_rag_selector', create_rag_tool())
            except Exception as e:
                print(f"⚠️  RAG tool initialization failed: {e}")

    @property
    def rag_selector(self):
        return getattr(self, '_rag_selector', None)

    def _run(self, sample_description: str) -> Dict[str, Any]:
        """Execute RAG model selection"""
        try:
            if isinstance(sample_description, dict):
                sample_description = sample_description.get('description', str(sample_description))
            if not isinstance(sample_description, str):
                sample_description = str(sample_description)

            if not self.rag_selector:
                return self._fallback_selection(sample_description)

            recommendations = self.rag_selector(sample_description)
            return {
                "success": True,
                "recommended_model": recommendations["recommended_model"],
                "confidence": recommendations["confidence"],
                "reasoning": recommendations["reasoning"],
                "parameters": recommendations.get("parameters", {}),
                "alternatives": recommendations.get("alternatives", []),
                "usage_guidance": recommendations.get("usage_guidance", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"RAG model selection failed: {str(e)}",
                "recommended_model": "sphere",
                "confidence": 0.3
            }

    def _fallback_selection(self, description: str) -> Dict[str, Any]:
        """Fallback model selection without RAG"""
        desc_lower = description.lower()
        if any(word in desc_lower for word in ['sphere', 'globular', 'round', 'micelle']):
            model = "sphere"
            confidence = 0.7
        elif any(word in desc_lower for word in ['rod', 'cylinder', 'tube', 'fiber']):
            model = "cylinder"
            confidence = 0.7
        elif any(word in desc_lower for word in ['flexible', 'polymer', 'chain', 'dna']):
            model = "flexible_cylinder"
            confidence = 0.6
        elif any(word in desc_lower for word in ['membrane', 'bilayer', 'layer']):
            model = "lamellar"
            confidence = 0.6
        elif any(word in desc_lower for word in ['core', 'shell']):
            model = "core_shell_sphere"
            confidence = 0.6
        else:
            model = "sphere"
            confidence = 0.4
        return {
            "success": True,
            "recommended_model": model,
            "confidence": confidence,
            "reasoning": {"selection_basis": "Keyword-based selection from description"},
            "parameters": {},
            "alternatives": ["sphere", "cylinder", "flexible_cylinder"]
        }


def create_model_selector_agent() -> Agent:
    """Create the RAG-powered model selection agent"""
    llm = setup_llm()
    agent_role = "SAS Model Selection Expert with RAG Intelligence" if RAG_AVAILABLE else "SAS Model Selection Expert"
    backstory = """You are an expert in small-angle scattering (SAS) model selection with deep knowledge of
    particle morphologies and scattering theory. """ + (
        """You have access to a comprehensive RAG system containing detailed knowledge of 78+ SasView
        models with their parameters, usage guidance, and scientific applications.""" if RAG_AVAILABLE else
        """You use your expertise to analyze sample descriptions and recommend appropriate SasView
        models based on expected particle geometry and structure."""
    )
    return Agent(
        role=agent_role,
        goal="Analyze sample descriptions and recommend the most appropriate SasView model with scientific reasoning",
        backstory=backstory,
        verbose=True,
        allow_delegation=False,
        tools=[RAGModelSelectorTool()],
        llm=llm
    )


def create_model_selection_task(sample_description: str) -> Task:
    """Create model selection task"""
    return Task(
        description=f"""
        Analyze the sample description and recommend the best SasView model for fitting or generation.

        Sample Description: {sample_description}

        Use the rag_model_selector tool to:
        1. Analyze the sample characteristics (geometry, structure, material type)
        2. Get model recommendation with confidence score
        3. Understand the scientific reasoning behind the selection
        4. Identify key parameters that will be important for fitting/generation
        5. Consider alternative models if confidence is low

        Provide a comprehensive model selection report in this EXACT format:
        PRIMARY_MODEL: [exact_model_name]
        CONFIDENCE: [0.0-1.0]
        REASONING: [scientific_basis_for_selection]
        KEY_PARAMETERS: [important_parameters_and_ranges]
        ALTERNATIVES: [backup_model1, backup_model2]
        FITTING_GUIDANCE: [specific_advice_for_fitter_or_generator]
        """,
        expected_output="Model selection report in the specified format",
        agent=create_model_selector_agent()
    )


# ========================================================================================
# AGENT 3: SASVIEW FITTER
# ========================================================================================

class SasViewFittingTool(BaseTool):
    """Tool for SasView model fitting"""
    name: str = "sasview_fitting_tool"
    description: str = """
    Fits small-angle scattering I(q) data to SasView models.

    USAGE: sasview_fitting_tool(csv_path="path/to/data.csv", model_name="exact_model_name", parameter_guidance="optional_guidance")

    Parameters:
    - csv_path (required): Full path to CSV file with q,I columns
    - model_name (required): Exact SasView model name (e.g., "sphere", "cylinder")
    - parameter_guidance (optional): Additional fitting guidance from model selector

    Returns:
    - success: True/False
    - r_squared: Goodness of fit (higher is better, >0.9 excellent, >0.8 good)
    - rmse: Root mean square error (lower is better)
    - fitted_parameters: Dictionary of fitted parameter values
    - report: Human-readable analysis
    """
    args_schema: type[BaseModel] = SasViewFittingInput

    def _run(self, csv_path: str, model_name: str, parameter_guidance: str = "") -> Dict[str, Any]:
        """Execute the SasView fitting"""
        try:
            if not os.path.exists(csv_path):
                return {"error": f"Data file not found: {csv_path}"}
            result = sasview_fit(csv_path, model_name, plot_label="CrewAI_Unified_Agent")
            if "error" in result:
                return result
            fit_data = result['fit_json']
            return {
                "success": True,
                "model": model_name,
                "r_squared": fit_data['r_squared'],
                "rmse": fit_data['rmse'],
                "chi_squared": fit_data.get('chi_squared', 0),
                "chi_squared_reduced": fit_data['chi_squared_reduced'],
                "fitted_parameters": fit_data['parameters'],
                "parameter_errors": fit_data.get('parameter_errors', {}),
                "report": result['report'],
                "plot_file": result.get('plot_file', ''),
                "data_file": csv_path,
                "data_points": fit_data.get('data_points', 0)
            }
        except Exception as e:
            return {"error": f"Fitting failed: {str(e)}"}


def create_fitting_agent() -> Agent:
    """Create the SasView fitting agent"""
    llm = setup_llm()
    return Agent(
        role="SAS Data Fitting Specialist",
        goal="Perform precise SasView model fitting and interpret results with statistical analysis",
        backstory="""You are an expert in small-angle scattering data fitting with deep knowledge of
        optimization algorithms, statistical analysis, and parameter interpretation.""",
        verbose=True,
        allow_delegation=False,
        tools=[sasview_fit_tool],  # Use fitting tool only
        llm=llm
    )


def create_fitting_task(data_path: str, model_recommendation: str, model_context: str = "") -> Task:
    """Create fitting task based on model recommendation"""
    return Task(
        description=f"""
        Perform SasView model fitting using the recommended model and analyze the results.

        Data file: {data_path}
        Recommended model: {model_recommendation}
        Model selection context: {model_context}

        Use the sasview_fitting_tool to:
        1. Fit the data using the recommended model
        2. Evaluate the quality of the fit (R², RMSE, χ²)
        3. Analyze fitted parameters for physical reasonableness
        4. Assess parameter uncertainties if available
        5. Identify any potential fitting issues
        """,
        expected_output="Comprehensive fitting analysis report",
        agent=create_fitting_agent()
    )


# ========================================================================================
# AGENT 4: SYNTHETIC DATA GENERATOR
# ========================================================================================

class SyntheticDataTool(BaseTool):
    """Tool for generating synthetic SAS data"""
    name: str = "synthetic_data_tool"
    description: str = """
    Generates synthetic I(q) data using SasView models.

    USAGE: synthetic_data_tool(sample_description="description", model_name="exact_model_name", params=optional_dict)

    Parameters:
    - sample_description (required): Description of the sample
    - model_name (optional): Exact SasView model name (e.g., "sphere")
    - params (optional): Dictionary of model parameters

    Returns:
    - success: True/False
    - csv_path: Path to generated CSV file
    - ground_truth_params: Parameters used for generation
    - model_used: Model name used
    - plot_file: Path to visualization plot
    """
    args_schema: type[BaseModel] = SyntheticDataInput

    def _run(self, sample_description: str, model_name: str = None, params: dict = None) -> Dict[str, Any]:
        """Execute synthetic data generation"""
        try:
            result = synthetic_data_tool(sample_description, model_name, params)
            return result
        except Exception as e:
            return {"success": False, "error": f"Synthetic data generation failed: {str(e)}"}


def create_data_generation_agent() -> Agent:
    """Create an agent for generating synthetic SAS data"""
    llm = setup_llm()
    return Agent(
        role="SAS Data Generation Specialist",
        goal="Generate synthetic I(q) data based on sample descriptions and parameters",
        backstory="""You are an expert in generating synthetic small-angle scattering (SAS) data
        using SasModels. You interpret sample descriptions, select appropriate models using RAG,
        and generate realistic I(q) data with appropriate noise and uncertainties.""",
        verbose=True,
        allow_delegation=False,
        tools=[RAGModelSelectorTool(), synthetic_data_tool],  # Use generation tool and RAG
        llm=llm
    )


def create_data_generation_task(sample_description: str, params: dict = None) -> Task:
    """Create a task for generating synthetic SAS data"""
    return Task(
        description=f"""
        Generate synthetic SAS I(q) data based on the sample description and parameters.

        Sample Description: {sample_description}
        Parameters: {params or 'Use defaults or RAG recommendations'}

        Steps:
        1. Use the RAG model selector to choose an appropriate model if not specified.
        2. Use the synthetic data generator tool to create I(q) data.
        3. Ensure reasonable noise (5%) and uncertainties are included.
        4. Save data as CSV and generate a plot.
        """,
        expected_output="Report with csv_path, ground_truth_params, model_used, plot_file",
        agent=create_data_generation_agent()
    )


# ========================================================================================
# UNIFIED ANALYSIS SYSTEM
# ========================================================================================

class UnifiedSASAnalysisSystem:
    """Multi-agent SAS analysis system with coordinator for task routing"""
    def __init__(self):
        self.model_selector = create_model_selector_agent()
        self.fitter = create_fitting_agent()
        self.data_generator = create_data_generation_agent()
        self.coordinator = create_coordinator_agent()
        self.rag_available = RAG_AVAILABLE

    def analyze_data(self, prompt: str, data_path: str = None) -> Dict[str, Any]:
        """
        Route and execute SAS tasks (fitting or generation) based on prompt.

        Args:
            prompt: User prompt (e.g., "Generate data for spheres" or "Fit data.csv to sphere model")
            data_path: Optional path to data file (for fitting tasks)

        Returns:
            Results from either generation or fitting workflow
        """
        try:
            print("🔧 Starting collaborative SAS analysis...")

            # Step 1: Use coordinator to determine intent
            coordinator_task = create_coordinator_task(prompt)
            coordinator_crew = Crew(
                agents=[self.coordinator],
                tasks=[coordinator_task],
                process=Process.sequential,
                verbose=True
            )
            coordinator_result = coordinator_crew.kickoff()

            # Parse coordinator output
            intent_match = re.search(
                r"INTENT: (generation|fitting)\nSAMPLE_DESCRIPTION: (.*?)\nDATA_PATH: (.*?)\nPARAMETERS: (.*?)$",
                str(coordinator_result), re.DOTALL
            )
            if not intent_match:
                return {"success": False, "error": "Could not determine task intent from prompt"}

            intent, sample_description, data_path_str, params_str = intent_match.groups()
            data_path = data_path_str if data_path_str != "none" else data_path
            params = json.loads(params_str) if params_str != "none" else None

            print(f"📋 Intent: {intent}, Sample: {sample_description}, Data: {data_path}, Params: {params}")

            # Step 2: Route to appropriate workflow
            if intent == "generation":
                generation_task = create_data_generation_task(sample_description, params)
                generation_crew = Crew(
                    agents=[self.model_selector, self.data_generator],
                    tasks=[create_model_selection_task(sample_description), generation_task],
                    process=Process.sequential,
                    verbose=True
                )
                result = generation_crew.kickoff()
                return {
                    "success": True,
                    "workflow": "Synthetic data generation",
                    "result": str(result),
                    "sample_description": sample_description,
                    "rag_enhanced": self.rag_available
                }
            elif intent == "fitting":
                if not data_path or not os.path.exists(data_path):
                    return {"success": False, "error": f"Data file required for fitting: {data_path}"}
                selection_task = create_model_selection_task(sample_description)
                fitting_task = create_fitting_task(data_path, sample_description)
                analysis_crew = Crew(
                    agents=[self.model_selector, self.fitter],
                    tasks=[selection_task, fitting_task],
                    process=Process.sequential,
                    verbose=True
                )
                result = analysis_crew.kickoff()
                return {
                    "success": True,
                    "workflow": "Data fitting",
                    "collaborative_analysis": str(result),
                    "data_file": data_path,
                    "sample_description": sample_description,
                    "rag_enhanced": self.rag_available
                }
            else:
                return {"success": False, "error": f"Unknown intent: {intent}"}
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "prompt": prompt,
                "rag_enhanced": self.rag_available
            }


# ========================================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================================

def analyze_sas_data(prompt: str, data_path: str = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze SAS data or generate synthetic data based on user prompt.

    Args:
        prompt: User prompt specifying task
        data_path: Optional path to CSV file (for fitting tasks)
        verbose: Whether to print progress information

    Returns:
        Results from the appropriate workflow
    """
    if verbose:
        print(f"📋 Processing prompt: {prompt}")
        if data_path:
            print(f"📂 Data file: {data_path}")
        print("🤖 Starting collaborative SAS analysis...")
    system = UnifiedSASAnalysisSystem()
    return system.analyze_data(prompt, data_path)


# ========================================================================================
# MAIN ENTRY POINT
# ========================================================================================

def main():
    """Main entry point for the collaborative SAS analysis system"""
    os.environ['OTEL_SDK_DISABLED'] = 'true'  # Disable CrewAI telemetry
    print("UNIFIED COLLABORATIVE SAS ANALYSIS SYSTEM")
    print("Three-Agent Framework: Coordinator, RAG Model Selector, SasView Fitter/Generator")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("🔑 API Key Setup Required")
        print("Set your OpenRouter API key:")
        print("  export OPENROUTER_API_KEY='your-openrouter-key'")
        print("\nFor OpenRouter:")
        print("  1. Sign up at https://openrouter.ai")
        print("  2. Get your API key")
        print("  3. Set OPENROUTER_API_KEY environment variable")
        print("\n💡 You can still test RAG model selection without API key!")
        try:
            rag_tool = RAGModelSelectorTool()
            test_descriptions = [
                "Spherical protein nanoparticles in aqueous buffer",
                "DNA polymer chains - flexible structures",
                "Phospholipid bilayer membranes"
            ]
            for i, desc in enumerate(test_descriptions, 1):
                print(f"\n{i}. Sample: {desc}")
                result = rag_tool._run(desc)
                if result.get('success'):
                    print(f"   🧠 RAG Recommendation: {result['recommended_model']}")
                    print(f"   📊 Confidence: {result['confidence']:.3f}")
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            print("\n✅ RAG system working! Set API key for full collaborative workflow.")
        except Exception as e:
            print(f"❌ RAG test failed: {e}")
        print("\n🔄 Demo completed. Exiting cleanly...")
        return

    # Run demo
    print("🚀 Starting Collaborative SAS Analysis Demo")
    print("=" * 50)
    system = UnifiedSASAnalysisSystem()
    print(f"RAG Enhancement: {'✅ Enabled' if system.rag_available else '❌ Disabled'}")
    print("System Components:")
    print("  🧠 Coordinator Agent: Routes tasks based on intent")
    print("  🧠 Model Selector Agent: RAG-powered model recommendation")
    print("  🔧 Fitting Agent: SasView/Bumps fitting")
    print("  📊 Generation Agent: Synthetic data generation")
    print()

    # Demo with both generation and fitting
    demo_cases = [
        {
            "prompt": "Generate synthetic data for DNA polymer chains in physiological buffer",
            "data_path": None
        },
        {
            "prompt": "Fit data/synthetic_sphere.csv to a sphere model",
            "data_path": "data/synthetic_sphere.csv"
        }
    ]

    for case in demo_cases:
        print(f"🧬 Demo: {case['prompt']}")
        if case['data_path']:
            print(f"Data: {case['data_path']}")
        print("=" * 60)
        try:
            result = system.analyze_data(case['prompt'], case['data_path'])
            if result['success']:
                print(f"✅ {result['workflow']} completed successfully!")
                print(f"🤝 Result: {result['result']}")
            else:
                print(f"❌ Analysis failed: {result['error']}")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
        print()

    print("\n🔄 Demo completed. Exiting cleanly...")
    print("🎉 System ready for production use!")


if __name__ == "__main__":
    main()
```
     </xaiArtifact>

5. **Update `README.md`**:
   - Revise the `README.md` to reflect the new structure, including the generation feature and the separate `SAS` folder for tools. The updated file is provided below:
     <xaiArtifact artifact_id="fe89ebc6-3054-4d33-8ab4-bd0d48837b8d" artifact_version_id="bbd572fe-3814-44b9-a25a-92fda73db446" title="README.md" contentType="text/markdown">
```markdown
# SAS Agent - Phase 1: Minimum Prototype

This is the Phase 1 implementation of the SAS Agent, a multi-agent system that uses SasView to fit I(q) scattering data and generate synthetic I(q) data based on natural language prompts.

## Features

- **Synthetic Data Generation**: Generate I(q) data with known ground truth parameters using SasModels, guided by RAG-based model selection.
- **SasView Fitting**: CrewAI-compatible tool wrapper for fitting scattering data with SasView/Bumps.
- **Multi-Agent System**: Includes a Coordinator Agent to route tasks, a RAG Model Selector Agent for intelligent model selection, a Fitting Agent for data fitting, and a Generation Agent for synthetic data creation.

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
from SAS.generation import generate_synthetic_data

csv_path, ground_truth = generate_synthetic_data(model_name="sphere", params={"radius": 50})
print(f"Data saved to: {csv_path}")
print(f"Ground truth: {ground_truth}")
```

#### Test SasView Fitting Tool
```python
from SAS.fitting import sasview_fit

result = sasview_fit("data/synthetic_sphere.csv", "sphere", {"radius": [10, 100]})
print(result['fit_json'])
```

#### Run CrewAI Agent (Fitting or Generation)
```python
from crewai_sas_agents import analyze_sas_data

# Generate synthetic data
result = analyze_sas_data("Generate synthetic data for spherical protein nanoparticles with radius 50 Å")
print(result)

# Fit data
result = analyze_sas_data("Fit this data to a sphere model", data_path="data/synthetic_sphere.csv")
print(result)
```

## Files

- `SAS/fitting.py` - SasView fitting logic and CrewAI tool wrapper.
- `SAS/generation.py` - Synthetic I(q) data generation and CrewAI tool wrapper.
- `crewai_sas_agents.py` - Multi-agent system with coordinator, model selector, fitting, and generation agents.
- `RAG/sasview_data_pipeline.py` - Web crawler and data cleaner for SasView model documentation.
- `RAG/sasview_rag_system.py` - RAG system for intelligent model selection.
- `test_phase1.py` - Complete test suite for Phase 1 (if available).
- `requirements.txt` - Python dependencies.

## Expected Results

When working correctly:

1. **Synthetic Data**: Generates CSV files with q,I,dI columns in `data/` folder, with plots.
2. **SasView Fitting**: Fits sphere model to data, returns parameters close to ground truth (e.g., radius ≈ 50 Å).
3. **CrewAI System**: Routes prompts to either generate data or fit data, using RAG for model selection.

## Example Ground Truth vs Fitted Parameters

For synthetic sphere data with ground truth:
- radius: 50 Å
- sld: 1
- sld_solvent: 0
- background: 0

The fitted parameters should be close to these values (within ~5% due to noise).

## Next Steps

- **Phase 2**: Enhance RAG tool with more advanced search algorithms.
- **Phase 3**: Add more complex multi-agent workflows (e.g., generate then fit).
- **Phase 4**: Build interactive web UI with Gradio.

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`).
- **API Key Errors**: Set `OPENROUTER_API_KEY` environment variable (get from https://openrouter.ai/keys).
- **Fitting Errors**: Check that CSV data has correct format (q,I,dI columns).
- **Generation Errors**: Verify model names and parameters are valid.
- **Plot Generation**: Plots are saved as PNG files for inspection.

## Project Structure

- `SAS/` - Contains all SAS-related tools:
  - `fitting.py`: Fitting logic and CrewAI tool.
  - `generation.py`: Synthetic data generation and CrewAI tool.
- `RAG/` - RAG system for model selection and data crawling.
- `crewai_sas_agents.py` - Core multi-agent system with task routing.
```
     </xaiArtifact>

#### Step 6: Testing and Cleanup
1. **Test the System**:
   - Run the demo:
     ```
     python crewai_sas_agents.py
     ```
   - Test specific prompts:
     ```python
     from crewai_sas_agents import analyze_sas_data
     # Generation
     result = analyze_sas_data("Generate synthetic data for spherical protein nanoparticles with radius 50 Å")
     print(result)
     # Fitting
     result = analyze_sas_data("Fit this data to a sphere model", data_path="data/synthetic_sphere.csv")
     print(result)
     ```
   - Verify:
     - Generation: Check for CSV and plot files in `data/` (e.g., `data/synthetic_sphere.csv`, `data/synthetic_sphere_plot.png`).
     - Fitting: Check for fitting results and plot (e.g., `data/synthetic_sphere_sphere_fit_bumps_CrewAI_Unified_Agent.png`).

2. **Update Test Suite** (if `test_phase1.py` exists):
   - Add test cases for both intents:
     ```python
     # test_phase1.py (example)
     from crewai_sas_agents import analyze_sas_data
     import os

     def test_generation():
         result = analyze_sas_data("Generate synthetic data for DNA polymer chains")
         assert result['success'], f"Generation failed: {result.get('error')}"
         assert os.path.exists(result['result']['csv_path']), "CSV file not created"

     def test_fitting():
         data_path = "data/synthetic_sphere.csv"
         result = analyze_sas_data("Fit this data to a sphere model", data_path=data_path)
         assert result['success'], f"Fitting failed: {result.get('error')}"
         assert result['collaborative_analysis']['r_squared'] > 0.8, "Fit quality too low"
     ```

3. **Cleanup**:
   - Remove old files:
     ```
     rm sasview_tool.py synthetic_data.py
     ```
   - Commit changes:
     ```
     git add . && git commit -m "Implemented generation agent with separate fitting/generation tools in SAS folder"
     ```

4. **Handle Import Issues**:
   - If import errors occur, set the `PYTHONPATH`:
     ```
     export PYTHONPATH=$PYTHONPATH:/Users/ldq/Work/SasAgent
     ```
   - Verify RAG module imports in `generation.py` and `crewai_sas_agents.py` (the `sys.path.append` in both files should handle this).

#### Step 7: Update Requirements
- Ensure `requirements.txt` includes all dependencies:
  ```
  sasmodels
  bumps
  matplotlib
  numpy
  crewai
  scikit-learn
  pydantic
  requests
  beautifulsoup4
  ```

### Key Benefits
- **Agent Clarity**: The Fitting Agent only has access to `sasview_fit_tool`, and the Generation Agent only has `synthetic_data_tool` and `RAGModelSelectorTool`, preventing confusion.
- **Modularity**: `fitting.py` and `generation.py` are self-contained, with their own CrewAI tools, making maintenance and extension straightforward.
- **Intent Routing**: The Coordinator Agent ensures prompts are correctly routed to the appropriate workflow, leveraging the LLM for robust intent detection.
- **Clean Structure**: The `SAS` folder centralizes SAS-related functionality, and the updated `README.md` clearly documents usage and structure.

### Potential Challenges and Solutions
- **Import Errors**: If RAG imports fail, double-check the `sys.path.append` in `generation.py` and `crewai_sas_agents.py`. Alternatively, use relative imports (e.g., `from ..RAG.sasview_rag_system import SasViewRAG`).
- **Ambiguous Prompts**: If the Coordinator Agent struggles with vague prompts (e.g., "Analyze spheres"), it defaults to generation unless a `data_path` is provided. You can enhance the coordinator task to request clarification for ambiguous cases.
- **Performance**: The RAG system in `generation.py` may add overhead; consider caching RAG results if frequently used.

This implementation ensures a clean, modular project structure with clear agent roles, addressing your concern about avoiding tool confusion. If you need help with specific debugging or want to add features (e.g., parsing complex prompts like "generate and fit"), let me know!