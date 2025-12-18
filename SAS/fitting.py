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


def sasview_fit_with_bumps(csv_path, model_name, param_constraints=None, fixed_params=None, model_config_path=None, plot_label=None, output_dir=None):
    """
    Fit SAS data using SasView models with Bumps optimization (the proper SasView way)

    Args:
        csv_path: Path to CSV file with q,I columns
        model_name: SasView model name (e.g., 'sphere', 'cylinder')
        param_constraints: User-specified parameter constraints
        fixed_params: Dictionary of parameters to fix during fitting (e.g., {"sld": 1.0})
        model_config_path: Optional path to model configuration JSON file
        plot_label: Optional label to append to plot filename
        output_dir: Optional directory to save plots (defaults to same dir as data)

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

        # Create a copy of param_names to track which parameters will be fitted
        fitting_param_names = param_names.copy()

        print(f"Fitting parameters: {param_names}")
        print(f"Initial parameters: {initial_params}")

        # Set up parameter ranges for Bumps
        for param_name in param_names:
            if hasattr(model, param_name):
                param = getattr(model, param_name)

                # Apply fixed parameters first (highest priority)
                if fixed_params and param_name in fixed_params:
                    fixed_value = fixed_params[param_name]
                    param.value = fixed_value
                    param.vary = False  # Fix the parameter - do not fit
                    # Remove from fitting_param_names to ensure it's not included in fitting
                    if param_name in fitting_param_names:
                        fitting_param_names.remove(param_name)
                    print(f"  {param_name}: FIXED at {fixed_value}")
                    continue

                # Apply user constraints if provided
                if param_constraints and param_name in param_constraints:
                    constraint = param_constraints[param_name]
                    if isinstance(constraint, (list, tuple)) and len(constraint) == 2:
                        lower, upper = constraint
                        param.range(*constraint)
                        param.value = (constraint[0] + constraint[1]) / 2
                    else:
                        param.value = constraint
                        # Set reasonable bounds around the fixed value
                        if constraint > 0:
                            lower, upper = constraint * 0.1, constraint * 10
                            param.range(lower, upper)
                        else:
                            lower, upper = constraint - abs(constraint), constraint + abs(constraint)
                            param.range(lower, upper)
                else:
                    # Use our calculated bounds and initial values
                    param_idx = param_names.index(param_name)
                    lower, upper = bounds[param_idx]
                    param.range(lower, upper)
                    param.value = initial_params[param_name]

                print(f"  {param_name}: {param.value} [{lower}, {upper}]")

        # Update param_names to only include parameters that will be fitted
        param_names = fitting_param_names
        print(f"Parameters to be fitted: {param_names}")

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

        # Get fitted parameters (include both fitted and fixed parameters)
        fitted_params = {}
        param_uncertainties = {}
        all_param_names = list(initial_params.keys())  # Get all parameter names including fixed ones
        for param_name in all_param_names:
            if hasattr(model, param_name):
                param = getattr(model, param_name)
                fitted_params[param_name] = float(param.value)
                # Get uncertainty for fitted parameters
                if param_name in param_names:  # Only fitted parameters have uncertainties
                    try:
                        # Try to get stderr from result object
                        if hasattr(result, 'dx') and result.dx is not None:
                            param_idx = param_names.index(param_name)
                            if param_idx < len(result.dx):
                                param_uncertainties[param_name] = float(result.dx[param_idx])
                        elif hasattr(result, 'stderr') and result.stderr is not None:
                            param_idx = param_names.index(param_name)
                            if param_idx < len(result.stderr):
                                param_uncertainties[param_name] = float(result.stderr[param_idx])
                        else:
                            param_uncertainties[param_name] = None
                    except Exception as e:
                        print(f"Warning: Could not get uncertainty for {param_name}: {e}")
                        param_uncertainties[param_name] = None
                else:
                    param_uncertainties[param_name] = None  # Fixed parameters don't have uncertainties

        print(f"Fitted parameters: {fitted_params}")
        print(f"Parameter uncertainties: {param_uncertainties}")

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
        n_params = len(fitting_param_names)  # Use only parameters that were actually fitted
        chi_squared_reduced = chi_squared / (n_data - n_params) if n_data > n_params else chi_squared

        # Relative RMSE
        mean_intensity = np.mean(intensity_data)
        relative_rmse = (rmse / mean_intensity) * 100 if mean_intensity > 0 else 0

        print(f"Final metrics: R²={r_squared:.4f}, RMSE={rmse:.3e}, χ²={chi_squared:.2f}")

    except Exception as e:
        return {"error": f"Failed to calculate final metrics: {str(e)}"}

    # Generate plot
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10 / 3 * 1.0, 10 / 3 * 1.6))

        # Main plot with error bars
        ax1.errorbar(q_data, intensity_data, yerr=error_data, fmt="o",
                     markerfacecolor="None", label="Data", markersize=5, alpha=0.7,
                     elinewidth=1, capsize=2, capthick=1)
        # Plot fitted intensity on top of the raw data by giving it a higher z-order
        ax1.plot(q_data, fitted_intensity, "-", label="Fit", linewidth=1.5, color="red", zorder=10)
        ax1.set_xlabel("q (1/Å)", fontsize=9, labelpad=0)
        ax1.set_ylabel("I(q)", fontsize=9, labelpad=0)
        ax1.set_xscale("log")  # Use log scale for q-axis
        ax1.set_yscale("log")
        # ax1.set_title(f"SAS Fit: {model_name} (R²={r_squared:.3f}, RMSE={relative_rmse:.1f}%) - Bumps")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=7)
        ax1.tick_params(axis='both', which='both', direction="in", labelsize=7)

        # Add fitted parameters annotation
        param_text_lines = []
        param_text_lines.append(f"Model: {model_name}")  # Add model name at the top
        for param_name, param_value in fitted_params.items():
            uncertainty = param_uncertainties.get(param_name)

            # Helper function to determine if scientific notation should be used for uncertainty
            def should_use_scientific(value, uncert):
                if uncert is None:
                    return False  # Don't use scientific notation if no uncertainty
                # Use scientific notation only for very small or large uncertainties
                return abs(uncert) < 0.01 or abs(uncert) > 10000

            if param_name in ['scale', 'background']:
                # Format scale and background with scientific notation if small
                if should_use_scientific(param_value, uncertainty):
                    if uncertainty is not None:
                        param_text_lines.append(f"{param_name}: {param_value:.4f} ± {uncertainty:.2e}")
                    else:
                        param_text_lines.append(f"{param_name}: {param_value:.4f}")
                else:
                    if uncertainty is not None:
                        param_text_lines.append(f"{param_name}: {param_value:.4f} ± {uncertainty:.4f}")
                    else:
                        param_text_lines.append(f"{param_name}: {param_value:.4f}")
            elif 'sld' in param_name.lower():
                # Format SLD parameters with appropriate precision
                if should_use_scientific(param_value, uncertainty):
                    if uncertainty is not None:
                        param_text_lines.append(f"{param_name}: {param_value:.2f} ± {uncertainty:.2e}")
                    else:
                        param_text_lines.append(f"{param_name}: {param_value:.2f}")
                else:
                    if uncertainty is not None:
                        param_text_lines.append(f"{param_name}: {param_value:.2f} ± {uncertainty:.2f}")
                    else:
                        param_text_lines.append(f"{param_name}: {param_value:.2f}")
            else:
                # Format structural parameters (radius, length, thickness, etc.)
                if should_use_scientific(param_value, uncertainty):
                    if uncertainty is not None:
                        param_text_lines.append(f"{param_name}: {param_value:.1f} ± {uncertainty:.2e}")
                    else:
                        param_text_lines.append(f"{param_name}: {param_value:.1f}")
                else:
                    if uncertainty is not None:
                        param_text_lines.append(f"{param_name}: {param_value:.1f} ± {uncertainty:.1f}")
                    else:
                        param_text_lines.append(f"{param_name}: {param_value:.1f}")

        param_text = "\n".join(param_text_lines)

        # Add parameter box in upper right corner
        ax1.text(0.05, 0.05, f"{param_text}",
                 transform=ax1.transAxes,
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.6),
                 fontsize=5, zorder=0)

        # Residuals plot
        residuals_norm = residuals / error_data
        ax2.semilogx(q_data, residuals_norm, "o", markersize=3, alpha=0.7)
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("q (1/Å)", fontsize=9, labelpad=0)
        ax2.set_ylabel("Normalized Residuals", fontsize=9, labelpad=0)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='both', direction="in", labelsize=7)

        plt.tight_layout()

        # Save plot as base64 for return
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # Also save plot as PNG file
        if output_dir and os.path.exists(output_dir):
            data_dir = output_dir
        else:
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
        "uncertainties": param_uncertainties,
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
        f"- Method: SasView/Bumps fitting (recommended)\n"
        f"- Fitted Parameters:\n"
    )

    for param_name, param_value in fitted_params.items():
        uncertainty = param_uncertainties.get(param_name)

        # Helper function to determine if scientific notation should be used
        def should_use_scientific_report(value, uncert):
            if uncert is None:
                return False  # Don't use scientific notation if no uncertainty
            # Use scientific notation only for very small or large uncertainties
            return abs(uncert) < 0.01 or abs(uncert) > 10000

        if uncertainty is not None:
            if should_use_scientific_report(param_value, uncertainty):
                report += f"  {param_name}: {param_value:.4f} ± {uncertainty:.2e}\n"
            else:
                report += f"  {param_name}: {param_value:.4f} ± {uncertainty:.4f}\n"
        else:
            report += f"  {param_name}: {param_value:.4f}\n"

    return {"plot_base64": plot_base64, "plot_file": plot_path, "fit_json": fit_result, "report": report}


# Main function that uses Bumps by default
def sasview_fit(csv_path, model_name, param_constraints=None, model_config_path=None, plot_label=None, output_dir=None):
    """
    Fit SAS data using SasView models with Bumps optimization

    Args:
        csv_path: Path to CSV file with q,I columns
        model_name: SasView model name (e.g., 'sphere', 'cylinder')
        param_constraints: User-specified parameter constraints
        model_config_path: Optional path to model configuration JSON file
        plot_label: Optional label to append to plot filename
        output_dir: Optional directory to save plots (defaults to same dir as data)

    Returns:
        dict: Fitting results with plot, parameters, and metrics
    """
    return sasview_fit_with_bumps(csv_path, model_name, param_constraints, model_config_path, plot_label, output_dir)


# Define as CrewAI tool
try:
    from crewai.tools import tool

    @tool("SasView Fit Tool")
    def sasview_fit_crewai_tool(csv_path: str, model_name: str, param_constraints: dict = None) -> dict:
        """Fit I(q) data from CSV to a SasView model using proper Bumps optimization.

        Args:
            csv_path: Path to CSV file with q,I columns
            model_name: SasView model name (e.g., 'sphere')
            param_constraints: Dict with parameter constraints, e.g., {'radius': [10, 100]}

        Returns:
            Dict with plot_base64, fit_json, and report
        """
        return sasview_fit(csv_path, model_name, param_constraints or {})

    sasview_fit_tool = sasview_fit_crewai_tool

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
