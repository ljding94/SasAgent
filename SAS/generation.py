from sasmodels.core import load_model, list_models
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel
import numpy as np
import os
import matplotlib.pyplot as plt


def get_available_models():
    """Get list of all available SasView models"""
    return list_models()


def get_model_info(model_name):
    """Get detailed information about a specific model"""
    try:
        model = load_model(model_name)
        info = {"name": model_name, "title": model.info.title, "description": model.info.description, "parameters": {}, "category": getattr(model.info, "category", "Unknown")}

        # Get parameter information
        for param in model.info.parameters.kernel_parameters:
            info["parameters"][param.name] = {
                "name": param.name,
                "units": param.units,
                "default": param.default,
                "limits": getattr(param, "limits", None),
                "description": getattr(param, "description", ""),
            }

        return info
    except Exception as e:
        print(f"Error getting info for model '{model_name}': {e}")
        return None


def get_model_defaults(model_name):
    """Get default parameters for any SasView model by using the model's own defaults"""
    try:
        # Load the model to get its parameter information
        model = load_model(model_name)

        # Get default parameters from the model itself
        params = {}

        # Get all kernel parameters (the actual model parameters)
        for param in model.info.parameters.kernel_parameters:
            params[param.name] = param.default

        # Add structure factor parameters if they exist
        if hasattr(model.info.parameters, "structure_factor") and model.info.parameters.structure_factor:
            for param in model.info.parameters.structure_factor_parameters:
                params[param.name] = param.default

        return params

    except Exception as e:
        print(f"Warning: Could not get defaults for model '{model_name}': {e}")
        # Fallback to basic sphere-like parameters for unknown models
        return {"radius": 50, "sld": 1, "sld_solvent": 0, "background": 0.01}


def generate_sasview_data(model_name, params=None, output_folder="data", noise_level=0.02, plot=True, q_values=None):
    """
    Simple wrapper to generate I(q) data for any SasView model.

    Args:
        model_name: Any valid SasView model name (e.g., 'sphere', 'hayter_msa', 'core_shell_sphere')
        params: Model parameters dict. If None, uses model's default values.
                Include 'background' in params to set background level (default: 0.01)
        output_folder: Output directory for CSV and plots
        noise_level: Deprecated. Noise is no longer added to synthetic data.
        plot: Whether to generate and save a plot
        q_values: Custom q array. If None, uses default logspace range

    Returns:
        tuple: (csv_path, params) - path to CSV file and ground truth parameters
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get model defaults if no parameters provided
    if params is None:
        params = get_model_defaults(model_name)

    # Ensure background parameter is included in params with realistic default
    if "background" not in params:
        params["background"] = 0.1  # Typical background level

    print(f"Using background level: {params['background']:.4f}")

    # Generate appropriate q range
    if q_values is not None:
        q = np.array(q_values)
    else:
        q = np.linspace(0.001, 1.0, 500)  # Linear spacing from 0.001 to 1.0 1/Å

    # Create SasModels data object and calculate intensity
    data = empty_data1D(q)
    model = load_model(model_name)
    direct_model = DirectModel(data, model)

    try:
        intensity_clean = direct_model(**params)
    except Exception as e:
        print(f"Error calculating intensity for {model_name}: {e}")
        print(f"Parameters used: {params}")
        raise e

    # Noise removed: always use clean intensity
    intensity_noisy = intensity_clean.copy()
    print("Generated noise-free synthetic data")

    # Save to CSV
    csv_path = os.path.join(output_folder, f"synthetic_{model_name}.csv")
    np.savetxt(csv_path, np.column_stack((q, intensity_noisy)), delimiter=",", header="q,I", comments="#")
    print(f"Saved data: q, I(q) columns -> {csv_path}")

    # Generate plot if requested
    if plot:
        plt.figure(figsize=(10 / 3 * 1.0, 10 / 3 * 0.8))

        plt.plot(q, intensity_noisy, "-", lw=1.2, label=f"Synthetic {model_name} data")

        plt.xlabel("q (1/Å)", fontsize=9, labelpad=0)
        plt.ylabel("I(q)", fontsize=9, labelpad=0)
        plt.yscale("log")
        plt.xscale("log")
        plt.tick_params(axis="both", which="both", direction="in", labelsize=7)
        # plt.title(f"Synthetic {model_name} Scattering Data")
        plt.grid(True, alpha=0.3)
        # plt.legend()

        # Add parameter info with model name
        param_text = f"{model_name}\n" + "\n".join([f"{k}: {v:.3g}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in params.items()])
        #plt.text(0.05, 0.05, f"{param_text}", transform=plt.gca().transAxes, verticalalignment="bottom", horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), fontsize=7) #, zorder=0)
        plt.text(0.98, 0.98, f"{param_text}", transform=plt.gca().transAxes, verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), fontsize=7) #, zorder=0)

        plot_path = os.path.join(output_folder, f"synthetic_{model_name}_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_path}")

    return csv_path, params, plot_path if plot else None


# Keep the original function for backward compatibility
def generate_synthetic_data(model_name="sphere", params=None, output_folder="data", noise_level=0.02, plot=True, q_values=None, background=None):
    """
    Generate synthetic SAS data using SasModels with Gaussian noise
    (Backward compatibility wrapper for generate_sasview_data)

    Note: background parameter is deprecated - include 'background' in params dict instead
    """
    # Handle legacy background parameter
    if background is not None:
        if params is None:
            params = {}
        params = params.copy()  # Don't modify the original dict
        params["background"] = background

    return generate_sasview_data(model_name, params, output_folder, noise_level, plot, q_values)


def generate_multiple_synthetic_datasets(models=None, output_folder="data", noise_level=0.02, plot=True, background=None):
    """
    Generate synthetic data for multiple models with Gaussian noise

    Args:
        models: List of model names or dict with model_name: params
        output_folder: Output directory
        noise_level: Gaussian noise level as fraction (0.02 = 2% noise)
        plot: Whether to generate plots
        background: Background level for all models (if None, uses 0.01)
                   Note: Deprecated - include 'background' in individual model params instead

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
                background=background,  # Still support legacy parameter
            )
            results[model_name] = (csv_path, ground_truth)
            print(f"✓ {model_name}: {csv_path}")
        except Exception as e:
            print(f"✗ Failed to generate {model_name}: {e}")
            results[model_name] = (None, None)

    return results


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
    print("🎯 Ready for collaborative SAS analysis testing!")


# Define as CrewAI tool
try:
    from crewai.tools import tool
    from pydantic import BaseModel
    from pathlib import Path

    class SyntheticDataInput(BaseModel):
        """Input schema for synthetic data generation tool"""

        sample_description: str
        model_name: str = None
        params: dict = None

    @tool("Synthetic SAS Data Generator")
    def synthetic_data_tool(sample_description: str, model_name: str = None, params: dict = None) -> dict:
        """Generate synthetic I(q) data for any SasView model.

        Args:
            sample_description: Description of the sample (used for model selection if no model specified)
            model_name: SasView model name (e.g., 'sphere', 'hayter_msa', 'core_shell_sphere')
            params: Optional dict of model parameters (uses model defaults if not provided)

        Returns:
            Dict with csv_path, ground_truth_params, model_used, plot_file
        """
        try:
            # Use simple keyword-based model selection if no model specified
            if not model_name:
                desc_lower = sample_description.lower()
                if any(word in desc_lower for word in ["sphere", "globular", "round", "micelle"]):
                    model_name = "sphere"
                elif any(word in desc_lower for word in ["rod", "cylinder", "tube", "fiber"]):
                    model_name = "cylinder"
                elif any(word in desc_lower for word in ["flexible", "polymer", "chain", "dna"]):
                    model_name = "flexible_cylinder"
                elif any(word in desc_lower for word in ["membrane", "bilayer", "layer"]):
                    model_name = "lamellar"
                elif any(word in desc_lower for word in ["core", "shell"]):
                    model_name = "core_shell_sphere"
                else:
                    model_name = "sphere"
                print(f"Auto-selected model: {model_name}")

            # Generate synthetic data using the simplified wrapper
            output_folder = "data"
            csv_path, ground_truth, plot_path = generate_sasview_data(
                model_name=model_name, params=params, output_folder=output_folder, noise_level=0.03, plot=True  # 3% noise
            )

            return {"success": True, "csv_path": csv_path, "ground_truth_params": ground_truth, "model_used": model_name, "plot_file": str(Path(csv_path).parent / f"synthetic_{model_name}_plot.png")}
        except Exception as e:
            return {"success": False, "error": f"Data generation failed: {str(e)}"}

except ImportError:
    print("CrewAI not available, generation tool not created")
    synthetic_data_tool = None
