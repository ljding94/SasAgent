from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel
import numpy as np
import os
import matplotlib.pyplot as plt


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
