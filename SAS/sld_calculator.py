# SAS/sld_calculator.py
import logging
from typing import Dict, Optional
from sas.qtgui.Calculators.SldPanel import neutronSldAlgorithm, xraySldAlgorithm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sld(
    formula: str,
    density: float,
    wavelength: Optional[float] = None,
    is_neutron: bool = True
) -> Dict[str, float]:
    """
    Calculate SLD using SasView's calculator.

    Args:
        formula: Molecular formula (e.g., 'H2O', 'H7D3O5', 'aa:AYL').
        density: Mass density in g/cm³.
        wavelength: Neutron/x-ray wavelength in Å (default: 6.0 for neutrons).
        is_neutron: True for neutron SLD, False for x-ray SLD.

    Returns:
        Dict with SLD results (e.g., {'sld_real': -0.56e-6, 'sld_imag': 0.0, ...}).

    Example:
        >>> calculate_sld("H2O", 1.0)
        {'sld_real': -0.56e-6, 'sld_imag': 0.0, 'inc_cross_section': ..., ...}
    """
    try:
        if not formula or not isinstance(density, (int, float)) or density <= 0:
            raise ValueError("Invalid formula or density (must be positive).")

        wavelength = wavelength or 6.0 if is_neutron else None

        if is_neutron:
            result = neutronSldAlgorithm(formula, density, wavelength)
            return {
                "sld_real": result.neutron_sld_real * 1e6,  # Convert Å⁻² to 10⁻⁶ Å⁻²
                "sld_imag": result.neutron_sld_imag * 1e6,  # Convert Å⁻² to 10⁻⁶ Å⁻²
                "inc_cross_section": result.neutron_inc_xs,
                "abs_cross_section": result.neutron_abs_xs,
                "length": result.neutron_length
            }
        else:
            result = xraySldAlgorithm(formula, density, wavelength)
            return {
                "sld_real": result.xray_sld_real * 1e6,  # Convert Å⁻² to 10⁻⁶ Å⁻²
                "sld_imag": result.xray_sld_imag * 1e6   # Convert Å⁻² to 10⁻⁶ Å⁻²
            }
    except Exception as e:
        logger.error(f"SLD calculation failed for formula '{formula}': {str(e)}")
        raise ValueError(f"SLD calculation failed: {str(e)}")


def calculate_sld_fallback(formula: str, density: float, wavelength: float = None, is_neutron: bool = True) -> Dict[str, float]:
    """
    Fallback SLD calculation using periodictable (if SasView fails).

    Args:
        formula: Molecular formula (e.g., 'H2O').
        density: Mass density in g/cm³.
        wavelength: Neutron/x-ray wavelength in Å.
        is_neutron: True for neutron SLD, False for x-ray SLD.

    Returns:
        Dict with basic SLD results.
    """
    try:
        import periodictable as pt
        mol = pt.formula(formula)
        if is_neutron:
            sld = mol.neutron.sld(density=density, wavelength=wavelength or 6.0)
        else:
            sld = mol.xray.sld(density=density)
        return {"sld_real": sld[0] * 1e6, "sld_imag": sld[1] * 1e6}  # Convert Å⁻² to 10⁻⁶ Å⁻²
    except ImportError:
        raise ValueError("Neither SasView SLD calculator nor periodictable is available")
    except Exception as e:
        raise ValueError(f"Fallback SLD calculation failed: {str(e)}")
