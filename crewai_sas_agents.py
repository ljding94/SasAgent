#!/usr/bin/env python3
"""
CrewAI SAS Analysis System - Unified Agent Framework
Multi-agent system: Coordinator + RAG model selector + SasView fitter + Synthetic data generator
"""

import os
import re
import json
from typing import Dict, Any
from pydantic import BaseModel

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Import tools from SAS package
try:
    from SAS.fitting import sasview_fit, sasview_fit_tool
    from SAS.generation import generate_sasview_data
    SAS_TOOLS_AVAILABLE = True
    print("✅ SAS tools available")
except ImportError as e:
    SAS_TOOLS_AVAILABLE = False
    print(f"⚠️  SAS tools not available: {e}")

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
    q_values: list = None  # User-specified q range [q_min, q_max, num_points] or full q array
    q_range: str = None    # String description like "0.01 to 1.0 with 100 points"


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
    - sample_description: Detailed description of the sample (e.g., "spherical protein nanoparticles 50nm radius in buffer")

    Returns:
    - recommended_model: Best model name
    - confidence: Confidence score (0-1)
    - reasoning: Scientific reasoning for selection
    - parameters: Key parameters to focus on with suggested values
    - alternatives: List of alternative models
    - q_suggestions: Suggested q-range for the sample type

    Example: rag_model_selector(sample_description="spherical gold nanoparticles 25nm radius in aqueous buffer solution")
    """
    args_schema: type[BaseModel] = RAGModelSelectorInput

    def __init__(self):
        super().__init__()
        # Initialize RAG selector outside of Pydantic field validation
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
            # Handle different input formats from CrewAI
            if isinstance(sample_description, dict):
                if 'description' in sample_description:
                    sample_description = sample_description['description']
                else:
                    sample_description = str(sample_description)

            if not isinstance(sample_description, str):
                sample_description = str(sample_description)

            if not self.rag_selector:
                # Fallback to basic heuristics
                return self._fallback_selection(sample_description)

            # Use RAG system
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
                "recommended_model": "sphere",  # Safe fallback
                "confidence": 0.3
            }

    def _fallback_selection(self, description: str) -> Dict[str, Any]:
        """Fallback model selection without RAG"""
        desc_lower = description.lower()

        # Simple keyword-based selection
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
        models with their parameters, usage guidance, and scientific applications.

        Your RAG system can:
        - Intelligently recommend models based on sample descriptions
        - Provide confidence scores and scientific reasoning
        - Suggest alternative models and parameter guidance
        - Explain when and why to use specific models

        Use the RAG system to provide scientifically sound model recommendations.""" if RAG_AVAILABLE else
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
        Analyze the sample description and recommend the best SasView model for fitting.

        Sample Description: {sample_description}

        Use the rag_model_selector tool to:
        1. Analyze the sample characteristics (geometry, structure, material type)
        2. Get model recommendation with confidence score
        3. Understand the scientific reasoning behind the selection
        4. Identify key parameters that will be important for fitting
        5. Consider alternative models if confidence is low

        Provide a comprehensive model selection report including:
        - Primary model recommendation with confidence score
        - Scientific reasoning for the selection
        - Key parameters to focus on during fitting
        - Alternative models to consider
        - Fitting guidance and tips
        """,
        expected_output="""
        A model selection report containing:
        1. Primary model recommendation with confidence score
        2. Scientific reasoning for model choice
        3. Key fitting parameters and their expected ranges
        4. Alternative models and when to use them
        5. Specific fitting guidance and potential challenges
        """,
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
    - model_name (required): Exact SasView model name (e.g., "sphere", "cylinder", "surface_fractal")
    - parameter_guidance (optional): Additional fitting guidance from model selector

    Returns:
    - success: True/False
    - r_squared: Goodness of fit (higher is better, >0.9 excellent, >0.8 good)
    - rmse: Root mean square error (lower is better)
    - fitted_parameters: Dictionary of fitted parameter values
    - report: Human-readable analysis

    Example: sasview_fitting_tool(csv_path="data/sample.csv", model_name="sphere", parameter_guidance="Focus on radius parameter")
    """
    args_schema: type[BaseModel] = SasViewFittingInput

    def _run(self, csv_path: str, model_name: str, parameter_guidance: str = "") -> Dict[str, Any]:
        """Execute the SasView fitting"""
        try:
            if not os.path.exists(csv_path):
                return {"error": f"Data file not found: {csv_path}"}

            # Call sasview_fit with agent label
            result = sasview_fit(csv_path, model_name, plot_label="CrewAI_Unified_Agent")

            if "error" in result:
                return result

            # Format result for agent consumption
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
        optimization algorithms, statistical analysis, and parameter interpretation.

        You specialize in:
        - Executing SasView model fits with recommended models
        - Evaluating fit quality using statistical metrics (R², RMSE, χ²)
        - Interpreting fitted parameters with physical meaning
        - Assessing parameter uncertainties and correlations
        - Identifying potential fitting issues and suggesting improvements

        You work with model recommendations from the model selection expert to ensure
        the best possible fitting results.
        """,
        verbose=True,
        allow_delegation=False,
        tools=[SasViewFittingTool()],
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

        If the fit quality is poor (R² < 0.85), suggest:
        - Potential reasons for poor fit
        - Parameter adjustments that might help
        - Whether an alternative model should be considered

        Provide detailed interpretation of:
        - What each fitted parameter means physically
        - Whether parameter values are reasonable for the sample
        - Confidence in the results based on statistical metrics
        """,
        expected_output="""
        A comprehensive fitting analysis report containing:
        1. Fitting statistics (R², RMSE, χ²) with quality assessment
        2. Fitted parameter values with physical interpretation
        3. Parameter uncertainty analysis (if available)
        4. Assessment of fit quality and reliability
        5. Physical reasonableness of results
        6. Recommendations for improvement if needed
        """,
        agent=create_fitting_agent()
    )


# ========================================================================================
# AGENT 4: SYNTHETIC DATA GENERATOR
# ========================================================================================

class SyntheticDataTool(BaseTool):
    """Tool for generating synthetic SAS data with flexible q-range and parameter control"""
    name: str = "synthetic_data_tool"
    description: str = """
    Generates synthetic I(q) data using SasView models with user-specified parameters and q-range.

    USAGE: synthetic_data_tool(sample_description="description", model_name="exact_model_name", params=optional_dict, q_values=optional_list, q_range="optional_string")

    Parameters:
    - sample_description (required): Description of the sample
    - model_name (optional): Exact SasView model name (e.g., "sphere")
    - params (optional): Dictionary of model parameters (e.g., {"radius": 50.0, "sld": 2.0})
    - q_values (optional): List for q-range as [q_min, q_max, num_points] (e.g., [0.01, 1.0, 100])
    - q_range (optional): String description of q-range (e.g., "0.01 to 1.0 with 150 points")

    Returns:
    - success: True/False
    - csv_path: Path to generated CSV file
    - ground_truth_params: Parameters used for generation
    - model_used: Model name used
    - q_info: Information about q-range used
    - plot_file: Path to visualization plot

    Examples:
    - Basic: synthetic_data_tool(sample_description="spherical particles")
    - With params: synthetic_data_tool(sample_description="gold spheres", params={"radius": 25.0})
    - With q-range: synthetic_data_tool(sample_description="spheres", q_values=[0.005, 2.0, 200])
    """
    args_schema: type[BaseModel] = SyntheticDataInput

    def _run(self, sample_description: str, model_name: str = None, params: dict = None,
             q_values: list = None, q_range: str = None) -> Dict[str, Any]:
        """Execute synthetic data generation with enhanced parameter and q-range control"""
        try:
            if SAS_TOOLS_AVAILABLE:
                # Use RAG to select model if not provided
                if not model_name:
                    try:
                        rag_tool = RAGModelSelectorTool()
                        rag_result = rag_tool._run(sample_description)
                        if rag_result.get('success'):
                            model_name = rag_result['recommended_model']
                            # Extract parameter values from RAG result if not provided
                            if not params and 'parameters' in rag_result:
                                params = self._extract_parameters_from_rag(rag_result['parameters'])
                        else:
                            model_name = "sphere"  # fallback
                    except Exception:
                        model_name = "sphere"  # fallback

                # Process q-range specifications
                q_info = self._process_q_range(q_values, q_range, sample_description)

                # Extract any parameter hints from sample description
                if not params:
                    params = self._extract_parameters_from_description(sample_description, model_name)

                # Extract background hints from description and add to params
                if params is None:
                    params = {}
                if 'background' not in params:
                    background = self._extract_background_from_description(sample_description)
                    if background is not None:
                        params['background'] = background

                # Generate synthetic data with enhanced parameters
                output_folder = "data/test_ai_generation"

                # Prepare generation arguments
                gen_args = {
                    "model_name": model_name,
                    "params": params,
                    "output_folder": output_folder,
                    "noise_level": 0.03,  # 3% noise
                    "plot": True,
                    "include_uncertainty": True
                }

                # Add q-range if specified
                if q_info.get('q_array') is not None:
                    gen_args["q_values"] = q_info['q_array']

                csv_path, ground_truth = generate_sasview_data(**gen_args)

                return {
                    "success": True,
                    "csv_path": csv_path,
                    "ground_truth_params": ground_truth,
                    "model_used": model_name,
                    "q_info": q_info,
                    "background_used": ground_truth.get('background'),
                    "plot_file": str(csv_path.replace('.csv', '_plot.png'))
                }
            else:
                return {"success": False, "error": "SAS generation tools not available"}
        except Exception as e:
            return {"success": False, "error": f"Synthetic data generation failed: {str(e)}"}

    def _extract_parameters_from_rag(self, rag_parameters: dict) -> dict:
        """Extract and convert RAG parameter recommendations to usable format"""
        params = {}
        for param_name, param_info in rag_parameters.items():
            if 'default' in param_info:
                try:
                    default_val = param_info['default']
                    if isinstance(default_val, str):
                        try:
                            params[param_name] = float(default_val)
                        except ValueError:
                            params[param_name] = default_val
                    else:
                        params[param_name] = default_val
                except (ValueError, KeyError):
                    continue
        return params

    def _process_q_range(self, q_values: list, q_range: str, description: str) -> dict:
        """Process q-range specifications and extract from description if needed"""
        import numpy as np
        import re

        q_info = {"source": "default", "q_array": None}

        # Process explicit q_values list [q_min, q_max, num_points]
        if q_values and len(q_values) >= 3:
            try:
                q_min, q_max, num_points = q_values[0], q_values[1], int(q_values[2])
                q_array = np.logspace(np.log10(q_min), np.log10(q_max), num_points)
                q_info = {
                    "source": "explicit_list",
                    "q_min": q_min,
                    "q_max": q_max,
                    "num_points": num_points,
                    "q_array": q_array.tolist()
                }
                return q_info
            except (ValueError, TypeError):
                pass

        # Process q_range string descriptions
        if q_range:
            try:
                # Parse patterns like "0.01 to 1.0 with 100 points"
                match = re.search(r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*with\s*(\d+)', q_range.lower())
                if match:
                    q_min, q_max, num_points = float(match.group(1)), float(match.group(2)), int(match.group(3))
                    q_array = np.logspace(np.log10(q_min), np.log10(q_max), num_points)
                    q_info = {
                        "source": "string_description",
                        "q_min": q_min,
                        "q_max": q_max,
                        "num_points": num_points,
                        "q_array": q_array.tolist(),
                        "original_string": q_range
                    }
                    return q_info
            except (ValueError, TypeError):
                pass

        # Extract q-range hints from sample description
        q_hints = self._extract_q_hints_from_description(description)
        if q_hints:
            q_info.update(q_hints)
            return q_info

        # Default: return None to use default q-range in generation function
        return q_info

    def _extract_q_hints_from_description(self, description: str) -> dict:
        """Extract q-range hints from natural language description"""
        import re
        import numpy as np

        desc_lower = description.lower()

        # Look for explicit q-range mentions
        q_patterns = [
            r'q.*?(\d+\.?\d*)\s*to\s*(\d+\.?\d*)',
            r'q.*?range.*?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*Å[-¹]?',
        ]

        for pattern in q_patterns:
            match = re.search(pattern, desc_lower)
            if match:
                try:
                    q_min, q_max = float(match.group(1)), float(match.group(2))
                    # Use 100 points as default
                    q_array = np.logspace(np.log10(q_min), np.log10(q_max), 100)
                    return {
                        "source": "description_extraction",
                        "q_min": q_min,
                        "q_max": q_max,
                        "num_points": 100,
                        "q_array": q_array.tolist()
                    }
                except (ValueError, TypeError):
                    continue

        # Look for instrument or resolution hints
        if any(word in desc_lower for word in ['sans', 'small angle', 'low q']):
            # SANS typically covers 0.01 - 1.0 Å⁻¹
            q_array = np.logspace(-2, 0, 100)  # 0.01 to 1.0
            return {
                "source": "instrument_hint_sans",
                "q_min": 0.01,
                "q_max": 1.0,
                "num_points": 100,
                "q_array": q_array.tolist()
            }
        elif any(word in desc_lower for word in ['saxs', 'high q', 'wide angle']):
            # SAXS can go to higher q
            q_array = np.logspace(-2, 0.5, 100)  # 0.01 to ~3.0
            return {
                "source": "instrument_hint_saxs",
                "q_min": 0.01,
                "q_max": 3.0,
                "num_points": 100,
                "q_array": q_array.tolist()
            }

        return {}

    def _extract_parameters_from_description(self, description: str, model_name: str) -> dict:
        """Extract parameter hints from natural language description"""
        import re

        params = {}
        desc_lower = description.lower()

        # Common parameter patterns
        size_patterns = [
            (r'radius.*?(\d+\.?\d*)\s*(?:nm|Å|angstrom)', 'radius'),
            (r'diameter.*?(\d+\.?\d*)\s*(?:nm|Å|angstrom)', 'radius'),  # Convert diameter to radius
            (r'length.*?(\d+\.?\d*)\s*(?:nm|Å|angstrom)', 'length'),
            (r'thickness.*?(\d+\.?\d*)\s*(?:nm|Å|angstrom)', 'thickness'),
            (r'(\d+\.?\d*)\s*(?:nm|Å|angstrom).*?radius', 'radius'),
            (r'(\d+\.?\d*)\s*(?:nm|Å|angstrom).*?diameter', 'radius'),
            (r'(\d+\.?\d*)\s*(?:nm|Å|angstrom).*?thick', 'thickness'),
        ]

        for pattern, param_name in size_patterns:
            match = re.search(pattern, desc_lower)
            if match:
                try:
                    value = float(match.group(1))
                    # Convert diameter to radius if needed
                    if 'diameter' in pattern and param_name == 'radius':
                        value = value / 2.0
                    params[param_name] = value
                except (ValueError, TypeError):
                    continue

        # SLD patterns (scattering length density)
        sld_patterns = [
            (r'sld.*?(\d+\.?\d*)', 'sld'),
            (r'scattering.*?length.*?density.*?(\d+\.?\d*)', 'sld'),
            (r'contrast.*?(\d+\.?\d*)', 'sld'),
        ]

        for pattern, param_name in sld_patterns:
            match = re.search(pattern, desc_lower)
            if match:
                try:
                    params[param_name] = float(match.group(1))
                except (ValueError, TypeError):
                    continue

        # Volume fraction for interaction models
        if any(word in desc_lower for word in ['volume fraction', 'concentration', 'vol frac']):
            vf_match = re.search(r'(?:volume fraction|concentration|vol frac).*?(\d+\.?\d*)', desc_lower)
            if vf_match:
                try:
                    vf = float(vf_match.group(1))
                    # Convert percentage to fraction if needed
                    if vf > 1.0:
                        vf = vf / 100.0
                    if 'volfraction' in model_name.lower():
                        params['volfraction'] = vf
                except (ValueError, TypeError):
                    pass

        return params

    def _extract_background_from_description(self, description: str) -> float:
        """Extract background level hints from natural language description"""
        import re

        desc_lower = description.lower()

        # Look for explicit background mentions
        bg_patterns = [
            (r'background.*?(\d+\.?\d*)', 'background'),
            (r'incoherent.*?(\d+\.?\d*)', 'background'),
            (r'baseline.*?(\d+\.?\d*)', 'background'),
        ]

        for pattern, _ in bg_patterns:
            match = re.search(pattern, desc_lower)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, TypeError):
                    continue

        # Instrument-specific background levels
        if any(word in desc_lower for word in ['sans', 'neutron']):
            return 0.0005  # Lower background for SANS
        elif any(word in desc_lower for word in ['saxs', 'x-ray', 'synchrotron']):
            return 0.01   # Higher background for SAXS
        elif any(word in desc_lower for word in ['solution', 'aqueous', 'buffer']):
            return 0.001  # Medium background for solution samples

        # Return None to use default
        return None


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
        tools=[RAGModelSelectorTool(), SyntheticDataTool()],  # Use generation tool and RAG
        llm=llm
    )


def create_data_generation_task(sample_description: str, params: dict = None, q_values: list = None) -> Task:
    """Create a task for generating synthetic SAS data with enhanced parameter and q-range control"""
    return Task(
        description=f"""
        Generate synthetic SAS I(q) data based on the sample description with enhanced parameter and q-range control.

        Sample Description: {sample_description}
        Parameters: {params or 'Extract from description or use RAG recommendations'}
        Q-range: {q_values or 'Extract from description or use defaults'}

        Steps:
        1. Use the RAG model selector to choose an appropriate model and extract parameter suggestions.
        2. Extract any parameter values mentioned in the sample description (e.g., "25nm radius").
        3. Extract any q-range specifications from the description (e.g., "q range 0.01 to 1.0").
        4. Use the enhanced synthetic data generator tool to create I(q) data with:
           - Extracted or suggested model parameters
           - Custom q-range if specified
           - Appropriate noise (3%) and uncertainties
        5. Save data as CSV and generate a plot with parameter and q-range information.

        Pay special attention to:
        - Size parameters (radius, diameter, length, thickness) with units
        - Scattering contrast (SLD values)
        - Concentration or volume fraction for interaction models
        - Q-range requirements for specific measurements or instruments
        """,
        expected_output="Report with csv_path, ground_truth_params, model_used, q_info, and plot_file",
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
        self.sas_tools_available = SAS_TOOLS_AVAILABLE

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
                r"INTENT:\s*(generation|fitting)\s*\nSAMPLE_DESCRIPTION:\s*(.*?)\s*\nDATA_PATH:\s*(.*?)\s*\nPARAMETERS:\s*(.*?)(?:\n|$)",
                str(coordinator_result), re.DOTALL
            )
            if not intent_match:
                return {"success": False, "error": "Could not determine task intent from prompt"}

            intent, sample_description, data_path_str, params_str = intent_match.groups()
            data_path = data_path_str.strip() if data_path_str.strip() != "none" else data_path
            params = json.loads(params_str.strip()) if params_str.strip() != "none" else None

            print(f"📋 Intent: {intent}, Sample: {sample_description}, Data: {data_path}, Params: {params}")

            # Step 2: Route to appropriate workflow
            if intent == "generation":
                generation_task = create_data_generation_task(sample_description, params)
                generation_crew = Crew(
                    agents=[self.data_generator],
                    tasks=[generation_task],
                    process=Process.sequential,
                    verbose=True
                )
                results = generation_crew.kickoff()
                return {
                    "success": True,
                    "task_type": "generation",
                    "results": str(results),
                    "sample_description": sample_description,
                    "rag_enhanced": self.rag_available
                }

            elif intent == "fitting":
                if not data_path or not os.path.exists(data_path):
                    return {"success": False, "error": f"Data file not found: {data_path}"}

                # Create collaborative tasks for model selection and fitting
                selection_task = create_model_selection_task(sample_description)
                fitting_task = create_fitting_task(data_path, "from_selector", sample_description)

                analysis_crew = Crew(
                    agents=[self.model_selector, self.fitter],
                    tasks=[selection_task, fitting_task],
                    process=Process.sequential,
                    verbose=True
                )
                results = analysis_crew.kickoff()
                return {
                    "success": True,
                    "task_type": "fitting",
                    "results": str(results),
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
                "data_path": data_path,
                "rag_enhanced": self.rag_available
            }


# ========================================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================================

def analyze_sas_data(prompt: str, data_path: str = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze SAS data or generate synthetic data based on user prompt.

    Args:
        prompt: User prompt specifying task (e.g., "Generate data for spheres" or "Fit my_data.csv to sphere model")
        data_path: Optional path to CSV file (for fitting tasks)
        verbose: Whether to print progress information

    Returns:
        Results from the appropriate workflow

    Example:
        # For generation:
        result = analyze_sas_data("Generate synthetic data for spherical gold nanoparticles")

        # For fitting:
        result = analyze_sas_data("Fit my_data.csv to spherical particles", "my_data.csv")
    """
    if verbose:
        print(f"� Processing prompt: {prompt}")
        if data_path:
            print(f"� Data file: {data_path}")
        print("🤖 Starting collaborative SAS analysis...")

    system = UnifiedSASAnalysisSystem()
    return system.analyze_data(prompt, data_path)


# ========================================================================================
# MAIN ENTRY POINT
# ========================================================================================

def main():
    """Main entry point for the collaborative SAS analysis system"""
    # Disable CrewAI telemetry to prevent hanging
    os.environ['OTEL_SDK_DISABLED'] = 'true'

    print("UNIFIED COLLABORATIVE SAS ANALYSIS SYSTEM")
    print("Multi-Agent Framework: Coordinator + RAG Model Selector + SasView Fitter + Synthetic Data Generator")
    print("=" * 80)

    # Check system status
    system = UnifiedSASAnalysisSystem()
    print("System Status:")
    print(f"  🔑 OpenRouter API: {'✅ Configured' if os.getenv('OPENROUTER_API_KEY') else '❌ Missing'}")
    print(f"  🧠 RAG Enhancement: {'✅ Enabled' if system.rag_available else '❌ Disabled'}")
    print(f"  🔧 SAS Tools: {'✅ Available' if system.sas_tools_available else '❌ Missing'}")
    print()

    if not os.getenv('OPENROUTER_API_KEY'):
        print("🔑 API Key Setup Required")
        print("Set your OpenRouter API key:")
        print("  export OPENROUTER_API_KEY='your-openrouter-key'")
        print("\nFor OpenRouter:")
        print("  1. Sign up at https://openrouter.ai")
        print("  2. Get your API key")
        print("  3. Set OPENROUTER_API_KEY environment variable")
        print()

    print("📋 Usage Examples:")
    print("=" * 40)
    print("# Python API usage:")
    print("from crewai_sas_agents import UnifiedSASAnalysisSystem")
    print("system = UnifiedSASAnalysisSystem()")
    print()
    print("# Generate synthetic data:")
    print("result = system.analyze_data('Generate data for spherical gold nanoparticles')")
    print()
    print("# Fit existing data:")
    print("result = system.analyze_data('Fit my_data.csv to spherical particles', 'path/to/my_data.csv')")
    print()
    print("📊 For Testing:")
    print("  python test_generation.py  # Test generation functionality")
    print("  python test_fitting.py     # Test fitting functionality")
    print()
    print("🎯 System ready for production use!")


if __name__ == "__main__":
    main()
