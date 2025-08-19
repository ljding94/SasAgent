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

# Disable CrewAI telemetry to prevent connection errors
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['CREWAI_TELEMETRY_DISABLED'] = 'true'
os.environ['DO_NOT_TRACK'] = '1'

# Configure logging to suppress telemetry errors
import logging
logging.basicConfig(level=logging.WARNING)
# Disable specific telemetry loggers
logging.getLogger('crewai.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('urllib3.connectionpool').setLevel(logging.CRITICAL)
logging.getLogger('requests.packages.urllib3').setLevel(logging.CRITICAL)

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Import tools from SAS package
try:
    from SAS.fitting import sasview_fit_with_bumps, sasview_fit_tool
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
    output_dir: str = None


class SyntheticDataInput(BaseModel):
    """Input schema for synthetic data generation tool"""
    sample_description: str
    model_name: str = None
    params: dict = None
    q_values: list = None  # User-specified q range [q_min, q_max, num_points] or full q array
    q_range: str = None    # String description like "0.01 to 1.0 with 100 points"
    folder_path: str = None  # Output folder path for generated files


# ========================================================================================
# SHARED UTILITIES
# ========================================================================================

def setup_llm(api_key: str = None, model: str = None):
    """Setup LLM for CrewAI agents using OpenRouter with configurable API key and model"""
    # Use provided API key or fall back to environment variable
    final_api_key = api_key or os.getenv('OPENROUTER_API_KEY')
    if not final_api_key:
        raise ValueError("No API key provided. Either pass api_key parameter or set OPENROUTER_API_KEY environment variable")

    # Use provided model or fall back to default
    final_model = model or "openai/gpt-4o-mini"

    # For OpenRouter, we need to specify the provider as "openrouter" for LiteLLM
    llm = LLM(
        model=f"openrouter/{final_model}",
        api_key=final_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=2000
    )
    return llm


# ========================================================================================
# AGENT 1: COORDINATOR
# ========================================================================================

def create_coordinator_agent(api_key: str = None, model: str = None) -> Agent:
    """Create a coordinator agent to route tasks based on user intent."""
    llm = setup_llm(api_key, model)
    return Agent(
        role="Task Coordinator",
        goal="Analyze user prompts and route to appropriate tasks (data generation or fitting)",
        backstory="""You are an expert in interpreting user prompts for small-angle scattering (SAS) tasks.
        You analyze natural language requests to determine whether the user wants to generate synthetic data
        or fit existing data, then delegate to the appropriate specialist agent.""",
        verbose=True,
        allow_delegation=False,
        memory=True,  # Enable memory for conversation context
        llm=llm
    )


def create_coordinator_task(prompt: str, folder_path: str = None, data_path: str = None, api_key: str = None, model: str = None) -> Task:
    """Create a task to classify the user prompt and decide the workflow."""
    return Task(
        description=f"""
        Analyze the following user prompt and determine whether it requests:
        1. Synthetic data generation (keywords: generate, create, synthetic, simulate)
        2. Data fitting (keywords: fit, analyze, model, curve fit)

        Prompt: {prompt}
        Output Folder: {folder_path or 'Use default output folder'}
        Data File Path: {data_path or 'none'}

        Steps:
        1. Identify keywords and context (e.g., presence of file path for fitting).
        2. Extract sample description and any parameters.
        3. If a data file path is provided above, use that path. Otherwise, extract from prompt if present.
        4. Output a structured decision in this EXACT format:

        INTENT: [generation|fitting]
        SAMPLE_DESCRIPTION: [extracted_description]
        DATA_PATH: [use_provided_data_file_path_or_extract_from_prompt_or_none]
        PARAMETERS: [extracted_parameters_or_none]
        """,
        expected_output="Structured decision in the specified format",
        agent=create_coordinator_agent(api_key, model)
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


def create_model_selector_agent(api_key: str = None, model: str = None) -> Agent:
    """Create the RAG-powered model selection agent"""
    llm = setup_llm(api_key, model)

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
        memory=True,  # Enable memory for conversation context
        tools=[RAGModelSelectorTool()],
        llm=llm
    )


def create_model_selection_task(sample_description: str, folder_path: str = None, api_key: str = None, model: str = None) -> Task:
    """Create model selection task"""
    return Task(
        description=f"""
        Analyze the sample description and select the most appropriate SAS model for data analysis.

        Sample Description: {sample_description}
        Output Folder: {folder_path or 'Use default output folder'}

        Steps:
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

        🎯 **Recommended Model:**
        - Primary model choice with confidence level
        - Scientific reasoning for the selection

        🔧 **Key Parameters:**
        - Important fitting parameters and expected ranges
        - Physical significance of each parameter

        🔄 **Alternatives:**
        - Alternative models to consider
        - When to use each alternative

        💡 **Fitting Guidance:**
        - Specific tips for successful fitting
        - Potential challenges and solutions

        Present in a clear, actionable format for the user.
        """,
        agent=create_model_selector_agent(api_key, model)
    )


# ========================================================================================
# AGENT 3: SASVIEW FITTER
# ========================================================================================

class SasViewFittingTool(BaseTool):
    """Tool for SasView model fitting"""

    name: str = "sasview_fitting_tool"
    description: str = """
    Fits small-angle scattering I(q) data to SasView models.

    USAGE: sasview_fitting_tool(csv_path="path/to/data.csv", model_name="exact_model_name", parameter_guidance="optional_guidance", output_dir="optional_output_dir")

    Parameters:
    - csv_path (required): Full path to CSV file with q,I columns
    - model_name (required): Exact SasView model name (e.g., "sphere", "cylinder", "surface_fractal")
    - parameter_guidance (optional): Additional fitting guidance from model selector
    - output_dir (optional): Directory to save plot files (defaults to cache/plots if available)

    Returns:
    - success: True/False
    - r_squared: Goodness of fit (higher is better, >0.9 excellent, >0.8 good)
    - rmse: Root mean square error (lower is better)
    - fitted_parameters: Dictionary of fitted parameter values
    - report: Human-readable analysis

    Example: sasview_fitting_tool(csv_path="data/sample.csv", model_name="sphere", parameter_guidance="Focus on radius parameter")
    """
    args_schema: type[BaseModel] = SasViewFittingInput

    def _run(self, csv_path: str, model_name: str, parameter_guidance: str = "", output_dir: str = None) -> Dict[str, Any]:
        """Execute the SasView fitting"""
        try:
            if not os.path.exists(csv_path):
                return {"error": f"Data file not found: {csv_path}"}

            # Always use plots directory for fitting output (consistent with test scripts)
            if not output_dir:
                # Try to use cache/plots directory if we're in the web app context
                cache_plots_dir = os.path.join(os.getcwd(), "cache", "plots")
                if os.path.exists(cache_plots_dir):
                    output_dir = cache_plots_dir
                else:
                    # Fallback to creating plots directory
                    output_dir = os.path.join(os.getcwd(), "plots")
                    os.makedirs(output_dir, exist_ok=True)
            else:
                # Even if output_dir is provided, redirect fitting plots to plots directory
                # This ensures consistency between test scripts and web app
                cache_plots_dir = os.path.join(os.getcwd(), "cache", "plots")
                if os.path.exists(cache_plots_dir):
                    output_dir = cache_plots_dir
                else:
                    # If cache/plots doesn't exist, but output_dir was provided, use it
                    pass

            # Parse parameter constraints from guidance text
            param_constraints = self._parse_parameter_guidance(parameter_guidance)

            print(f"📁 Fitting plots will be saved to: {output_dir}")

            # Call sasview_fit with agent label, output directory, and parsed constraints
            result = sasview_fit_with_bumps(
                csv_path,
                model_name,
                param_constraints=param_constraints if param_constraints else None,
                plot_label="CrewAI_Unified_Agent",
                output_dir=output_dir
            )

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
                "data_points": fit_data.get('data_points', 0),
                "parameter_constraints": param_constraints  # Include constraints in result
            }

        except Exception as e:
            return {"error": f"Fitting failed: {str(e)}"}

    def _parse_parameter_guidance(self, guidance_text: str) -> Dict[str, float]:
        """Parse parameter constraints from natural language guidance"""
        if not guidance_text:
            return {}

        import re
        constraints = {}
        text_lower = guidance_text.lower()

        # Look for "kuhn length" pattern first (most specific)
        kuhn_match = re.search(r'kuhn\s*length\s*(?:to|=|:)?\s*([0-9.]+)', text_lower)
        if kuhn_match:
            constraints['kuhn_length'] = float(kuhn_match.group(1))

        # Look for standalone "length" pattern (but not "kuhn length")
        length_match = re.search(r'(?<!kuhn\s)(?<!kuhn_)length\s*(?:to|=|:)?\s*([0-9.]+)', text_lower)
        if length_match:
            constraints['length'] = float(length_match.group(1))

        # Look for explicit parameter = value patterns
        explicit_params = re.findall(r'(\w+)\s*[=:]\s*([0-9.]+)', text_lower)
        for param_name, value in explicit_params:
            if param_name in ['kuhn_length', 'length', 'radius', 'scale', 'background']:
                constraints[param_name] = float(value)

        return constraints


def create_fitting_agent(api_key: str = None, model: str = None) -> Agent:
    """Create the SasView fitting agent"""
    llm = setup_llm(api_key, model)

    return Agent(
        role="SAS Data Fitting Specialist",
        goal="Perform precise SasView model fitting and interpret results with statistical analysis",
        backstory="""You are an expert in small-angle scattering data fitting with deep knowledge of
        optimization algorithms, statistical analysis, and parameter interpretation.

        You specialize in:
        - Executing SasView model fits with recommended models
        - Evaluating fit quality using statistical metrics (R^2, RMSE, chi^2)
        - Interpreting fitted parameters with physical meaning
        - Assessing parameter uncertainties and correlations
        - Identifying potential fitting issues and suggesting improvements

        You work with model recommendations from the model selection expert to ensure
        the best possible fitting results.
        """,
        verbose=True,
        allow_delegation=False,
        memory=True,  # Enable memory for conversation context
        tools=[SasViewFittingTool()],
        llm=llm
    )


def create_fitting_task(data_path: str, model_recommendation: str = "auto", model_context: str = "", folder_path: str = None, api_key: str = None, model: str = None, original_prompt: str = "") -> Task:
    """Create fitting task based on model recommendation"""
    return Task(
        description=f"""
        Perform SasView model fitting using the recommended model and analyze the results.

        Data file: {data_path}
        Recommended model: {model_recommendation}
        Model selection context: {model_context}
        Original user prompt: {original_prompt}
        Output Folder: {folder_path or 'Use default output folder'}

        IMPORTANT: Parse the original user prompt for parameter constraints or guidance.
        Look for phrases like "kuhn length 10", "length 100", "radius = 50", etc.
        Use this parameter guidance when calling the sasview_fitting_tool.

        Use the sasview_fitting_tool to:
        1. Fit the data using the recommended model
        2. Apply any parameter constraints extracted from the user prompt
        3. Evaluate the quality of the fit (R^2, RMSE, chi^2)
        4. Analyze fitted parameters for physical reasonableness
        5. Assess parameter uncertainties if available
        6. Identify any potential fitting issues

        CRITICAL: Always explicitly state which model was used for fitting at the beginning of your results.
        The model name should be prominently displayed in the "Fitting Results" section.

        If the fit quality is poor (R^2 < 0.85), suggest:
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

        📊 **Fitting Results:**
        - Model used for fitting (e.g., "sphere", "cylinder", etc.)
        - Fit quality metrics (R^2, RMSE, chi^2) with assessment
        - Overall quality rating (Excellent/Good/Fair/Poor)

        🔧 **Model Parameters:**
        - Fitted parameter values with physical interpretation
        - Parameter uncertainties (if available)
        - Physical reasonableness assessment

        🔬 **Scientific Analysis:**
        - What the results tell us about the sample
        - Confidence level in the results
        - Key insights from the fitted parameters

        💡 **Recommendations:**
        - Assessment of fit reliability
        - Suggestions for improvement if needed
        - Alternative approaches if applicable

        IMPORTANT: Always explicitly state which model was used for fitting in the Fitting Results section.
        Present results in a clear, user-friendly format focusing on scientific insights.
        """,
        agent=create_fitting_agent(api_key, model)
    )


# ========================================================================================
# AGENT 4: SYNTHETIC DATA GENERATOR
# ========================================================================================

class SyntheticDataTool(BaseTool):
    """Tool for generating synthetic SAS data with flexible q-range and parameter control"""
    name: str = "synthetic_data_tool"
    description: str = """
    Generates synthetic I(q) data using SasView models with user-specified parameters and q-range.

    CRITICAL UNIT CONVERSION RULES:
    - ALL length parameters MUST be in Angstroms (Å) in SasView
    - Convert nm to Å by multiplying by 10 (e.g., 30nm = 300Å)
    - Convert μm to Å by multiplying by 10000 (e.g., 2μm = 20000Å)
    - SLD values are in 10⁻⁶ Å⁻²
    - Q-values are in Å⁻¹

    USAGE: synthetic_data_tool(sample_description="description", model_name="exact_model_name", params=optional_dict, q_values=optional_list, q_range="optional_string", folder_path="output_folder")

    Parameters:
    - sample_description (required): Description of the sample
    - model_name (optional): Exact SasView model name (e.g., "sphere")
    - params (optional): Dictionary of model parameters (e.g., {"radius": 300.0, "sld": 2.0}) - RADIUS IN ANGSTROMS!
    - q_values (optional): DO NOT USE unless specifically requested. Tool automatically calculates optimal q-range based on particle size (0.01/L to 10/L where L is characteristic size)
    - q_range (optional): String description if custom q-range needed (e.g., "0.01 to 1.0 with 150 points")
    - folder_path (optional): Output folder path for generated files (default: "data/test_ai_generation")

    SMART Q-RANGE SELECTION:
    - Tool automatically calculates optimal q-range based on characteristic size
    - For spheres: uses radius as characteristic size
    - For cylinders: uses radius as characteristic size
    - For lamellae: uses thickness as characteristic size
    - Formula: q_min = 0.01/L, q_max = 10/L where L is characteristic size in Angstroms
    - ONLY provide q_values if user specifically requests a custom range

    Returns:
    - success: True/False
    - csv_path: Path to generated CSV file
    - ground_truth_params: Parameters used for generation
    - model_used: Model name used
    - q_info: Information about q-range used
    - plot_file: Path to visualization plot

    Examples:
    - Basic: synthetic_data_tool(sample_description="spherical particles")
    - With params: synthetic_data_tool(sample_description="gold spheres 25nm radius", params={"radius": 250.0}) # 25nm = 250Å
    - With q-range: synthetic_data_tool(sample_description="spheres", q_values=[0.005, 2.0, 200])
    - With folder: synthetic_data_tool(sample_description="spheres", folder_path="/path/to/output")
    """
    args_schema: type[BaseModel] = SyntheticDataInput

    def _run(self, sample_description: str, model_name: str = None, params: dict = None,
             q_values: list = None, q_range: str = None, folder_path: str = None) -> Dict[str, Any]:
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

                # Process q-range specifications (initial pass)
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

                # Apply unit conversion to ensure all length parameters are in Angstroms
                params = self._convert_units_to_angstroms(params, sample_description)

                # If no specific q-range was provided, calculate smart q-range based on parameters
                if q_info.get("source") == "default" or not q_info.get("q_array"):
                    smart_q_info = self._calculate_smart_q_range(model_name, params, sample_description)
                    if smart_q_info.get("q_array"):
                        q_info = smart_q_info

                # Generate synthetic data with enhanced parameters
                # Use provided folder_path or default to legacy directory
                output_folder = folder_path or "data/test_ai_generation"

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

                csv_path, ground_truth, plot_path = generate_sasview_data(**gen_args)

                return {
                    "success": True,
                    "csv_path": csv_path,
                    "ground_truth_params": ground_truth,
                    "model_used": model_name,
                    "q_info": q_info,
                    "background_used": ground_truth.get('background'),
                    "plot_file": plot_path if plot_path else ""
                }
            else:
                return {"success": False, "error": "SAS generation tools not available"}
        except Exception as e:
            return {"success": False, "error": f"Synthetic data generation failed: {str(e)}"}

    def _convert_units_to_angstroms(self, params: dict, sample_description: str) -> dict:
        """Convert length parameters from nm/μm to Angstroms based on description context"""
        if not params:
            return params

        converted_params = params.copy()

        # Common length parameter names in SasView models
        length_params = ['radius', 'radius_core', 'radius_shell', 'length', 'thickness',
                         'radius_effective', 'radius_lg', 'radius_sm', 'kuhn_length',
                         'radius_bell', 'thickness_shell', 'core_radius', 'shell_thickness']

        # Check description for unit indicators
        description_lower = sample_description.lower()
        has_nm = 'nm' in description_lower
        has_um = 'μm' in description_lower or 'um' in description_lower
        has_angstrom = 'å' in description_lower or 'angstrom' in description_lower

        # Extract numeric values with units from description
        import re

        # Pattern to match numbers with units (e.g., "25nm", "2.5μm", "50 nm")
        unit_pattern = r'(\d+\.?\d*)\s*(nm|μm|um|å|angstrom)'
        matches = re.findall(unit_pattern, description_lower)

        unit_conversions = {}
        for value_str, unit in matches:
            try:
                value = float(value_str)
                if unit == 'nm':
                    unit_conversions[value] = value * 10  # nm to Å
                elif unit in ['μm', 'um']:
                    unit_conversions[value] = value * 10000  # μm to Å
                elif unit in ['å', 'angstrom']:
                    unit_conversions[value] = value  # already in Å
            except ValueError:
                continue

        # Apply conversions to length parameters
        for param_name in length_params:
            if param_name in converted_params:
                param_value = converted_params[param_name]
                try:
                    param_value = float(param_value)

                    # Check if this value needs conversion based on description context
                    if param_value in unit_conversions:
                        converted_params[param_name] = unit_conversions[param_value]
                        print(f"🔄 Converted {param_name}: {param_value} → {unit_conversions[param_value]} Å")
                    elif has_nm and not has_angstrom and param_value < 1000:
                        # Likely in nm, convert to Å
                        converted_params[param_name] = param_value * 10
                        print(f"🔄 Auto-converted {param_name}: {param_value}nm → {param_value * 10} Å")
                    elif has_um and param_value < 100:
                        # Likely in μm, convert to Å
                        converted_params[param_name] = param_value * 10000
                        print(f"🔄 Auto-converted {param_name}: {param_value}μm → {param_value * 10000} Å")

                except (ValueError, TypeError):
                    # Keep original value if conversion fails
                    pass

        return converted_params

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
                q_array = np.linspace(q_min, q_max, num_points)
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
                    q_array = np.linspace(q_min, q_max, num_points)
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
                    q_array = np.linspace(q_min, q_max, 100)
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
            q_array = np.linspace(0.01, 1.0, 100)  # 0.01 to 1.0
            return {
                "source": "instrument_hint_sans",
                "q_min": 0.01,
                "q_max": 1.0,
                "num_points": 100,
                "q_array": q_array.tolist()
            }
        elif any(word in desc_lower for word in ['saxs', 'high q', 'wide angle']):
            # SAXS can go to higher q
            q_array = np.linspace(0.01, 3.0, 100)  # 0.01 to ~3.0
            return {
                "source": "instrument_hint_saxs",
                "q_min": 0.01,
                "q_max": 3.0,
                "num_points": 100,
                "q_array": q_array.tolist()
            }

        return {}

    def _calculate_smart_q_range(self, model_name: str, params: dict, description: str) -> dict:
        """Calculate smart q-range based on characteristic size: q = 0.01/L to 10/L"""
        import numpy as np

        # Model-specific characteristic size mapping
        size_param_map = {
            # Spherical models
            'sphere': 'radius',
            'fuzzy_sphere': 'radius',
            'core_shell_sphere': 'radius',
            'vesicle': 'radius',
            'hollow_sphere': 'radius',
            'multilayer_vesicle': 'radius',
            'onion': 'radius',

            # Cylindrical models
            'cylinder': 'radius',  # Use radius as characteristic size for cylinders
            'hollow_cylinder': 'radius',
            'core_shell_cylinder': 'radius',
            'barbell': 'radius',
            'capped_cylinder': 'radius',
            'flexible_cylinder': 'radius',
            'flexible_cylinder_elliptical': 'radius',
            'worm_like_chain': 'radius',

            # Ellipsoidal models
            'ellipsoid': 'radius_polar',  # Use the smaller radius
            'triaxial_ellipsoid': 'radius_polar',
            'core_shell_ellipsoid': 'radius_polar',

            # Lamellar models
            'lamellar': 'thickness',
            'lamellar_hg': 'thickness',
            'lamellar_hg_stack_caille': 'thickness',
            'lamellar_stack_caille': 'thickness',
            'lamellar_stack_paracrystal': 'thickness',

            # Parallelepipeds
            'parallelepiped': 'length_a',  # Use smallest dimension
            'rectangular_prism': 'length_a',
            'hollow_rectangular_prism': 'length_a',

            # Other models
            'guinier': 'radius',  # Rg can be treated as characteristic size
            'power_law': None,  # No characteristic size
        }

        # Get characteristic size parameter name for this model
        size_param = size_param_map.get(model_name)

        if not size_param or not params:
            # No characteristic size available, use instrument hints or default
            return self._get_instrument_based_q_range(description)

        # Try to get the characteristic size from parameters
        characteristic_size = None

        # Look for the specific parameter
        if size_param in params:
            try:
                characteristic_size = float(params[size_param])
            except (ValueError, TypeError):
                pass

        # For ellipsoidal models, try alternative radius parameters
        if not characteristic_size and model_name in ['ellipsoid', 'triaxial_ellipsoid']:
            for alt_param in ['radius', 'radius_equatorial', 'radius_polar']:
                if alt_param in params:
                    try:
                        characteristic_size = float(params[alt_param])
                        break
                    except (ValueError, TypeError):
                        continue

        # For cylinder models, consider both radius and length, use the smaller one
        if not characteristic_size and 'cylinder' in model_name.lower():
            radius = params.get('radius')
            length = params.get('length')
            if radius and length:
                try:
                    # Use the smaller dimension as characteristic size
                    characteristic_size = min(float(radius), float(length))
                except (ValueError, TypeError):
                    pass
            elif radius:
                try:
                    characteristic_size = float(radius)
                except (ValueError, TypeError):
                    pass

        # Extract size from description if not found in parameters
        if not characteristic_size:
            characteristic_size = self._extract_size_from_description(description, model_name)

        if not characteristic_size:
            # No characteristic size found, use instrument-based q-range
            return self._get_instrument_based_q_range(description)

        # Calculate smart q-range: 0.01/L to 10/L
        q_min = 0.01 / characteristic_size
        q_max = 10.0 / characteristic_size

        # Apply reasonable bounds to avoid extreme ranges
        q_min = max(q_min, 0.001)  # Don't go below 0.001 Å⁻¹
        q_max = min(q_max, 10.0)   # Don't go above 10 Å⁻¹

        # Ensure q_min < q_max
        if q_min >= q_max:
            q_min = 0.01
            q_max = 1.0

        # Generate q-array with 150 points for good resolution
        num_points = 150
        q_array = np.linspace(q_min, q_max, num_points)

        return {
            "source": "smart_size_based",
            "q_min": q_min,
            "q_max": q_max,
            "num_points": num_points,
            "q_array": q_array.tolist(),
            "characteristic_size": characteristic_size,
            "size_parameter": size_param
        }

    def _extract_size_from_description(self, description: str, model_name: str) -> float:
        """Extract characteristic size from description text"""
        import re

        desc_lower = description.lower()

        # Size patterns with unit conversion
        size_patterns = [
            r'radius.*?(\d+\.?\d*)\s*(nm|μm|um|å|angstrom)',
            r'diameter.*?(\d+\.?\d*)\s*(nm|μm|um|å|angstrom)',
            r'thickness.*?(\d+\.?\d*)\s*(nm|μm|um|å|angstrom)',
            r'length.*?(\d+\.?\d*)\s*(nm|μm|um|å|angstrom)',
            r'(\d+\.?\d*)\s*(nm|μm|um|å|angstrom).*?radius',
            r'(\d+\.?\d*)\s*(nm|μm|um|å|angstrom).*?diameter',
            r'(\d+\.?\d*)\s*(nm|μm|um|å|angstrom).*?thick',
        ]

        for pattern in size_patterns:
            match = re.search(pattern, desc_lower)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Convert to Angstroms
                    if unit == 'nm':
                        value *= 10
                    elif unit in ['μm', 'um']:
                        value *= 10000
                    elif unit in ['å', 'angstrom']:
                        pass  # already in Angstroms

                    # For diameter, convert to radius
                    if 'diameter' in pattern:
                        value = value / 2

                    return value
                except (ValueError, TypeError):
                    continue

        return None

    def _get_instrument_based_q_range(self, description: str) -> dict:
        """Get q-range based on instrument type hints in description"""
        import numpy as np

        desc_lower = description.lower()

        if any(word in desc_lower for word in ['sans', 'small angle', 'low q']):
            # SANS typically covers 0.01 - 1.0 Å⁻¹
            q_array = np.linspace(0.01, 1.0, 120)  # 0.01 to 1.0
            return {
                "source": "instrument_hint_sans",
                "q_min": 0.01,
                "q_max": 1.0,
                "num_points": 120,
                "q_array": q_array.tolist()
            }
        elif any(word in desc_lower for word in ['saxs', 'high q', 'wide angle']):
            # SAXS can go to higher q
            q_array = np.linspace(0.01, 3.0, 120)  # 0.01 to ~3.0
            return {
                "source": "instrument_hint_saxs",
                "q_min": 0.01,
                "q_max": 3.0,
                "num_points": 120,
                "q_array": q_array.tolist()
            }
        else:
            # Default range
            q_array = np.linspace(0.01, 1.0, 120)  # 0.01 to 1.0
            return {
                "source": "default",
                "q_min": 0.01,
                "q_max": 1.0,
                "num_points": 120,
                "q_array": q_array.tolist()
            }

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


def create_data_generation_agent(api_key: str = None, model: str = None) -> Agent:
    """Create an agent for generating synthetic SAS data"""
    llm = setup_llm(api_key, model)
    return Agent(
        role="SAS Data Generation Specialist",
        goal="Generate synthetic I(q) data based on sample descriptions and parameters",
        backstory="""You are an expert in generating synthetic small-angle scattering (SAS) data
        using SasModels. You interpret sample descriptions, select appropriate models using RAG,
        and generate realistic I(q) data with appropriate noise and uncertainties.

        CRITICAL: You are an expert in unit conversions for SAS data:
        - SasView uses Angstroms (Å) for ALL length parameters
        - Always convert nm to Å by multiplying by 10 (e.g., 30nm = 300Å)
        - Always convert μm to Å by multiplying by 10000 (e.g., 2μm = 20000Å)
        - When you see "25nm radius", use radius=250.0 in the parameters
        - When you see "100nm length", use length=1000.0 in the parameters
        - SLD values are in 10⁻⁶ Å⁻² units
        - Q-values are in Å⁻¹ units""",
        verbose=True,
        allow_delegation=False,
        tools=[RAGModelSelectorTool(), SyntheticDataTool()],  # Use generation tool and RAG
        llm=llm
    )


def create_data_generation_task(sample_description: str, params: dict = None, q_values: list = None, folder_path: str = None, api_key: str = None, model: str = None) -> Task:
    """Create a task for generating synthetic SAS data with enhanced parameter and q-range control"""
    return Task(
        description=f"""
        Generate synthetic SAS I(q) data based on the sample description with enhanced parameter and q-range control.

        Sample Description: {sample_description}
        Parameters: {params or 'Extract from description or use RAG recommendations'}
        Q-range: {q_values or 'Extract from description or use defaults'}
        Output Folder: {folder_path or 'Use default folder (data/test_ai_generation)'}

        CRITICAL UNIT CONVERSION REQUIREMENTS:
        - SasView uses Angstroms (Å) for ALL length parameters
        - Convert nm to Å by multiplying by 10: 30nm = 300Å
        - Convert μm to Å by multiplying by 10000: 2μm = 20000Å
        - Examples: "25nm radius" → radius=250.0, "100nm length" → length=1000.0

        Steps:
        1. Use the RAG model selector to choose an appropriate model and extract parameter suggestions.
        2. Extract any parameter values mentioned in the sample description (e.g., "25nm radius").
           - CONVERT ALL LENGTH UNITS TO ANGSTROMS before passing to the tool!
        3. Extract any q-range specifications from the description (e.g., "q range 0.01 to 1.0").
        4. Use the enhanced synthetic data generator tool to create I(q) data with:
           - Extracted or suggested model parameters (WITH PROPER UNIT CONVERSION)
           - DO NOT provide q_values parameter unless specifically requested - tool automatically calculates optimal q-range
           - Appropriate noise (3%) and uncertainties
           - Save to specified folder path: {folder_path or 'default folder'}
        5. Save data as CSV and generate a plot with parameter and q-range information.

        IMPORTANT: When calling synthetic_data_tool, pass folder_path="{folder_path}" if a folder path is specified.

        Pay special attention to:
        - Size parameters (radius, diameter, length, thickness) with units - CONVERT TO ANGSTROMS!
        - Scattering contrast (SLD values)
        - Concentration or volume fraction for interaction models
        - Q-range requirements for specific measurements or instruments
        """,
        expected_output="""
        Provide a comprehensive, user-friendly summary focusing on scientific insights:

        📊 **Generation Summary:**
        - Sample characteristics and what makes this system interesting
        - Model selection: which model was chosen and the scientific reasoning
        - Key parameters: values with physical meaning and units
        - Data quality: q-range coverage and measurement characteristics

        📁 **Results:**
        - Data file: Successfully generated synthetic scattering data
        - Visualization: Created scattering profile plot showing key features

        🔬 **Scientific Insights:**
        - What the parameters tell us about the sample structure
        - Expected scattering features and their physical origin
        - Practical considerations for analysis and interpretation

        Focus on scientific value and user understanding. Avoid technical file paths.
        Present as a professional analysis report that would be valuable to a scientist.
        """,
        agent=create_data_generation_agent(api_key, model)
    )


# ========================================================================================
# UNIFIED ANALYSIS SYSTEM
# ========================================================================================

class UnifiedSASAnalysisSystem:
    """Multi-agent SAS analysis system with coordinator for task routing"""
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the SAS analysis system with optional API configuration"""
        self.api_key = api_key
        self.model = model
        self.model_selector = create_model_selector_agent(api_key, model)
        self.fitter = create_fitting_agent(api_key, model)
        self.data_generator = create_data_generation_agent(api_key, model)
        self.coordinator = create_coordinator_agent(api_key, model)
        self.rag_available = RAG_AVAILABLE
        self.sas_tools_available = SAS_TOOLS_AVAILABLE

    def analyze_data(self, prompt: str, data_path: str = None, output_folder: str = None, chat_history: list = None) -> Dict[str, Any]:
        """
        Route and execute SAS tasks (fitting or generation) based on prompt.

        Args:
            prompt: User prompt (e.g., "Generate data for spheres" or "Fit data.csv to sphere model")
            data_path: Optional path to data file (for fitting tasks)
            output_folder: Optional output folder for generated files (for generation tasks)
            chat_history: Optional chat history for context and memory

        Returns:
            Results from either generation or fitting workflow
        """
        try:
            print("🔧 Starting collaborative SAS analysis...")

            # Enhance prompt with chat history context if available
            enhanced_prompt = prompt
            if chat_history and len(chat_history) > 0:
                print(f"💭 Using chat history context ({len(chat_history)} previous messages)")
                try:
                    # Build history context safely - handle different chat history formats
                    history_parts = []
                    for msg in chat_history[-2:]:  # Last 2 exchanges only to avoid too long prompts
                        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                            user_msg, agent_msg = str(msg[0]), str(msg[1])
                            # Only add if both messages exist and are meaningful
                            if user_msg and agent_msg and len(user_msg.strip()) > 0 and len(agent_msg.strip()) > 0:
                                # Truncate very long messages to avoid prompt bloat
                                user_msg = user_msg[:200] + "..." if len(user_msg) > 200 else user_msg
                                agent_msg = agent_msg[:200] + "..." if len(agent_msg) > 200 else agent_msg
                                history_parts.append(f"Previous - User: {user_msg}")
                                history_parts.append(f"Previous - Agent: {agent_msg}")

                    if history_parts:
                        history_context = "\n".join(history_parts)
                        enhanced_prompt = f"{history_context}\n\nCurrent Request: {prompt}"
                        print(f"📝 Enhanced prompt with {len(history_parts)//2} previous exchanges")
                    else:
                        print("📝 No valid history found, using original prompt")
                except Exception as e:
                    print(f"⚠️ Error processing chat history: {e}, using original prompt")
                    enhanced_prompt = prompt
            else:
                print("📝 No chat history provided, using original prompt")

            # Step 1: Use coordinator to determine intent
            coordinator_task = create_coordinator_task(enhanced_prompt, output_folder, data_path, self.api_key, self.model)
            coordinator_crew = Crew(
                agents=[self.coordinator],
                tasks=[coordinator_task],
                process=Process.sequential,
                verbose=True,
                name="SAS_Task_Coordinator"
            )
            coordinator_result = coordinator_crew.kickoff()

            # Debug: Print coordinator result
            print(f"🔍 Coordinator result: {str(coordinator_result)[:200]}...")

            # Parse coordinator output
            intent_match = re.search(
                r"INTENT:\s*(generation|fitting)\s*\nSAMPLE_DESCRIPTION:\s*(.*?)\s*\nDATA_PATH:\s*(.*?)\s*\nPARAMETERS:\s*(.*?)(?:\n|$)",
                str(coordinator_result), re.DOTALL
            )
            if not intent_match:
                print(f"❌ Failed to parse coordinator output: {str(coordinator_result)}")
                return {"success": False, "error": "Could not determine task intent from prompt"}

            intent, sample_description, data_path_str, params_str = intent_match.groups()
            data_path = data_path_str.strip() if data_path_str.strip() != "none" else data_path

            # Handle parameters - try JSON first, fall back to plain text
            params = None
            if params_str.strip() != "none":
                try:
                    params = json.loads(params_str.strip())
                except (json.JSONDecodeError, ValueError):
                    # If JSON parsing fails, keep as string
                    params = params_str.strip()

            print(f"📋 Intent: {intent}, Sample: {sample_description}, Data: {data_path}, Params: {params}")

            # Step 2: Route to appropriate workflow
            if intent == "generation":
                generation_task = create_data_generation_task(sample_description, params, folder_path=output_folder, api_key=self.api_key, model=self.model)
                generation_crew = Crew(
                    agents=[self.data_generator],
                    tasks=[generation_task],
                    process=Process.sequential,
                    verbose=True,
                    name="SAS_Data_Generator"
                )
                results = generation_crew.kickoff()

                # Extract structured information from the CrewAI result
                result_info = {
                    "success": True,
                    "task_type": "generation",
                    "results": str(results),
                    "sample_description": sample_description,
                    "rag_enhanced": self.rag_available
                }

                # Try to extract plot_file from the generation task result
                # The result might contain tool outputs with plot_file information
                try:
                    # Check if results has tool outputs
                    if hasattr(results, 'tasks_output') and results.tasks_output:
                        for task_output in results.tasks_output:
                            if hasattr(task_output, 'tool_calls') and task_output.tool_calls:
                                for tool_call in task_output.tool_calls:
                                    if hasattr(tool_call, 'result') and isinstance(tool_call.result, dict):
                                        tool_result = tool_call.result
                                        if tool_result.get('success') and 'plot_file' in tool_result:
                                            result_info['plot_file'] = tool_result['plot_file']
                                        if tool_result.get('success') and 'csv_path' in tool_result:
                                            result_info['csv_path'] = tool_result['csv_path']
                                        if tool_result.get('success') and 'model_used' in tool_result:
                                            result_info['model_used'] = tool_result['model_used']
                except Exception as e:
                    # If extraction fails, continue with basic result
                    print(f"Note: Could not extract tool results: {e}")

                return result_info

            elif intent == "fitting":
                if not data_path or not os.path.exists(data_path):
                    return {"success": False, "error": f"Data file not found: {data_path}"}

                # Extract model information from chat history if available
                previous_model_info = None
                if chat_history and len(chat_history) > 0:
                    try:
                        # Look for previous fitting results in chat history
                        for msg in reversed(chat_history):  # Check most recent first
                            if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                                _, agent_response = msg
                                if isinstance(agent_response, dict) and agent_response.get('task_type') == 'fitting':
                                    if 'sample_description' in agent_response:
                                        previous_model_info = agent_response['sample_description']
                                        print(f"🔍 Found previous model from history: {previous_model_info}")
                                        break
                                elif isinstance(agent_response, str):
                                    # Check if response contains model information
                                    model_match = re.search(r'Model.*?:\s*(\w+)', str(agent_response), re.IGNORECASE)
                                    if model_match:
                                        previous_model_info = model_match.group(1)
                                        print(f"🔍 Extracted model from history: {previous_model_info}")
                                        break
                    except Exception as e:
                        print(f"⚠️ Could not extract model info from history: {e}")

                # Use previous model info if available, otherwise use current sample description
                model_context = previous_model_info if previous_model_info else sample_description

                # Create collaborative tasks for model selection and fitting
                selection_task = create_model_selection_task(model_context, output_folder, self.api_key, self.model)

                # Combine original user prompt with current parameter guidance
                combined_prompt = prompt
                if previous_model_info and previous_model_info != sample_description:
                    combined_prompt = f"Using {previous_model_info} model. {prompt}"

                fitting_task = create_fitting_task(data_path, "from_selector", model_context, output_folder, self.api_key, self.model, original_prompt=combined_prompt)

                analysis_crew = Crew(
                    agents=[self.model_selector, self.fitter],
                    tasks=[selection_task, fitting_task],
                    process=Process.sequential,
                    verbose=True,
                    name="SAS_Model_Analyzer"
                )
                results = analysis_crew.kickoff()

                # Extract structured information from the CrewAI result
                result_info = {
                    "success": True,
                    "task_type": "fitting",
                    "results": str(results),
                    "data_file": data_path,
                    "sample_description": sample_description,
                    "rag_enhanced": self.rag_available
                }

                # Try to extract plot_file and fitted parameters from the fitting task result
                try:
                    # Check if results has tool outputs
                    if hasattr(results, 'tasks_output') and results.tasks_output:
                        for task_output in results.tasks_output:
                            if hasattr(task_output, 'tool_calls') and task_output.tool_calls:
                                for tool_call in task_output.tool_calls:
                                    if hasattr(tool_call, 'result') and isinstance(tool_call.result, dict):
                                        tool_result = tool_call.result
                                        if tool_result.get('success'):
                                            # Extract fitting results
                                            if 'plot_file' in tool_result:
                                                result_info['plot_file'] = tool_result['plot_file']
                                            if 'fitted_parameters' in tool_result:
                                                result_info['fitted_parameters'] = tool_result['fitted_parameters']
                                            if 'r_squared' in tool_result:
                                                result_info['r_squared'] = tool_result['r_squared']
                                            if 'rmse' in tool_result:
                                                result_info['rmse'] = tool_result['rmse']
                                            if 'model' in tool_result:
                                                result_info['model_used'] = tool_result['model']
                                            if 'chi_squared' in tool_result:
                                                result_info['chi_squared'] = tool_result['chi_squared']
                                            if 'chi_squared_reduced' in tool_result:
                                                result_info['chi_squared_reduced'] = tool_result['chi_squared_reduced']
                except Exception as e:
                    # If extraction fails, continue with basic result
                    print(f"Note: Could not extract tool results: {e}")

                return result_info

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

def analyze_sas_data(prompt: str, data_path: str = None, output_folder: str = None, chat_history: list = None, verbose: bool = True, api_key: str = None, model: str = None) -> Dict[str, Any]:
    """
    Analyze SAS data or generate synthetic data based on user prompt.

    Args:
        prompt: User prompt specifying task (e.g., "Generate data for spheres" or "Fit my_data.csv to sphere model")
        data_path: Optional path to CSV file (for fitting tasks)
        output_folder: Optional output folder for generated files (for generation tasks)
        chat_history: Optional chat history for context and memory
        verbose: Whether to print progress information
        api_key: Optional API key for LLM configuration
        model: Optional model selection for LLM

    Returns:
        Results from the appropriate workflow

    Example:
        # For generation:
        result = analyze_sas_data("Generate synthetic data for spherical gold nanoparticles")

        # For generation with custom output folder:
        result = analyze_sas_data("Generate synthetic data for spheres", output_folder="/path/to/output")

        # For fitting:
        result = analyze_sas_data("Fit my_data.csv to spherical particles", "my_data.csv")

        # With custom API configuration:
        result = analyze_sas_data("Generate data for spheres", api_key="your-key", model="gpt-4")
    """
    if verbose:
        print(f"🔍 Processing prompt: {prompt}")
        if data_path:
            print(f"📊 Data file: {data_path}")
        if output_folder:
            print(f"📁 Output folder: {output_folder}")
        print("🤖 Starting collaborative SAS analysis...")

    system = UnifiedSASAnalysisSystem(api_key, model)
    return system.analyze_data(prompt, data_path, output_folder, chat_history)


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
