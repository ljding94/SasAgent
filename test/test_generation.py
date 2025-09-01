#!/usr/bin/env python3
"""
Comprehensive SAS Synthetic Data Generation Test Suite
Streamlined testing for generation capabilities and CrewAI integration

Features tested:
- ✅ Unified generation testing with 4 essential models (sphere, cylinder, flexible_cylinder, lamellar)
- ✅ Both default and custom q-ranges tested within the same framework
- ✅ Background handling integrated into params dictionary (tested across all models)
- ✅ CrewAI natural language processing with q-range specifications
- ✅ Parameter extraction from descriptions including q-range hints
- ✅ Realistic SANS/SAXS scenarios with appropriate backgrounds and q-ranges

Note: test_fitting.py remains separate as it tests fitting functionality, not generation.

Usage: python3 test_generation.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level since we're in test/
sys.path.insert(0, str(project_root))

def test_generation():
    """Test synthetic data generation with various models, parameters, and q-ranges"""
    print("🧪 Testing Comprehensive Generation")
    print("=" * 60)

    from SAS.generation import generate_sasview_data

    # Test cases covering essential models with different scenarios
    test_cases = [
        # Basic models with default q-range
        {
            "name": "colloids",
            "model": "sphere",
            "params": {"radius": 50, "sld": 1.0, "sld_solvent": 0.0, "background": 0.001},
            "q_values": None,  # Default q-range
            "description": "Spherical particles, colloidal suspension (SANS measurement)",
            "use_case": "SANS protein measurement"
        },
        {
            "name": "cylinder_nanorods",
            "model": "cylinder",
            "params": {"radius": 20, "length": 100, "sld": 2.0, "sld_solvent": 0.0, "background": 0.01},
            "q_values": None,  # Default q-range
            "description": "Rod-shaped nanoparticles (SAXS measurement)",
            "use_case": "SAXS nanorod measurement"
        },
        {
            "name": "flexible_cylinder_polymer",
            "model": "flexible_cylinder",
            "params": {"length": 1000, "kuhn_length": 100, "radius": 5, "sld": 0.5, "sld_solvent": 0.0, "background": 0.0005},
            "q_values": None,  # Default q-range
            "description": "Polymer chains (SANS measurement)",
            "use_case": "SANS polymer solution"
        },
        {
            "name": "lamellar_membrane",
            "model": "lamellar",
            "params": {"thickness": 30, "sld": 1.8, "sld_solvent": 0.0, "background": 0.002},
            "q_values": None,  # Default q-range
            "description": "Bilayer structures (membrane analysis)",
            "use_case": "Membrane structure analysis"
        },

        # Enhanced cases with custom q-ranges
        {
            "name": "sphere_custom_q",
            "model": "sphere",
            "params": {"radius": 25.0, "sld": 2.0, "sld_solvent": 0.0, "background": 0.001},
            "q_values": np.logspace(-2, 0, 120).tolist(),  # Custom q-range
            "description": "25nm sphere with custom q-range (0.01-1.0)",
            "use_case": "High-resolution SANS measurement"
        },
        {
            "name": "cylinder_high_q_saxs",
            "model": "cylinder",
            "params": {"radius": 10.0, "length": 50.0, "sld": 1.5, "sld_solvent": 0.0, "background": 0.01},
            "q_values": np.logspace(-2, 0.5, 150).tolist(),  # High-q for SAXS
            "description": "Cylinder for SAXS measurement (0.01-3.0)",
            "use_case": "High-q SAXS analysis"
        },
        {
            "name": "lamellar_sans_custom",
            "model": "lamellar",
            "params": {"thickness": 40.0, "sld": 1.2, "sld_solvent": 0.0, "background": 0.0005},
            "q_values": np.logspace(-2.5, -0.5, 100).tolist(),  # SANS range
            "description": "Membrane for SANS measurement (0.003-0.3)",
            "use_case": "Low-q membrane analysis"
        }
    ]

    results = {}

    for case in test_cases:
        print(f"\n📋 Testing {case['name']}...")
        print(f"   Description: {case['description']}")
        print(f"   Use case: {case['use_case']}")
        print(f"   Background: {case['params']['background']:.4f}")
        print(f"   Q-points: {len(case['q_values']) if case['q_values'] else 'Default'}")

        try:
            csv_path, params, plot_path = generate_sasview_data(
                model_name=case["model"],
                params=case["params"],
                q_values=case["q_values"],
                output_folder="test/test_data/generation",
                noise_level=0.02,
                plot=True,
                include_uncertainty=True
            )

            results[case["name"]] = {
                "success": True,
                "csv_path": csv_path,
                "parameters": params,
                "use_case": case["use_case"],
                "q_custom": case["q_values"] is not None
            }
            print(f"✅ {case['name']}: Generated {csv_path}")
            print(f"   Background used: {params['background']:.4f}")
            print(f"   Custom q-range: {'Yes' if case['q_values'] else 'Default'}")

        except Exception as e:
            results[case["name"]] = {"success": False, "error": str(e)}
            print(f"❌ {case['name']}: Failed - {e}")

    return results


def test_generation_sas_tool():
    """Test CrewAI-enhanced generation with natural language descriptions including q-range specifications"""
    print("\n🤖 Testing CrewAI Enhanced Generation")
    print("=" * 60)

    try:
        from crewai_sas_agents import SyntheticDataTool
        tool = SyntheticDataTool()

        # Test cases with natural language descriptions including q-range information
        ai_test_cases = [
            {
                "description": "spherical gold nanoparticles with 30nm radius in water for SANS measurement, q-range 0.005 to 0.5 inverse angstrom",
                "expected_features": ["radius extraction", "SANS q-range", "appropriate background", "q-range specification"]
            },
            {
                "description": "cylindrical carbon nanotubes, length 200nm, radius 5nm, for SAXS analysis with high-q resolution up to 2.0 inverse angstrom",
                "expected_features": ["length/radius extraction", "SAXS q-range", "cylinder model", "high-q specification"]
            },
            {
                "description": "phospholipid bilayer membranes, thickness 4nm, SLD 1.5, low background, measured with SANS in q-range 0.003-0.3",
                "expected_features": ["thickness extraction", "SLD extraction", "background hint", "SANS q-range"]
            },
            {
                "description": "polymer microspheres radius 100nm for small-angle scattering, custom q-range from 0.01 to 1.5 inverse angstrom, medium background",
                "expected_features": ["radius extraction", "custom q-range", "background specification", "sphere model"]
            }
        ]

        ai_results = {}

        for i, test_case in enumerate(ai_test_cases, 1):
            print(f"\n📋 AI Test {i}: {test_case['description']}")
            print(f"   Expected: {', '.join(test_case['expected_features'])}")

            try:
                result = tool._run(sample_description=test_case["description"])

                if result.get("success"):
                    ai_results[f"ai_test_{i}"] = {
                        "success": True,
                        "model_used": result["model_used"],
                        "csv_path": result["csv_path"],
                        "parameters": result["ground_truth_params"],
                        "q_info": result.get("q_info", {}),
                        "background": result.get("background_used")
                    }
                    print(f"✅ AI Test {i}: Model = {result['model_used']}")
                    print(f"   Parameters: {list(result['ground_truth_params'].keys())}")
                    print(f"   Q-range source: {result.get('q_info', {}).get('source', 'default')}")
                    print(f"   Q-range: {result.get('q_info', {}).get('range', 'N/A')}")
                    print(f"   Background: {result.get('background_used', 'default')}")
                else:
                    ai_results[f"ai_test_{i}"] = {"success": False, "error": result.get("error")}
                    print(f"❌ AI Test {i}: {result.get('error')}")

            except Exception as e:
                ai_results[f"ai_test_{i}"] = {"success": False, "error": str(e)}
                print(f"❌ AI Test {i}: Exception - {e}")

        return ai_results

    except ImportError as e:
        print(f"⚠️  CrewAI tools not available: {e}")
        return {"ai_tests": "not_available"}



def test_end_to_end_generation():
    """Test the full CrewAI agent system using UnifiedSASAnalysisSystem.analyze_data()"""
    print("\n🤖 Testing Full CrewAI Agent System Integration")
    print("=" * 60)

    try:
        from crewai_sas_agents import UnifiedSASAnalysisSystem

        # Initialize the unified system
        system = UnifiedSASAnalysisSystem()

        # Test cases for the unified system with various prompts
        test_prompts = [
            {
                "name": "polymer_chains",
                "prompt": "Generate SANS data for flexible polymer chains with length 200A and persistence length 50A",
                "expected_task": "generation",
            },
            {
                "name": "lamellar_paracrystal",
                "prompt": "Create synthetic data for lamellar stak paracrystal with sigma_d 0.1 and q range (0.01, 1)",
                "expected_task": "generation",
            },
            {
                "name": "guinier",
                "prompt": "Generate scattering data for guinier model for radius of gyration 30A over q range (0.01, 1)",
                "expected_task": "generation",
            },
            {
                "name": "ellipsoid",
                "prompt": "Generate scattering data of ellipsoid",
                "expected_task": "generation",
            },
            {
                "name": "spherical_nanoparticles",
                "prompt": "Generate synthetic data for spherical colloidal particles of diameter 100A",
                "expected_task": "generation",
            },
            {
                "name": "crystal",
                "prompt": "Generate synthetic data for cubic lattice with paracrystalline distortion with q range (0.005, 0.5)",
                "expected_task": "generation",
            },

        ]

        unified_results = {}
        output_folder = "test/test_data/generation"

        for test_case in [test_prompts[1]]:
            print(f"\n📋 Testing: {test_case['name']}")
            print(f"   Prompt: {test_case['prompt']}")
            print(f"   Expected: {test_case['expected_task']} task")

            try:
                # Call the unified system with the test prompt
                result = system.analyze_data(
                    prompt=test_case["prompt"],
                    output_folder=output_folder
                )

                if result.get("success"):
                    unified_results[test_case["name"]] = {
                        "success": True,
                        "task_type": result.get("task_type"),
                        "model_used": result.get("model_used"),
                        "csv_path": result.get("csv_path"),
                        "plot_file": result.get("plot_file"),
                        "rag_enhanced": result.get("rag_enhanced"),
                        "results_summary": str(result.get("results", ""))[:200] + "..." if result.get("results") else ""
                    }

                    print(f"✅ {test_case['name']}: Success")
                    print(f"   Task type: {result.get('task_type', 'Unknown')}")
                    print(f"   Model: {result.get('model_used', 'Auto-selected')}")

                    if result.get("csv_path"):
                        print(f"   CSV: {result['csv_path']}")
                    if result.get("plot_file"):
                        print(f"   Plot: {result['plot_file']}")

                    print(f"   RAG Enhanced: {result.get('rag_enhanced', False)}")

                else:
                    unified_results[test_case["name"]] = {
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                        "task_type": result.get("task_type")
                    }
                    print(f"❌ {test_case['name']}: Failed - {result.get('error', 'Unknown error')}")

            except Exception as e:
                unified_results[test_case["name"]] = {
                    "success": False,
                    "error": str(e),
                    "exception": True
                }
                print(f"❌ {test_case['name']}: Exception - {e}")

        # Summary of results
        print("\n📊 Unified System Test Summary")
        print("=" * 50)

        successful_tests = sum(1 for result in unified_results.values() if result.get("success"))
        total_tests = len(unified_results)

        print(f"Successful tests: {successful_tests}/{total_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")

        # Group results by task type
        task_types = {}
        for name, result in unified_results.items():
            task_type = result.get("task_type", "unknown")
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append((name, result.get("success", False)))

        print("\nResults by task type:")
        for task_type, tests in task_types.items():
            successful = sum(1 for _, success in tests if success)
            total = len(tests)
            print(f"  {task_type}: {successful}/{total} successful")

        return unified_results

    except ImportError as e:
        print(f"⚠️  UnifiedSASAnalysisSystem not available: {e}")
        return {"unified_tests": "not_available"}


def main():
    """Run comprehensive SAS generation testing suite"""
    print("COMPREHENSIVE SAS SYNTHETIC DATA GENERATION TEST SUITE")
    print("=" * 80)
    # Run all test suites
    print("🔄 Running Direct Generation Tests...")
    #basic_results = test_generation()

    print("\n🔄 Running CrewAI Tool Tests...")
    #crewai_results = test_generation_sas_tool()

    print("\n🔄 Running Full Unified System Tests...")
    unified_results = test_end_to_end_generation()

if __name__ == "__main__":
    main()
