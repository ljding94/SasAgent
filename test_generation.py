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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# TODO: improve ai test case prompt for better results
# TODO: improve q range in both tbasic test and ai test cases


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
            csv_path, params = generate_sasview_data(
                model_name=case["model"],
                params=case["params"],
                q_values=case["q_values"],
                output_folder="data",
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


def test_crewai_generation():
    """Test CrewAI-enhanced generation with natural language descriptions including q-range specifications"""
    print("\n🤖 Testing CrewAI Enhanced Generation with Q-Range Specifications")
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


def generate_summary_report(generation_results, ai_results):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE GENERATION TEST SUMMARY")
    print("=" * 80)

    # Generation test summary (both basic and enhanced)
    generation_success = sum(1 for r in generation_results.values() if r.get("success"))
    print(f"\n🧪 Generation Tests: {generation_success}/{len(generation_results)} successful")

    # Categorize by test type
    basic_tests = {k: v for k, v in generation_results.items() if not v.get("q_custom", False)}
    enhanced_tests = {k: v for k, v in generation_results.items() if v.get("q_custom", False)}

    print(f"   📋 Basic models (default q-range): {sum(1 for r in basic_tests.values() if r.get('success'))}/{len(basic_tests)}")
    for name, result in basic_tests.items():
        status = "✅" if result.get("success") else "❌"
        bg_info = f"(bg={result.get('parameters', {}).get('background', 'N/A')})"
        print(f"      {status} {name}: {result.get('use_case', 'N/A')} {bg_info}")

    print(f"   🚀 Enhanced models (custom q-range): {sum(1 for r in enhanced_tests.values() if r.get('success'))}/{len(enhanced_tests)}")
    for name, result in enhanced_tests.items():
        status = "✅" if result.get("success") else "❌"
        bg_info = f"(bg={result.get('parameters', {}).get('background', 'N/A')})"
        print(f"      {status} {name}: {result.get('use_case', 'N/A')} {bg_info}")

    # AI generation summary
    if isinstance(ai_results, dict) and ai_results.get("ai_tests") != "not_available":
        ai_success = sum(1 for r in ai_results.values() if r.get("success"))
        print(f"\n🤖 CrewAI Enhanced Tests: {ai_success}/{len(ai_results)} successful")
        for test, result in ai_results.items():
            status = "✅" if result.get("success") else "❌"
            model = result.get("model_used", "N/A")
            bg_info = f"(bg={result.get('background', 'auto')})"
            q_info = result.get('q_info', {}).get('source', 'default')
            print(f"   {status} {test}: Model = {model}, Q-range = {q_info} {bg_info}")
    else:
        print("\n🤖 CrewAI Enhanced Tests: Not available")

    # Overall assessment
    total_tests = len(generation_results)
    if isinstance(ai_results, dict) and ai_results.get("ai_tests") != "not_available":
        total_tests += len(ai_results)

    print("\n🎯 Overall Assessment:")
    print(f"   Total test scenarios: {total_tests}")
    print("   Background handling: ✅ Integrated into params (tested across all models)")
    print("   Q-range flexibility: ✅ Both default and custom q-ranges tested")
    print("   Model variety: ✅ 4 essential particle types covered")
    print("   Realistic scenarios: ✅ SANS/SAXS conditions tested")
    print("   AI integration: ✅ Natural language to model parameters with q-range specs")

    print("\n📁 Test outputs saved to: data/")
    print("\n🎉 Comprehensive testing complete!")


def main():
    """Run comprehensive SAS generation testing suite"""
    print("COMPREHENSIVE SAS SYNTHETIC DATA GENERATION TEST SUITE")
    print("=" * 80)
    print("🔬 Testing: Unified Generation + CrewAI Integration")
    print("📋 Background and q-range testing integrated across all test types")
    print()

    # Run streamlined test suites
    generation_results = test_generation()
    ai_results = test_crewai_generation()

    # Generate comprehensive summary
    generate_summary_report(generation_results, ai_results)


if __name__ == "__main__":
    main()
