#!/usr/bin/env python3
"""
Test script for SAS data fitting functionality
Tests both direct fitting functions and RAG-enhanced model selection for fitting
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level since we're in test/
sys.path.insert(0, str(project_root))


def generate_test_data():
    """Generate some test data for fitting"""
    print("🔧 Generating Test Data for Fitting")
    print("=" * 50)

    from SAS.generation import generate_sasview_data

    # Generate test datasets with known parameters
    test_datasets = {
        "sphere_test": {
            "model": "sphere",
            "params": {"radius": 50.0, "background": 0.01},
            "q_values": np.linspace(0.001, 0.3 , 200).tolist()  # 0.001 to 1.0, 100 points
        },
        "cylinder_test": {
            "model": "cylinder",
            "params": {"radius": 20.0, "length": 400.0,  "background": 0.01},
            "q_values": np.linspace(0.01, 0.5, 200).tolist()  # 0.01 to ~3.16, 120 points (SAXS-like)
        },
        "flexible_cylinder_test": {
            "model": "flexible_cylinder",
            "params": {"length": 1000.0, "kuhn_length": 100.0, "radius": 20.0, "background": 0.01},
            "q_values": np.linspace(0.01, 0.6, 200).tolist()  # 0.003 to ~0.5, 80 points (SANS-like)
        },
        "lamellar_test": {
            "model": "lamellar",
            "params": {"thickness": 40.0, "background": 0.01},
            "q_values": np.linspace(0.01, 1.0, 200).tolist()  # 0.0016 to ~0.32, 90 points (low-q SANS)
        }
    }

    generated_files = {}

    for dataset_name, config in test_datasets.items():
        print(f"\n📋 Generating {dataset_name}...")
        try:
            csv_path, ground_truth, plot_path = generate_sasview_data(
                model_name=config["model"],
                params=config["params"],
                q_values=config.get("q_values"),  # Pass custom q_values if provided
                output_folder="test/test_data/fitting",
                noise_level=0.03,  # 3% noise for realistic fitting challenge
                plot=True,
                include_uncertainty=True
            )

            generated_files[dataset_name] = {
                "csv_path": csv_path,
                "model": config["model"],
                "ground_truth": ground_truth
            }
            print(f"✅ Generated: {csv_path}")
            print(f"   Ground truth: {ground_truth}")

        except Exception as e:
            print(f"❌ Failed to generate {dataset_name}: {e}")
            generated_files[dataset_name] = {"error": str(e)}

    return generated_files


def test_basic_fitting():
    """Test basic SasView fitting functionality"""
    print("\n🔧 Testing Basic Fitting Functionality")
    print("=" * 50)

    try:
        from SAS.fitting import sasview_fit

        # First, generate test data
        test_files = generate_test_data()

        fitting_results = {}

        for dataset_name, file_info in test_files.items():
            if "error" in file_info:
                continue

            csv_path = file_info["csv_path"]
            expected_model = file_info["model"]
            ground_truth = file_info["ground_truth"]

            print(f"\n📋 Fitting {dataset_name} with {expected_model} model...")

            try:
                # Test fitting with the correct model
                result = sasview_fit(
                    csv_path,
                    expected_model,
                    plot_label=f"Test_{dataset_name}"
                )

                if "error" not in result:
                    fit_data = result['fit_json']
                    r_squared = fit_data['r_squared']
                    rmse = fit_data['rmse']
                    fitted_params = fit_data['parameters']

                    print("✅ Fitting successful!")
                    print(f"   R² = {r_squared:.4f}")
                    print(f"   RMSE = {rmse:.2e}")
                    print(f"   Parameters fitted: {list(fitted_params.keys())}")

                    # Evaluate fit quality
                    fit_quality = "Excellent" if r_squared > 0.95 else "Good" if r_squared > 0.85 else "Poor"
                    print(f"Fit quality: {fit_quality}")

                    # Compare key physical parameters (ignore scale and highly correlated params)
                    key_params = {
                        "sphere": ["radius"],
                        "cylinder": ["radius", "length"],
                        "flexible_cylinder": ["length", "kuhn_length", "radius"],
                        "lamellar": ["thickness"]
                    }

                    model_key_params = key_params.get(expected_model, [])
                    param_recovery_good = True

                    print("   Key parameter comparison:")
                    for param in model_key_params:
                        if param in ground_truth and param in fitted_params:
                            true_val = ground_truth[param]
                            fitted_val = fitted_params[param]

                            if abs(true_val) < 1e-10:  # effectively zero
                                diff_abs = abs(fitted_val - true_val)
                                print(f"     {param}: true={true_val:.3f}, fitted={fitted_val:.3f} (diff={diff_abs:.3f})")
                                param_ok = diff_abs < 0.1
                            else:
                                diff_pct = abs(fitted_val - true_val) / abs(true_val) * 100
                                print(f"     {param}: true={true_val:.3f}, fitted={fitted_val:.3f} ({diff_pct:.1f}% diff)")
                                param_ok = diff_pct < 50.0  # Allow 50% tolerance for complex fits

                            if not param_ok:
                                param_recovery_good = False

                    # Overall assessment
                    overall_success = r_squared > 0.8 and param_recovery_good
                    status = "✅ PASS" if overall_success else "⚠️  MARGINAL" if r_squared > 0.5 else "❌ FAIL"
                    print(f"   Overall assessment: {status}")

                    fitting_results[dataset_name] = {
                        "success": True,
                        "r_squared": r_squared,
                        "rmse": rmse,
                        "fitted_parameters": fitted_params,
                        "ground_truth": ground_truth,
                        "fit_quality": fit_quality,
                        "overall_success": overall_success
                    }
                else:
                    print(f"❌ Fitting failed: {result['error']}")
                    fitting_results[dataset_name] = {
                        "success": False,
                        "error": result['error']
                    }

            except Exception as e:
                print(f"❌ Fitting exception: {e}")
                fitting_results[dataset_name] = {
                    "success": False,
                    "error": str(e)
                }

        return fitting_results

    except ImportError as e:
        print(f"❌ SAS fitting tools not available: {e}")
        return {"error": "Fitting tools not available"}


def test_rag_enhanced_fitting():
    """Test RAG-enhanced model selection for fitting"""
    print("\n🧠 Testing RAG-Enhanced Fitting Workflow")
    print("=" * 50)

    try:
        from crewai_sas_agents import RAGModelSelectorTool, SasViewFittingTool

        # Use previously generated test data
        test_data_dir = Path("data/test_fitting")
        if not test_data_dir.exists():
            print("❌ No test data found. Run generate_test_data() first.")
            return {"error": "No test data available"}

        csv_files = list(test_data_dir.glob("*.csv"))
        if not csv_files:
            print("❌ No CSV files found in test data directory.")
            return {"error": "No CSV files found"}

        rag_tool = RAGModelSelectorTool()
        fitting_tool = SasViewFittingTool()

        # Test sample descriptions that should match our test data
        test_cases = [
            {
                "description": "spherical nanoparticles in solution, analyze with sphere model",
                "expected_file_pattern": "sphere"
            },
            {
                "description": "cylindrical rod-like particles, analyze with cylinder model",
                "expected_file_pattern": "cylinder"
            },
            {
                "description": "flexible polymer chains in solution, analyze with flexible_cylinder model",
                "expected_file_pattern": "flexible_cylinder"
            },
            {
                "description": "lamellar bilayer structures, analyze with lamellar model",
                "expected_file_pattern": "lamellar"
            }
        ]

        results = {}

        for test_case in test_cases:
            description = test_case["description"]
            pattern = test_case["expected_file_pattern"]

            print(f"\n📋 Sample: {description}")

            # Find matching CSV file
            matching_files = [f for f in csv_files if pattern in f.name]
            if not matching_files:
                print(f"   ❌ No matching data file found for pattern: {pattern}")
                continue

            csv_path = str(matching_files[0])
            print(f"   📊 Using data file: {csv_path}")

            # Get RAG model recommendation
            rag_result = rag_tool._run(description)
            if rag_result.get('success'):
                recommended_model = rag_result['recommended_model']
                confidence = rag_result['confidence']
                print(f"   🧠 RAG Recommended: {recommended_model} (confidence: {confidence:.3f})")

                # Test fitting with recommended model
                fitting_result = fitting_tool._run(
                    csv_path=csv_path,
                    model_name=recommended_model,
                    parameter_guidance="RAG-guided fitting test"
                )

                if fitting_result.get('success'):
                    r_squared = fitting_result['r_squared']
                    rmse = fitting_result['rmse']
                    print("   ✅ Fitting successful!")
                    print(f"      R² = {r_squared:.4f}")
                    print(f"      RMSE = {rmse:.2e}")

                    results[description] = {
                        "success": True,
                        "recommended_model": recommended_model,
                        "confidence": confidence,
                        "r_squared": r_squared,
                        "rmse": rmse,
                        "data_file": csv_path
                    }
                else:
                    error = fitting_result.get('error', 'Unknown fitting error')
                    print(f"   ❌ Fitting failed: {error}")
                    results[description] = {
                        "success": False,
                        "recommended_model": recommended_model,
                        "error": error
                    }
            else:
                error = rag_result.get('error', 'Unknown RAG error')
                print(f"   ❌ RAG failed: {error}")
                results[description] = {
                    "success": False,
                    "error": error
                }

        return results

    except ImportError as e:
        print(f"❌ RAG/CrewAI tools not available: {e}")
        return {"error": "RAG tools not available"}


def test_model_selection_accuracy():
    """Test how well RAG model selection matches expected models"""
    print("\n🎯 Testing Model Selection Accuracy")
    print("=" * 50)

    try:
        from crewai_sas_agents import RAGModelSelectorTool

        rag_tool = RAGModelSelectorTool()

        # Test cases with expected models
        test_cases = [
            ("spherical gold nanoparticles using sphere model", "sphere"),
            ("cylindrical carbon nanotubes using cylinder model", "cylinder"),
            ("flexible polymer chains using flexible_cylinder model", "flexible_cylinder"),
            ("core-shell particles using core_shell_sphere model", "core_shell_sphere"),
            ("charged spherical particles using hayter_msa model", "hayter_msa"),
            ("lamellar bilayer structures using lamellar model", "lamellar")
        ]

        results = {}
        correct_predictions = 0

        for description, expected_model in test_cases:
            print(f"\n📋 Description: {description}")
            print(f"   Expected: {expected_model}")

            rag_result = rag_tool._run(description)
            if rag_result.get('success'):
                recommended = rag_result['recommended_model']
                confidence = rag_result['confidence']

                # Check if it's correct or a reasonable alternative
                is_correct = (recommended == expected_model)
                if is_correct:
                    correct_predictions += 1
                    print(f"   ✅ Recommended: {recommended} (confidence: {confidence:.3f}) - CORRECT")
                else:
                    print(f"   ⚠️  Recommended: {recommended} (confidence: {confidence:.3f}) - DIFFERENT")

                results[description] = {
                    "success": True,
                    "expected": expected_model,
                    "recommended": recommended,
                    "confidence": confidence,
                    "correct": is_correct
                }
            else:
                print(f"   ❌ Failed: {rag_result.get('error')}")
                results[description] = {
                    "success": False,
                    "error": rag_result.get('error')
                }

        accuracy = correct_predictions / len(test_cases) * 100
        print(f"\n📊 Overall Accuracy: {correct_predictions}/{len(test_cases)} ({accuracy:.1f}%)")

        return {
            "results": results,
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_tests": len(test_cases)
        }

    except ImportError as e:
        print(f"❌ RAG tools not available: {e}")
        return {"error": "RAG tools not available"}


def test_crewai_collaborative_analysis():
    """Test the full CrewAI collaborative analysis system"""
    print("\n🤖 Testing CrewAI Collaborative Analysis System")
    print("=" * 50)

    try:
        from crewai_sas_agents import analyze_sas_data
        import tempfile

        # Create a temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"   📁 Using temporary directory: {temp_dir}")

            # First, generate some test data to analyze
            test_files = generate_test_data()

            if not test_files or all('error' in info for info in test_files.values()):
                print("   ❌ No test data available for CrewAI testing")
                return {"error": "No test data available"}

            results = {}

            # Test cases for collaborative analysis
            analysis_prompts = [
                {
                    "name": "sphere_analysis",
                    "prompt": "Analyze this spherical nanoparticle data. Determine the best model, fit the parameters, and provide scientific insights about the particle size and structure.",
                    "data_pattern": "sphere"
                },
                {
                    "name": "cylinder_analysis",
                    "prompt": "This data is from rod-like nanoparticles. Please select an appropriate model, perform fitting, and comment on the aspect ratio and structure.",
                    "data_pattern": "cylinder"
                },
                {
                    "name": "automated_analysis",
                    "prompt": "I have SAS data but I'm not sure what structure it represents. Please analyze it automatically - select the best model, fit the data, and tell me what you found.",
                    "data_pattern": "flexible_cylinder"  # Use a more complex case
                }
            ]

            for test_case in analysis_prompts:
                prompt = test_case["prompt"]
                pattern = test_case["data_pattern"]
                test_name = test_case["name"]

                print(f"\n📋 Test: {test_name}")
                print(f"   Prompt: {prompt[:80]}...")

                # Find matching test data
                matching_data = None
                for dataset_name, file_info in test_files.items():
                    if pattern in dataset_name and 'error' not in file_info:
                        matching_data = file_info
                        break

                if not matching_data:
                    print(f"   ❌ No matching data found for pattern: {pattern}")
                    results[test_name] = {"error": f"No data for {pattern}"}
                    continue

                data_path = matching_data["csv_path"]
                print(f"   📊 Using data: {data_path}")

                try:
                    # Test the full CrewAI collaborative analysis
                    analysis_result = analyze_sas_data(
                        prompt=prompt,
                        data_path=data_path,
                        output_folder=temp_dir,
                        verbose=True,
                        api_key=os.getenv('OPENROUTER_API_KEY'),  # Use env variable
                        model='openai/gpt-4o-mini'  # Use a reliable model for testing
                    )

                    print("   🤖 CrewAI analysis completed!")
                    print(f"   Result type: {type(analysis_result)}")

                    # Analyze the result
                    success = False
                    analysis_details = {}

                    if hasattr(analysis_result, 'raw'):
                        # CrewAI CrewOutput object
                        raw_output = str(analysis_result.raw)
                        analysis_details['raw_output'] = raw_output[:500] + "..." if len(raw_output) > 500 else raw_output
                        success = True
                        print(f"   ✅ Got CrewAI raw output ({len(raw_output)} chars)")
                    elif isinstance(analysis_result, dict):
                        analysis_details = analysis_result
                        success = analysis_result.get('success', True)
                        print(f"   ✅ Got structured result with keys: {list(analysis_result.keys())}")
                    elif isinstance(analysis_result, str):
                        analysis_details['text_output'] = analysis_result
                        success = True
                        print(f"   ✅ Got text output ({len(analysis_result)} chars)")
                    else:
                        analysis_details['output'] = str(analysis_result)
                        success = True
                        print(f"   ✅ Got result: {type(analysis_result)}")

                    # Check for generated files in temp directory
                    generated_files = []
                    for file_path in Path(temp_dir).rglob('*'):
                        if file_path.is_file():
                            generated_files.append(str(file_path))

                    analysis_details['generated_files'] = generated_files
                    analysis_details['file_count'] = len(generated_files)

                    if generated_files:
                        print(f"   📁 Generated {len(generated_files)} files")

                    results[test_name] = {
                        "success": success,
                        "data_file": data_path,
                        "analysis_details": analysis_details,
                        "prompt_used": prompt
                    }

                except Exception as e:
                    error_msg = str(e)
                    print(f"   ❌ CrewAI analysis failed: {error_msg}")
                    results[test_name] = {
                        "success": False,
                        "error": error_msg,
                        "data_file": data_path,
                        "prompt_used": prompt
                    }

            # Summary
            successful_tests = sum(1 for r in results.values() if r.get('success'))
            total_tests = len(results)

            print(f"\n📊 CrewAI Analysis Summary: {successful_tests}/{total_tests} tests successful")

            return {
                "results": results,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "successful_tests": successful_tests,
                "total_tests": total_tests
            }

    except ImportError as e:
        print(f"❌ CrewAI system not available: {e}")
        return {"error": "CrewAI system not available"}
    except Exception as e:
        print(f"❌ Error in CrewAI testing: {e}")
        return {"error": str(e)}


def main():
    """Main test runner"""
    print("SAS DATA FITTING TEST SUITE")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs("test/test_data/fitting", exist_ok=True)

    # Run tests
    results = {}

    # Test 1: Basic fitting functionality
    results['basic_fitting'] = test_basic_fitting()

    # Test 2: RAG-enhanced fitting workflow
    results['rag_fitting'] = test_rag_enhanced_fitting()

    # Test 3: Model selection accuracy
    results['model_selection'] = test_model_selection_accuracy()

    # Test 4: Full CrewAI collaborative analysis
    results['crewai_analysis'] = test_crewai_collaborative_analysis()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    # Basic fitting summary
    basic_results = results['basic_fitting']
    if isinstance(basic_results, dict) and 'error' not in basic_results:
        successful = sum(1 for r in basic_results.values() if isinstance(r, dict) and r.get('overall_success'))
        total = len(basic_results)
        print(f"Basic Fitting: {successful}/{total} datasets fitted successfully")

        # Show detailed results
        for dataset, result in basic_results.items():
            if isinstance(result, dict) and result.get('success'):
                r_squared = result.get('r_squared', 0)
                quality = result.get('fit_quality', 'Unknown')
                overall = "✅ PASS" if result.get('overall_success') else "⚠️ MARGINAL"
                print(f"  {overall} {dataset}: R² = {r_squared:.4f} ({quality})")
            else:
                print(f"  ❌ {dataset}: FAILED")
    else:
        print(f"Basic Fitting: Not available ({basic_results.get('error', 'Unknown error')})")
    rag_results = results['rag_fitting']
    if isinstance(rag_results, dict) and 'error' not in rag_results:
        successful = sum(1 for r in rag_results.values() if isinstance(r, dict) and r.get('success'))
        total = len(rag_results)
        print(f"RAG-Enhanced Fitting: {successful}/{total} workflows successful")
    elif isinstance(rag_results, dict) and 'error' in rag_results:
        print(f"RAG-Enhanced Fitting: Not available ({rag_results['error']})")

    # Model selection accuracy
    selection_results = results['model_selection']
    if isinstance(selection_results, dict) and 'accuracy' in selection_results:
        accuracy = selection_results['accuracy']
        print(f"Model Selection Accuracy: {accuracy:.1f}%")
    elif isinstance(selection_results, dict) and 'error' in selection_results:
        print(f"Model Selection: Not available ({selection_results['error']})")

    # CrewAI collaborative analysis
    crewai_results = results['crewai_analysis']
    if isinstance(crewai_results, dict) and 'success_rate' in crewai_results:
        success_rate = crewai_results['success_rate'] * 100
        successful = crewai_results['successful_tests']
        total = crewai_results['total_tests']
        print(f"CrewAI Collaborative Analysis: {successful}/{total} workflows successful ({success_rate:.1f}%)")
    elif isinstance(crewai_results, dict) and 'error' in crewai_results:
        print(f"CrewAI Analysis: Not available ({crewai_results['error']})")

    print("\n🎯 Fitting testing complete!")
    print("📁 Test files saved to: test/test_data/fitting/")


if __name__ == "__main__":
    main()
