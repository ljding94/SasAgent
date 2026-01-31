#!/usr/bin/env python3
"""
Test SLD Calculation Functionality
Compare direct SasView SLD calculator vs CrewAI agent approach
"""

import os
import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level since we're in test/
sys.path.insert(0, str(project_root))

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable CrewAI telemetry
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['CREWAI_TELEMETRY_DISABLED'] = 'true'
os.environ['DO_NOT_TRACK'] = '1'


def test_direct_sld_calculation():
    """Test direct SasView SLD calculator function"""
    print("\n🧪 Test 1: Direct SasView SLD Calculator")
    print("=" * 50)

    try:
        # Import the SLD calculator tool directly
        from crewai_sas_agents import SLDCalculatorTool

        # Test cases with both chemical formula and neutron/X-ray flag
        test_cases = [
            {"chemical": "THF", "is_neutron": True},
            {"chemical": "THF", "is_neutron": False},
            {"chemical": "D2O", "is_neutron": True},
            {"chemical": "D2O", "is_neutron": False},
            {"chemical": "H2O", "is_neutron": True},
            {"chemical": "H2O", "is_neutron": False},
            {"chemical": "C12H6", "is_neutron": True},
            {"chemical": "C12H6", "is_neutron": False}
        ]

        sld_tool = SLDCalculatorTool()
        direct_results = {}

        for test_case in test_cases:
            chemical = test_case["chemical"]
            is_neutron = test_case["is_neutron"]
            scattering_type = "Neutron" if is_neutron else "X-ray"
            test_key = f"{chemical}_{scattering_type}"

            print(f"\n📋 Testing: {chemical} ({scattering_type})")
            try:
                result = sld_tool._run(formula=chemical, is_neutron=is_neutron)

                if result.get('success'):
                    sld_data = result['result']
                    sld_real = sld_data['sld_real']

                    print("  ✅ Success!")
                    print(f"     SLD Real: {sld_real:.4f} × 10⁻⁶ Å⁻²")
                    print(f"     Density: {result.get('density', 'N/A')} g/cm³")

                    direct_results[test_key] = {
                        'chemical': chemical,
                        'is_neutron': is_neutron,
                        'sld_real': sld_real,
                        'density': result.get('density'),
                        'success': True
                    }
                else:
                    print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                    direct_results[test_key] = {
                        'chemical': chemical,
                        'is_neutron': is_neutron,
                        'success': False,
                        'error': result.get('error')
                    }

            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
                direct_results[test_key] = {
                    'chemical': chemical,
                    'is_neutron': is_neutron,
                    'success': False,
                    'error': str(e)
                }

        print("\n📊 Direct SLD calculation test completed")
        print(f"   Successful: {sum(1 for r in direct_results.values() if r.get('success'))}/{len(test_cases)}")

        return direct_results

    except ImportError as e:
        print(f"❌ Could not import SLD calculator tool: {e}")
        print("   This may indicate the tool is not available or not properly installed")
        return {}
    except Exception as e:
        print(f"❌ Direct SLD calculation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def test_agent_sld_calculation(llm_model="google/gemini-2.5-flash"):
    """Test CrewAI agent-based SLD calculation"""
    print("\n🤖 Test 2: CrewAI Agent SLD Calculation")
    print("=" * 50)

    try:
        from crewai_sas_agents import analyze_sas_data

        # Test cases with both chemical formula and neutron/X-ray flag
        test_cases = [
            {"chemical": "THF", "is_neutron": True},
            #{"chemical": "THF", "is_neutron": False},
            {"chemical": "D2O", "is_neutron": True},
            #{"chemical": "D2O", "is_neutron": False},
            #{"chemical": "H2O", "is_neutron": True},
            #{"chemical": "C12H6", "is_neutron": True},
        ]

        agent_results = {}
        output_folder = project_root / "test" / "test_data" / "sld_calculations" / llm_model.replace("/", "_")
        output_folder.mkdir(parents=True, exist_ok=True)
        output_folder = str(output_folder)

        # Check if API key is available
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("⚠️  No OPENROUTER_API_KEY found")
            print("   Set OPENROUTER_API_KEY environment variable for agent-based tests")
            return {}

        for test_case in test_cases:
            chemical = test_case["chemical"]
            is_neutron = test_case["is_neutron"]
            scattering_type = "Neutron" if is_neutron else "X-ray"
            test_key = f"{chemical}_{scattering_type}"

            print(f"\n📋 Testing: {chemical} ({scattering_type})")

            # Create SLD calculation prompt
            scattering_desc = "neutron" if is_neutron else "x-ray"
            prompt = f"Calculate the {scattering_desc} SLD (scattering length density) for the molecular formula {chemical}"

            try:
                result = analyze_sas_data(
                    prompt=prompt,
                    data_path=None,  # No data file for SLD calculation
                    output_folder=output_folder,
                    verbose=False,  # Suppress verbose crew output
                    api_key=api_key,
                    model=llm_model
                )

                print(f"  📋 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Error'}")

                if result.get('success'):
                    task_type = result.get('task_type')
                    print(f"  📊 Task type: {task_type}")

                    if task_type == 'sld_calculation':
                        # Extract SLD values from the results
                        results_data = result.get('results', {})
                        print(f"  📝 Results keys: {list(results_data.keys()) if isinstance(results_data, dict) else type(results_data)}")

                        # Try to extract SLD values directly from results
                        sld_values = None
                        if isinstance(results_data, dict):
                            sld_values = {
                                'sld_real': results_data.get('sld_real'),
                                'sld_imag': results_data.get('sld_imag')
                            }
                            # Filter out None values
                            sld_values = {k: v for k, v in sld_values.items() if v is not None}

                        # If direct extraction failed, try parsing text output
                        if not sld_values:
                            output_text = result.get('output', '')
                            if not output_text:
                                # Try to convert results dict to string
                                output_text = str(results_data) if results_data else ''
                            print(f"  📝 Output preview: {output_text[:200]}...")
                            sld_values = extract_sld_from_output(output_text, chemical)

                        if sld_values:
                            print("  ✅ Success!")
                            print(f"     SLD Real: {sld_values.get('sld_real', 'N/A')}")

                            agent_results[test_key] = {
                                'chemical': chemical,
                                'is_neutron': is_neutron,
                                'sld_real': sld_values.get('sld_real'),
                                #'sld_imag': sld_values.get('sld_imag'),
                                'success': True,
                                'results': results_data
                            }
                        else:
                            print("  ⚠️  Could not extract SLD values from output")
                            print(f"  📝 Full results data: {results_data}")
                            agent_results[test_key] = {
                                'chemical': chemical,
                                'is_neutron': is_neutron,
                                'success': False,
                                'error': 'Could not extract SLD values',
                                'results': results_data
                            }
                    else:
                        print(f"  ⚠️  Unexpected task type: {task_type}")
                        agent_results[test_key] = {
                            'chemical': chemical,
                            'is_neutron': is_neutron,
                            'success': False,
                            'error': f'Wrong task type: {task_type}',
                            'output': result.get('output', '')
                        }
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"  ❌ Failed: {error_msg}")
                    agent_results[test_key] = {
                        'chemical': chemical,
                        'is_neutron': is_neutron,
                        'success': False,
                        'error': error_msg
                    }

            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
                agent_results[test_key] = {
                    'chemical': chemical,
                    'is_neutron': is_neutron,
                    'success': False,
                    'error': str(e)
                }

        print("\n📊 Agent SLD calculation test completed")
        successful_count = sum(1 for r in agent_results.values() if r.get('success'))
        print(f"   Successful: {successful_count}/{len(test_cases)}")

        # Save test results to file
        _save_agent_test_results(agent_results, output_folder, llm_model, successful_count, len(test_cases))

        return agent_results

    except ImportError as e:
        print(f"❌ Could not import CrewAI agents: {e}")
        return {}
    except Exception as e:
        print(f"❌ Agent SLD calculation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def _save_agent_test_results(agent_results, output_folder, llm_model, successful_count, total_count, run_number=0):
    """Save agent SLD calculation test results to files"""
    import json
    from datetime import datetime

    try:
        output_path = Path(output_folder)

        # Create summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'llm_model': llm_model,
            'total_tests': total_count,
            'successful': successful_count,
            'failed': total_count - successful_count,
            'success_rate': f"{(successful_count/total_count)*100:.1f}%",
            'tests': {}
        }

        # Organize results by chemical
        for test_key, result in agent_results.items():
            chemical = result.get('chemical', 'unknown')
            is_neutron = result.get('is_neutron', True)
            scattering_type = 'Neutron' if is_neutron else 'X-ray'

            if chemical not in summary['tests']:
                summary['tests'][chemical] = {'neutron': {}, 'xray': {}}

            test_type = 'neutron' if is_neutron else 'xray'
            summary['tests'][chemical][test_type] = {
                'success': result.get('success', False),
                'sld_real': result.get('sld_real'),
                'sld_imag': result.get('sld_imag'),
                'error': result.get('error')
            }

        # Save JSON summary
        json_file = output_path / f'{run_number}_test_summary.json'
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Saved test summary to: {json_file}")

        # Save detailed results JSON
        detailed_file = output_path / f'{run_number}_test_results_detailed.json'
        with open(detailed_file, 'w') as f:
            # Remove Path objects and non-serializable items for JSON
            clean_results = {}
            for key, val in agent_results.items():
                clean_val = val.copy()
                if 'results' in clean_val:
                    clean_val['results'] = str(clean_val['results'])
                clean_results[key] = clean_val
            json.dump(clean_results, f, indent=2)
        print(f"📄 Saved detailed results to: {detailed_file}")

        # Save human-readable report
        report_file = output_path / f'{run_number}_test_report.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SAS Agent SLD Calculation Test Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"LLM Model: {llm_model}\n")
            f.write(f"Total Tests: {total_count}\n")
            f.write(f"Successful: {successful_count}\n")
            f.write(f"Failed: {total_count - successful_count}\n")
            f.write(f"Success Rate: {summary['success_rate']}\n")
            f.write("-" * 70 + "\n\n")

            f.write("Results by Chemical:\n")
            f.write("-" * 70 + "\n")
            for chemical in sorted(summary['tests'].keys()):
                f.write(f"\n{chemical}:\n")
                for scattering_type in ['neutron', 'xray']:
                    result = summary['tests'][chemical][scattering_type]
                    type_name = 'Neutron' if scattering_type == 'neutron' else 'X-ray'
                    # Check if result is empty (no test for this type)
                    if not result:
                        continue
                    status = "✅ SUCCESS" if result.get('success', False) else "❌ FAILED"
                    f.write(f"  {type_name}: {status}\n")
                    if result.get('success'):
                        f.write(f"    SLD Real: {result.get('sld_real')}\n")
                    else:
                        f.write(f"    Error: {result.get('error')}\n")

        print(f"📄 Saved test report to: {report_file}")

    except Exception as e:
        print(f"⚠️  Could not save test results: {e}")


def extract_sld_from_output(output_text, chemical):
    """Extract SLD values from agent output text"""
    import re

    # Look for SLD values in the output
    sld_values = {}

    # Enhanced patterns for real SLD from the actual agent output format
    real_patterns = [
        # "**Neutron SLD (real part): 2.43 x 10⁻⁶ Å⁻²**" - with bold markers around everything
        r'\*\*Neutron\s+SLD\s*\([^)]*\):\s*([+-]?[0-9.]+)\s*(?:x|×)\s*10[⁻\-][0-9]',
        # "**Neutron SLD (real component):** 2.421 x 10⁻⁶ Å⁻²"
        r'\*\*Neutron\s+SLD[^:]*\*\*:\s*([+-]?[0-9.]+)\s*(?:x|×)\s*10[⁻\-][0-9]',
        # "**Neutron SLD (real component):** 2.421"
        r'\*\*Neutron\s+SLD[^:]*\*\*:\s*([+-]?[0-9.]+)',
        # Simplest first: just "is" followed by number and x 10
        r'is\s+([0-9.]+)\s+x\s+10',
        # Plain text format: "is 2.528 x 10⁻⁶ Å⁻²"
        r'is\s+([+-]?[0-9.]+)\s*(?:x|×)\s*10[⁻\-][0-9]',
        # Pattern for "SLD is calculated as **2.528 x 10⁻⁶ Å⁻²**"
        r'(?:is\s+)?(?:calculated\s+)?as\s+\*\*([+-]?[0-9.]+)\s*(?:x|×)\s*10[⁻\-][0-9]',
        # "**Real SLD (ρ'):** 8.840"
        r'\*\*Real\s+SLD[^:]*\*\*:\s*([+-]?[0-9.]+)',
        # "**Real Component (sld_real)**: 0.1830"
        r'\*\*Real\s+Component[^:]*\*\*:\s*([+-]?[0-9.]+)',
        # "Real SLD: 8.840"
        r'Real\s+SLD[^:]*:\s*([+-]?[0-9.]+)',
        # "Real Component (sld_real): 0.1830"
        r'Real\s+Component[^:]*:\s*([+-]?[0-9.]+)',
        # "sld_real: 0.1830"
        r'sld_real[^:]*:\s*([+-]?[0-9.]+)',
        # "- **Real Component**: 0.1830"
        r'-\s*\*\*Real[^:]*\*\*:\s*([+-]?[0-9.]+)',
        # "Real SLD: 8.840 x 10⁻⁶"
        r'Real\s+SLD.*?([+-]?[0-9.]+)\s*(?:x|×)\s*10[⁻\-]?[0-9]*',
        # "Real ... 8.840 x 10⁻⁶ Å⁻²" with × symbol
        r'Real.*?([+-]?[0-9.]+)\s*[×x]\s*10[⁻\-]?[0-9]*\s*Å[⁻\-]?²',
        # "Real ... 8.840 x 10⁻⁶ Å⁻²"
        r'Real.*?([+-]?[0-9.]+)\s*x?\s*10[⁻\-]?[0-9]*\s*Å[⁻\-]?²',
        # "Neutron SLD ... 2.421 x 10⁻⁶"
        r'Neutron\s+SLD[^:]*:\s*([+-]?[0-9.]+)\s*(?:x|×)\s*10[⁻\-]?[0-9]*',
    ]

    # Try to find real SLD
    for i, pattern in enumerate(real_patterns):
        match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                sld_values['sld_real'] = float(match.group(1))
                print(f"   🔍 Extracted real SLD: {match.group(1)} (pattern {i+1}: {pattern[:40]}...)")
                break
            except ValueError:
                continue

    if not sld_values:
        print(f"   🔍 DEBUG: Failed to match any pattern. First 200 chars: {output_text[:200]}")

    return sld_values


def test_known_sld_values():
    """Test against known SLD values for validation"""
    print("\n📚 Test 4: Validation Against Known SLD Values")
    print("=" * 50)

    # Known SLD values (approximate, at neutron wavelength ~6 Å)
    known_values = {
        "D2O": 6.4,     # Heavy water - well known value
        "H2O": -0.56,   # Water - well known negative value
        "THF": 0.183,   # Tetrahydrofuran with density 0.889 g/cm³
        # THF and C12H6 values may vary depending on density assumptions
    }

    try:
        from crewai_sas_agents import SLDCalculatorTool
        sld_tool = SLDCalculatorTool()

        validation_results = {}

        for chemical, expected_sld in known_values.items():
            print(f"\n📋 Validating: {chemical} (expected SLD ≈ {expected_sld})")

            try:
                result = sld_tool._run(formula=chemical)

                if result.get('success'):
                    calculated_sld = result['result']['sld_real']
                    relative_error = abs(calculated_sld - expected_sld) / abs(expected_sld) * 100

                    print(f"  📊 Calculated: {calculated_sld:.4f} × 10⁻⁶ Å⁻²")
                    print(f"  📊 Expected:   {expected_sld:.4f} × 10⁻⁶ Å⁻²")
                    print(f"  📊 Relative error: {relative_error:.1f}%")

                    # Consider good if within 10% of known value
                    is_good = relative_error < 10
                    status = "✅ GOOD" if is_good else "⚠️  DIFFERENT"
                    print(f"  📊 Status: {status}")

                    validation_results[chemical] = {
                        'calculated': calculated_sld,
                        'expected': expected_sld,
                        'relative_error': relative_error,
                        'is_good': is_good
                    }
                else:
                    print(f"  ❌ Calculation failed: {result.get('error')}")
                    validation_results[chemical] = {'failed': True}

            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
                validation_results[chemical] = {'failed': True, 'error': str(e)}

        # Summary
        successful = [r for r in validation_results.values() if not r.get('failed')]
        good_matches = [r for r in successful if r.get('is_good')]

        print("\n📈 Validation Summary:")
        print(f"   Total tested: {len(known_values)}")
        print(f"   Successful calculations: {len(successful)}")
        print(f"   Good matches (< 10% error): {len(good_matches)}")

        if successful:
            avg_error = sum(r['relative_error'] for r in successful) / len(successful)
            print(f"   Average relative error: {avg_error:.1f}%")

        return len(good_matches) >= len(known_values) // 2  # At least half should be good

    except ImportError as e:
        print(f"❌ Could not import SLD calculator: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        return False


def cleanup_test_environment():
    """Clean up test directories"""
    print("\n🧹 Cleaning up test environment...")

    test_cache_dir = project_root / "test_cache"
    if test_cache_dir.exists():
        import shutil
        shutil.rmtree(test_cache_dir)
        print(f"  🗑️  Removed: {test_cache_dir}")


def test_end_to_end_sld_calculation(llm_model="google/gemini-2.5-flash", run_number=0):
    """Test the full CrewAI agent system using UnifiedSASAnalysisSystem.analyze_data() for SLD calculation tasks"""
    print("\n🤖 Testing Full CrewAI Agent System Integration - SLD Calculation")
    print("=" * 60)

    try:
        from crewai_sas_agents import UnifiedSASAnalysisSystem

        # Initialize the unified system with the specified model
        system = UnifiedSASAnalysisSystem(model=llm_model)

        # Test cases with both chemical formula and neutron/X-ray flag
        test_cases = [
            {
                "name": "heavy_water_sld",
                "prompt": "What is the neutron SLD for heavy water?",
                "expected_formula": "D2O",
            },
            {
                "name": "heavy_water_sld_xray",
                "prompt": "What is the X-ray SLD for heavy water?",
                "expected_formula": "D2O",
            },
            {
                "name": "thf_sld",
                "prompt": "What is the SLD of tetrahydrofuran",
                "expected_formula": "C4H8O",
            },
            {
                "name": "thf_sld_xray",
                "prompt": "What is the X-ray SLD of tetrahydrofuran",
                "expected_formula": "C4H8O",
            },
            {
                "name": "dmso_sld",
                "prompt": "What is the SLD of DMSO (Dimethyl sulfoxide)",
                "expected_formula": "C4H8O",
            },
            {
                "name": "PAN",
                "prompt": "What is the SLD of PAN polymer C3H3N, use density 1.184 g/cm3",
                "expected_formula": "C3H3N",
            },
        ]

        sld_results = {}
        output_folder = project_root / "test" / "test_data" / "sld_calculations" / f"{llm_model.replace('/', '_')}"
        output_folder.mkdir(parents=True, exist_ok=True)

        test_cases = [test_cases[1]]
        for test_case in test_cases:
            print(f"\n📋 Testing: {test_case['name']}")
            print(f"   Prompt: {test_case['prompt']}")

            try:
                # Call the unified system with the test prompt
                result = system.analyze_data(
                    prompt=test_case["prompt"]
                )

                if result.get('success'):
                    task_type = result.get('task_type')
                    print(f"  📊 Task type: {task_type}")

                    if task_type == 'sld_calculation':
                        results_data = result.get('results', {})

                        # Get output text to check for success
                        output_text = result.get('output', '')
                        if not output_text and isinstance(results_data, dict):
                            output_text = results_data.get('output', '')
                        if not output_text:
                            output_text = str(results_data) if results_data else ''

                        print(f"  📝 Output preview: {output_text[:300]}...")

                        # Check for success indicators - look for SLD-related content with numbers
                        import re
                        success_indicators = [
                            r'SLD.*?[0-9.]+\s*(?:x|×)\s*10[⁻\-][0-9]',  # SLD with scientific notation
                            r'scattering\s+length\s+density.*?[0-9.]+',  # "scattering length density" with number
                            r'neutron\s+SLD.*?[0-9.]+',  # "neutron SLD" with number
                            r'calculated.*?SLD',  # "calculated" and "SLD" together
                            r'[0-9.]+\s*g/cm',  # density with g/cm units
                        ]

                        has_sld_calculation = any(re.search(pattern, output_text, re.IGNORECASE) for pattern in success_indicators)

                        # Try to extract actual SLD value
                        sld_value = None
                        sld_values = extract_sld_from_output(output_text, test_case['name'])
                        if sld_values:
                            sld_value = sld_values.get('sld_real')

                        if has_sld_calculation and len(output_text) > 50:
                            print("  ✅ Success! SLD calculation completed")
                            if sld_value:
                                print(f"     SLD Real: {sld_value}")
                            else:
                                print(f"     SLD value present in output")

                            sld_results[test_case["name"]] = {
                                "success": True,
                                "sld_real": sld_value,
                                "task_type": task_type,
                                "output": output_text[:500],  # Store truncated output
                                "results": str(results_data)[:500] if results_data else None
                            }
                        else:
                            print("  ⚠️  Could not verify SLD calculation in output")
                            sld_results[test_case["name"]] = {
                                "success": False,
                                "error": "No SLD calculation indicators found",
                                "output": output_text[:500],
                                "results": str(results_data)[:500] if results_data else None
                            }
                    else:
                        print(f"  ⚠️  Unexpected task type: {task_type}")
                        sld_results[test_case["name"]] = {
                            "success": False,
                            "error": f"Wrong task type: {task_type}"
                        }
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"  ❌ Failed: {error_msg}")
                    sld_results[test_case["name"]] = {
                        "success": False,
                        "error": error_msg
                    }

            except Exception as e:
                sld_results[test_case["name"]] = {
                    "success": False,
                    "error": str(e),
                    "exception": True
                }
                print(f"  ❌ Exception: {e}")

        print("\n📊 End-to-end SLD calculation test completed")
        successful_count = sum(1 for r in sld_results.values() if r.get('success'))
        print(f"   Successful: {successful_count}/{len(test_cases)}")

        # Save test results to file
        if sld_results:
            _save_agent_test_results(sld_results, str(output_folder), llm_model, successful_count, len(test_cases), run_number)

        return sld_results

    except ImportError as e:
        print(f"⚠️  UnifiedSASAnalysisSystem not available: {e}")
        return {"sld_tests": "not_available"}


def main():
    """Run all SLD calculation tests"""
    print("🧪 SAS Agent SLD Calculation Tests")
    print("=" * 60)
    print("Testing chemicals: THF, D2O, H2O, C12H6")
    print("Comparing direct SasView calculator vs CrewAI agent approach")

    try:
        # Test 1: Direct SLD calculation
        print("\n" + "="*60)
        #direct_results = test_direct_sld_calculation()

        # Test 2: Agent-based SLD calculation
        print("\n" + "="*60)
        #agent_results = test_agent_sld_calculation()

        # Test 3: Comparison (commented out - requires both tests)
        # print("\n" + "="*60)
        # comparison_success = compare_sld_results(direct_results, agent_results)

        # Test 4: Validation against known values
        print("\n" + "="*60)
        #validation_success = test_known_sld_values()

        # Test 5: End-to-end unified system test
        print("\n" + "="*60)
        llm_model="google/gemini-2.5-flash"
        #llm_model = "x-ai/grok-4.1-fast"
        #llm_model = "openai/gpt-5-mini"

        for run_num in range(7,8):
            end_to_end_results = test_end_to_end_sld_calculation(llm_model=llm_model, run_number=run_num)


    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        cleanup_test_environment()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
