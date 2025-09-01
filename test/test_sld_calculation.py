#!/usr/bin/env python3
"""
Test SLD Calculation Functionality
Compare direct SasView SLD calculator vs CrewAI agent approach
"""

import os
import sys

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

        # Test chemicals - use SLDCalculatorTool for both direct and agent tests
        # since it handles name normalization properly (THF -> C4H8O)
        test_chemicals = [
            "THF",      # Tetrahydrofuran
            "D2O",      # Heavy water
            "H2O",      # Water
            "C12H6"     # Acenaphthylene or similar aromatic compound
        ]

        sld_tool = SLDCalculatorTool()
        direct_results = {}

        for chemical in test_chemicals:
            print(f"\n📋 Testing: {chemical}")
            try:
                result = sld_tool._run(formula=chemical)

                if result.get('success'):
                    sld_data = result['result']
                    sld_real = sld_data['sld_real']
                    sld_imag = sld_data['sld_imag']

                    print("  ✅ Success!")
                    print(f"     SLD Real: {sld_real:.4f} × 10⁻⁶ Å⁻²")
                    print(f"     SLD Imag: {sld_imag:.6f} × 10⁻⁶ Å⁻²")
                    print(f"     Density: {result.get('density', 'N/A')} g/cm³")

                    direct_results[chemical] = {
                        'sld_real': sld_real,
                        'sld_imag': sld_imag,
                        'density': result.get('density'),
                        'success': True
                    }
                else:
                    print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                    direct_results[chemical] = {'success': False, 'error': result.get('error')}

            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
                direct_results[chemical] = {'success': False, 'error': str(e)}

        print("\n📊 Direct SLD calculation test completed")
        print(f"   Successful: {sum(1 for r in direct_results.values() if r.get('success'))}/{len(test_chemicals)}")

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


def test_agent_sld_calculation():
    """Test CrewAI agent-based SLD calculation"""
    print("\n🤖 Test 2: CrewAI Agent SLD Calculation")
    print("=" * 50)

    try:
        from crewai_sas_agents import analyze_sas_data

        # Test chemicals
        test_chemicals = [
            "THF",      # Tetrahydrofuran
            "D2O",      # Heavy water
            "H2O",      # Water
            "C12H6"     # Acenaphthylene or similar aromatic compound
        ]

        agent_results = {}
        output_folder = "/Users/ldq/Work/SasAgent/test_cache/sld_calculations"
        os.makedirs(output_folder, exist_ok=True)

        # Check if API key is available
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("⚠️  No OPENROUTER_API_KEY found")
            print("   Set OPENROUTER_API_KEY environment variable for agent-based tests")
            return {}

        for chemical in test_chemicals:
            print(f"\n📋 Testing: {chemical}")

            # Create SLD calculation prompt
            prompt = f"Calculate the SLD (scattering length density) for {chemical}"

            try:
                result = analyze_sas_data(
                    prompt=prompt,
                    data_path=None,  # No data file for SLD calculation
                    output_folder=output_folder,
                    verbose=True,
                    api_key=api_key,
                    model="openai/gpt-4o-mini"
                )

                print(f"  📋 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Error'}")

                if result.get('success'):
                    task_type = result.get('task_type')
                    print(f"  📊 Task type: {task_type}")

                    if task_type == 'sld_calculation':
                        # Extract SLD values from the result
                        output_text = result.get('output', '')
                        print(f"  📝 Output preview: {output_text[:200]}...")

                        # Try to extract SLD values from the output
                        sld_values = extract_sld_from_output(output_text, chemical)

                        if sld_values:
                            print("  ✅ Success!")
                            print(f"     SLD Real: {sld_values.get('sld_real', 'N/A')}")
                            print(f"     SLD Imag: {sld_values.get('sld_imag', 'N/A')}")

                            agent_results[chemical] = {
                                'sld_real': sld_values.get('sld_real'),
                                'sld_imag': sld_values.get('sld_imag'),
                                'success': True,
                                'output': output_text
                            }
                        else:
                            print("  ⚠️  Could not extract SLD values from output")
                            agent_results[chemical] = {
                                'success': False,
                                'error': 'Could not extract SLD values',
                                'output': output_text
                            }
                    else:
                        print(f"  ⚠️  Unexpected task type: {task_type}")
                        agent_results[chemical] = {
                            'success': False,
                            'error': f'Wrong task type: {task_type}',
                            'output': result.get('output', '')
                        }
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"  ❌ Failed: {error_msg}")
                    agent_results[chemical] = {'success': False, 'error': error_msg}

            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
                agent_results[chemical] = {'success': False, 'error': str(e)}

        print("\n📊 Agent SLD calculation test completed")
        print(f"   Successful: {sum(1 for r in agent_results.values() if r.get('success'))}/{len(test_chemicals)}")

        return agent_results

    except ImportError as e:
        print(f"❌ Could not import CrewAI agents: {e}")
        return {}
    except Exception as e:
        print(f"❌ Agent SLD calculation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def extract_sld_from_output(output_text, chemical):
    """Extract SLD values from agent output text"""
    import re

    # Look for SLD values in the output
    sld_values = {}

    # Enhanced patterns for real SLD from the actual agent output format
    real_patterns = [
        r'\*\*Real\s+Component[^:]*\*\*:\s*([+-]?[0-9.]+)',  # "**Real Component (sld_real)**: 0.1830"
        r'Real\s+Component[^:]*:\s*([+-]?[0-9.]+)',          # "Real Component (sld_real): 0.1830"
        r'sld_real[^:]*:\s*([+-]?[0-9.]+)',                  # "sld_real: 0.1830"
        r'-\s*\*\*Real[^:]*\*\*:\s*([+-]?[0-9.]+)',         # "- **Real Component**: 0.1830"
        r'Real.*?([+-]?[0-9.]+)\s*[×x]\s*10[⁻\-]?[0-9]*\s*Å[⁻\-]?²',  # Extract from formatted text with × symbol
        r'Real.*?([+-]?[0-9.]+)\s*x?\s*10[⁻\-]?[0-9]*\s*Å[⁻\-]?²',  # Extract from formatted text
    ]

    # Enhanced patterns for imaginary SLD
    imag_patterns = [
        r'\*\*Imaginary\s+Component[^:]*\*\*:\s*([+-]?[0-9.]+)',  # "**Imaginary Component (sld_imag)**: 5.5230"
        r'Imaginary\s+Component[^:]*:\s*([+-]?[0-9.]+)',          # "Imaginary Component (sld_imag): 5.5230"
        r'sld_imag[^:]*:\s*([+-]?[0-9.]+)',                       # "sld_imag: 5.5230"
        r'-\s*\*\*Imaginary[^:]*\*\*:\s*([+-]?[0-9.]+)',         # "- **Imaginary Component**: 5.5230"
        r'Imaginary.*?([+-]?[0-9.]+)\s*[×x]\s*10[⁻\-]?[0-9]*\s*Å[⁻\-]?²',  # Extract from formatted text with × symbol
        r'Imaginary.*?([+-]?[0-9.]+)\s*x?\s*10[⁻\-]?[0-9]*\s*Å[⁻\-]?²',  # Extract from formatted text
    ]

    # Try to find real SLD
    for pattern in real_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                sld_values['sld_real'] = float(match.group(1))
                print(f"   🔍 Extracted real SLD: {match.group(1)} using pattern: {pattern[:30]}...")
                break
            except ValueError:
                continue

    # Try to find imaginary SLD
    for pattern in imag_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                sld_values['sld_imag'] = float(match.group(1))
                print(f"   🔍 Extracted imag SLD: {match.group(1)} using pattern: {pattern[:30]}...")
                break
            except ValueError:
                continue

    return sld_values if sld_values else None


def compare_sld_results(direct_results, agent_results):
    """Compare direct vs agent SLD calculation results"""
    print("\n📊 Test 3: Comparison of Direct vs Agent Results")
    print("=" * 50)

    if not direct_results and not agent_results:
        print("❌ No results to compare")
        return False

    # Get all chemicals that were tested
    all_chemicals = set(direct_results.keys()) | set(agent_results.keys())

    print(f"\n{'Chemical':<10} {'Direct SLD':<15} {'Agent SLD':<15} {'Match':<8} {'Status':<15}")
    print("-" * 70)

    matches = 0
    total_compared = 0

    for chemical in sorted(all_chemicals):
        direct_data = direct_results.get(chemical, {})
        agent_data = agent_results.get(chemical, {})

        # Check if both succeeded
        direct_success = direct_data.get('success', False)
        agent_success = agent_data.get('success', False)

        if direct_success and agent_success:
            direct_sld = direct_data.get('sld_real', 0)
            agent_sld = agent_data.get('sld_real', 0)

            # Check if values are close (within 5% tolerance)
            if direct_sld != 0:
                relative_diff = abs(direct_sld - agent_sld) / abs(direct_sld)
                match = relative_diff < 0.05  # 5% tolerance
            else:
                match = abs(direct_sld - agent_sld) < 0.1

            status = "✅ MATCH" if match else "❌ DIFFER"
            if match:
                matches += 1
            total_compared += 1

            print(f"{chemical:<10} {direct_sld:<15.4f} {agent_sld:<15.4f} {str(match):<8} {status:<15}")

        elif direct_success:
            direct_sld = direct_data.get('sld_real', 0)
            print(f"{chemical:<10} {direct_sld:<15.4f} {'FAILED':<15} {'N/A':<8} {'Agent Failed':<15}")

        elif agent_success:
            agent_sld = agent_data.get('sld_real', 0)
            print(f"{chemical:<10} {'FAILED':<15} {agent_sld:<15.4f} {'N/A':<8} {'Direct Failed':<15}")

        else:
            print(f"{chemical:<10} {'FAILED':<15} {'FAILED':<15} {'N/A':<8} {'Both Failed':<15}")

    if total_compared > 0:
        match_percentage = (matches / total_compared) * 100
        print("\n📈 Comparison Summary:")
        print(f"   Total compared: {total_compared}")
        print(f"   Matches: {matches}")
        print(f"   Match rate: {match_percentage:.1f}%")

        return match_percentage >= 80  # Consider success if 80%+ match
    else:
        print("\n❌ No successful comparisons possible")
        return False


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

    test_cache_dir = "/Users/ldq/Work/SasAgent/test_cache"
    if os.path.exists(test_cache_dir):
        import shutil
        shutil.rmtree(test_cache_dir)
        print(f"  🗑️  Removed: {test_cache_dir}")


def test_end_to_end_sld_calculation():
    """Test the full CrewAI agent system using UnifiedSASAnalysisSystem.analyze_data() for SLD calculation tasks"""
    print("\n🤖 Testing Full CrewAI Agent System Integration - SLD Calculation")
    print("=" * 60)

    try:
        from crewai_sas_agents import UnifiedSASAnalysisSystem

        # Initialize the unified system
        system = UnifiedSASAnalysisSystem()

        # Test cases for the unified system with SLD calculation tasks
        test_cases = [
            {
                "name": "heavy_water_sld",
                "prompt": "What is the neutron SLD for heavy water?",
                "expected_formula": "D2O",
            },
            {
                "name": "thf_sld",
                "prompt": "What is the SLD of tetrahydrofuran",
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

            #{
            #    "name": "ladder_polymer_sld",
            #    "prompt": "What is the SLD of C15H14 polymer",
            #    "expected_formula": "C15H14",
            #},

        ]

        sld_results = {}

        for test_case in test_cases[-1:]:
            print(f"\n📋 Testing: {test_case['name']}")
            print(f"   Prompt: {test_case['prompt']}")

            try:
                # Call the unified system with the test prompt
                result = system.analyze_data(
                    prompt=test_case["prompt"]
                )

            except Exception as e:
                sld_results[test_case["name"]] = {
                    "success": False,
                    "error": str(e),
                    "exception": True
                }
                print(f"❌ {test_case['name']}: Exception - {e}")

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
        direct_results = test_direct_sld_calculation()

        # Test 2: Agent-based SLD calculation (commented out - requires API key)
        # print("\n" + "="*60)
        # agent_results = test_agent_sld_calculation()

        # Test 3: Comparison (commented out - requires both tests)
        # print("\n" + "="*60)
        # comparison_success = compare_sld_results(direct_results, agent_results)

        # Test 4: Validation against known values
        print("\n" + "="*60)
        validation_success = test_known_sld_values()

        # Test 5: End-to-end unified system test
        print("\n" + "="*60)
        end_to_end_results = test_end_to_end_sld_calculation()


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
