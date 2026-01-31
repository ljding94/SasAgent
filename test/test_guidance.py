#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level since we're in test/
sys.path.insert(0, str(project_root))

from crewai_sas_agents import UnifiedSASAnalysisSystem


def test_end_to_end_guidance():
    """Test the system's response to guidance/help inquiries"""
    print("🤖 Testing System Guidance Response")
    print("=" * 50)

    try:
        system = UnifiedSASAnalysisSystem()
        print("✅ System initialized successfully")

        # Test the inquiry prompt
        prompt = "what can you do for me?"
        print(f"\n📝 Testing prompt: '{prompt}'")
        print("\n🔄 Analyzing prompt...")

        result = system.analyze_data(prompt=prompt)

        if result and result.get("success"):
            print("\n✅ System Response Received!")
            print("=" * 50)

            # Display key information about the response
            print(f"Task Type: {result.get('task_type', 'Unknown')}")
            print(f"Success: {result.get('success', False)}")

            # Display the actual response content
            if result.get("results"):
                print("\n📋 System Response:")
                print("-" * 30)
                response = result.get("results", "")
                # Truncate very long responses for readability
                print(response)

            # Display any additional metadata
            if result.get("rag_enhanced"):
                print(f"\n🔍 RAG Enhanced: {result.get('rag_enhanced')}")

            return 0

        else:
            print("\n❌ System Response Failed!")
            if result:
                print(f"Error: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Run the guidance test"""
    print("SAS AGENT GUIDANCE TEST")
    print("=" * 60)

    exit_code = test_end_to_end_guidance()

    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ Guidance test completed successfully!")
    else:
        print("❌ Guidance test failed!")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)










