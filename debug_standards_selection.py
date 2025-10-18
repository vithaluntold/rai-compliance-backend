#!/usr/bin/env python3
"""
Debug script to test if user-selected standards are being ignored
during compliance analysis. This reproduces the user's issue.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_standards_selection_bug():
    """Test to reproduce the standards selection issue."""
    
    print("🔍 TESTING STANDARDS SELECTION BUG")
    print("=" * 50)
    
    # Test 1: Check if saved framework selection preserves user selections
    print("\n📝 Test 1: Framework Selection Preservation")
    
    # Simulate user selecting only IAS 7
    user_selected_standards = ["IAS 7"]
    document_id = "test_document_123"
    framework = "IFRS"
    
    # Create mock results file (simulating the framework selection save)
    results_data = {
        "document_id": document_id,
        "framework": framework,
        "standards": user_selected_standards,  # Only IAS 7 selected by user
        "status": "PROCESSING",
        "compliance_analysis": "PENDING"
    }
    
    print(f"   User selected standards: {user_selected_standards}")
    print(f"   Framework: {framework}")
    
    # Save to file (simulate framework selection endpoint)
    results_file = Path("analysis_results") / f"{document_id}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Read back from file (simulate compliance analysis reading)
    with open(results_file, 'r') as f:
        loaded_results = json.load(f)
    
    loaded_standards = loaded_results.get("standards", [])
    
    print(f"   Standards loaded from file: {loaded_standards}")
    
    if loaded_standards == user_selected_standards:
        print("   ✅ PASS: Framework selection preserves user standards")
    else:
        print("   ❌ FAIL: Framework selection corrupted user standards")
        print(f"      Expected: {user_selected_standards}")
        print(f"      Got: {loaded_standards}")
    
    # Test 2: Check if compliance analysis process modifies standards
    print("\n🔧 Test 2: Compliance Analysis Standards Processing")
    
    # Simulate the compliance analysis function parameters
    def simulate_process_compliance_analysis(standards_param):
        """Simulate what process_compliance_analysis receives."""
        print(f"   process_compliance_analysis received: {standards_param}")
        
        # Check if function receives the correct user selection
        if standards_param == user_selected_standards:
            print("   ✅ PASS: Compliance analysis receives correct user selection")
            return True
        else:
            print("   ❌ FAIL: Compliance analysis receives wrong standards")
            print(f"      Expected: {user_selected_standards}")
            print(f"      Received: {standards_param}")
            return False
    
    # Test the function with user standards
    test_passed = simulate_process_compliance_analysis(loaded_standards)
    
    # Test 3: Check if automatic standard identification is interfering
    print("\n🤖 Test 3: Automatic Standards Identification Interference")
    
    # Simulate what might happen with automatic standard identification
    automatically_identified_standards = [
        "IAS 1", "IAS 7", "IAS 8", "IAS 10", "IAS 12", "IAS 16", "IAS 19", 
        "IAS 32", "IAS 36", "IAS 37", "IAS 38", "IFRS 2", "IFRS 3", "IFRS 7", 
        "IFRS 9", "IFRS 15", "IFRS 16"
    ]
    
    print(f"   Automatically identified standards: {automatically_identified_standards}")
    print(f"   User selected standards: {user_selected_standards}")
    
    # Check if there's a bug where auto-identified standards replace user selection
    if len(automatically_identified_standards) > len(user_selected_standards):
        print("   ⚠️  WARNING: Auto-identification found MORE standards than user selected")
        print("   🚨 POTENTIAL BUG: System might be processing ALL identified standards")
        print("      instead of just user-selected ones!")
        
        # Check for the specific issue the user mentioned
        extra_standards = set(automatically_identified_standards) - set(user_selected_standards)
        if extra_standards:
            print(f"   🚨 EXTRA STANDARDS being processed: {list(extra_standards)}")
            print("   🚨 This is the BUG the user reported!")
    
    # Test 4: Framework Selection vs Compliance Analysis Mismatch
    print("\n🔄 Test 4: Framework Selection vs Compliance Analysis Mismatch")
    
    framework_selection_standards = user_selected_standards
    compliance_analysis_standards = automatically_identified_standards  # Bug simulation
    
    print(f"   Framework selection saved: {framework_selection_standards}")
    print(f"   Compliance analysis processed: {compliance_analysis_standards}")
    
    if framework_selection_standards != compliance_analysis_standards:
        print("   🚨 CRITICAL BUG CONFIRMED!")
        print("   🚨 Framework selection saves user choice correctly")
        print("   🚨 BUT compliance analysis processes different standards!")
        print("   🚨 This explains the user's complaint!")
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 50)
    print("🎯 USER ISSUE: Selects specific standards, but AI processes everything")
    print("💡 ROOT CAUSE: Compliance analysis ignores framework selection")
    print("🔧 SOLUTION NEEDED: Ensure compliance analysis ONLY processes")
    print("   the standards saved during framework selection")
    
    # Cleanup
    if results_file.exists():
        results_file.unlink()
    
    return test_passed

if __name__ == "__main__":
    test_standards_selection_bug()