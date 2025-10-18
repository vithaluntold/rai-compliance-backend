#!/usr/bin/env python3
"""
Test script to reproduce the exact issue: User selects specific standards
but AI compliance analysis processes different/additional standards.

This will test the actual API endpoints to confirm the bug.
"""

import requests
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000/api/v1/analysis"

def test_standards_selection_bug():
    """Test the exact workflow that causes the standards selection bug."""
    
    print("🔍 TESTING STANDARDS SELECTION BUG - LIVE API TEST")
    print("=" * 60)
    
    # Test document ID from existing analysis results
    test_document_id = "RAI-02102025-E1JZP-HV77U"  # From recent analysis
    
    print(f"📄 Using existing test document: {test_document_id}")
    
    # Step 1: Check document status
    print(f"\n📋 Step 1: Checking document status...")
    try:
        response = requests.get(f"{BASE_URL}/analysis/documents/{test_document_id}", timeout=10)
        print(f"   📡 Response status: {response.status_code}")
        if response.status_code == 200:
            doc_status = response.json()
            print(f"   ✅ Document exists - Status: {doc_status.get('status')}")
            print(f"   📊 Current framework: {doc_status.get('framework', 'None')}")
            print(f"   📋 Current standards: {doc_status.get('standards', 'None')}")
        else:
            print(f"   ❌ Document check failed: {response.status_code}")
            print(f"   📄 Response text: {response.text[:200]}")
            print("   🔄 Continuing with framework selection test anyway...")
            # Don't return False - continue with test
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error - is backend running?: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error checking document: {e}")
        print("   🔄 Continuing with framework selection test anyway...")
        # Don't return False - continue with test
    
    # Step 2: Test framework selection with ONLY IAS 7
    print(f"\n🎯 Step 2: Selecting ONLY IAS 7 standard...")
    
    framework_request = {
        "framework": "IFRS",
        "standards": ["IAS 7"],  # User selects ONLY IAS 7
        "specialInstructions": "Test compliance analysis with IAS 7 only",
        "extensiveSearch": False,
        "processingMode": "smart"
    }
    
    print(f"   📤 Sending framework selection: {framework_request}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/analysis/documents/{test_document_id}/select-framework",
            json=framework_request
        )
        
        if response.status_code == 200:
            selection_result = response.json()
            print(f"   ✅ Framework selection successful")
            print(f"   📋 Saved standards: {selection_result.get('standards', [])}")
            
            # CRITICAL CHECK: Verify only IAS 7 was saved
            saved_standards = selection_result.get('standards', [])
            if saved_standards == ["IAS 7"]:
                print(f"   ✅ PASS: Framework selection correctly saved only IAS 7")
            else:
                print(f"   ❌ FAIL: Framework selection saved wrong standards!")
                print(f"      Expected: ['IAS 7']")
                print(f"      Got: {saved_standards}")
                return False
        else:
            print(f"   ❌ Framework selection failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error in framework selection: {e}")
        return False
    
    # Step 3: Wait for compliance analysis to start and monitor progress
    print(f"\n⏱️ Step 3: Monitoring compliance analysis progress...")
    
    max_wait_time = 120  # 2 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            progress_response = requests.get(f"{BASE_URL}/analysis/progress/{test_document_id}")
            if progress_response.status_code == 200:
                progress_data = progress_response.json()
                status = progress_data.get('status', 'UNKNOWN')
                current_standard = progress_data.get('overall_progress', {}).get('current_standard', 'Unknown')
                
                print(f"   📊 Status: {status}, Current: {current_standard}")
                
                # Check if analysis is processing wrong standards
                if 'standards_detail' in progress_data:
                    processing_standards = [std.get('standard_id', 'Unknown') 
                                          for std in progress_data['standards_detail']]
                    print(f"   🔍 Standards being processed: {processing_standards}")
                    
                    # CRITICAL BUG CHECK: Are extra standards being processed?
                    if len(processing_standards) > 1 or (processing_standards and processing_standards[0] != "IAS 7"):
                        print(f"   🚨 BUG DETECTED! Processing extra standards!")
                        print(f"      User selected: ['IAS 7']")
                        print(f"      System processing: {processing_standards}")
                        return False
                
                if status == 'COMPLETED':
                    print(f"   ✅ Analysis completed successfully")
                    break
                elif status == 'FAILED':
                    print(f"   ❌ Analysis failed")
                    return False
                    
            time.sleep(5)  # Wait 5 seconds between checks
            
        except Exception as e:
            print(f"   ⚠️ Error checking progress: {e}")
            time.sleep(5)
    
    # Step 4: Check final results
    print(f"\n📊 Step 4: Checking final analysis results...")
    
    try:
        final_response = requests.get(f"{BASE_URL}/analysis/documents/{test_document_id}")
        if final_response.status_code == 200:
            final_data = final_response.json()
            final_status = final_data.get('status', 'Unknown')
            final_standards = final_data.get('standards', [])
            sections = final_data.get('sections', [])
            
            print(f"   📋 Final status: {final_status}")
            print(f"   📊 Final standards: {final_standards}")
            print(f"   📄 Sections analyzed: {len(sections)}")
            
            # CRITICAL VERIFICATION: Check if only IAS 7 sections were analyzed
            if sections:
                analyzed_standards = set()
                for section in sections:
                    standard = section.get('standard', section.get('section', 'Unknown'))
                    analyzed_standards.add(standard)
                
                print(f"   🔍 Standards in analysis results: {list(analyzed_standards)}")
                
                # Check for the bug
                if len(analyzed_standards) > 1 or 'IAS 7' not in analyzed_standards:
                    print(f"   🚨 CRITICAL BUG CONFIRMED!")
                    print(f"      User selected: ['IAS 7']")
                    print(f"      System analyzed: {list(analyzed_standards)}")
                    print(f"      ✨ This is the exact issue the user reported!")
                    return False
                else:
                    print(f"   ✅ SUCCESS: Only IAS 7 was analyzed as requested")
                    return True
            else:
                print(f"   ⚠️ No sections found in results")
                return False
        else:
            print(f"   ❌ Failed to get final results: {final_response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking final results: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 STARTING LIVE API TEST FOR STANDARDS SELECTION BUG")
    print("This test will reproduce the exact user workflow:")
    print("1. User selects ONLY IAS 7")
    print("2. System should analyze ONLY IAS 7")  
    print("3. If system processes other standards = BUG CONFIRMED")
    print()
    
    success = test_standards_selection_bug()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST PASSED: No bug detected - system respects user selection")
    else:
        print("🚨 TEST FAILED: BUG CONFIRMED - system ignores user selection")
        print("   The user's complaint is valid!")
        print("   🔧 Fix needed: Ensure compliance analysis processes ONLY user-selected standards")
    print("=" * 60)

if __name__ == "__main__":
    main()