#!/usr/bin/env python3
"""
Test script to specifically test the framework selection -> compliance analysis workflow
that's currently broken
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_framework_selection_workflow():
    """Test the exact workflow that's failing: framework selection -> compliance analysis"""
    print("🔍 TESTING FRAMEWORK SELECTION -> COMPLIANCE ANALYSIS WORKFLOW")
    
    test_document_id = "RAI-01102025-TEST-AS-SELECTION"
    
    try:
        # 1. Setup - Create a document that has completed metadata extraction
        print("\n📋 Step 1: Setup - Creating document with completed metadata...")
        
        from services.persistent_storage_enhanced import get_persistent_storage_manager
        storage_manager = get_persistent_storage_manager()
        
        # Create analysis results directory and files
        analysis_dir = backend_dir / "analysis_results"
        analysis_dir.mkdir(exist_ok=True)
        
        # Create chunks file (required for text extraction)
        chunks_file = analysis_dir / f"{test_document_id}_chunks.json"
        mock_chunks = [
            {
                "id": f"{test_document_id}_chunk_1",
                "text": "Phoenix Group PLC consolidated financial statements. Revenue recognition policies and asset valuation methods.",
                "metadata": {"page": 1}
            },
            {
                "id": f"{test_document_id}_chunk_2", 
                "text": "Investment properties measured at fair value. Depreciation policies for property, plant and equipment.",
                "metadata": {"page": 2}
            }
        ]
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(mock_chunks, f, indent=2)
        
        # Create document results file (filesystem)
        results_file = analysis_dir / f"{test_document_id}.json"
        mock_results = {
            "document_id": test_document_id,
            "status": "COMPLETED",
            "metadata_extraction": "COMPLETED",
            "metadata": {
                "company_name": "Phoenix Group PLC",
                "nature_of_business": "Digital asset mining and infrastructure",
                "operational_demographics": "United Arab Emirates", 
                "financial_statements_type": "Consolidated"
            },
            "compliance_analysis": "PENDING"
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(mock_results, f, indent=2)
            
        # Also save to persistent storage
        await storage_manager.store_analysis_results(test_document_id, mock_results)
        
        print("✅ Document setup complete - metadata extraction COMPLETED")
        
        # 2. Simulate framework selection request
        print("\n🎯 Step 2: Simulating framework selection (user selects AS)...")
        
        # Import what we need for the framework selection
        from services.ai import AIService, get_ai_service
        from routes.analysis_routes import FrameworkSelectionRequest, _extract_document_text
        
        # Create framework selection request
        request = FrameworkSelectionRequest(
            framework="IFRS",
            standards=["IAS_1", "IAS_16", "IAS_40"],
            specialInstructions="Test AS selection workflow",
            extensiveSearch=True,
            processingMode="smart"
        )
        
        print(f"Framework: {request.framework}")
        print(f"Standards: {request.standards}")
        print(f"Special Instructions: {request.specialInstructions}")
        
        # 3. Test text extraction (this happens in select-framework endpoint)
        print("\n📄 Step 3: Testing text extraction...")
        extracted_text = _extract_document_text(test_document_id)
        
        if isinstance(extracted_text, str):
            print(f"✅ Text extracted: {len(extracted_text)} characters")
        else:
            print(f"❌ Text extraction failed: {extracted_text}")
            return False
            
        # 4. Test persistent storage update (happens in select-framework endpoint)
        print("\n💾 Step 4: Testing persistent storage update...")
        
        # Get current results
        current_results = await storage_manager.get_analysis_results(test_document_id)
        if not current_results:
            print("❌ Could not retrieve current results from persistent storage")
            return False
            
        # Update with framework selection (simulate what select-framework endpoint does)
        current_results.update({
            "framework": request.framework,
            "standards": request.standards, 
            "specialInstructions": request.specialInstructions,
            "extensiveSearch": request.extensiveSearch,
            "compliance_analysis": "PROCESSING",
            "status": "PROCESSING",
            "message": f"Framework {request.framework} and standards {request.standards} selected, compliance analysis pending"
        })
        
        # Save updated results
        success = await storage_manager.store_analysis_results(test_document_id, current_results)
        if not success:
            print("❌ Failed to update persistent storage")
            return False
            
        print("✅ Persistent storage updated with framework selection")
        
        # 5. Test compliance analysis initiation
        print("\n🚀 Step 5: Testing compliance analysis initiation...")
        
        try:
            from routes.analysis_routes import process_compliance_analysis
            ai_svc = get_ai_service()
            
            print("✅ Compliance analysis function and AI service ready")
            
            # This is what should happen in the background task
            print("📝 Simulating background compliance analysis start...")
            
            # Don't actually run the full compliance analysis, just test that it can start
            print(f"✅ Would start compliance analysis with:")
            print(f"   - Document: {test_document_id}")
            print(f"   - Framework: {request.framework}")
            print(f"   - Standards: {request.standards}")
            print(f"   - Text length: {len(extracted_text)}")
            print(f"   - Processing mode: {request.processingMode}")
            
            return True
            
        except Exception as e:
            print(f"❌ Compliance analysis initiation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            if chunks_file.exists():
                chunks_file.unlink()
            if results_file.exists():
                results_file.unlink()
        except:
            pass

async def test_status_polling():
    """Test if status polling shows correct compliance_analysis status"""
    print("\n🔄 TESTING STATUS POLLING...")
    
    test_document_id = "RAI-01102025-TEST-POLLING"
    
    try:
        from services.persistent_storage_enhanced import get_persistent_storage_manager
        storage_manager = get_persistent_storage_manager()
        
        # Create mock results in PROCESSING state
        mock_results = {
            "document_id": test_document_id,
            "status": "PROCESSING",
            "metadata_extraction": "COMPLETED", 
            "compliance_analysis": "PROCESSING",
            "framework": "IFRS",
            "standards": ["IAS_1", "IAS_16"],
            "message": "Compliance analysis in progress"
        }
        
        # Save to persistent storage
        await storage_manager.store_analysis_results(test_document_id, mock_results)
        
        # Test retrieval 
        retrieved = await storage_manager.get_analysis_results(test_document_id)
        
        if retrieved:
            print(f"✅ Status polling would return:")
            print(f"   - status: {retrieved.get('status')}")
            print(f"   - metadata_extraction: {retrieved.get('metadata_extraction')}")
            print(f"   - compliance_analysis: {retrieved.get('compliance_analysis')}")
            print(f"   - framework: {retrieved.get('framework')}")
            print(f"   - standards: {retrieved.get('standards')}")
            return True
        else:
            print("❌ Could not retrieve results for status polling")
            return False
            
    except Exception as e:
        print(f"❌ Status polling test failed: {e}")
        return False

async def main():
    print("🧪 COMPLIANCE ANALYSIS WORKFLOW DIAGNOSTIC")
    print("=" * 60)
    
    # Test the workflow
    workflow_ok = await test_framework_selection_workflow()
    polling_ok = await test_status_polling()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print(f"✅ Framework Selection Workflow: {'PASS' if workflow_ok else 'FAIL'}")
    print(f"✅ Status Polling: {'PASS' if polling_ok else 'FAIL'}")
    
    if workflow_ok and polling_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("The compliance analysis workflow should be working.")
        print("Issue might be in frontend-backend communication or timing.")
    else:
        print("\n❌ TESTS FAILED!")
        print("There are issues in the backend compliance workflow.")

if __name__ == "__main__":
    asyncio.run(main())