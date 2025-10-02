#!/usr/bin/env python3
"""
Test the actual status endpoint that frontend calls
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_status_endpoint():
    """Test the actual GET /documents/{document_id} endpoint that frontend calls"""
    print("🔍 TESTING ACTUAL STATUS ENDPOINT")
    
    test_document_id = "RAI-01102025-TEST-STATUS-ENDPOINT"
    
    try:
        from services.persistent_storage_enhanced import get_persistent_storage_manager
        from routes.analysis_routes import get_document_status, _process_analysis_results
        
        storage_manager = get_persistent_storage_manager()
        
        # 1. Test with no results (should return not found or fallback)
        print("\n1️⃣ Testing with no existing results...")
        result = await get_document_status(test_document_id)
        print(f"Result type: {type(result)}")
        if hasattr(result, 'status_code'):
            print(f"Status code: {result.status_code}")
        else:
            print(f"Result: {result}")
        
        # 2. Create mock results and test
        print(f"\n2️⃣ Creating mock results for {test_document_id}...")
        mock_results = {
            "document_id": test_document_id,
            "status": "PROCESSING",
            "metadata_extraction": "COMPLETED",
            "compliance_analysis": "PROCESSING", 
            "framework": "IFRS",
            "standards": ["IAS_1", "IAS_16"],
            "message": "Compliance analysis in progress",
            "specialInstructions": "Test workflow",
            "extensiveSearch": True
        }
        
        # Save to persistent storage
        success = await storage_manager.store_analysis_results(test_document_id, mock_results)
        print(f"Saved to persistent storage: {success}")
        
        # 3. Test status endpoint with results
        print(f"\n3️⃣ Testing status endpoint with results...")
        result = await get_document_status(test_document_id)
        print(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            print("📊 Status endpoint returned:")
            for key, value in result.items():
                if key != "sections":  # Skip sections as it's usually empty/long
                    print(f"   {key}: {value}")
        else:
            print(f"Unexpected result type: {result}")
        
        # 4. Test _process_analysis_results function directly
        print(f"\n4️⃣ Testing _process_analysis_results function...")
        processed = _process_analysis_results(test_document_id, mock_results)
        print("🔧 _process_analysis_results returned:")
        for key, value in processed.items():
            if key != "sections":
                print(f"   {key}: {value}")
                
        return True
        
    except Exception as e:
        print(f"❌ Status endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_framework_selection_endpoint_simulation():
    """Simulate the complete framework selection endpoint"""
    print("\n🎯 TESTING COMPLETE FRAMEWORK SELECTION SIMULATION")
    
    test_document_id = "RAI-01102025-TEST-FRAMEWORK-ENDPOINT"
    
    try:
        from services.persistent_storage_enhanced import get_persistent_storage_manager
        from routes.analysis_routes import FrameworkSelectionRequest
        from services.ai import get_ai_service
        
        storage_manager = get_persistent_storage_manager()
        
        # Setup document with completed metadata
        print("📋 Setting up document with completed metadata...")
        initial_results = {
            "document_id": test_document_id,
            "status": "COMPLETED",
            "metadata_extraction": "COMPLETED",
            "compliance_analysis": "PENDING",
            "metadata": {
                "company_name": "Test Company",
                "nature_of_business": "Test Business"
            }
        }
        
        await storage_manager.store_analysis_results(test_document_id, initial_results)
        
        # Create chunks file for text extraction
        analysis_dir = backend_dir / "analysis_results"
        analysis_dir.mkdir(exist_ok=True)
        chunks_file = analysis_dir / f"{test_document_id}_chunks.json"
        
        mock_chunks = [{"id": "test_chunk", "text": "Test financial content for analysis"}]
        with open(chunks_file, 'w') as f:
            json.dump(mock_chunks, f)
        
        print("✅ Initial setup complete")
        
        # Simulate framework selection request processing
        print("\n🔧 Simulating framework selection request processing...")
        
        request = FrameworkSelectionRequest(
            framework="IFRS",
            standards=["IAS_1", "IAS_16"],
            specialInstructions="Test framework selection",
            extensiveSearch=True,
            processingMode="smart"
        )
        
        # 1. Get current results (what select-framework does)
        current_results = await storage_manager.get_analysis_results(test_document_id)
        print(f"Retrieved current results: {current_results is not None}")
        
        # 2. Update with framework selection (what select-framework does)
        if current_results:
            current_results.update({
                "framework": request.framework,
                "standards": request.standards,
                "specialInstructions": request.specialInstructions,
                "extensiveSearch": request.extensiveSearch,
                "compliance_analysis": "PROCESSING",
                "status": "PROCESSING",
                "message": f"Framework {request.framework} and standards {request.standards} selected, compliance analysis pending"
            })
            
            success = await storage_manager.store_analysis_results(test_document_id, current_results)
            print(f"Updated results saved: {success}")
        
        # 3. Test status retrieval immediately after framework selection
        print("\n📊 Testing status retrieval after framework selection...")
        
        from routes.analysis_routes import get_document_status
        status_result = await get_document_status(test_document_id)
        
        if isinstance(status_result, dict):
            print("✅ Status after framework selection:")
            print(f"   status: {status_result.get('status')}")
            print(f"   metadata_extraction: {status_result.get('metadata_extraction')}")
            print(f"   compliance_analysis: {status_result.get('compliance_analysis')}")
            print(f"   framework: {status_result.get('framework')}")
            print(f"   standards: {status_result.get('standards')}")
            print(f"   message: {status_result.get('message')}")
            
            # This is what the frontend should see after selecting AS
            if status_result.get('compliance_analysis') == 'PROCESSING':
                print("🎉 SUCCESS: compliance_analysis shows PROCESSING - frontend should see this!")
            else:
                print(f"❌ ISSUE: compliance_analysis is '{status_result.get('compliance_analysis')}', expected 'PROCESSING'")
                
        else:
            print(f"❌ Unexpected status result: {status_result}")
            
        return True
        
    except Exception as e:
        print(f"❌ Framework selection simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if chunks_file.exists():
                chunks_file.unlink()
        except:
            pass

async def main():
    print("🧪 STATUS ENDPOINT DIAGNOSTIC")
    print("=" * 60)
    
    endpoint_ok = await test_status_endpoint()
    simulation_ok = await test_framework_selection_endpoint_simulation()
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC RESULTS:")
    print(f"✅ Status Endpoint: {'PASS' if endpoint_ok else 'FAIL'}")  
    print(f"✅ Framework Selection Flow: {'PASS' if simulation_ok else 'FAIL'}")
    
    if endpoint_ok and simulation_ok:
        print("\n🎉 BACKEND WORKFLOW IS WORKING!")
        print("The issue is likely in frontend-backend communication timing.")
    else:
        print("\n❌ BACKEND WORKFLOW HAS ISSUES!")

if __name__ == "__main__":
    asyncio.run(main())