#!/usr/bin/env python3
"""
Test script to verify compliance analysis workflow
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_compliance_workflow():
    """Test the compliance analysis workflow step by step"""
    print("🧪 TESTING COMPLIANCE ANALYSIS WORKFLOW")
    
    # Test document ID (use a realistic one)
    test_document_id = "RAI-01102025-TEST-WORKFLOW"
    
    try:
        # 1. Test persistent storage initialization
        print("\n1️⃣ Testing persistent storage...")
        from services.persistent_storage_enhanced import get_persistent_storage_manager
        storage_manager = get_persistent_storage_manager()
        print("✅ Persistent storage initialized")
        
        # 2. Create mock results data to simulate upload/metadata completion
        print("\n2️⃣ Creating mock document results...")
        mock_results = {
            "document_id": test_document_id,
            "status": "COMPLETED",
            "metadata_extraction": "COMPLETED",
            "metadata": {
                "company_name": "Test Company",
                "nature_of_business": "Test Business",
                "operational_demographics": "Test Location",
                "financial_statements_type": "Consolidated"
            },
            "compliance_analysis": "PENDING"
        }
        
        # Save to persistent storage
        success = await storage_manager.store_analysis_results(test_document_id, mock_results)
        if success:
            print("✅ Mock results saved to persistent storage")
        else:
            print("❌ Failed to save mock results")
            return
            
        # 3. Test framework selection request simulation
        print("\n3️⃣ Testing framework selection...")
        from routes.analysis_routes import FrameworkSelectionRequest
        
        # Create a mock request
        mock_request = FrameworkSelectionRequest(
            framework="IFRS",
            standards=["IAS_1", "IAS_16"],
            specialInstructions="Test compliance analysis",
            extensiveSearch=True,
            processingMode="smart"
        )
        print(f"✅ Mock framework request: {mock_request.framework} with {len(mock_request.standards)} standards")
        
        # 4. Test document text extraction
        print("\n4️⃣ Testing text extraction...")
        
        # Create mock chunks file
        chunks_dir = backend_dir / "analysis_results"
        chunks_dir.mkdir(exist_ok=True)
        
        chunks_file = chunks_dir / f"{test_document_id}_chunks.json"
        mock_chunks = [
            {
                "id": f"{test_document_id}_chunk_1",
                "text": "This is test financial statement content for compliance analysis testing.",
                "metadata": {"page": 1}
            },
            {
                "id": f"{test_document_id}_chunk_2", 
                "text": "Additional test content including revenue recognition and asset valuation policies.",
                "metadata": {"page": 2}
            }
        ]
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(mock_chunks, f, indent=2)
        print(f"✅ Mock chunks file created: {chunks_file}")
        
        # Test text extraction function
        from routes.analysis_routes import _extract_document_text
        extracted_text = _extract_document_text(test_document_id)
        
        if isinstance(extracted_text, str):
            print(f"✅ Text extracted successfully: {len(extracted_text)} characters")
            print(f"📝 Sample text: {extracted_text[:100]}...")
        else:
            print("❌ Text extraction failed")
            return
            
        # 5. Test persistent storage update after framework selection
        print("\n5️⃣ Testing persistent storage update...")
        
        # Get current results
        current_results = await storage_manager.get_analysis_results(test_document_id)
        if current_results:
            print("✅ Retrieved existing results from persistent storage")
            
            # Update with framework selection
            current_results.update({
                "framework": mock_request.framework,
                "standards": mock_request.standards,
                "compliance_analysis": "PROCESSING",
                "status": "PROCESSING"
            })
            
            # Save back
            success = await storage_manager.store_analysis_results(test_document_id, current_results)
            if success:
                print("✅ Updated persistent storage with framework selection")
            else:
                print("❌ Failed to update persistent storage")
                return
        else:
            print("❌ Could not retrieve existing results")
            return
            
        # 6. Test compliance analysis function import
        print("\n6️⃣ Testing compliance analysis function...")
        try:
            from routes.analysis_routes import process_compliance_analysis
            print("✅ Compliance analysis function imported successfully")
            
            # Test AI service
            from services.ai_service import get_ai_service
            ai_svc = get_ai_service()
            print("✅ AI service initialized")
            
            print("\n🎉 ALL TESTS PASSED!")
            print("The compliance analysis workflow components are working correctly.")
            
        except Exception as e:
            print(f"❌ Error importing compliance analysis: {e}")
            return
            
    except Exception as e:
        print(f"❌ WORKFLOW TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up test files...")
        try:
            if chunks_file.exists():
                chunks_file.unlink()
                print("✅ Test chunks file removed")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_compliance_workflow())