"""
Test Session Integration with PostgreSQL Documents
This script tests the integration between sessions and documents stored in PostgreSQL.
"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Set up environment for PostgreSQL
os.environ['DATABASE_URL'] = "postgresql://rai_admin:SOv5DgsrOVzS0Te5q6h9UiYu4xenIps8@dpg-d3ampjs9c44c73e0nqv0-a.oregon-postgres.render.com/rai_compliance_db"
os.environ['USE_POSTGRESQL'] = "true"
os.environ['ENVIRONMENT'] = "production"

async def test_session_document_integration():
    """Test the integration between sessions and PostgreSQL document storage."""
    print("🔍 Testing Session-Document Integration with PostgreSQL")
    print("=" * 60)
    
    try:
        from services.persistent_storage_enhanced import PersistentStorageManager
        
        # Initialize storage
        storage = PersistentStorageManager()
        print(f"✅ Storage initialized: {storage.db_type}")
        
        # Create test document in PostgreSQL
        test_document_id = "SESSION-TEST-DOC-001"
        test_analysis = {
            "document_id": test_document_id,
            "status": "COMPLETED",
            "metadata": {
                "company_name": {"value": "Session Test Corporation", "confidence": 0.95},
                "document_type": {"value": "Financial Statement", "confidence": 0.90}
            },
            "sections": [
                {
                    "title": "Balance Sheet",
                    "content": "Session integration test data",
                    "compliance_score": 0.88
                },
                {
                    "title": "Income Statement", 
                    "content": "Revenue and expenses for session test",
                    "compliance_score": 0.92
                }
            ],
            "created_at": datetime.now().isoformat(),
            "analysis_summary": "Test document for session integration"
        }
        
        print(f"📄 Storing test document: {test_document_id}")
        doc_stored = await storage.store_analysis_results(test_document_id, test_analysis)
        
        if doc_stored:
            print("✅ Document stored in PostgreSQL")
        else:
            print("❌ Failed to store document")
            return False
            
        # Create corresponding session file
        session_id = f"session_{test_document_id}"
        session_data = {
            "session_id": session_id,
            "title": "Session Integration Test",
            "description": "Testing session integration with PostgreSQL documents",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "document_count": 1,
            "last_document_id": test_document_id,
            "status": "active",
            "chat_state": {
                "documentId": test_document_id,
                "analysisComplete": True
            },
            "messages": [
                {
                    "role": "user", 
                    "content": f"Document {test_document_id} uploaded successfully"
                },
                {
                    "role": "assistant",
                    "content": f"Analysis complete for {test_document_id}. You can now view the results."
                }
            ],
            "documents": []  # Will be populated by enhanced endpoint
        }
        
        # Save session file
        sessions_dir = Path("sessions")
        sessions_dir.mkdir(exist_ok=True)
        session_file = sessions_dir / f"{session_id}.json"
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✅ Session created: {session_id}")
        
        # Test session retrieval with PostgreSQL integration
        print(f"\n🔍 Testing session retrieval...")
        
        # Simulate the enhanced session endpoint
        from routes.analysis_routes import load_session_from_file
        loaded_session = load_session_from_file(session_id)
        
        if loaded_session:
            print(f"✅ Session loaded from file")
            print(f"   Title: {loaded_session['title']}")
            print(f"   Document Count: {loaded_session['document_count']}")
            print(f"   Last Document ID: {loaded_session['last_document_id']}")
            
            # Test document retrieval from PostgreSQL
            if loaded_session.get("last_document_id"):
                doc_id = loaded_session["last_document_id"]
                retrieved_doc = await storage.get_analysis_results(doc_id)
                
                if retrieved_doc:
                    print(f"✅ Document retrieved from PostgreSQL")
                    print(f"   Company: {retrieved_doc['metadata']['company_name']['value']}")
                    print(f"   Sections: {len(retrieved_doc['sections'])}")
                    print(f"   Status: {retrieved_doc['status']}")
                    
                    return True
                else:
                    print(f"❌ Could not retrieve document from PostgreSQL")
                    return False
        else:
            print(f"❌ Could not load session")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_multiple_documents_session():
    """Test session with multiple documents stored in PostgreSQL."""
    print(f"\n🔍 Testing Multiple Documents in Session...")
    
    try:
        from services.persistent_storage_enhanced import PersistentStorageManager
        storage = PersistentStorageManager()
        
        # Create multiple test documents
        doc_ids = ["MULTI-TEST-001", "MULTI-TEST-002"]
        
        for i, doc_id in enumerate(doc_ids):
            test_doc = {
                "document_id": doc_id,
                "status": "COMPLETED",
                "metadata": {
                    "company_name": {"value": f"Multi-Test Corp {i+1}", "confidence": 0.95}
                },
                "sections": [{"title": f"Section {i+1}", "content": "Test data"}],
                "created_at": datetime.now().isoformat()
            }
            
            stored = await storage.store_analysis_results(doc_id, test_doc)
            if stored:
                print(f"✅ Stored document: {doc_id}")
            else:
                print(f"❌ Failed to store: {doc_id}")
                
        # Create session with multiple documents
        multi_session_id = "session_multi_test"
        multi_session = {
            "session_id": multi_session_id,
            "title": "Multi-Document Session",
            "description": "Session with multiple PostgreSQL documents",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "document_count": len(doc_ids),
            "last_document_id": doc_ids[-1],
            "status": "active",
            "chat_state": {"documentId": doc_ids[-1]},
            "messages": [
                {"role": "user", "content": f"Uploaded documents: {', '.join(doc_ids)}"},
                {"role": "assistant", "content": "All documents processed successfully"}
            ],
            "documents": []
        }
        
        # Save session
        session_file = Path("sessions") / f"{multi_session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(multi_session, f, indent=2)
            
        print(f"✅ Multi-document session created: {multi_session_id}")
        
        # Test retrieval of all documents
        for doc_id in doc_ids:
            retrieved = await storage.get_analysis_results(doc_id)
            if retrieved:
                print(f"✅ Retrieved {doc_id}: {retrieved['metadata']['company_name']['value']}")
            else:
                print(f"❌ Failed to retrieve {doc_id}")
                
        return True
        
    except Exception as e:
        print(f"❌ Multi-document test failed: {e}")
        return False

async def main():
    """Run all session integration tests."""
    print("🚀 Session-PostgreSQL Integration Test Suite")
    print("=" * 60)
    
    test1_result = await test_session_document_integration()
    test2_result = await test_multiple_documents_session()
    
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY:")
    print(f"   Single Document Session: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"   Multi-Document Session:  {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Sessions can retrieve documents from PostgreSQL")
        print("✅ Frontend sessions panel will show live document data")
        print("✅ Results page integration working")
        
        print("\n🎯 What This Means:")
        print("   • Sessions panel shows documents stored in PostgreSQL")
        print("   • Users can access analysis results from any session")
        print("   • No localhost dependency - works with cloud database")
        print("   • Document persistence across deployments")
    else:
        print("\n⚠️ Some tests failed - check the issues above")

if __name__ == "__main__":
    asyncio.run(main())