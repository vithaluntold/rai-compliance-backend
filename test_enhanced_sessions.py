"""
Test Enhanced Session System with All Features
This script tests the complete enhanced session system including file tracking,
conversation history, user choices, AI responses, and document section queries.
"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import requests

# Set up environment for PostgreSQL
os.environ['DATABASE_URL'] = "postgresql://rai_admin:SOv5DgsrOVzS0Te5q6h9UiYu4xenIps8@dpg-d3ampjs9c44c73e0nqv0-a.oregon-postgres.render.com/rai_compliance_db"
os.environ['USE_POSTGRESQL'] = "true"
os.environ['ENVIRONMENT'] = "production"

BASE_URL = "http://localhost:8001/api/v1"

def test_session_creation():
    """Test creating a session with custom analysis context."""
    print("🔧 Testing Enhanced Session Creation...")
    
    session_data = {
        "title": "Enhanced Session Test",
        "description": "Testing comprehensive session features",
        "analysis_context": {
            "accounting_standard": "IND AS",
            "custom_instructions": "Focus on compliance and risk assessment. Pay special attention to financial statement disclosures.",
            "selected_frameworks": ["IND AS 107", "IND AS 109"],
            "analysis_preferences": {
                "detail_level": "comprehensive",
                "include_recommendations": True
            }
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/sessions/create", json=session_data, timeout=10)
        
        if response.status_code == 200:
            session = response.json()
            print(f"✅ Session created: {session['session_id']}")
            print(f"   Title: {session['title']}")
            print(f"   Status: {session['status']}")
            return session['session_id']
        else:
            print(f"❌ Session creation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return None

def test_user_choice_recording(session_id):
    """Test recording user choices in session."""
    print(f"\n📝 Testing User Choice Recording for {session_id}...")
    
    user_choice = {
        "choice_type": "accounting_standard",
        "value": "IND AS",
        "timestamp": datetime.now().isoformat(),
        "context": "User selected IND AS from dropdown menu"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/user-choice", json=user_choice, timeout=10)
        
        if response.status_code == 200:
            print("✅ User choice recorded successfully")
            return True
        else:
            print(f"❌ User choice recording failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ User choice error: {e}")
        return False

def test_conversation_message(session_id):
    """Test adding conversation messages to session."""
    print(f"\n💬 Testing Conversation History for {session_id}...")
    
    user_message = {
        "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_001",
        "role": "user",
        "content": "Can you analyze the cash flow statement for compliance with IND AS 107?",
        "timestamp": datetime.now().isoformat(),
        "message_type": "analysis_request",
        "metadata": {"section": "cash_flow", "standard": "IND AS 107"}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/conversation", json=user_message, timeout=10)
        
        if response.status_code == 200:
            print("✅ Conversation message added successfully")
            
            # Add AI response
            ai_message = {
                "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_002", 
                "role": "assistant",
                "content": "I'll analyze the cash flow statement against IND AS 107 requirements. Based on the document sections, here are the key compliance points...",
                "timestamp": datetime.now().isoformat(),
                "message_type": "analysis_response",
                "metadata": {"confidence": 0.92, "sources": ["Section 3: Cash Flow Statement"]}
            }
            
            response2 = requests.post(f"{BASE_URL}/sessions/{session_id}/conversation", json=ai_message, timeout=10)
            if response2.status_code == 200:
                print("✅ AI response message added successfully")
                return True
            
        else:
            print(f"❌ Conversation message failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Conversation error: {e}")
        return False

async def test_document_section_query(session_id):
    """Test document section query functionality."""
    print(f"\n🔍 Testing Document Section Query for {session_id}...")
    
    # First create a test document in PostgreSQL
    try:
        from services.persistent_storage_enhanced import PersistentStorageManager
        storage = PersistentStorageManager()
        
        test_doc_id = "ENHANCED-SESSION-DOC-001"
        test_analysis = {
            "document_id": test_doc_id,
            "status": "COMPLETED",
            "metadata": {
                "company_name": {"value": "Enhanced Session Test Corp", "confidence": 0.95},
                "document_type": {"value": "Annual Report", "confidence": 0.90}
            },
            "sections": [
                {
                    "title": "Cash Flow Statement",
                    "content": "Cash flows from operating activities: Rs. 1,50,000. Cash flows from investing activities: Rs. -75,000. Cash flows from financing activities: Rs. 25,000.",
                    "page_start": 15,
                    "page_end": 18,
                    "compliance_score": 0.88
                },
                {
                    "title": "Notes to Financial Statements",
                    "content": "Significant accounting policies, contingent liabilities, and subsequent events are disclosed as per IND AS requirements.",
                    "page_start": 25, 
                    "page_end": 35,
                    "compliance_score": 0.92
                }
            ],
            "created_at": datetime.now().isoformat()
        }
        
        doc_stored = await storage.store_analysis_results(test_doc_id, test_analysis)
        if doc_stored:
            print(f"✅ Test document stored: {test_doc_id}")
        
        # Test section query
        query_data = {
            "document_id": test_doc_id,
            "question": "What are the cash flows from operating activities and how do they comply with IND AS 107?",
            "start_page": 15,
            "end_page": 18,
            "section_name": "Cash Flow Statement", 
            "custom_instructions": "Provide detailed analysis with specific reference to IND AS 107 disclosure requirements."
        }
        
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/document-query", json=query_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Document section query successful")
            print(f"   Query ID: {result['ai_response']['query_id']}")
            print(f"   Confidence: {result['ai_response']['confidence']}")
            print(f"   Sections Referenced: {result['ai_response']['sections_referenced']}")
            return True
        else:
            print(f"❌ Document query failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Document query error: {e}")
        return False

def test_session_sharing(session_id):
    """Test session sharing functionality."""
    print(f"\n🤝 Testing Session Sharing for {session_id}...")
    
    shared_users = ["user2@company.com", "manager@company.com"]
    
    try:
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/share", json=shared_users, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Session shared successfully")
            print(f"   Shared with: {len(result['shared_with'])} users")
            return True
        else:
            print(f"❌ Session sharing failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Session sharing error: {e}")
        return False

def test_enhanced_session_retrieval(session_id):
    """Test retrieving enhanced session with all data."""
    print(f"\n📊 Testing Enhanced Session Retrieval for {session_id}...")
    
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}", timeout=15)
        
        if response.status_code == 200:
            session = response.json()
            print("✅ Enhanced session retrieved successfully")
            print(f"   Title: {session['title']}")
            print(f"   Document Count: {session['document_count']}")
            print(f"   Conversation Messages: {len(session.get('conversation_history', []))}")
            print(f"   User Choices: {len(session.get('user_choices', []))}")
            print(f"   AI Responses: {len(session.get('ai_responses', []))}")
            print(f"   Uploaded Files: {len(session.get('uploaded_files', []))}")
            print(f"   Status: {session['status']}")
            
            # Check analysis context
            if 'analysis_context' in session:
                context = session['analysis_context']
                print(f"   Accounting Standard: {context.get('accounting_standard')}")
                print(f"   Custom Instructions: {context.get('custom_instructions')[:50]}..." if context.get('custom_instructions') else "   Custom Instructions: None")
            
            return True
        else:
            print(f"❌ Session retrieval failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Session retrieval error: {e}")
        return False

async def main():
    """Run complete enhanced session system test."""
    print("🚀 Enhanced Session System Comprehensive Test")
    print("=" * 60)
    
    # Test 1: Create enhanced session
    session_id = test_session_creation()
    if not session_id:
        print("❌ Cannot proceed without session creation")
        return
    
    # Test 2: Record user choice
    choice_success = test_user_choice_recording(session_id)
    
    # Test 3: Add conversation messages
    conversation_success = test_conversation_message(session_id)
    
    # Test 4: Document section query
    query_success = await test_document_section_query(session_id)
    
    # Test 5: Session sharing
    sharing_success = test_session_sharing(session_id)
    
    # Test 6: Enhanced session retrieval
    retrieval_success = test_enhanced_session_retrieval(session_id)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 ENHANCED SESSION SYSTEM TEST RESULTS:")
    print(f"   Session Creation: {'✅ PASS' if session_id else '❌ FAIL'}")
    print(f"   User Choice Recording: {'✅ PASS' if choice_success else '❌ FAIL'}")
    print(f"   Conversation History: {'✅ PASS' if conversation_success else '❌ FAIL'}")
    print(f"   Document Section Query: {'✅ PASS' if query_success else '❌ FAIL'}")
    print(f"   Session Sharing: {'✅ PASS' if sharing_success else '❌ FAIL'}")
    print(f"   Enhanced Retrieval: {'✅ PASS' if retrieval_success else '❌ FAIL'}")
    
    all_passed = all([session_id, choice_success, conversation_success, query_success, sharing_success, retrieval_success])
    
    if all_passed:
        print("\n🎉 ALL ENHANCED SESSION FEATURES WORKING!")
        print("\n✅ Your session system now includes:")
        print("   • Complete conversation history tracking")
        print("   • User choice and AI response recording")
        print("   • Document section-specific queries")
        print("   • Custom analysis instructions")
        print("   • Session sharing capabilities")
        print("   • Enhanced session management")
        print("   • PostgreSQL integration for all data")
        
        print(f"\n🔗 Test Session ID: {session_id}")
        print("   Use this session ID to test frontend integration")
    else:
        print("\n⚠️ Some enhanced features need attention - check the failures above")

if __name__ == "__main__":
    asyncio.run(main())