#!/usr/bin/env python3
"""
HOLISTIC END-TO-END INTEGRATION TEST
Simulates the exact frontend workflow with real timing and polling
"""
import os
import sys
import json
import asyncio
import aiohttp
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test configuration
BASE_URL = "http://localhost:8000/api/v1/analysis"  # Local test
# BASE_URL = "https://rai-compliance-backend.onrender.com/api/v1/analysis"  # Production test
TEST_DOCUMENT = "Phoenix-Group-PLC-2024-Consolidated-Financial-Statements.pdf"

class IntegrationTester:
    def __init__(self):
        self.session = None
        self.document_id = None
        self.upload_time = None
        self.metadata_complete_time = None
        self.polling_log = []
        
    async def start_session(self):
        """Start HTTP session"""
        self.session = aiohttp.ClientSession()
        
    async def stop_session(self):
        """Stop HTTP session"""
        if self.session:
            await self.session.close()
            
    async def upload_document(self):
        """Step 1: Upload document and get document_id"""
        print("🚀 STEP 1: UPLOADING DOCUMENT")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        
        # Check if test file exists
        test_file_path = Path(TEST_DOCUMENT)
        if not test_file_path.exists():
            # Create a dummy PDF for testing
            dummy_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            test_file_path.write_bytes(dummy_content)
            print(f"📄 Created dummy test file: {test_file_path}")
        
        # Upload file
        with open(test_file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=TEST_DOCUMENT, content_type='application/pdf')
            data.add_field('processing_mode', 'smart')
            
            self.upload_time = time.time()
            
            async with self.session.post(f"{BASE_URL}/upload", data=data) as response:
                result = await response.json()
                
                print(f"📊 Upload Response Status: {response.status}")
                print(f"📊 Upload Response: {json.dumps(result, indent=2)}")
                
                if response.status == 200 and 'document_id' in result:
                    self.document_id = result['document_id']
                    print(f"✅ Upload successful! Document ID: {self.document_id}")
                    return True
                else:
                    print(f"❌ Upload failed: {result}")
                    return False
                    
    async def poll_document_status(self, max_duration=300):
        """Step 2: Poll document status until metadata is complete"""
        print(f"\n🔄 STEP 2: POLLING DOCUMENT STATUS")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"Document ID: {self.document_id}")
        
        start_time = time.time()
        poll_count = 0
        
        while (time.time() - start_time) < max_duration:
            poll_count += 1
            poll_time = time.time()
            
            try:
                async with self.session.get(f"{BASE_URL}/documents/{self.document_id}") as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Log this polling result
                        poll_entry = {
                            'poll_number': poll_count,
                            'timestamp': time.strftime('%H:%M:%S'),
                            'elapsed_since_upload': round(poll_time - self.upload_time, 1),
                            'status': result.get('status', 'UNKNOWN'),
                            'metadata_extraction': result.get('metadata_extraction', 'UNKNOWN'),
                            'compliance_analysis': result.get('compliance_analysis', 'UNKNOWN'),
                            'metadata_fields_present': bool(result.get('metadata', {})),
                            'metadata_field_count': len(result.get('metadata', {})),
                            'message': result.get('message', 'No message')
                        }
                        self.polling_log.append(poll_entry)
                        
                        print(f"📊 Poll #{poll_count} at {poll_entry['timestamp']} (+{poll_entry['elapsed_since_upload']}s)")
                        print(f"   Status: {poll_entry['status']}")
                        print(f"   Metadata Extraction: {poll_entry['metadata_extraction']}")
                        print(f"   Metadata Fields: {poll_entry['metadata_field_count']} present")
                        print(f"   Message: {poll_entry['message']}")
                        
                        # Check if metadata is complete
                        if poll_entry['metadata_extraction'] == 'COMPLETED':
                            self.metadata_complete_time = poll_time
                            print(f"✅ METADATA EXTRACTION COMPLETED!")
                            print(f"⏱️  Total time from upload: {round(poll_time - self.upload_time, 1)} seconds")
                            
                            # Show the actual metadata
                            metadata = result.get('metadata', {})
                            if metadata:
                                print(f"📋 EXTRACTED METADATA:")
                                for field, value in metadata.items():
                                    if isinstance(value, dict):
                                        print(f"   {field}: {value.get('value', value)}")
                                    else:
                                        print(f"   {field}: {value}")
                            else:
                                print("❌ No metadata found in response!")
                                
                            return result
                        
                        # Check for errors
                        elif poll_entry['status'] in ['FAILED', 'ERROR']:
                            print(f"❌ Processing failed: {poll_entry['message']}")
                            return result
                            
                    else:
                        print(f"❌ Poll #{poll_count} failed with status {response.status}")
                        error_text = await response.text()
                        print(f"   Error: {error_text}")
                        
            except Exception as e:
                print(f"❌ Poll #{poll_count} exception: {str(e)}")
                
            # Wait before next poll (simulate frontend polling interval)
            await asyncio.sleep(2)  # Frontend typically polls every 2-3 seconds
            
        print(f"⏰ Polling timeout after {max_duration} seconds")
        return None
        
    async def test_framework_selection(self):
        """Step 3: Test framework selection after metadata is complete"""
        if not self.metadata_complete_time:
            print("❌ Cannot test framework selection - metadata not completed")
            return False
            
        print(f"\n🎯 STEP 3: TESTING FRAMEWORK SELECTION")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        
        # Test framework selection
        framework_data = {
            "framework": "IFRS",
            "standards": ["IFRS 1", "IFRS 15"],
            "specialInstructions": "",
            "extensiveSearch": False
        }
        
        try:
            async with self.session.post(
                f"{BASE_URL}/documents/{self.document_id}/select-framework",
                json=framework_data
            ) as response:
                result = await response.json()
                
                print(f"📊 Framework Selection Status: {response.status}")
                print(f"📊 Framework Selection Response: {json.dumps(result, indent=2)}")
                
                if response.status == 200:
                    print("✅ Framework selection successful!")
                    return True
                else:
                    print(f"❌ Framework selection failed: {result}")
                    return False
                    
        except Exception as e:
            print(f"❌ Framework selection exception: {str(e)}")
            return False
            
    def generate_report(self):
        """Generate detailed test report"""
        print(f"\n📊 INTEGRATION TEST REPORT")
        print("=" * 60)
        
        if self.upload_time and self.metadata_complete_time:
            total_time = self.metadata_complete_time - self.upload_time
            print(f"⏱️  Total Processing Time: {round(total_time, 1)} seconds")
        else:
            print(f"⏱️  Total Processing Time: INCOMPLETE")
            
        print(f"🔄 Total Polls: {len(self.polling_log)}")
        
        if self.polling_log:
            print(f"\n📈 POLLING TIMELINE:")
            for entry in self.polling_log:
                status_icon = "✅" if entry['metadata_extraction'] == 'COMPLETED' else "🔄" if entry['metadata_extraction'] == 'IN_PROGRESS' else "⏳"
                print(f"  {status_icon} {entry['timestamp']} (+{entry['elapsed_since_upload']:5.1f}s): {entry['status']} | Metadata: {entry['metadata_extraction']} | Fields: {entry['metadata_field_count']}")
        
        # Identify issues
        print(f"\n🔍 INTEGRATION ANALYSIS:")
        
        metadata_polls = [p for p in self.polling_log if p['metadata_extraction'] == 'COMPLETED']
        if not metadata_polls:
            print("❌ CRITICAL: Metadata never became available to frontend")
            
        empty_metadata_polls = [p for p in self.polling_log if p['metadata_extraction'] == 'COMPLETED' and p['metadata_field_count'] == 0]
        if empty_metadata_polls:
            print(f"⚠️  WARNING: Metadata marked complete but empty fields in {len(empty_metadata_polls)} polls")
            
        rapid_status_changes = []
        for i in range(1, len(self.polling_log)):
            if self.polling_log[i]['status'] != self.polling_log[i-1]['status']:
                rapid_status_changes.append((self.polling_log[i-1], self.polling_log[i]))
        
        if rapid_status_changes:
            print(f"⚡ STATUS CHANGES: {len(rapid_status_changes)} status transitions detected")
            for prev, curr in rapid_status_changes:
                print(f"   {prev['timestamp']} → {curr['timestamp']}: {prev['status']} → {curr['status']}")
        
        print(f"\n💾 Detailed polling log saved to integration_test_log.json")
        
        # Save detailed log
        with open('integration_test_log.json', 'w') as f:
            json.dump({
                'test_summary': {
                    'document_id': self.document_id,
                    'upload_time': self.upload_time,
                    'metadata_complete_time': self.metadata_complete_time,
                    'total_processing_time': (self.metadata_complete_time - self.upload_time) if self.metadata_complete_time and self.upload_time else None,
                    'total_polls': len(self.polling_log)
                },
                'polling_log': self.polling_log
            }, f, indent=2)

async def main():
    """Run the holistic integration test"""
    print("🧪 HOLISTIC END-TO-END INTEGRATION TEST")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print(f"Test document: {TEST_DOCUMENT}")
    
    tester = IntegrationTester()
    
    try:
        await tester.start_session()
        
        # Step 1: Upload
        if not await tester.upload_document():
            print("❌ Test failed at upload step")
            return
            
        # Step 2: Poll until metadata complete
        final_status = await tester.poll_document_status()
        
        # Step 3: Test framework selection (if metadata completed)
        if final_status and tester.metadata_complete_time:
            await tester.test_framework_selection()
            
    except Exception as e:
        print(f"❌ Test exception: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
    finally:
        await tester.stop_session()
        tester.generate_report()

if __name__ == "__main__":
    asyncio.run(main())