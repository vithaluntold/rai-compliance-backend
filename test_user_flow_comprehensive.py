#!/usr/bin/env python3
"""
Comprehensive User Flow Testing for RAI Compliance Platform
Tests the complete user journey from upload to results page navigation

This test simulates exactly what a user does:
1. Upload a PDF document 
2. Wait for processing to complete
3. Navigate through the analysis workflow
4. Click buttons to reach the results page
5. Verify all data is properly displayed

Tests both backend APIs and simulates frontend behavior
"""

import json
import requests
import time
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Setup logging for comprehensive test tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('user_flow_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UserFlowTester:
    """Comprehensive user flow tester that simulates real user behavior"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.document_id = None
        self.test_results = {
            "upload_test": {"status": "pending", "details": {}},
            "processing_test": {"status": "pending", "details": {}},
            "navigation_test": {"status": "pending", "details": {}},
            "results_page_test": {"status": "pending", "details": {}},
            "end_to_end_test": {"status": "pending", "details": {}}
        }
        
    def log_test_step(self, step: str, status: str, details: Dict[str, Any] = None):
        """Log test step with emoji indicators like the backend logging"""
        if status == "start":
            logger.info(f"🚀 TEST STEP: {step}")
        elif status == "success":
            logger.info(f"✅ TEST COMPLETE: {step}")
            if details:
                for key, value in details.items():
                    logger.info(f"   📊 {key}: {value}")
        elif status == "failed":
            logger.error(f"❌ TEST FAILED: {step}")
            if details:
                for key, value in details.items():
                    logger.error(f"   ⚠️ {key}: {value}")
        elif status == "progress":
            logger.info(f"⏳ TEST PROGRESS: {step}")
            if details:
                for key, value in details.items():
                    logger.info(f"   📈 {key}: {value}")

    def check_backend_health(self) -> bool:
        """Check if backend is running and accessible"""
        self.log_test_step("Backend Health Check", "start")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                self.log_test_step("Backend Health Check", "success", {
                    "Status Code": response.status_code,
                    "Response Time": f"{response.elapsed.total_seconds():.2f}s"
                })
                return True
            else:
                self.log_test_step("Backend Health Check", "failed", {
                    "Status Code": response.status_code,
                    "Response": response.text[:200]
                })
                return False
        except Exception as e:
            self.log_test_step("Backend Health Check", "failed", {
                "Error": str(e)
            })
            return False

    def test_upload_flow(self, test_file_path: str) -> bool:
        """
        Test 1: File Upload Flow
        Simulates user selecting a file and uploading it
        """
        self.log_test_step("File Upload Flow Test", "start")
        
        if not os.path.exists(test_file_path):
            self.log_test_step("File Upload Flow Test", "failed", {
                "Error": f"Test file not found: {test_file_path}"
            })
            self.test_results["upload_test"]["status"] = "failed"
            return False
        
        try:
            # Step 1: Simulate file selection (frontend behavior)
            file_size = os.path.getsize(test_file_path)
            self.log_test_step("Upload - File Selection", "progress", {
                "File Path": test_file_path,
                "File Size": f"{file_size / 1024 / 1024:.2f} MB"
            })
            
            # Step 2: Upload file to backend (what happens when user clicks "Upload")
            with open(test_file_path, 'rb') as file:
                files = {'file': file}
                upload_start_time = time.time()
                
                response = self.session.post(
                    f"{self.base_url}/api/v1/analysis/upload",
                    files=files,
                    timeout=60
                )
                
                upload_duration = time.time() - upload_start_time
            
            if response.status_code == 200:
                upload_data = response.json()
                self.document_id = upload_data.get('document_id')
                
                self.log_test_step("File Upload Flow Test", "success", {
                    "Document ID": self.document_id,
                    "Upload Time": f"{upload_duration:.2f}s",
                    "Response Status": upload_data.get('status', 'unknown')
                })
                
                self.test_results["upload_test"] = {
                    "status": "passed",
                    "details": {
                        "document_id": self.document_id,
                        "upload_duration": upload_duration,
                        "file_size_mb": file_size / 1024 / 1024
                    }
                }
                return True
            else:
                self.log_test_step("File Upload Flow Test", "failed", {
                    "Status Code": response.status_code,
                    "Error Response": response.text[:500]
                })
                self.test_results["upload_test"]["status"] = "failed"
                return False
                
        except Exception as e:
            self.log_test_step("File Upload Flow Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["upload_test"]["status"] = "failed"
            return False

    def test_processing_flow(self, timeout_minutes: int = 5) -> bool:
        """
        Test 2: Document Processing Flow
        Simulates waiting for backend processing to complete
        """
        if not self.document_id:
            self.log_test_step("Processing Flow Test", "failed", {
                "Error": "No document ID available from upload test"
            })
            return False
            
        self.log_test_step("Document Processing Flow Test", "start")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        poll_interval = 5  # Check every 5 seconds
        
        try:
            while time.time() - start_time < timeout_seconds:
                # Check processing status (what frontend does in polling)
                status_response = self.session.get(
                    f"{self.base_url}/api/v1/analysis/status/{self.document_id}",
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    processing_status = status_data.get('status', 'unknown')
                    
                    elapsed = time.time() - start_time
                    self.log_test_step("Processing Status Check", "progress", {
                        "Status": processing_status,
                        "Elapsed Time": f"{elapsed:.1f}s",
                        "Metadata Status": status_data.get('metadata_extraction', 'unknown')
                    })
                    
                    # Check for completion (what frontend looks for)
                    if processing_status in ['COMPLETED', 'FAILED']:
                        if processing_status == 'COMPLETED':
                            self.log_test_step("Document Processing Flow Test", "success", {
                                "Final Status": processing_status,
                                "Total Processing Time": f"{elapsed:.1f}s",
                                "Sections Generated": len(status_data.get('sections', []))
                            })
                            
                            self.test_results["processing_test"] = {
                                "status": "passed",
                                "details": {
                                    "processing_time_seconds": elapsed,
                                    "final_status": processing_status,
                                    "sections_count": len(status_data.get('sections', []))
                                }
                            }
                            return True
                        else:
                            self.log_test_step("Document Processing Flow Test", "failed", {
                                "Final Status": processing_status,
                                "Error Details": status_data.get('error', 'Processing failed')
                            })
                            self.test_results["processing_test"]["status"] = "failed"
                            return False
                else:
                    self.log_test_step("Status Check Failed", "failed", {
                        "Status Code": status_response.status_code
                    })
                
                time.sleep(poll_interval)
            
            # Timeout reached
            self.log_test_step("Document Processing Flow Test", "failed", {
                "Error": f"Processing timeout after {timeout_minutes} minutes"
            })
            self.test_results["processing_test"]["status"] = "failed"
            return False
            
        except Exception as e:
            self.log_test_step("Document Processing Flow Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["processing_test"]["status"] = "failed"
            return False

    def test_navigation_flow(self) -> bool:
        """
        Test 3: Navigation Flow
        Simulates user clicking through the interface to reach results
        """
        if not self.document_id:
            self.log_test_step("Navigation Flow Test", "failed", {
                "Error": "No document ID available"
            })
            return False
            
        self.log_test_step("Navigation Flow Test", "start")
        
        try:
            # Step 1: Simulate framework selection (what user does in chat interface)
            framework_data = {
                "framework": "IAS_40",
                "standards": ["Investment Property Disclosure", "Fair Value Measurement"]
            }
            
            self.log_test_step("Framework Selection", "progress", {
                "Framework": framework_data["framework"],
                "Standards Count": len(framework_data["standards"])
            })
            
            # Step 2: Simulate starting analysis (clicking "Start Analysis" button)
            analysis_response = self.session.post(
                f"{self.base_url}/api/v1/analysis/analyze",
                json={
                    "document_id": self.document_id,
                    "framework": framework_data["framework"],
                    "standards": framework_data["standards"],
                    "processing_mode": "smart"
                },
                timeout=30
            )
            
            if analysis_response.status_code == 200:
                analysis_data = analysis_response.json()
                
                self.log_test_step("Analysis Trigger", "success", {
                    "Analysis Status": analysis_data.get('status', 'unknown'),
                    "Task ID": analysis_data.get('task_id', 'N/A')
                })
                
                # Step 3: Simulate checking if results are ready (what "View Results" button checks)
                results_response = self.session.get(
                    f"{self.base_url}/api/v1/analysis/results/{self.document_id}",
                    timeout=10
                )
                
                if results_response.status_code == 200:
                    results_data = results_response.json()
                    
                    self.log_test_step("Navigation Flow Test", "success", {
                        "Results Available": True,
                        "Sections Count": len(results_data.get('sections', [])),
                        "Document ID": self.document_id
                    })
                    
                    self.test_results["navigation_test"] = {
                        "status": "passed",
                        "details": {
                            "framework_selected": framework_data["framework"],
                            "analysis_triggered": True,
                            "results_accessible": True
                        }
                    }
                    return True
                else:
                    self.log_test_step("Navigation Flow Test", "failed", {
                        "Results Response Code": results_response.status_code,
                        "Error": "Results not accessible"
                    })
                    self.test_results["navigation_test"]["status"] = "failed"
                    return False
            else:
                self.log_test_step("Navigation Flow Test", "failed", {
                    "Analysis Response Code": analysis_response.status_code,
                    "Error": "Could not trigger analysis"
                })
                self.test_results["navigation_test"]["status"] = "failed"
                return False
                
        except Exception as e:
            self.log_test_step("Navigation Flow Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["navigation_test"]["status"] = "failed"
            return False

    def test_results_page_flow(self) -> bool:
        """
        Test 4: Results Page Flow
        Simulates what happens when user navigates to /results/[documentId]
        """
        if not self.document_id:
            self.log_test_step("Results Page Flow Test", "failed", {
                "Error": "No document ID available"
            })
            return False
            
        self.log_test_step("Results Page Flow Test", "start")
        
        try:
            # Step 1: Fetch results data (what results page does on load)
            results_response = self.session.get(
                f"{self.base_url}/api/v1/analysis/results/{self.document_id}",
                timeout=10
            )
            
            if results_response.status_code == 200:
                results_data = results_response.json()
                
                # Step 2: Verify data structure (what frontend expects)
                required_fields = ['status', 'document_id', 'sections']
                missing_fields = [field for field in required_fields if field not in results_data]
                
                if missing_fields:
                    self.log_test_step("Results Page Flow Test", "failed", {
                        "Missing Fields": missing_fields,
                        "Available Fields": list(results_data.keys())
                    })
                    self.test_results["results_page_test"]["status"] = "failed"
                    return False
                
                # Step 3: Validate sections data (what results page renders)
                sections = results_data.get('sections', [])
                if not sections:
                    self.log_test_step("Results Page Flow Test", "failed", {
                        "Error": "No sections available in results data"
                    })
                    self.test_results["results_page_test"]["status"] = "failed"
                    return False
                
                # Step 4: Check section items (what gets displayed to user)
                total_items = 0
                sections_with_items = 0
                
                for section in sections:
                    items = section.get('items', [])
                    if items:
                        sections_with_items += 1
                        total_items += len(items)
                
                # Step 5: Test export functionality (user clicking export buttons)
                export_endpoints = [
                    f"/api/v1/analysis/export/{self.document_id}/pdf",
                    f"/api/v1/analysis/export/{self.document_id}/excel"
                ]
                
                export_tests = {}
                for endpoint in export_endpoints:
                    try:
                        export_response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                        export_tests[endpoint] = {
                            "status_code": export_response.status_code,
                            "content_type": export_response.headers.get('content-type', 'unknown')
                        }
                    except Exception as e:
                        export_tests[endpoint] = {"error": str(e)}
                
                self.log_test_step("Results Page Flow Test", "success", {
                    "Total Sections": len(sections),
                    "Sections with Items": sections_with_items,
                    "Total Items": total_items,
                    "Export Tests": export_tests
                })
                
                self.test_results["results_page_test"] = {
                    "status": "passed",
                    "details": {
                        "sections_count": len(sections),
                        "total_items": total_items,
                        "export_functionality": export_tests
                    }
                }
                return True
            else:
                self.log_test_step("Results Page Flow Test", "failed", {
                    "Status Code": results_response.status_code,
                    "Error": "Could not fetch results data"
                })
                self.test_results["results_page_test"]["status"] = "failed"
                return False
                
        except Exception as e:
            self.log_test_step("Results Page Flow Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["results_page_test"]["status"] = "failed"
            return False

    def run_end_to_end_test(self, test_file_path: str) -> bool:
        """
        Test 5: Complete End-to-End User Journey
        Runs all tests in sequence to simulate real user behavior
        """
        self.log_test_step("End-to-End User Journey Test", "start")
        
        # Reset test state
        self.document_id = None
        
        tests = [
            ("Backend Health Check", self.check_backend_health),
            ("Upload Flow", lambda: self.test_upload_flow(test_file_path)),
            ("Processing Flow", self.test_processing_flow),
            ("Navigation Flow", self.test_navigation_flow),
            ("Results Page Flow", self.test_results_page_flow)
        ]
        
        passed_tests = 0
        failed_tests = []
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                    self.log_test_step(f"E2E - {test_name}", "success")
                else:
                    failed_tests.append(test_name)
                    self.log_test_step(f"E2E - {test_name}", "failed")
                    # Continue with remaining tests even if one fails
            except Exception as e:
                failed_tests.append(f"{test_name} (Exception: {str(e)})")
                self.log_test_step(f"E2E - {test_name}", "failed", {"Exception": str(e)})
        
        success_rate = (passed_tests / len(tests)) * 100
        
        if len(failed_tests) == 0:
            self.log_test_step("End-to-End User Journey Test", "success", {
                "Passed Tests": f"{passed_tests}/{len(tests)}",
                "Success Rate": f"{success_rate:.1f}%",
                "Document ID": self.document_id
            })
            
            self.test_results["end_to_end_test"] = {
                "status": "passed",
                "details": {
                    "passed_tests": passed_tests,
                    "total_tests": len(tests),
                    "success_rate": success_rate,
                    "final_document_id": self.document_id
                }
            }
            return True
        else:
            self.log_test_step("End-to-End User Journey Test", "failed", {
                "Passed Tests": f"{passed_tests}/{len(tests)}",
                "Failed Tests": failed_tests,
                "Success Rate": f"{success_rate:.1f}%"
            })
            
            self.test_results["end_to_end_test"] = {
                "status": "failed",
                "details": {
                    "passed_tests": passed_tests,
                    "total_tests": len(tests),
                    "failed_tests": failed_tests,
                    "success_rate": success_rate
                }
            }
            return False

    def generate_test_report(self) -> str:
        """Generate a comprehensive test report"""
        report = "=" * 80 + "\n"
        report += "RAI COMPLIANCE PLATFORM - USER FLOW TEST REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Test Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Backend URL: {self.base_url}\n"
        report += f"Final Document ID: {self.document_id}\n\n"
        
        for test_name, test_data in self.test_results.items():
            status_emoji = "✅" if test_data["status"] == "passed" else "❌" if test_data["status"] == "failed" else "⏳"
            report += f"{status_emoji} {test_name.replace('_', ' ').title()}: {test_data['status'].upper()}\n"
            
            if test_data.get("details"):
                for key, value in test_data["details"].items():
                    report += f"   📊 {key}: {value}\n"
            report += "\n"
        
        report += "=" * 80 + "\n"
        report += "USER FLOW ANALYSIS SUMMARY\n"
        report += "=" * 80 + "\n"
        
        passed_count = sum(1 for test in self.test_results.values() if test["status"] == "passed")
        total_count = len(self.test_results)
        
        report += f"Tests Passed: {passed_count}/{total_count}\n"
        report += f"Overall Success Rate: {(passed_count/total_count)*100:.1f}%\n\n"
        
        if passed_count == total_count:
            report += "🎉 ALL TESTS PASSED! User flow is working correctly.\n"
        else:
            report += "⚠️ SOME TESTS FAILED! Review failed tests for user flow issues.\n"
        
        return report


def main():
    """Main function to run comprehensive user flow tests"""
    print("=" * 80)
    print("RAI COMPLIANCE PLATFORM - COMPREHENSIVE USER FLOW TESTING")
    print("=" * 80)
    
    # Configuration
    BACKEND_URL = "http://localhost:8000"  # Change if backend runs on different port
    TEST_FILE_PATH = "c:/Users/saivi/OneDrive/Documents/Audricc all/uploads/RAI-1757795217-3ADC237A.pdf"
    
    # Initialize tester
    tester = UserFlowTester(BACKEND_URL)
    
    # Check if test file exists
    if not os.path.exists(TEST_FILE_PATH):
        logger.error(f"❌ Test file not found: {TEST_FILE_PATH}")
        logger.info("📋 Please update TEST_FILE_PATH in the script to point to a valid PDF file")
        return
    
    print(f"\n📋 Test Configuration:")
    print(f"   Backend URL: {BACKEND_URL}")
    print(f"   Test File: {TEST_FILE_PATH}")
    print(f"   Test File Size: {os.path.getsize(TEST_FILE_PATH) / 1024 / 1024:.2f} MB")
    print("\n" + "=" * 80)
    
    # Run comprehensive test suite
    success = tester.run_end_to_end_test(TEST_FILE_PATH)
    
    # Generate and display report
    report = tester.generate_test_report()
    print("\n" + report)
    
    # Save report to file
    report_file = "user_flow_test_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"📊 Full test report saved to: {report_file}")
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()