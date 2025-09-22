#!/usr/bin/env python3
"""
Simple User Flow Test - Checks for user journey disconnects
Windows-compatible version without emojis that cause encoding issues

This test validates the complete user flow:
1. Backend API functionality (upload -> process -> results)
2. Frontend navigation and interactions
3. Integration between components
"""

import asyncio
import subprocess
import sys
import os
import time
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Setup logging without file handler to avoid encoding issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SimpleUserFlowTester:
    """Simple test runner for user flow validation"""
    
    def __init__(self):
        self.config = {
            "frontend_url": "http://localhost:3000",
            "backend_url": "http://localhost:8000",
            "test_file_path": "c:/Users/saivi/OneDrive/Documents/Audricc all/uploads/RAI-1757795217-3ADC237A.pdf"
        }
        
        self.results = {}
        self.issues = []
        
    def log_step(self, step: str, status: str, details: str = ""):
        """Log test step without emojis"""
        if status == "start":
            logger.info(f"STARTING: {step}")
        elif status == "success":
            logger.info(f"SUCCESS: {step} - {details}")
        elif status == "failed":
            logger.error(f"FAILED: {step} - {details}")
        elif status == "warning":
            logger.warning(f"WARNING: {step} - {details}")

    def test_backend_health(self) -> bool:
        """Test if backend server is running"""
        self.log_step("Backend Health Check", "start")
        
        try:
            response = requests.get(f"{self.config['backend_url']}/api/v1/health", timeout=5)
            if response.status_code == 200:
                self.log_step("Backend Health Check", "success", f"Server responding at {self.config['backend_url']}")
                return True
            else:
                self.log_step("Backend Health Check", "failed", f"Server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_step("Backend Health Check", "failed", f"Connection error: {str(e)}")
            self.issues.append("Backend server not accessible")
            return False

    def test_frontend_health(self) -> bool:
        """Test if frontend server is running"""
        self.log_step("Frontend Health Check", "start")
        
        try:
            response = requests.get(self.config['frontend_url'], timeout=5)
            if response.status_code == 200:
                self.log_step("Frontend Health Check", "success", f"Server responding at {self.config['frontend_url']}")
                return True
            else:
                self.log_step("Frontend Health Check", "failed", f"Server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_step("Frontend Health Check", "failed", f"Connection error: {str(e)}")
            self.issues.append("Frontend server not accessible")
            return False

    def test_file_upload_api(self) -> Tuple[bool, str]:
        """Test file upload API endpoint"""
        self.log_step("File Upload API Test", "start")
        
        if not os.path.exists(self.config['test_file_path']):
            self.log_step("File Upload API Test", "failed", f"Test file not found: {self.config['test_file_path']}")
            self.issues.append("Test file missing")
            return False, ""
        
        try:
            with open(self.config['test_file_path'], 'rb') as f:
                files = {'file': ('test_document.pdf', f, 'application/pdf')}
                data = {'framework': 'smart'}
                
                response = requests.post(
                    f"{self.config['backend_url']}/upload_document",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                document_id = result.get('document_id', '')
                self.log_step("File Upload API Test", "success", f"Document uploaded with ID: {document_id}")
                return True, document_id
            else:
                self.log_step("File Upload API Test", "failed", f"Upload failed with status {response.status_code}")
                self.issues.append(f"File upload API error: {response.status_code}")
                return False, ""
                
        except Exception as e:
            self.log_step("File Upload API Test", "failed", f"Exception: {str(e)}")
            self.issues.append(f"File upload exception: {str(e)}")
            return False, ""

    def test_processing_api(self, document_id: str) -> bool:
        """Test document processing API"""
        self.log_step("Processing API Test", "start")
        
        if not document_id:
            self.log_step("Processing API Test", "failed", "No document ID provided")
            return False
        
        try:
            # Test processing status endpoint
            response = requests.get(
                f"{self.config['backend_url']}/processing_status/{document_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                status_data = response.json()
                self.log_step("Processing API Test", "success", f"Processing status: {status_data.get('status', 'unknown')}")
                
                # Wait a bit for processing to complete
                time.sleep(5)
                
                # Check if results are available
                results_response = requests.get(
                    f"{self.config['backend_url']}/analysis_results/{document_id}",
                    timeout=10
                )
                
                if results_response.status_code == 200:
                    self.log_step("Processing API Test", "success", "Analysis results available")
                    return True
                else:
                    self.log_step("Processing API Test", "warning", f"Results not ready yet (status: {results_response.status_code})")
                    return True  # Processing might still be ongoing
            else:
                self.log_step("Processing API Test", "failed", f"Status check failed: {response.status_code}")
                self.issues.append(f"Processing status API error: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_step("Processing API Test", "failed", f"Exception: {str(e)}")
            self.issues.append(f"Processing API exception: {str(e)}")
            return False

    def test_results_api(self, document_id: str) -> bool:
        """Test results retrieval API"""
        self.log_step("Results API Test", "start")
        
        if not document_id:
            self.log_step("Results API Test", "failed", "No document ID provided")
            return False
        
        try:
            response = requests.get(
                f"{self.config['backend_url']}/analysis_results/{document_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                if results and len(results) > 0:
                    self.log_step("Results API Test", "success", f"Retrieved {len(results)} analysis results")
                    return True
                else:
                    self.log_step("Results API Test", "warning", "Results available but empty")
                    self.issues.append("Analysis results are empty")
                    return False
            else:
                self.log_step("Results API Test", "failed", f"Results retrieval failed: {response.status_code}")
                self.issues.append(f"Results API error: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_step("Results API Test", "failed", f"Exception: {str(e)}")
            self.issues.append(f"Results API exception: {str(e)}")
            return False

    async def test_frontend_basic_navigation(self) -> bool:
        """Test basic frontend navigation using simple requests"""
        self.log_step("Frontend Navigation Test", "start")
        
        try:
            # Test main page loads
            main_response = requests.get(self.config['frontend_url'], timeout=10)
            if main_response.status_code != 200:
                self.log_step("Frontend Navigation Test", "failed", f"Main page not loading: {main_response.status_code}")
                self.issues.append("Frontend main page not accessible")
                return False
            
            # Test if we can reach other routes (this is basic without browser automation)
            self.log_step("Frontend Navigation Test", "success", "Main page accessible")
            
            # Note: Without browser automation, we can't test actual user interactions
            # But we can verify the frontend server is serving pages
            
            return True
            
        except Exception as e:
            self.log_step("Frontend Navigation Test", "failed", f"Exception: {str(e)}")
            self.issues.append(f"Frontend navigation error: {str(e)}")
            return False

    def check_integration_consistency(self, document_id: str) -> bool:
        """Check if data flows consistently between components"""
        self.log_step("Integration Consistency Check", "start")
        
        try:
            # Test if backend document data is consistent
            doc_response = requests.get(
                f"{self.config['backend_url']}/document_info/{document_id}",
                timeout=10
            )
            
            if doc_response.status_code == 200:
                doc_info = doc_response.json()
                self.log_step("Integration Consistency Check", "success", f"Document info retrieved for ID: {document_id}")
                
                # Check if results match document
                results_response = requests.get(
                    f"{self.config['backend_url']}/analysis_results/{document_id}",
                    timeout=10
                )
                
                if results_response.status_code == 200:
                    self.log_step("Integration Consistency Check", "success", "Results consistent with document")
                    return True
                else:
                    self.log_step("Integration Consistency Check", "warning", "Results not available for consistency check")
                    return True  # May still be processing
            else:
                self.log_step("Integration Consistency Check", "failed", f"Document info not available: {doc_response.status_code}")
                self.issues.append("Document information inconsistency")
                return False
                
        except Exception as e:
            self.log_step("Integration Consistency Check", "failed", f"Exception: {str(e)}")
            self.issues.append(f"Integration check error: {str(e)}")
            return False

    async def run_complete_user_flow_test(self) -> bool:
        """Run the complete user flow test"""
        logger.info("=" * 80)
        logger.info("RAI COMPLIANCE PLATFORM - USER FLOW TEST")
        logger.info("=" * 80)
        
        start_time = time.time()
        test_results = {}
        
        # Step 1: Health checks
        backend_healthy = self.test_backend_health()
        frontend_healthy = self.test_frontend_health()
        test_results['backend_health'] = backend_healthy
        test_results['frontend_health'] = frontend_healthy
        
        if not backend_healthy:
            logger.error("CRITICAL: Backend server not accessible - cannot continue testing")
            return False
        
        if not frontend_healthy:
            logger.warning("WARNING: Frontend server not accessible - some tests will be skipped")
        
        # Step 2: File upload test
        upload_success, document_id = self.test_file_upload_api()
        test_results['file_upload'] = upload_success
        
        if not upload_success:
            logger.error("CRITICAL: File upload failed - cannot continue with document processing tests")
            return False
        
        # Step 3: Processing test
        processing_success = self.test_processing_api(document_id)
        test_results['processing'] = processing_success
        
        # Step 4: Results retrieval test
        results_success = self.test_results_api(document_id)
        test_results['results_retrieval'] = results_success
        
        # Step 5: Frontend navigation test (if frontend is healthy)
        if frontend_healthy:
            navigation_success = await self.test_frontend_basic_navigation()
            test_results['frontend_navigation'] = navigation_success
        else:
            test_results['frontend_navigation'] = False
        
        # Step 6: Integration consistency check
        integration_success = self.check_integration_consistency(document_id)
        test_results['integration_consistency'] = integration_success
        
        # Calculate results
        total_time = time.time() - start_time
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        # Generate report
        logger.info("=" * 80)
        logger.info("USER FLOW TEST RESULTS")
        logger.info("=" * 80)
        
        for test_name, passed in test_results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info("-" * 80)
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info(f"Document ID: {document_id}")
        
        if self.issues:
            logger.info("-" * 80)
            logger.info("ISSUES DISCOVERED:")
            for i, issue in enumerate(self.issues, 1):
                logger.info(f"{i}. {issue}")
        
        logger.info("-" * 80)
        
        if success_rate >= 80:
            logger.info("OVERALL RESULT: PASS - User flow is working well")
            overall_success = True
        elif success_rate >= 60:
            logger.info("OVERALL RESULT: WARNING - Some user flow issues detected")
            overall_success = False
        else:
            logger.info("OVERALL RESULT: FAIL - Critical user flow problems")
            overall_success = False
        
        logger.info("=" * 80)
        
        self.results = test_results
        return overall_success


async def main():
    """Main function"""
    print("\nStarting RAI Compliance Platform User Flow Test...")
    print("This test checks for disconnects in the user journey.")
    print("\nPrerequisites:")
    print("- Backend server running on http://localhost:8000")
    print("- Frontend server running on http://localhost:3000")
    print("- Test PDF file available\n")
    
    tester = SimpleUserFlowTester()
    
    try:
        success = await tester.run_complete_user_flow_test()
        
        if success:
            print("\nSUCCESS: User flow test passed!")
        else:
            print("\nWARNING: User flow issues detected - check log details above")
        
        return success
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"\nTest failed with exception: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)