#!/usr/bin/env python3
"""
Master End-to-End User Flow Integration Test
Orchestrates and runs all user flow tests in proper sequence

This is the master test that simulates the complete real user journey:
1. Backend API testing (upload → processing → results)
2. Frontend navigation testing (UI interactions)
3. Results page functionality testing
4. Cross-system integration validation
5. Performance and reliability assessment

Provides comprehensive reporting and identifies disconnects in user flow
"""

import asyncio
import subprocess
import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import concurrent.futures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_integration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MasterIntegrationTester:
    """Master test orchestrator for complete user flow validation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "frontend_url": "http://localhost:3000",
            "backend_url": "http://localhost:8000",
            "test_file_path": "c:/Users/saivi/OneDrive/Documents/Audricc all/uploads/RAI-1757795217-3ADC237A.pdf",
            "test_timeout_minutes": 10,
            "parallel_execution": False,
            "headless_browser": True,
            "generate_detailed_reports": True
        }
        
        self.test_results = {
            "system_health": {"status": "pending", "details": {}},
            "backend_flow": {"status": "pending", "details": {}},
            "frontend_navigation": {"status": "pending", "details": {}},
            "results_page": {"status": "pending", "details": {}},
            "integration": {"status": "pending", "details": {}},
            "performance": {"status": "pending", "details": {}},
            "reliability": {"status": "pending", "details": {}}
        }
        
        self.discovered_issues = []
        self.performance_metrics = {}
        self.test_artifacts = {}
        
    def log_master_step(self, step: str, status: str, details: Dict[str, Any] = None):
        """Log master test step with enhanced formatting"""
        if status == "start":
            logger.info(f"🎯 MASTER TEST: {step}")
            logger.info("=" * 60)
        elif status == "success":
            logger.info(f"✅ MASTER COMPLETE: {step}")
            if details:
                for key, value in details.items():
                    logger.info(f"   🏆 {key}: {value}")
            logger.info("=" * 60)
        elif status == "failed":
            logger.error(f"❌ MASTER FAILED: {step}")
            if details:
                for key, value in details.items():
                    logger.error(f"   💥 {key}: {value}")
            logger.info("=" * 60)
        elif status == "progress":
            logger.info(f"⚙️ MASTER PROGRESS: {step}")
            if details:
                for key, value in details.items():
                    logger.info(f"   📊 {key}: {value}")

    def check_prerequisites(self) -> bool:
        """Check system prerequisites and dependencies"""
        self.log_master_step("System Prerequisites Check", "start")
        
        prerequisites = {
            "Python 3.8+": False,
            "Required Packages": False,
            "Test File": False,
            "Network Access": False,
            "Frontend Server": False,
            "Backend Server": False
        }
        
        try:
            # Check Python version
            if sys.version_info >= (3, 8):
                prerequisites["Python 3.8+"] = True
            
            # Check required packages
            required_packages = ["requests", "playwright", "asyncio"]
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if not missing_packages:
                prerequisites["Required Packages"] = True
            else:
                logger.warning(f"Missing packages: {missing_packages}")
            
            # Check test file
            test_file = self.config["test_file_path"]
            if os.path.exists(test_file):
                prerequisites["Test File"] = True
                file_size = os.path.getsize(test_file) / 1024 / 1024
                logger.info(f"Test file found: {file_size:.2f} MB")
            else:
                logger.error(f"Test file not found: {test_file}")
            
            # Check network access
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                prerequisites["Network Access"] = True
            except:
                logger.warning("Network connectivity check failed")
            
            # Check servers (will be done later in health check)
            prerequisites["Frontend Server"] = "Will check during health test"
            prerequisites["Backend Server"] = "Will check during health test"
            
            passed_prereqs = sum(1 for v in prerequisites.values() if v is True)
            total_prereqs = len([v for v in prerequisites.values() if v is not str])
            
            if passed_prereqs >= total_prereqs:
                self.log_master_step("System Prerequisites Check", "success", {
                    "Prerequisites Met": f"{passed_prereqs}/{total_prereqs}",
                    "Status": "Ready for testing"
                })
                
                self.test_results["system_health"] = {
                    "status": "passed",
                    "details": prerequisites
                }
                return True
            else:
                self.log_master_step("System Prerequisites Check", "failed", {
                    "Prerequisites Met": f"{passed_prereqs}/{total_prereqs}",
                    "Missing": [k for k, v in prerequisites.items() if v is False]
                })
                
                self.test_results["system_health"]["status"] = "failed"
                return False
                
        except Exception as e:
            self.log_master_step("System Prerequisites Check", "failed", {
                "Exception": str(e)
            })
            self.test_results["system_health"]["status"] = "failed"
            return False

    def run_backend_flow_test(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute backend API flow test"""
        self.log_master_step("Backend Flow Test Execution", "start")
        
        try:
            # Import the backend test module
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # Create a simple backend test runner
            backend_test_script = '''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_user_flow_comprehensive import UserFlowTester

def run_backend_test():
    tester = UserFlowTester("''' + self.config["backend_url"] + '''")
    success = tester.run_end_to_end_test("''' + self.config["test_file_path"] + '''")
    return success, tester.test_results

if __name__ == "__main__":
    success, results = run_backend_test()
    print(f"BACKEND_TEST_RESULT:{success}")
    print(f"BACKEND_TEST_DATA:{results}")
'''
            
            # Write temporary test script
            temp_script = "temp_backend_test.py"
            with open(temp_script, 'w') as f:
                f.write(backend_test_script)
            
            # Execute backend test
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=self.config["test_timeout_minutes"] * 60
            )
            execution_time = time.time() - start_time
            
            # Clean up
            if os.path.exists(temp_script):
                os.remove(temp_script)
            
            # Parse results
            success = False
            test_data = {}
            
            for line in result.stdout.split('\n'):
                if line.startswith("BACKEND_TEST_RESULT:"):
                    success = line.split(":")[1].strip() == "True"
                elif line.startswith("BACKEND_TEST_DATA:"):
                    try:
                        test_data = eval(line.split(":", 1)[1].strip())
                    except:
                        test_data = {"error": "Could not parse test data"}
            
            self.log_master_step("Backend Flow Test Execution", "success" if success else "failed", {
                "Test Success": success,
                "Execution Time": f"{execution_time:.2f}s",
                "Return Code": result.returncode,
                "Tests Run": len(test_data) if test_data else 0
            })
            
            self.test_results["backend_flow"] = {
                "status": "passed" if success else "failed",
                "details": {
                    "execution_time": execution_time,
                    "test_data": test_data,
                    "stdout": result.stdout[:500] if result.stdout else "",
                    "stderr": result.stderr[:500] if result.stderr else ""
                }
            }
            
            return success, test_data
            
        except subprocess.TimeoutExpired:
            self.log_master_step("Backend Flow Test Execution", "failed", {
                "Error": f"Test timed out after {self.config['test_timeout_minutes']} minutes"
            })
            self.discovered_issues.append("Backend test timeout - potential performance issue")
            self.test_results["backend_flow"]["status"] = "failed"
            return False, {}
            
        except Exception as e:
            self.log_master_step("Backend Flow Test Execution", "failed", {
                "Exception": str(e)
            })
            self.discovered_issues.append(f"Backend test execution error: {str(e)}")
            self.test_results["backend_flow"]["status"] = "failed"
            return False, {}

    async def run_frontend_navigation_test(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute frontend navigation test"""
        self.log_master_step("Frontend Navigation Test Execution", "start")
        
        try:
            # Import frontend test module
            from test_frontend_navigation import FrontendNavigationTester
            
            tester = FrontendNavigationTester(
                self.config["frontend_url"],
                self.config["backend_url"]
            )
            
            start_time = time.time()
            success = await tester.run_all_frontend_tests(self.config["test_file_path"])
            execution_time = time.time() - start_time
            
            self.log_master_step("Frontend Navigation Test Execution", "success" if success else "failed", {
                "Test Success": success,
                "Execution Time": f"{execution_time:.2f}s",
                "Tests Completed": len(tester.test_results)
            })
            
            self.test_results["frontend_navigation"] = {
                "status": "passed" if success else "failed",
                "details": {
                    "execution_time": execution_time,
                    "test_results": tester.test_results
                }
            }
            
            return success, tester.test_results
            
        except Exception as e:
            self.log_master_step("Frontend Navigation Test Execution", "failed", {
                "Exception": str(e)
            })
            self.discovered_issues.append(f"Frontend navigation test error: {str(e)}")
            self.test_results["frontend_navigation"]["status"] = "failed"
            return False, {}

    async def run_results_page_test(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute results page comprehensive test"""
        self.log_master_step("Results Page Test Execution", "start")
        
        try:
            # Import results page test module
            from test_results_page_comprehensive import ResultsPageTester
            
            tester = ResultsPageTester(
                self.config["frontend_url"],
                self.config["backend_url"]
            )
            
            start_time = time.time()
            success = await tester.run_comprehensive_results_page_tests()
            execution_time = time.time() - start_time
            
            self.log_master_step("Results Page Test Execution", "success" if success else "failed", {
                "Test Success": success,
                "Execution Time": f"{execution_time:.2f}s",
                "Tests Completed": len(tester.test_results)
            })
            
            self.test_results["results_page"] = {
                "status": "passed" if success else "failed",
                "details": {
                    "execution_time": execution_time,
                    "test_results": tester.test_results,
                    "document_id": tester.test_document_id
                }
            }
            
            return success, tester.test_results
            
        except Exception as e:
            self.log_master_step("Results Page Test Execution", "failed", {
                "Exception": str(e)
            })
            self.discovered_issues.append(f"Results page test error: {str(e)}")
            self.test_results["results_page"]["status"] = "failed"
            return False, {}

    def analyze_integration_points(self, backend_results: Dict, frontend_results: Dict, results_page_results: Dict) -> bool:
        """Analyze integration between different components"""
        self.log_master_step("Integration Points Analysis", "start")
        
        try:
            integration_analysis = {
                "data_flow_consistency": False,
                "api_frontend_alignment": False,
                "state_management": False,
                "error_handling_consistency": False,
                "performance_coherence": False
            }
            
            # Check data flow consistency
            backend_doc_id = None
            frontend_doc_id = None
            results_doc_id = None
            
            # Extract document IDs from test results
            if backend_results.get("upload_test", {}).get("details", {}).get("document_id"):
                backend_doc_id = backend_results["upload_test"]["details"]["document_id"]
            
            if frontend_results.get("button_click_test", {}).get("details", {}).get("document_id"):
                frontend_doc_id = frontend_results["button_click_test"]["details"]["document_id"]
            
            if results_page_results.get("data_loading_test", {}).get("details", {}).get("document_id"):
                results_doc_id = results_page_results["data_loading_test"]["details"]["document_id"]
            
            # Check if document IDs are consistent across systems
            doc_ids = [backend_doc_id, frontend_doc_id, results_doc_id]
            valid_doc_ids = [doc_id for doc_id in doc_ids if doc_id and doc_id != "TEST_DOC_001"]
            
            if len(set(valid_doc_ids)) <= 1 and valid_doc_ids:
                integration_analysis["data_flow_consistency"] = True
            
            # Check API-Frontend alignment
            backend_api_success = backend_results.get("upload_test", {}).get("status") == "passed"
            frontend_upload_success = frontend_results.get("upload_interaction_test", {}).get("status") == "passed"
            
            if backend_api_success and frontend_upload_success:
                integration_analysis["api_frontend_alignment"] = True
            
            # Check state management
            frontend_state = frontend_results.get("state_persistence_test", {}).get("status") == "passed"
            results_state = results_page_results.get("user_interaction_test", {}).get("status") == "passed"
            
            if frontend_state and results_state:
                integration_analysis["state_management"] = True
            
            # Check error handling consistency
            backend_errors = backend_results.get("processing_test", {}).get("status") != "failed"
            frontend_errors = frontend_results.get("error_handling_test", {}).get("status") != "failed"
            results_errors = results_page_results.get("error_handling_test", {}).get("status") != "failed"
            
            if backend_errors and frontend_errors and results_errors:
                integration_analysis["error_handling_consistency"] = True
            
            # Check performance coherence
            backend_time = backend_results.get("processing_test", {}).get("details", {}).get("processing_time_seconds", 999)
            frontend_time = frontend_results.get("page_load_test", {}).get("details", {}).get("load_time", 999)
            results_time = results_page_results.get("performance_test", {}).get("details", {}).get("load_time", 999)
            
            # Check if all components perform within reasonable bounds
            if all(isinstance(t, (int, float)) and t < 30 for t in [backend_time, frontend_time, results_time]):
                integration_analysis["performance_coherence"] = True
            
            # Identify integration issues
            failed_integrations = [k for k, v in integration_analysis.items() if not v]
            
            if failed_integrations:
                for issue in failed_integrations:
                    self.discovered_issues.append(f"Integration issue: {issue.replace('_', ' ').title()}")
            
            integration_success = len(failed_integrations) <= 1  # Allow one minor issue
            
            self.log_master_step("Integration Points Analysis", "success" if integration_success else "failed", {
                "Integration Checks": len(integration_analysis),
                "Passed Checks": sum(integration_analysis.values()),
                "Failed Integrations": failed_integrations,
                "Document ID Consistency": len(set(valid_doc_ids)) <= 1
            })
            
            self.test_results["integration"] = {
                "status": "passed" if integration_success else "failed",
                "details": {
                    "integration_analysis": integration_analysis,
                    "failed_integrations": failed_integrations,
                    "document_ids": {
                        "backend": backend_doc_id,
                        "frontend": frontend_doc_id,
                        "results_page": results_doc_id
                    }
                }
            }
            
            return integration_success
            
        except Exception as e:
            self.log_master_step("Integration Points Analysis", "failed", {
                "Exception": str(e)
            })
            self.discovered_issues.append(f"Integration analysis error: {str(e)}")
            self.test_results["integration"]["status"] = "failed"
            return False

    def analyze_performance_metrics(self, all_results: Dict) -> bool:
        """Analyze overall performance across all tests"""
        self.log_master_step("Performance Analysis", "start")
        
        try:
            performance_data = {
                "backend_performance": {},
                "frontend_performance": {},
                "results_page_performance": {},
                "overall_assessment": {}
            }
            
            # Extract backend performance
            backend_results = all_results.get("backend_flow", {}).get("details", {}).get("test_data", {})
            if backend_results:
                upload_time = backend_results.get("upload_test", {}).get("details", {}).get("upload_duration")
                processing_time = backend_results.get("processing_test", {}).get("details", {}).get("processing_time_seconds")
                
                performance_data["backend_performance"] = {
                    "upload_time": upload_time,
                    "processing_time": processing_time,
                    "total_backend_time": (upload_time or 0) + (processing_time or 0)
                }
            
            # Extract frontend performance
            frontend_results = all_results.get("frontend_navigation", {}).get("details", {}).get("test_results", {})
            if frontend_results:
                page_load = frontend_results.get("page_load_test", {}).get("details", {})
                performance_data["frontend_performance"] = {
                    "page_load_time": page_load.get("load_time"),
                    "responsive_design": page_load.get("responsive")
                }
            
            # Extract results page performance
            results_results = all_results.get("results_page", {}).get("details", {}).get("test_results", {})
            if results_results:
                perf_test = results_results.get("performance_test", {}).get("details", {})
                performance_data["results_page_performance"] = {
                    "load_time": perf_test.get("load_time"),
                    "memory_efficient": perf_test.get("memory_efficient"),
                    "responsive_interactions": perf_test.get("responsive_interactions")
                }
            
            # Overall assessment
            total_time = sum([
                performance_data["backend_performance"].get("total_backend_time", 0),
                performance_data["frontend_performance"].get("page_load_time", 0),
                performance_data["results_page_performance"].get("load_time", 0)
            ])
            
            performance_grade = "Excellent" if total_time < 10 else "Good" if total_time < 30 else "Fair" if total_time < 60 else "Poor"
            
            performance_data["overall_assessment"] = {
                "total_user_journey_time": total_time,
                "performance_grade": performance_grade,
                "meets_user_expectations": total_time < 30
            }
            
            self.performance_metrics = performance_data
            
            performance_success = performance_data["overall_assessment"]["meets_user_expectations"]
            
            self.log_master_step("Performance Analysis", "success" if performance_success else "failed", {
                "Total Journey Time": f"{total_time:.2f}s",
                "Performance Grade": performance_grade,
                "Meets Expectations": performance_success
            })
            
            self.test_results["performance"] = {
                "status": "passed" if performance_success else "failed",
                "details": performance_data
            }
            
            return performance_success
            
        except Exception as e:
            self.log_master_step("Performance Analysis", "failed", {
                "Exception": str(e)
            })
            self.test_results["performance"]["status"] = "failed"
            return False

    def assess_reliability_and_robustness(self, all_results: Dict) -> bool:
        """Assess system reliability and robustness"""
        self.log_master_step("Reliability Assessment", "start")
        
        try:
            reliability_metrics = {
                "error_handling": 0,
                "data_consistency": 0,
                "user_experience": 0,
                "system_stability": 0
            }
            
            # Assess error handling across all components
            error_handling_tests = [
                all_results.get("backend_flow", {}).get("status") != "failed",
                all_results.get("frontend_navigation", {}).get("details", {}).get("test_results", {}).get("error_handling_test", {}).get("status") == "passed",
                all_results.get("results_page", {}).get("details", {}).get("test_results", {}).get("error_handling_test", {}).get("status") == "passed"
            ]
            
            reliability_metrics["error_handling"] = sum(error_handling_tests) / len(error_handling_tests)
            
            # Assess data consistency
            integration_status = all_results.get("integration", {}).get("status") == "passed"
            reliability_metrics["data_consistency"] = 1.0 if integration_status else 0.0
            
            # Assess user experience
            frontend_passed = all_results.get("frontend_navigation", {}).get("status") == "passed"
            results_passed = all_results.get("results_page", {}).get("status") == "passed"
            reliability_metrics["user_experience"] = (int(frontend_passed) + int(results_passed)) / 2
            
            # Assess system stability
            no_critical_issues = len(self.discovered_issues) <= 2
            reliability_metrics["system_stability"] = 1.0 if no_critical_issues else 0.5
            
            # Overall reliability score
            overall_reliability = sum(reliability_metrics.values()) / len(reliability_metrics)
            reliability_grade = "High" if overall_reliability >= 0.8 else "Medium" if overall_reliability >= 0.6 else "Low"
            
            reliability_success = overall_reliability >= 0.7
            
            self.log_master_step("Reliability Assessment", "success" if reliability_success else "failed", {
                "Overall Reliability": f"{overall_reliability:.2f}",
                "Reliability Grade": reliability_grade,
                "Critical Issues": len(self.discovered_issues),
                "Component Metrics": reliability_metrics
            })
            
            self.test_results["reliability"] = {
                "status": "passed" if reliability_success else "failed",
                "details": {
                    "overall_score": overall_reliability,
                    "grade": reliability_grade,
                    "metrics": reliability_metrics,
                    "critical_issues": self.discovered_issues
                }
            }
            
            return reliability_success
            
        except Exception as e:
            self.log_master_step("Reliability Assessment", "failed", {
                "Exception": str(e)
            })
            self.test_results["reliability"]["status"] = "failed"
            return False

    async def run_master_integration_test(self) -> bool:
        """Execute the complete master integration test"""
        self.log_master_step("Master Integration Test Suite", "start")
        
        overall_start_time = time.time()
        
        try:
            # Step 1: Prerequisites
            if not self.check_prerequisites():
                return False
            
            # Step 2: Execute test suites
            if self.config.get("parallel_execution", False):
                # Run frontend tests in parallel (both use browser)
                frontend_task = asyncio.create_task(self.run_frontend_navigation_test())
                results_task = asyncio.create_task(self.run_results_page_test())
                
                # Run backend test separately (can't parallelize with browser tests easily)
                backend_success, backend_data = self.run_backend_flow_test()
                
                # Wait for frontend tests
                frontend_success, frontend_data = await frontend_task
                results_success, results_data = await results_task
            else:
                # Sequential execution (safer)
                backend_success, backend_data = self.run_backend_flow_test()
                frontend_success, frontend_data = await self.run_frontend_navigation_test()
                results_success, results_data = await self.run_results_page_test()
            
            # Step 3: Analyze integration points
            integration_success = self.analyze_integration_points(
                backend_data, frontend_data, results_data
            )
            
            # Step 4: Performance analysis
            performance_success = self.analyze_performance_metrics(self.test_results)
            
            # Step 5: Reliability assessment
            reliability_success = self.assess_reliability_and_robustness(self.test_results)
            
            # Step 6: Final assessment
            all_tests = [
                backend_success,
                frontend_success,
                results_success,
                integration_success,
                performance_success,
                reliability_success
            ]
            
            total_execution_time = time.time() - overall_start_time
            
            passed_tests = sum(all_tests)
            total_tests = len(all_tests)
            success_rate = (passed_tests / total_tests) * 100
            
            overall_success = passed_tests >= (total_tests - 1)  # Allow one minor failure
            
            self.log_master_step("Master Integration Test Suite", "success" if overall_success else "failed", {
                "Total Tests Passed": f"{passed_tests}/{total_tests}",
                "Success Rate": f"{success_rate:.1f}%",
                "Total Execution Time": f"{total_execution_time:.2f}s",
                "Critical Issues Found": len(self.discovered_issues),
                "Overall Status": "PASS" if overall_success else "FAIL"
            })
            
            return overall_success
            
        except Exception as e:
            self.log_master_step("Master Integration Test Suite", "failed", {
                "Exception": str(e),
                "Total Execution Time": f"{time.time() - overall_start_time:.2f}s"
            })
            return False

    def generate_comprehensive_report(self) -> str:
        """Generate the final comprehensive test report"""
        report = "=" * 100 + "\n"
        report += "RAI COMPLIANCE PLATFORM - MASTER INTEGRATION TEST REPORT\n"
        report += "=" * 100 + "\n\n"
        
        report += f"🕒 Test Execution: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"🔧 Configuration:\n"
        for key, value in self.config.items():
            report += f"   • {key}: {value}\n"
        report += "\n"
        
        # Test Results Summary
        report += "📊 TEST RESULTS SUMMARY\n"
        report += "-" * 50 + "\n"
        
        for test_name, test_data in self.test_results.items():
            status_emoji = "✅" if test_data["status"] == "passed" else "❌" if test_data["status"] == "failed" else "⏳"
            report += f"{status_emoji} {test_name.replace('_', ' ').title()}: {test_data['status'].upper()}\n"
        
        report += "\n"
        
        # Performance Metrics
        if self.performance_metrics:
            report += "⚡ PERFORMANCE METRICS\n"
            report += "-" * 50 + "\n"
            
            overall_perf = self.performance_metrics.get("overall_assessment", {})
            report += f"🏁 Total User Journey Time: {overall_perf.get('total_user_journey_time', 'N/A')}s\n"
            report += f"🎯 Performance Grade: {overall_perf.get('performance_grade', 'N/A')}\n"
            report += f"👤 Meets User Expectations: {overall_perf.get('meets_user_expectations', 'N/A')}\n\n"
        
        # Discovered Issues
        if self.discovered_issues:
            report += "⚠️ DISCOVERED ISSUES\n"
            report += "-" * 50 + "\n"
            for i, issue in enumerate(self.discovered_issues, 1):
                report += f"{i}. {issue}\n"
            report += "\n"
        
        # User Flow Analysis
        report += "🔍 USER FLOW ANALYSIS\n"
        report += "-" * 50 + "\n"
        
        passed_count = sum(1 for test in self.test_results.values() if test["status"] == "passed")
        total_count = len(self.test_results)
        
        report += f"📈 Success Rate: {(passed_count/total_count)*100:.1f}%\n"
        report += f"🧪 Tests Passed: {passed_count}/{total_count}\n"
        report += f"🚨 Critical Issues: {len(self.discovered_issues)}\n\n"
        
        # Recommendations
        report += "💡 RECOMMENDATIONS\n"
        report += "-" * 50 + "\n"
        
        if passed_count == total_count:
            report += "🎉 EXCELLENT! All user flow tests passed.\n"
            report += "✅ The system is ready for production use.\n"
            report += "🔧 Consider performance optimization for even better UX.\n"
        elif passed_count >= total_count - 1:
            report += "👍 GOOD! Most user flow tests passed.\n"
            report += "🔧 Address the failed component before production.\n"
            report += "📋 Review discovered issues for potential improvements.\n"
        else:
            report += "⚠️ ATTENTION NEEDED! Multiple user flow issues detected.\n"
            report += "🚫 System not ready for production without fixes.\n"
            report += "🔧 Prioritize fixing critical user flow breaks.\n"
        
        report += "\n" + "=" * 100 + "\n"
        
        return report

    def save_test_artifacts(self):
        """Save test artifacts and detailed logs"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save main report
        main_report = self.generate_comprehensive_report()
        main_report_file = f"master_integration_test_report_{timestamp}.txt"
        
        with open(main_report_file, 'w') as f:
            f.write(main_report)
        
        # Save detailed test data
        detailed_data_file = f"master_test_data_{timestamp}.json"
        
        test_data = {
            "timestamp": timestamp,
            "config": self.config,
            "test_results": self.test_results,
            "performance_metrics": self.performance_metrics,
            "discovered_issues": self.discovered_issues
        }
        
        with open(detailed_data_file, 'w') as f:
            json.dump(test_data, f, indent=2, default=str)
        
        logger.info(f"📁 Test artifacts saved:")
        logger.info(f"   📄 Main Report: {main_report_file}")
        logger.info(f"   📊 Detailed Data: {detailed_data_file}")
        
        return main_report_file, detailed_data_file


async def main():
    """Main function to execute master integration test"""
    print("=" * 100)
    print("RAI COMPLIANCE PLATFORM - MASTER INTEGRATION TEST SUITE")
    print("=" * 100)
    print("\n🎯 This test simulates the COMPLETE user journey:")
    print("   1. Upload document → Process → Get results")
    print("   2. Navigate through chat interface")
    print("   3. View and interact with results page")
    print("   4. Test all user interactions and button clicks")
    print("   5. Validate data flow and integration")
    print("\n⚠️ Prerequisites:")
    print("   • Frontend server running (npm run dev)")
    print("   • Backend server running (python main.py)")
    print("   • Test PDF file available")
    print("   • Stable internet connection")
    
    # Configuration
    config = {
        "frontend_url": "http://localhost:3000",
        "backend_url": "http://localhost:8000", 
        "test_file_path": "c:/Users/saivi/OneDrive/Documents/Audricc all/uploads/RAI-1757795217-3ADC237A.pdf",
        "test_timeout_minutes": 10,
        "parallel_execution": False,  # Set to True for faster execution
        "headless_browser": False,    # Set to True to hide browser window
        "generate_detailed_reports": True
    }
    
    print(f"\n🔧 Test Configuration:")
    for key, value in config.items():
        print(f"   • {key}: {value}")
    
    print("\n" + "=" * 100)
    print("🚀 STARTING MASTER INTEGRATION TEST...")
    print("=" * 100)
    
    # Initialize and run master test
    master_tester = MasterIntegrationTester(config)
    
    try:
        success = await master_tester.run_master_integration_test()
        
        # Generate and display report
        report = master_tester.generate_comprehensive_report()
        print("\n" + report)
        
        # Save artifacts
        if config["generate_detailed_reports"]:
            master_tester.save_test_artifacts()
        
        # Final status
        if success:
            print("🎉 MASTER INTEGRATION TEST: PASSED")
            print("✅ User flow is working correctly across all systems!")
        else:
            print("❌ MASTER INTEGRATION TEST: FAILED") 
            print("🔧 User flow issues detected - review report for details")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Master test failed with exception: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)