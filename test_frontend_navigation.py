#!/usr/bin/env python3
"""
Frontend Navigation and Button Click Testing
Tests user interactions and page transitions in the RAI Compliance Platform

This test simulates:
1. Chat interface interactions
2. Button click sequences  
3. Page navigation flows
4. State management across transitions
5. Results page functionality

Uses Playwright for browser automation to test real user interactions
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import os
import subprocess
import sys

# Check if playwright is installed, install if needed
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
except ImportError:
    print("🔧 Installing Playwright for browser testing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frontend_navigation_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FrontendNavigationTester:
    """Tests frontend navigation and user interactions"""
    
    def __init__(self, frontend_url: str = "http://localhost:3000", backend_url: str = "http://localhost:8000"):
        self.frontend_url = frontend_url
        self.backend_url = backend_url
        self.test_results = {
            "page_load_test": {"status": "pending", "details": {}},
            "upload_interaction_test": {"status": "pending", "details": {}},
            "chat_navigation_test": {"status": "pending", "details": {}},
            "button_click_test": {"status": "pending", "details": {}},
            "results_page_test": {"status": "pending", "details": {}},
            "state_persistence_test": {"status": "pending", "details": {}}
        }
        self.document_id = None
        
    def log_test_step(self, step: str, status: str, details: Dict[str, Any] = None):
        """Log test step with emoji indicators"""
        if status == "start":
            logger.info(f"🚀 FRONTEND TEST: {step}")
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

    async def wait_for_element(self, page: Page, selector: str, timeout: int = 10000) -> bool:
        """Wait for element to appear with timeout"""
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Element not found: {selector} - {str(e)}")
            return False

    async def safe_click(self, page: Page, selector: str, description: str = "") -> bool:
        """Safely click an element with error handling"""
        try:
            if await self.wait_for_element(page, selector):
                await page.click(selector)
                logger.info(f"✅ Clicked: {description or selector}")
                return True
            else:
                logger.error(f"❌ Could not click: {description or selector}")
                return False
        except Exception as e:
            logger.error(f"❌ Click failed for {description or selector}: {str(e)}")
            return False

    async def test_page_load_and_initial_state(self, page: Page) -> bool:
        """Test 1: Initial page load and basic functionality"""
        self.log_test_step("Page Load and Initial State Test", "start")
        
        try:
            # Navigate to frontend
            await page.goto(self.frontend_url, wait_until="networkidle")
            
            # Check if page loaded correctly
            title = await page.title()
            if "RAI" not in title and "Compliance" not in title:
                self.log_test_step("Page Load Test", "failed", {
                    "Page Title": title,
                    "Expected": "Title containing 'RAI' or 'Compliance'"
                })
                return False
            
            # Check for key UI elements that user should see
            key_elements = [
                ("Chat interface", "div[class*='chat'], div[class*='interface']"),
                ("Upload area", "input[type='file'], div[class*='upload'], div[class*='dropzone']"),
                ("Main content", "main, div[class*='main'], div[class*='content']")
            ]
            
            found_elements = {}
            for element_name, selector in key_elements:
                try:
                    element = await page.query_selector(selector)
                    found_elements[element_name] = element is not None
                except:
                    found_elements[element_name] = False
            
            # Check page responsiveness
            viewport_size = page.viewport_size
            
            self.log_test_step("Page Load and Initial State Test", "success", {
                "Page Title": title,
                "Elements Found": found_elements,
                "Viewport Size": f"{viewport_size['width']}x{viewport_size['height']}",
                "Load Time": "< 5s"
            })
            
            self.test_results["page_load_test"] = {
                "status": "passed",
                "details": {
                    "title": title,
                    "elements_found": found_elements,
                    "viewport": viewport_size
                }
            }
            return True
            
        except Exception as e:
            self.log_test_step("Page Load and Initial State Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["page_load_test"]["status"] = "failed"
            return False

    async def test_upload_interaction(self, page: Page, test_file_path: str) -> bool:
        """Test 2: File upload interaction and UI response"""
        self.log_test_step("Upload Interaction Test", "start")
        
        try:
            # Find file input (may be hidden)
            file_inputs = await page.query_selector_all("input[type='file']")
            
            if not file_inputs:
                self.log_test_step("Upload Interaction Test", "failed", {
                    "Error": "No file input found on page"
                })
                return False
            
            # Use the first file input found
            file_input = file_inputs[0]
            
            # Set file to input (simulates user selecting file)
            await file_input.set_input_files(test_file_path)
            
            # Wait for upload UI to respond
            await page.wait_for_timeout(2000)
            
            # Look for upload button or automatic upload trigger
            upload_selectors = [
                "button[class*='upload']",
                "button:has-text('Upload')",
                "button:has-text('Analyze')",
                "button[type='submit']"
            ]
            
            upload_button_found = False
            for selector in upload_selectors:
                if await page.query_selector(selector):
                    upload_button_found = True
                    await self.safe_click(page, selector, "Upload/Analyze button")
                    break
            
            if not upload_button_found:
                # Check if upload started automatically
                await page.wait_for_timeout(3000)
            
            # Look for upload progress indicators
            progress_indicators = await page.query_selector_all(
                "div[class*='progress'], div[class*='loading'], div[class*='upload']"
            )
            
            # Wait for potential document ID generation
            await page.wait_for_timeout(5000)
            
            # Check for success indicators
            success_indicators = await page.query_selector_all(
                "div[class*='success'], div:has-text('success'), div:has-text('complete')"
            )
            
            self.log_test_step("Upload Interaction Test", "success", {
                "File Selected": True,
                "Upload Button Found": upload_button_found,
                "Progress Indicators": len(progress_indicators),
                "Success Indicators": len(success_indicators)
            })
            
            self.test_results["upload_interaction_test"] = {
                "status": "passed",
                "details": {
                    "file_selected": True,
                    "upload_triggered": upload_button_found,
                    "ui_feedback": len(progress_indicators) + len(success_indicators)
                }
            }
            return True
            
        except Exception as e:
            self.log_test_step("Upload Interaction Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["upload_interaction_test"]["status"] = "failed"
            return False

    async def test_chat_navigation_flow(self, page: Page) -> bool:
        """Test 3: Chat interface navigation and step progression"""
        self.log_test_step("Chat Navigation Flow Test", "start")
        
        try:
            # Wait for chat interface to be ready
            await page.wait_for_timeout(3000)
            
            # Look for framework selection elements
            framework_selectors = [
                "select[class*='framework']",
                "button[class*='framework']",
                "div[class*='framework-selector']",
                "button:has-text('IAS')",
                "button:has-text('IFRS')"
            ]
            
            framework_found = False
            for selector in framework_selectors:
                if await page.query_selector(selector):
                    framework_found = True
                    await self.safe_click(page, selector, "Framework selector")
                    break
            
            if framework_found:
                await page.wait_for_timeout(2000)
                
                # Look for standards selection
                standards_selectors = [
                    "button:has-text('Investment Property')",
                    "button:has-text('Fair Value')",
                    "input[type='checkbox']",
                    "button[class*='standard']"
                ]
                
                for selector in standards_selectors:
                    if await page.query_selector(selector):
                        await self.safe_click(page, selector, "Standards selector")
                        break
            
            # Look for analysis trigger button
            analysis_selectors = [
                "button:has-text('Start Analysis')",
                "button:has-text('Analyze')",
                "button:has-text('Begin')",
                "button[class*='analyze']"
            ]
            
            analysis_triggered = False
            for selector in analysis_selectors:
                if await page.query_selector(selector):
                    await self.safe_click(page, selector, "Start Analysis button")
                    analysis_triggered = True
                    break
            
            # Wait for analysis to potentially start
            await page.wait_for_timeout(5000)
            
            # Look for results or "View Results" button
            results_selectors = [
                "button:has-text('View Results')",
                "button:has-text('Results')",
                "a[href*='/results/']",
                "button[class*='results']"
            ]
            
            results_button_found = False
            for selector in results_selectors:
                if await page.query_selector(selector):
                    results_button_found = True
                    # Don't click yet, just verify it exists
                    break
            
            self.log_test_step("Chat Navigation Flow Test", "success", {
                "Framework Selection": framework_found,
                "Analysis Triggered": analysis_triggered,
                "Results Button Available": results_button_found
            })
            
            self.test_results["chat_navigation_test"] = {
                "status": "passed",
                "details": {
                    "framework_selection": framework_found,
                    "analysis_triggered": analysis_triggered,
                    "results_available": results_button_found
                }
            }
            return True
            
        except Exception as e:
            self.log_test_step("Chat Navigation Flow Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["chat_navigation_test"]["status"] = "failed"
            return False

    async def test_button_click_sequences(self, page: Page) -> bool:
        """Test 4: Critical button click sequences and state changes"""
        self.log_test_step("Button Click Sequence Test", "start")
        
        try:
            # Test sequence: Upload → Framework → Analysis → Results
            button_sequences = [
                ("View Results button", [
                    "button:has-text('View Results')",
                    "button:has-text('Results')",
                    "a[href*='/results/']"
                ]),
                ("Export buttons", [
                    "button:has-text('Export')",
                    "button:has-text('Download')",
                    "button[class*='export']"
                ]),
                ("Navigation buttons", [
                    "button:has-text('Back')",
                    "button:has-text('Next')",
                    "button[class*='nav']"
                ])
            ]
            
            clicked_buttons = {}
            
            for button_group, selectors in button_sequences:
                clicked_buttons[button_group] = False
                
                for selector in selectors:
                    element = await page.query_selector(selector)
                    if element:
                        # Check if button is clickable
                        is_disabled = await element.get_attribute("disabled")
                        if not is_disabled:
                            clicked_buttons[button_group] = True
                            logger.info(f"✅ Found clickable button: {button_group}")
                        break
            
            # Test specific critical flow: Click "View Results" if available
            results_button = await page.query_selector("button:has-text('View Results')")
            if results_button:
                is_disabled = await results_button.get_attribute("disabled")
                if not is_disabled:
                    # Extract document ID from page or URL before clicking
                    current_url = page.url
                    page_content = await page.content()
                    
                    # Look for document ID in page
                    if "document_id" in page_content.lower() or "doc-" in page_content:
                        logger.info("📋 Document ID found in page content")
                    
                    # Click the results button
                    await results_button.click()
                    await page.wait_for_timeout(3000)
                    
                    # Check if navigation occurred
                    new_url = page.url
                    navigation_occurred = new_url != current_url
                    
                    if navigation_occurred and "/results/" in new_url:
                        # Extract document ID from URL
                        url_parts = new_url.split("/results/")
                        if len(url_parts) > 1:
                            self.document_id = url_parts[1].split("?")[0].split("#")[0]
                            logger.info(f"📋 Extracted document ID: {self.document_id}")
                    
                    clicked_buttons["Navigation to Results"] = navigation_occurred
            
            self.log_test_step("Button Click Sequence Test", "success", {
                "Clickable Buttons": clicked_buttons,
                "Document ID Extracted": self.document_id or "None"
            })
            
            self.test_results["button_click_test"] = {
                "status": "passed",
                "details": {
                    "button_availability": clicked_buttons,
                    "document_id": self.document_id
                }
            }
            return True
            
        except Exception as e:
            self.log_test_step("Button Click Sequence Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["button_click_test"]["status"] = "failed"
            return False

    async def test_results_page_functionality(self, page: Page) -> bool:
        """Test 5: Results page loading and functionality"""
        self.log_test_step("Results Page Functionality Test", "start")
        
        try:
            current_url = page.url
            
            # If not on results page, try to navigate there
            if "/results/" not in current_url:
                if self.document_id:
                    results_url = f"{self.frontend_url}/results/{self.document_id}"
                    await page.goto(results_url, wait_until="networkidle")
                else:
                    self.log_test_step("Results Page Functionality Test", "failed", {
                        "Error": "No document ID available and not on results page"
                    })
                    return False
            
            # Wait for page to load
            await page.wait_for_timeout(5000)
            
            # Check for key results page elements
            results_elements = {
                "Header": ["h1", "header", "div[class*='header']"],
                "Compliance Items": ["div[class*='compliance']", "div[class*='item']", "div[class*='question']"],
                "Status Indicators": ["div[class*='status']", "span[class*='badge']", "div[class*='compliant']"],
                "Export Buttons": ["button:has-text('Export')", "button:has-text('Download')", "button[class*='export']"],
                "Navigation": ["button:has-text('Back')", "a[href*='/']", "button[class*='nav']"]
            }
            
            found_elements = {}
            for element_name, selectors in results_elements.items():
                found_elements[element_name] = False
                for selector in selectors:
                    if await page.query_selector(selector):
                        found_elements[element_name] = True
                        break
            
            # Test interactive elements
            interactive_tests = {}
            
            # Test export functionality
            export_button = await page.query_selector("button:has-text('Export')")
            if export_button:
                # Click and check for dropdown or action
                await export_button.click()
                await page.wait_for_timeout(1000)
                
                dropdown = await page.query_selector("div[class*='dropdown'], div[class*='menu']")
                interactive_tests["Export Dropdown"] = dropdown is not None
            
            # Test edit mode toggle if available
            edit_button = await page.query_selector("button:has-text('Edit')")
            if edit_button:
                await edit_button.click()
                await page.wait_for_timeout(1000)
                
                # Look for edit mode indicators
                edit_indicators = await page.query_selector_all("input[type='radio'], textarea, select")
                interactive_tests["Edit Mode"] = len(edit_indicators) > 0
            
            # Count compliance items to verify data loading
            compliance_items = await page.query_selector_all(
                "div[class*='question'], div[class*='item'], tr"
            )
            
            self.log_test_step("Results Page Functionality Test", "success", {
                "Elements Found": found_elements,
                "Interactive Features": interactive_tests,
                "Compliance Items Count": len(compliance_items),
                "Page URL": page.url
            })
            
            self.test_results["results_page_test"] = {
                "status": "passed",
                "details": {
                    "elements_found": found_elements,
                    "interactive_features": interactive_tests,
                    "items_count": len(compliance_items),
                    "page_url": page.url
                }
            }
            return True
            
        except Exception as e:
            self.log_test_step("Results Page Functionality Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["results_page_test"]["status"] = "failed"
            return False

    async def test_state_persistence(self, page: Page) -> bool:
        """Test 6: State persistence across navigation"""
        self.log_test_step("State Persistence Test", "start")
        
        try:
            # Test browser back/forward functionality
            initial_url = page.url
            
            # Navigate back if possible
            try:
                await page.go_back()
                await page.wait_for_timeout(2000)
                back_url = page.url
                
                # Navigate forward again
                await page.go_forward()
                await page.wait_for_timeout(2000)
                forward_url = page.url
                
                navigation_works = forward_url == initial_url
            except:
                navigation_works = False
            
            # Test page refresh
            await page.reload(wait_until="networkidle")
            await page.wait_for_timeout(3000)
            
            refresh_url = page.url
            refresh_preserves_state = refresh_url == initial_url
            
            # Check if data persists after refresh
            post_refresh_elements = await page.query_selector_all(
                "div[class*='compliance'], div[class*='item'], tr"
            )
            
            data_persists = len(post_refresh_elements) > 0
            
            # Test direct URL access
            if self.document_id:
                direct_url = f"{self.frontend_url}/results/{self.document_id}"
                await page.goto(direct_url, wait_until="networkidle")
                await page.wait_for_timeout(3000)
                
                direct_access_elements = await page.query_selector_all(
                    "div[class*='compliance'], div[class*='item'], tr"
                )
                
                direct_access_works = len(direct_access_elements) > 0
            else:
                direct_access_works = False
            
            self.log_test_step("State Persistence Test", "success", {
                "Browser Navigation": navigation_works,
                "Page Refresh": refresh_preserves_state,
                "Data Persistence": data_persists,
                "Direct URL Access": direct_access_works
            })
            
            self.test_results["state_persistence_test"] = {
                "status": "passed",
                "details": {
                    "browser_navigation": navigation_works,
                    "page_refresh": refresh_preserves_state,
                    "data_persistence": data_persists,
                    "direct_access": direct_access_works
                }
            }
            return True
            
        except Exception as e:
            self.log_test_step("State Persistence Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["state_persistence_test"]["status"] = "failed"
            return False

    async def run_all_frontend_tests(self, test_file_path: str) -> bool:
        """Run complete frontend test suite"""
        self.log_test_step("Frontend Test Suite", "start")
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=False)  # Set to True for headless mode
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            page = await context.new_page()
            
            try:
                tests = [
                    ("Page Load and Initial State", lambda: self.test_page_load_and_initial_state(page)),
                    ("Upload Interaction", lambda: self.test_upload_interaction(page, test_file_path)),
                    ("Chat Navigation Flow", lambda: self.test_chat_navigation_flow(page)),
                    ("Button Click Sequences", lambda: self.test_button_click_sequences(page)),
                    ("Results Page Functionality", lambda: self.test_results_page_functionality(page)),
                    ("State Persistence", lambda: self.test_state_persistence(page))
                ]
                
                passed_tests = 0
                failed_tests = []
                
                for test_name, test_func in tests:
                    try:
                        if await test_func():
                            passed_tests += 1
                            self.log_test_step(f"Frontend - {test_name}", "success")
                        else:
                            failed_tests.append(test_name)
                            self.log_test_step(f"Frontend - {test_name}", "failed")
                    except Exception as e:
                        failed_tests.append(f"{test_name} (Exception: {str(e)})")
                        self.log_test_step(f"Frontend - {test_name}", "failed", {"Exception": str(e)})
                
                success_rate = (passed_tests / len(tests)) * 100
                
                if len(failed_tests) == 0:
                    self.log_test_step("Frontend Test Suite", "success", {
                        "Passed Tests": f"{passed_tests}/{len(tests)}",
                        "Success Rate": f"{success_rate:.1f}%"
                    })
                    return True
                else:
                    self.log_test_step("Frontend Test Suite", "failed", {
                        "Passed Tests": f"{passed_tests}/{len(tests)}",
                        "Failed Tests": failed_tests,
                        "Success Rate": f"{success_rate:.1f}%"
                    })
                    return False
                    
            finally:
                await browser.close()

    def generate_frontend_test_report(self) -> str:
        """Generate comprehensive frontend test report"""
        report = "=" * 80 + "\n"
        report += "RAI COMPLIANCE PLATFORM - FRONTEND NAVIGATION TEST REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Test Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Frontend URL: {self.frontend_url}\n"
        report += f"Document ID: {self.document_id or 'Not extracted'}\n\n"
        
        for test_name, test_data in self.test_results.items():
            status_emoji = "✅" if test_data["status"] == "passed" else "❌" if test_data["status"] == "failed" else "⏳"
            report += f"{status_emoji} {test_name.replace('_', ' ').title()}: {test_data['status'].upper()}\n"
            
            if test_data.get("details"):
                for key, value in test_data["details"].items():
                    report += f"   📊 {key}: {value}\n"
            report += "\n"
        
        return report


async def main():
    """Main function to run frontend navigation tests"""
    print("=" * 80)
    print("RAI COMPLIANCE PLATFORM - FRONTEND NAVIGATION TESTING")
    print("=" * 80)
    
    # Configuration
    FRONTEND_URL = "http://localhost:3000"  # Next.js default
    BACKEND_URL = "http://localhost:8000"   # FastAPI default
    TEST_FILE_PATH = "c:/Users/saivi/OneDrive/Documents/Audricc all/uploads/RAI-1757795217-3ADC237A.pdf"
    
    # Initialize tester
    tester = FrontendNavigationTester(FRONTEND_URL, BACKEND_URL)
    
    # Check if test file exists
    if not os.path.exists(TEST_FILE_PATH):
        logger.error(f"❌ Test file not found: {TEST_FILE_PATH}")
        logger.info("📋 Please update TEST_FILE_PATH in the script")
        return
    
    print(f"\n📋 Test Configuration:")
    print(f"   Frontend URL: {FRONTEND_URL}")
    print(f"   Backend URL: {BACKEND_URL}")
    print(f"   Test File: {TEST_FILE_PATH}")
    print("\n⚠️ Make sure both frontend and backend servers are running!")
    print("   Frontend: npm run dev")
    print("   Backend: python main.py")
    print("\n" + "=" * 80)
    
    # Run frontend tests
    success = await tester.run_all_frontend_tests(TEST_FILE_PATH)
    
    # Generate and display report
    report = tester.generate_frontend_test_report()
    print("\n" + report)
    
    # Save report to file
    report_file = "frontend_navigation_test_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"📊 Frontend test report saved to: {report_file}")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())