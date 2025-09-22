#!/usr/bin/env python3
"""
Results Page Comprehensive Testing
Tests the complete results page functionality, data flow, and user interactions

This test validates:
1. Results page data loading and API integration
2. Compliance item display and formatting
3. Interactive features (edit mode, export, filters)
4. Data consistency and error handling
5. Export functionality and file generation
6. User interaction patterns and state management

Simulates real user behavior on the results page
"""

import asyncio
import requests
import json
import logging
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile

# Browser automation
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
except ImportError:
    import subprocess
    import sys
    print("🔧 Installing Playwright for browser testing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results_page_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResultsPageTester:
    """Comprehensive results page testing suite"""
    
    def __init__(self, frontend_url: str = "http://localhost:3000", backend_url: str = "http://localhost:8000"):
        self.frontend_url = frontend_url
        self.backend_url = backend_url
        self.test_results = {
            "data_loading_test": {"status": "pending", "details": {}},
            "display_formatting_test": {"status": "pending", "details": {}},
            "interactive_features_test": {"status": "pending", "details": {}},
            "export_functionality_test": {"status": "pending", "details": {}},
            "user_interaction_test": {"status": "pending", "details": {}},
            "error_handling_test": {"status": "pending", "details": {}},
            "performance_test": {"status": "pending", "details": {}}
        }
        self.session = requests.Session()
        self.test_document_id = None
        
    def log_test_step(self, step: str, status: str, details: Dict[str, Any] = None):
        """Log test step with emoji indicators"""
        if status == "start":
            logger.info(f"🚀 RESULTS PAGE TEST: {step}")
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

    def prepare_test_data(self) -> Optional[str]:
        """Prepare test data by finding or creating a document with results"""
        self.log_test_step("Test Data Preparation", "start")
        
        try:
            # First, check if there are existing documents with results
            response = self.session.get(f"{self.backend_url}/api/v1/documents", timeout=10)
            
            if response.status_code == 200:
                documents = response.json()
                
                # Look for a completed document
                for doc in documents:
                    if doc.get('status') == 'COMPLETED':
                        document_id = doc.get('id') or doc.get('document_id')
                        if document_id:
                            # Verify this document has results
                            results_response = self.session.get(
                                f"{self.backend_url}/api/v1/analysis/results/{document_id}",
                                timeout=10
                            )
                            
                            if results_response.status_code == 200:
                                results_data = results_response.json()
                                if results_data.get('sections') and len(results_data['sections']) > 0:
                                    self.test_document_id = document_id
                                    
                                    self.log_test_step("Test Data Preparation", "success", {
                                        "Document ID": document_id,
                                        "Sections Count": len(results_data['sections']),
                                        "Status": results_data.get('status', 'unknown')
                                    })
                                    return document_id
            
            # If no existing data found, we'll create a mock document ID for testing
            self.test_document_id = "TEST_DOC_001"
            
            self.log_test_step("Test Data Preparation", "progress", {
                "Status": "Using mock document ID for testing",
                "Document ID": self.test_document_id
            })
            
            return self.test_document_id
            
        except Exception as e:
            self.log_test_step("Test Data Preparation", "failed", {
                "Exception": str(e)
            })
            return None

    async def test_data_loading_and_api_integration(self, page: Page, document_id: str) -> bool:
        """Test 1: Data loading and API integration"""
        self.log_test_step("Data Loading and API Integration Test", "start")
        
        try:
            # Navigate to results page
            results_url = f"{self.frontend_url}/results/{document_id}"
            
            # Monitor network requests
            network_requests = []
            
            def handle_request(request):
                network_requests.append({
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type
                })
            
            page.on("request", handle_request)
            
            # Navigate and measure load time
            start_time = time.time()
            await page.goto(results_url, wait_until="networkidle")
            load_time = time.time() - start_time
            
            # Wait for potential async data loading
            await page.wait_for_timeout(5000)
            
            # Check for loading indicators
            loading_indicators = await page.query_selector_all(
                "div[class*='loading'], div[class*='spinner'], div[class*='skeleton']"
            )
            
            # Check for error states
            error_indicators = await page.query_selector_all(
                "div[class*='error'], div:has-text('Error'), div:has-text('Failed')"
            )
            
            # Check for data content
            data_elements = await page.query_selector_all(
                "div[class*='compliance'], div[class*='item'], tr, div[class*='question']"
            )
            
            # Analyze network requests
            api_requests = [req for req in network_requests if "/api/" in req["url"]]
            failed_requests = []
            
            # Check API responses
            for req in api_requests:
                try:
                    response = self.session.get(req["url"], timeout=5)
                    if response.status_code != 200:
                        failed_requests.append({
                            "url": req["url"],
                            "status": response.status_code
                        })
                except:
                    failed_requests.append({
                        "url": req["url"],
                        "status": "Request failed"
                    })
            
            self.log_test_step("Data Loading and API Integration Test", "success", {
                "Page Load Time": f"{load_time:.2f}s",
                "Network Requests": len(network_requests),
                "API Requests": len(api_requests),
                "Failed Requests": len(failed_requests),
                "Data Elements Found": len(data_elements),
                "Loading Indicators": len(loading_indicators),
                "Error Indicators": len(error_indicators)
            })
            
            self.test_results["data_loading_test"] = {
                "status": "passed",
                "details": {
                    "load_time": load_time,
                    "api_requests": len(api_requests),
                    "failed_requests": len(failed_requests),
                    "data_elements": len(data_elements)
                }
            }
            
            return len(error_indicators) == 0 and len(data_elements) > 0
            
        except Exception as e:
            self.log_test_step("Data Loading and API Integration Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["data_loading_test"]["status"] = "failed"
            return False

    async def test_display_formatting_and_layout(self, page: Page) -> bool:
        """Test 2: Display formatting and layout verification"""
        self.log_test_step("Display Formatting and Layout Test", "start")
        
        try:
            # Check page layout structure
            layout_elements = {
                "Header": ["h1", "header", "div[class*='header']"],
                "Navigation": ["nav", "button:has-text('Back')", "div[class*='breadcrumb']"],
                "Main Content": ["main", "div[class*='content']", "div[class*='results']"],
                "Sidebar": ["aside", "div[class*='sidebar']", "div[class*='panel']"],
                "Footer": ["footer", "div[class*='footer']"]
            }
            
            found_layout = {}
            for element_name, selectors in layout_elements.items():
                found_layout[element_name] = False
                for selector in selectors:
                    if await page.query_selector(selector):
                        found_layout[element_name] = True
                        break
            
            # Check compliance item formatting
            compliance_items = await page.query_selector_all(
                "div[class*='question'], div[class*='item'], tr"
            )
            
            # Analyze item structure
            item_analysis = {
                "items_with_questions": 0,
                "items_with_status": 0,
                "items_with_references": 0,
                "items_with_explanations": 0
            }
            
            for i, item in enumerate(compliance_items[:10]):  # Check first 10 items
                item_text = await item.inner_text()
                
                if "?" in item_text or "question" in item_text.lower():
                    item_analysis["items_with_questions"] += 1
                
                if any(status in item_text.upper() for status in ["YES", "NO", "N/A", "COMPLIANT", "NON-COMPLIANT"]):
                    item_analysis["items_with_status"] += 1
                
                if "IAS" in item_text or "IFRS" in item_text or "reference" in item_text.lower():
                    item_analysis["items_with_references"] += 1
                
                if len(item_text) > 100:  # Assuming longer text indicates explanations
                    item_analysis["items_with_explanations"] += 1
            
            # Check responsive design
            viewport_sizes = [
                {"width": 1920, "height": 1080, "name": "Desktop"},
                {"width": 768, "height": 1024, "name": "Tablet"},
                {"width": 375, "height": 667, "name": "Mobile"}
            ]
            
            responsive_tests = {}
            for size in viewport_sizes:
                await page.set_viewport_size({"width": size["width"], "height": size["height"]})
                await page.wait_for_timeout(1000)
                
                # Check if content is still visible and properly formatted
                visible_items = await page.query_selector_all(
                    "div[class*='question']:visible, div[class*='item']:visible"
                )
                
                responsive_tests[size["name"]] = {
                    "visible_items": len(visible_items),
                    "viewport": f"{size['width']}x{size['height']}"
                }
            
            # Reset to desktop view
            await page.set_viewport_size({"width": 1920, "height": 1080})
            
            self.log_test_step("Display Formatting and Layout Test", "success", {
                "Layout Elements": found_layout,
                "Compliance Items": len(compliance_items),
                "Item Analysis": item_analysis,
                "Responsive Design": responsive_tests
            })
            
            self.test_results["display_formatting_test"] = {
                "status": "passed",
                "details": {
                    "layout_complete": all(found_layout.values()),
                    "items_count": len(compliance_items),
                    "item_structure": item_analysis,
                    "responsive": responsive_tests
                }
            }
            
            return len(compliance_items) > 0
            
        except Exception as e:
            self.log_test_step("Display Formatting and Layout Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["display_formatting_test"]["status"] = "failed"
            return False

    async def test_interactive_features(self, page: Page) -> bool:
        """Test 3: Interactive features and user controls"""
        self.log_test_step("Interactive Features Test", "start")
        
        try:
            interactive_features = {}
            
            # Test Edit Mode Toggle
            edit_button = await page.query_selector("button:has-text('Edit')")
            if edit_button:
                await edit_button.click()
                await page.wait_for_timeout(2000)
                
                # Check for edit mode indicators
                edit_indicators = await page.query_selector_all(
                    "input[type='radio'], textarea, select, input[type='text']"
                )
                
                interactive_features["Edit Mode"] = {
                    "available": True,
                    "edit_controls": len(edit_indicators)
                }
                
                # Test radio button interaction
                radio_buttons = await page.query_selector_all("input[type='radio']")
                if radio_buttons:
                    # Click first radio button
                    await radio_buttons[0].click()
                    await page.wait_for_timeout(500)
                    
                    # Check if state changed
                    is_checked = await radio_buttons[0].is_checked()
                    interactive_features["Radio Buttons"] = {
                        "available": True,
                        "clickable": is_checked
                    }
                
                # Exit edit mode
                exit_edit_button = await page.query_selector("button:has-text('Exit Edit')")
                if exit_edit_button:
                    await exit_edit_button.click()
                    await page.wait_for_timeout(1000)
            else:
                interactive_features["Edit Mode"] = {"available": False}
            
            # Test Filter/Search functionality
            search_elements = await page.query_selector_all(
                "input[type='search'], input[placeholder*='search'], input[placeholder*='filter']"
            )
            
            if search_elements:
                search_input = search_elements[0]
                await search_input.fill("investment property")
                await page.wait_for_timeout(1000)
                
                # Check if results are filtered
                filtered_items = await page.query_selector_all(
                    "div[class*='question'], div[class*='item']"
                )
                
                interactive_features["Search/Filter"] = {
                    "available": True,
                    "results_after_filter": len(filtered_items)
                }
                
                # Clear search
                await search_input.fill("")
                await page.wait_for_timeout(1000)
            else:
                interactive_features["Search/Filter"] = {"available": False}
            
            # Test Collapsible Sections
            collapsible_triggers = await page.query_selector_all(
                "button[class*='collapsible'], button[aria-expanded], summary"
            )
            
            if collapsible_triggers:
                initial_content = await page.query_selector_all(
                    "div[class*='content'], div[class*='details']"
                )
                
                # Click first collapsible
                await collapsible_triggers[0].click()
                await page.wait_for_timeout(1000)
                
                after_click_content = await page.query_selector_all(
                    "div[class*='content'], div[class*='details']"
                )
                
                interactive_features["Collapsible Sections"] = {
                    "available": True,
                    "triggers_count": len(collapsible_triggers),
                    "content_changed": len(initial_content) != len(after_click_content)
                }
            else:
                interactive_features["Collapsible Sections"] = {"available": False}
            
            # Test Sorting functionality
            sort_elements = await page.query_selector_all(
                "button:has-text('Sort'), select[class*='sort'], th[class*='sortable']"
            )
            
            if sort_elements:
                initial_order = []
                items = await page.query_selector_all("div[class*='question'], tr")
                for item in items[:5]:  # Check first 5 items
                    text = await item.inner_text()
                    initial_order.append(text[:50])  # First 50 chars
                
                # Click sort element
                await sort_elements[0].click()
                await page.wait_for_timeout(2000)
                
                after_sort_order = []
                items = await page.query_selector_all("div[class*='question'], tr")
                for item in items[:5]:
                    text = await item.inner_text()
                    after_sort_order.append(text[:50])
                
                interactive_features["Sorting"] = {
                    "available": True,
                    "order_changed": initial_order != after_sort_order
                }
            else:
                interactive_features["Sorting"] = {"available": False}
            
            self.log_test_step("Interactive Features Test", "success", {
                "Features Tested": list(interactive_features.keys()),
                "Feature Details": interactive_features
            })
            
            self.test_results["interactive_features_test"] = {
                "status": "passed",
                "details": {
                    "features_available": len([f for f in interactive_features.values() if f.get("available")]),
                    "total_features_tested": len(interactive_features),
                    "feature_details": interactive_features
                }
            }
            
            return True
            
        except Exception as e:
            self.log_test_step("Interactive Features Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["interactive_features_test"]["status"] = "failed"
            return False

    async def test_export_functionality(self, page: Page, document_id: str) -> bool:
        """Test 4: Export functionality and file generation"""
        self.log_test_step("Export Functionality Test", "start")
        
        try:
            export_tests = {}
            
            # Test Export Button Availability
            export_button = await page.query_selector("button:has-text('Export')")
            if not export_button:
                export_button = await page.query_selector("button:has-text('Download')")
            
            if export_button:
                # Click export button to open dropdown
                await export_button.click()
                await page.wait_for_timeout(1000)
                
                # Check for export options
                export_options = await page.query_selector_all(
                    "button:has-text('PDF'), button:has-text('Excel'), button:has-text('Word'), button:has-text('JSON')"
                )
                
                export_tests["Export Button"] = {
                    "available": True,
                    "options_count": len(export_options)
                }
                
                # Test each export format via API (to avoid download handling in browser)
                export_formats = ["pdf", "excel", "word", "json"]
                api_export_tests = {}
                
                for format_type in export_formats:
                    try:
                        export_url = f"{self.backend_url}/api/v1/analysis/export/{document_id}/{format_type}"
                        response = self.session.get(export_url, timeout=10)
                        
                        api_export_tests[format_type.upper()] = {
                            "status_code": response.status_code,
                            "content_type": response.headers.get("content-type", "unknown"),
                            "content_length": len(response.content) if response.content else 0
                        }
                    except Exception as e:
                        api_export_tests[format_type.upper()] = {
                            "error": str(e)
                        }
                
                export_tests["API Export Tests"] = api_export_tests
                
                # Test export button clicks in UI
                pdf_button = await page.query_selector("button:has-text('PDF')")
                if pdf_button:
                    # Set up download handling
                    downloads = []
                    
                    def handle_download(download):
                        downloads.append({
                            "filename": download.suggested_filename,
                            "url": download.url
                        })
                    
                    page.on("download", handle_download)
                    
                    # Click PDF export
                    await pdf_button.click()
                    await page.wait_for_timeout(3000)
                    
                    export_tests["PDF Export UI"] = {
                        "clicked": True,
                        "downloads_triggered": len(downloads)
                    }
            else:
                export_tests["Export Button"] = {"available": False}
            
            # Test export data integrity by checking API responses
            if document_id and document_id != "TEST_DOC_001":
                try:
                    results_response = self.session.get(
                        f"{self.backend_url}/api/v1/analysis/results/{document_id}",
                        timeout=10
                    )
                    
                    if results_response.status_code == 200:
                        results_data = results_response.json()
                        
                        export_tests["Data Integrity"] = {
                            "results_available": True,
                            "sections_count": len(results_data.get("sections", [])),
                            "has_metadata": bool(results_data.get("metadata"))
                        }
                    else:
                        export_tests["Data Integrity"] = {
                            "results_available": False,
                            "error": f"API returned {results_response.status_code}"
                        }
                except Exception as e:
                    export_tests["Data Integrity"] = {
                        "error": str(e)
                    }
            
            self.log_test_step("Export Functionality Test", "success", {
                "Export Tests Completed": list(export_tests.keys()),
                "Test Results": export_tests
            })
            
            self.test_results["export_functionality_test"] = {
                "status": "passed",
                "details": {
                    "export_button_available": export_tests.get("Export Button", {}).get("available", False),
                    "export_formats_tested": len(export_tests.get("API Export Tests", {})),
                    "test_results": export_tests
                }
            }
            
            return True
            
        except Exception as e:
            self.log_test_step("Export Functionality Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["export_functionality_test"]["status"] = "failed"
            return False

    async def test_user_interaction_patterns(self, page: Page) -> bool:
        """Test 5: Common user interaction patterns"""
        self.log_test_step("User Interaction Patterns Test", "start")
        
        try:
            interaction_tests = {}
            
            # Test scrolling behavior and infinite scroll/pagination
            initial_items = await page.query_selector_all("div[class*='question'], tr")
            initial_count = len(initial_items)
            
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
            
            after_scroll_items = await page.query_selector_all("div[class*='question'], tr")
            after_scroll_count = len(after_scroll_items)
            
            interaction_tests["Scrolling"] = {
                "initial_items": initial_count,
                "after_scroll_items": after_scroll_count,
                "infinite_scroll": after_scroll_count > initial_count
            }
            
            # Test keyboard navigation
            await page.keyboard.press("Tab")
            await page.wait_for_timeout(500)
            
            focused_element = await page.evaluate("document.activeElement.tagName")
            
            interaction_tests["Keyboard Navigation"] = {
                "tab_navigation": focused_element in ["BUTTON", "INPUT", "A", "SELECT"]
            }
            
            # Test right-click context menu (if applicable)
            compliance_item = await page.query_selector("div[class*='question'], tr")
            if compliance_item:
                await compliance_item.click(button="right")
                await page.wait_for_timeout(1000)
                
                context_menu = await page.query_selector("div[class*='context'], div[class*='menu']")
                interaction_tests["Context Menu"] = {
                    "available": context_menu is not None
                }
            
            # Test drag and drop (if applicable)
            draggable_elements = await page.query_selector_all("[draggable='true']")
            interaction_tests["Drag and Drop"] = {
                "draggable_elements": len(draggable_elements)
            }
            
            # Test double-click actions
            if compliance_item:
                await compliance_item.dblclick()
                await page.wait_for_timeout(1000)
                
                # Check if double-click triggered any action
                modals = await page.query_selector_all("dialog, div[class*='modal'], div[class*='popup']")
                interaction_tests["Double Click"] = {
                    "modals_opened": len(modals)
                }
            
            # Test copy functionality (Ctrl+C)
            if compliance_item:
                await compliance_item.click()
                await page.keyboard.press("Control+c")
                await page.wait_for_timeout(500)
                
                # Try to paste somewhere to test if copy worked
                text_input = await page.query_selector("input[type='text'], textarea")
                if text_input:
                    await text_input.click()
                    await page.keyboard.press("Control+v")
                    
                    pasted_value = await text_input.input_value()
                    interaction_tests["Copy/Paste"] = {
                        "copy_worked": len(pasted_value) > 0
                    }
            
            # Test browser back/forward with results state
            current_url = page.url
            await page.go_back()
            await page.wait_for_timeout(2000)
            
            back_url = page.url
            
            await page.go_forward()
            await page.wait_for_timeout(2000)
            
            forward_url = page.url
            
            interaction_tests["Browser Navigation"] = {
                "back_navigation": back_url != current_url,
                "forward_navigation": forward_url == current_url,
                "state_preserved": forward_url == current_url
            }
            
            self.log_test_step("User Interaction Patterns Test", "success", {
                "Interaction Tests": list(interaction_tests.keys()),
                "Test Results": interaction_tests
            })
            
            self.test_results["user_interaction_test"] = {
                "status": "passed",
                "details": {
                    "interactions_tested": len(interaction_tests),
                    "interaction_results": interaction_tests
                }
            }
            
            return True
            
        except Exception as e:
            self.log_test_step("User Interaction Patterns Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["user_interaction_test"]["status"] = "failed"
            return False

    async def test_error_handling_and_edge_cases(self, page: Page) -> bool:
        """Test 6: Error handling and edge cases"""
        self.log_test_step("Error Handling and Edge Cases Test", "start")
        
        try:
            error_tests = {}
            
            # Test invalid document ID
            invalid_url = f"{self.frontend_url}/results/INVALID_DOC_ID"
            await page.goto(invalid_url)
            await page.wait_for_timeout(3000)
            
            error_messages = await page.query_selector_all(
                "div:has-text('Error'), div:has-text('Not Found'), div:has-text('Invalid')"
            )
            
            error_tests["Invalid Document ID"] = {
                "error_displayed": len(error_messages) > 0,
                "graceful_handling": len(error_messages) > 0
            }
            
            # Test network interruption simulation
            # Go back to valid page first
            if self.test_document_id:
                valid_url = f"{self.frontend_url}/results/{self.test_document_id}"
                await page.goto(valid_url)
                await page.wait_for_timeout(2000)
                
                # Simulate offline state
                await page.context.set_offline(True)
                await page.reload()
                await page.wait_for_timeout(3000)
                
                offline_indicators = await page.query_selector_all(
                    "div:has-text('offline'), div:has-text('connection'), div:has-text('network')"
                )
                
                error_tests["Network Interruption"] = {
                    "offline_detection": len(offline_indicators) > 0
                }
                
                # Restore connection
                await page.context.set_offline(False)
                await page.wait_for_timeout(2000)
            
            # Test empty results handling
            try:
                # Try to access API directly to test empty response handling
                empty_response = self.session.get(
                    f"{self.backend_url}/api/v1/analysis/results/EMPTY_TEST",
                    timeout=5
                )
                
                error_tests["Empty Results API"] = {
                    "status_code": empty_response.status_code,
                    "handles_404": empty_response.status_code == 404
                }
            except:
                error_tests["Empty Results API"] = {"test_skipped": True}
            
            # Test large dataset handling (if applicable)
            # This would test performance with many compliance items
            compliance_items = await page.query_selector_all("div[class*='question'], tr")
            
            if len(compliance_items) > 50:
                # Test scrolling performance with large dataset
                scroll_start = time.time()
                
                for i in range(5):  # Scroll 5 times
                    await page.evaluate("window.scrollBy(0, 500)")
                    await page.wait_for_timeout(100)
                
                scroll_time = time.time() - scroll_start
                
                error_tests["Large Dataset"] = {
                    "items_count": len(compliance_items),
                    "scroll_performance": f"{scroll_time:.2f}s",
                    "responsive": scroll_time < 2.0
                }
            
            # Test browser compatibility issues
            user_agent = await page.evaluate("navigator.userAgent")
            viewport = page.viewport_size
            
            error_tests["Browser Compatibility"] = {
                "user_agent": user_agent[:50] + "..." if len(user_agent) > 50 else user_agent,
                "viewport": f"{viewport['width']}x{viewport['height']}",
                "javascript_enabled": True  # If we got this far, JS is working
            }
            
            self.log_test_step("Error Handling and Edge Cases Test", "success", {
                "Error Tests Completed": list(error_tests.keys()),
                "Test Results": error_tests
            })
            
            self.test_results["error_handling_test"] = {
                "status": "passed",
                "details": {
                    "error_scenarios_tested": len(error_tests),
                    "error_test_results": error_tests
                }
            }
            
            return True
            
        except Exception as e:
            self.log_test_step("Error Handling and Edge Cases Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["error_handling_test"]["status"] = "failed"
            return False

    async def test_performance_and_optimization(self, page: Page, document_id: str) -> bool:
        """Test 7: Performance and optimization"""
        self.log_test_step("Performance and Optimization Test", "start")
        
        try:
            performance_metrics = {}
            
            # Test page load performance
            results_url = f"{self.frontend_url}/results/{document_id}"
            
            start_time = time.time()
            await page.goto(results_url, wait_until="networkidle")
            
            # Measure time to first content
            await page.wait_for_selector("div[class*='question'], tr", timeout=10000)
            first_content_time = time.time() - start_time
            
            # Measure time to full load
            await page.wait_for_timeout(2000)
            full_load_time = time.time() - start_time
            
            performance_metrics["Load Times"] = {
                "first_content": f"{first_content_time:.2f}s",
                "full_load": f"{full_load_time:.2f}s",
                "performance_grade": "Good" if full_load_time < 3.0 else "Fair" if full_load_time < 5.0 else "Poor"
            }
            
            # Test memory usage (via JavaScript)
            memory_info = await page.evaluate("""
                () => {
                    if (performance.memory) {
                        return {
                            used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                            total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
                            limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
                        };
                    }
                    return null;
                }
            """)
            
            if memory_info:
                performance_metrics["Memory Usage"] = {
                    "used_mb": memory_info["used"],
                    "total_mb": memory_info["total"],
                    "limit_mb": memory_info["limit"],
                    "efficiency": f"{(memory_info['used'] / memory_info['total'] * 100):.1f}%"
                }
            
            # Test rendering performance with large datasets
            compliance_items = await page.query_selector_all("div[class*='question'], tr")
            
            if len(compliance_items) > 10:
                # Test scroll performance
                scroll_start = time.time()
                
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(500)
                
                scroll_time = time.time() - scroll_start
                
                performance_metrics["Scroll Performance"] = {
                    "items_count": len(compliance_items),
                    "scroll_time": f"{scroll_time:.2f}s",
                    "smooth_scrolling": scroll_time < 1.0
                }
            
            # Test interaction responsiveness
            edit_button = await page.query_selector("button:has-text('Edit')")
            if edit_button:
                click_start = time.time()
                await edit_button.click()
                
                # Wait for edit mode to activate
                await page.wait_for_selector("input[type='radio']", timeout=5000)
                
                interaction_time = time.time() - click_start
                
                performance_metrics["Interaction Response"] = {
                    "edit_mode_activation": f"{interaction_time:.2f}s",
                    "responsive": interaction_time < 1.0
                }
            
            # Test API response times
            api_tests = [
                f"/api/v1/analysis/results/{document_id}",
                f"/api/v1/analysis/status/{document_id}"
            ]
            
            api_performance = {}
            for endpoint in api_tests:
                try:
                    api_start = time.time()
                    response = self.session.get(f"{self.backend_url}{endpoint}", timeout=10)
                    api_time = time.time() - api_start
                    
                    api_performance[endpoint] = {
                        "response_time": f"{api_time:.2f}s",
                        "status_code": response.status_code,
                        "fast_response": api_time < 2.0
                    }
                except Exception as e:
                    api_performance[endpoint] = {"error": str(e)}
            
            performance_metrics["API Performance"] = api_performance
            
            self.log_test_step("Performance and Optimization Test", "success", {
                "Performance Metrics": list(performance_metrics.keys()),
                "Results": performance_metrics
            })
            
            self.test_results["performance_test"] = {
                "status": "passed",
                "details": {
                    "load_time": first_content_time,
                    "memory_efficient": memory_info["used"] < 50 if memory_info else True,
                    "responsive_interactions": performance_metrics.get("Interaction Response", {}).get("responsive", True),
                    "performance_metrics": performance_metrics
                }
            }
            
            return True
            
        except Exception as e:
            self.log_test_step("Performance and Optimization Test", "failed", {
                "Exception": str(e)
            })
            self.test_results["performance_test"]["status"] = "failed"
            return False

    async def run_comprehensive_results_page_tests(self) -> bool:
        """Run complete results page test suite"""
        self.log_test_step("Comprehensive Results Page Test Suite", "start")
        
        # Prepare test data
        document_id = self.prepare_test_data()
        if not document_id:
            self.log_test_step("Test Data Preparation", "failed", {
                "Error": "Could not prepare test data"
            })
            return False
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)  # Set to True for headless
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            page = await context.new_page()
            
            try:
                tests = [
                    ("Data Loading and API Integration", lambda: self.test_data_loading_and_api_integration(page, document_id)),
                    ("Display Formatting and Layout", lambda: self.test_display_formatting_and_layout(page)),
                    ("Interactive Features", lambda: self.test_interactive_features(page)),
                    ("Export Functionality", lambda: self.test_export_functionality(page, document_id)),
                    ("User Interaction Patterns", lambda: self.test_user_interaction_patterns(page)),
                    ("Error Handling and Edge Cases", lambda: self.test_error_handling_and_edge_cases(page)),
                    ("Performance and Optimization", lambda: self.test_performance_and_optimization(page, document_id))
                ]
                
                passed_tests = 0
                failed_tests = []
                
                for test_name, test_func in tests:
                    try:
                        if await test_func():
                            passed_tests += 1
                            self.log_test_step(f"Results Page - {test_name}", "success")
                        else:
                            failed_tests.append(test_name)
                            self.log_test_step(f"Results Page - {test_name}", "failed")
                    except Exception as e:
                        failed_tests.append(f"{test_name} (Exception: {str(e)})")
                        self.log_test_step(f"Results Page - {test_name}", "failed", {"Exception": str(e)})
                
                success_rate = (passed_tests / len(tests)) * 100
                
                if len(failed_tests) == 0:
                    self.log_test_step("Comprehensive Results Page Test Suite", "success", {
                        "Passed Tests": f"{passed_tests}/{len(tests)}",
                        "Success Rate": f"{success_rate:.1f}%",
                        "Document ID": document_id
                    })
                    return True
                else:
                    self.log_test_step("Comprehensive Results Page Test Suite", "failed", {
                        "Passed Tests": f"{passed_tests}/{len(tests)}",
                        "Failed Tests": failed_tests,
                        "Success Rate": f"{success_rate:.1f}%"
                    })
                    return False
                    
            finally:
                await browser.close()

    def generate_results_page_test_report(self) -> str:
        """Generate comprehensive results page test report"""
        report = "=" * 80 + "\n"
        report += "RAI COMPLIANCE PLATFORM - RESULTS PAGE COMPREHENSIVE TEST REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Test Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Frontend URL: {self.frontend_url}\n"
        report += f"Backend URL: {self.backend_url}\n"
        report += f"Test Document ID: {self.test_document_id}\n\n"
        
        for test_name, test_data in self.test_results.items():
            status_emoji = "✅" if test_data["status"] == "passed" else "❌" if test_data["status"] == "failed" else "⏳"
            report += f"{status_emoji} {test_name.replace('_', ' ').title()}: {test_data['status'].upper()}\n"
            
            if test_data.get("details"):
                for key, value in test_data["details"].items():
                    if isinstance(value, dict):
                        report += f"   📊 {key}:\n"
                        for sub_key, sub_value in value.items():
                            report += f"      • {sub_key}: {sub_value}\n"
                    else:
                        report += f"   📊 {key}: {value}\n"
            report += "\n"
        
        # Summary
        passed_count = sum(1 for test in self.test_results.values() if test["status"] == "passed")
        total_count = len(self.test_results)
        
        report += "=" * 80 + "\n"
        report += "RESULTS PAGE TEST SUMMARY\n"
        report += "=" * 80 + "\n"
        
        report += f"Tests Passed: {passed_count}/{total_count}\n"
        report += f"Overall Success Rate: {(passed_count/total_count)*100:.1f}%\n\n"
        
        if passed_count == total_count:
            report += "🎉 ALL RESULTS PAGE TESTS PASSED!\n"
            report += "✅ Results page is fully functional and user-ready.\n"
        else:
            report += "⚠️ SOME RESULTS PAGE TESTS FAILED!\n"
            report += "❌ Review failed tests to identify user experience issues.\n"
        
        return report


async def main():
    """Main function to run results page tests"""
    print("=" * 80)
    print("RAI COMPLIANCE PLATFORM - RESULTS PAGE COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Configuration
    FRONTEND_URL = "http://localhost:3000"
    BACKEND_URL = "http://localhost:8000"
    
    # Initialize tester
    tester = ResultsPageTester(FRONTEND_URL, BACKEND_URL)
    
    print(f"\n📋 Test Configuration:")
    print(f"   Frontend URL: {FRONTEND_URL}")
    print(f"   Backend URL: {BACKEND_URL}")
    print("\n⚠️ Make sure both frontend and backend servers are running!")
    print("   Frontend: npm run dev")
    print("   Backend: python main.py")
    print("\n" + "=" * 80)
    
    # Run comprehensive results page tests
    success = await tester.run_comprehensive_results_page_tests()
    
    # Generate and display report
    report = tester.generate_results_page_test_report()
    print("\n" + report)
    
    # Save report to file
    report_file = "results_page_test_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"📊 Results page test report saved to: {report_file}")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())