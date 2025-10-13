"""
Compliance Analyzer Service - Extracted from AI.py

Handles compliance analysis against checklist questions using the exclusive pipeline:
- Financial Statement Detector (mandatory)
- Standard Identifier integration 
- Smart Metadata Extractor
- Azure OpenAI analysis with proper response formatting
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI  # type: ignore

from services.ai_prompts import ai_prompts
from services.hybrid_financial_detector import detect_financial_statements_hybrid
from services.rate_limiter import (
    check_rate_limit_with_backoff,
    check_duplicate_question,
    record_failure,
    reset_circuit_breaker,
    RateLimitError,
    CircuitBreakerOpenError,
    EXPONENTIAL_BACKOFF_BASE,
)

logger = logging.getLogger(__name__)


class ComplianceAnalyzer:
    """
    Handles compliance analysis for checklist questions using exclusive pipeline approach.
    """
    
    def __init__(self, openai_client: AzureOpenAI, deployment_name: str):
        self.openai_client = openai_client
        self.deployment_name = deployment_name
        
    def analyze_chunk(
        self, 
        chunk: str, 
        question: str, 
        standard_id: Optional[str] = None,
        current_document_id: Optional[str] = None,
        standard_specific_context_func = None,
        enhanced_evidence = None
    ) -> dict:
        """
        Analyze a chunk of annual report content against a compliance checklist question.
        Enhanced with intelligent document analysis, rate limiting, and duplicate prevention.
        Returns a JSON response with:
          - status: "YES", "NO", or "N/A"
          - confidence: float between 0 and 1
          - explanation: str explaining the analysis
          - evidence: str containing relevant text from the document
          - suggestion: str containing suggestion when status is "NO"
        """
        try:
            # Ensure document_id is set
            if not current_document_id:
                logger.error(
                    "analyze_chunk called with no current_document_id set. Cannot perform vector search."
                )
                return {
                    "status": "Error",
                    "confidence": 0.0,
                    "explanation": "No document ID provided for vector search.",
                    "evidence": "",
                    "suggestion": "Check backend logic to ensure document_id is always set.",
                }

            # Check for duplicate questions
            if check_duplicate_question(question, current_document_id):
                logger.warning(
                    f"Skipping duplicate question for document {current_document_id}"
                )
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": "This question has already been processed for this document.",
                    "evidence": "Duplicate question detected.",
                    "suggestion": "Question already analyzed - check previous results.",
                }

            # Check circuit breaker before proceeding
            try:
                from services.rate_limiter import check_circuit_breaker
                check_circuit_breaker()
            except CircuitBreakerOpenError as e:
                logger.error(f"Circuit breaker is open: {str(e)}")
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": f"Analysis temporarily unavailable: {str(e)}",
                    "evidence": "Circuit breaker open due to repeated failures.",
                    "suggestion": "Please wait and retry the analysis later.",
                }

            # 🎯 EXCLUSIVE PIPELINE: ONLY Financial Statement Detector + Standard Identifier + Smart Metadata Extractor
            context = ""
            financial_content = None
            
            # STEP 1: Get document text for processing
            document_text_for_processing = None
            
            # Option A: Use provided chunk if substantial
            if chunk and len(chunk) > 1000:
                document_text_for_processing = chunk
                logger.info(f"🎯 Using provided document chunk ({len(chunk)} chars)")
            
            # Option B: Extract full document text if we have document_id
            elif current_document_id:
                try:
                    # Import the document text extraction function
                    from routes.analysis_routes import _extract_document_text
                    
                    extracted_text = _extract_document_text(current_document_id)
                    if isinstance(extracted_text, str) and len(extracted_text) > 1000:
                        document_text_for_processing = extracted_text
                        logger.info(f"🎯 Extracted full document text ({len(extracted_text)} chars)")
                    else:
                        logger.error(f"❌ Document text extraction failed or insufficient content")
                        return {
                            "status": "N/A",
                            "confidence": 0.0,
                            "explanation": "Could not extract document text for analysis",
                            "evidence": "",
                            "suggestion": "Ensure document is properly uploaded and processed.",
                        }
                except Exception as e:
                    logger.error(f"❌ Could not extract document text: {str(e)}")
                    return {
                        "status": "N/A",
                        "confidence": 0.0,
                        "explanation": f"Document text extraction failed: {str(e)}",
                        "evidence": "",
                        "suggestion": "Check document upload and processing status.",
                    }
            
            # MANDATORY: Document text is required for exclusive pipeline
            if not document_text_for_processing:
                logger.error("❌ EXCLUSIVE PIPELINE FAILURE: No document text available")
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": "No document content available for exclusive pipeline processing",
                    "evidence": "",
                    "suggestion": "Upload and process document before analysis.",
                }
            
            # STEP 2: MANDATORY Financial Statement Detection
            try:
                logger.info(f"🎯 EXCLUSIVE PIPELINE: Hybrid Financial Statement Detector (MANDATORY)")
                
                # Use hybrid financial statement detection
                financial_content = detect_financial_statements_hybrid(
                    document_text=document_text_for_processing, 
                    document_id=current_document_id
                )
                
                if financial_content and financial_content.statements:
                    # Extract validated financial statement content for AI analysis
                    validated_financial_content = self._get_content_for_compliance_analysis(financial_content)
                    
                    # EXTRACT STRUCTURED FINANCIAL DATA FOR JSON OUTPUT
                    try:
                        logger.info("📊 Extracting structured financial data for JSON compliance analysis...")
                        structured_financial_data = self._extract_structured_financial_data(financial_content)
                        
                        if structured_financial_data and structured_financial_data.get("financial_data"):
                            logger.info(f"✅ STRUCTURED FINANCIAL DATA EXTRACTED: {len(structured_financial_data['financial_data'])} statements")
                            
                            # Store structured data for inclusion in final result
                            self.structured_financial_data = structured_financial_data
                        else:
                            logger.warning("⚠️ No structured financial data could be extracted")
                            self.structured_financial_data = {"financial_data": [], "extraction_status": "no_data"}
                    
                    except Exception as extraction_error:
                        logger.error(f"❌ Structured financial data extraction failed: {extraction_error}")
                        self.structured_financial_data = {"financial_data": [], "extraction_status": "extraction_failed", "error": str(extraction_error)}
                    
                    if validated_financial_content and not validated_financial_content.startswith("ERROR:"):
                        # SUCCESS: Use ONLY validated financial statement content
                        context = validated_financial_content
                        logger.info(f"✅ FINANCIAL STATEMENTS EXTRACTED: {len(validated_financial_content)} characters")
                        logger.info(f"📊 Content type: {financial_content.content_type}, Confidence: {financial_content.total_confidence:.1f}%")
                        logger.info(f"📋 Statements: {[stmt.statement_type for stmt in financial_content.statements]}")
                    else:
                        logger.error(f"❌ Financial statement detector error: {validated_financial_content}")
                        return {
                            "status": "N/A",
                            "confidence": 0.0,
                            "explanation": "Financial statement detection failed - no valid financial content found",
                            "evidence": "",
                            "suggestion": "Ensure document contains proper financial statements (Balance Sheet, Income Statement, Cash Flow).",
                        }
                else:
                    logger.error("❌ No financial statements detected in document")
                    return {
                        "status": "N/A",
                        "confidence": 0.0,
                        "explanation": "No financial statements found in document",
                        "evidence": "",
                        "suggestion": "Upload a document containing financial statements (Balance Sheet, Income Statement, Statement of Cash Flows).",
                    }
                    
            except Exception as e:
                logger.error(f"❌ Financial statement detector critical failure: {str(e)}", exc_info=True)
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": f"Financial statement detection system failed: {str(e)}",
                    "evidence": "",
                    "suggestion": "Contact system administrator - financial statement detector is not functioning.",
                }

            # STEP 3: MANDATORY Standard Identifier Integration (Notes to Accounts)
            logger.info(f"🎯 EXCLUSIVE PIPELINE: Standard Identifier integration for {standard_id}")
            standard_context = ""
            if standard_specific_context_func:
                standard_context = standard_specific_context_func(question, standard_id)
            
            if standard_context:
                # Combine financial statements with standard-specific Notes content
                context = f"=== FINANCIAL STATEMENTS ===\n{context}\n\n=== STANDARD-SPECIFIC NOTES ({standard_id}) ===\n{standard_context}"
                logger.info(f"✅ Enhanced with {len(standard_context)} chars of standard-specific context")
            else:
                logger.warning(f"⚠️ No standard-specific context found for {standard_id} - using financial statements only")
            
            # FINAL VALIDATION: Ensure we have context
            if not context:
                logger.error("❌ EXCLUSIVE PIPELINE FAILURE: No context generated")
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": "Exclusive pipeline failed to generate any context for analysis",
                    "evidence": "",
                    "suggestion": "Check financial statement detector and standard identifier systems.",
                }

            # 🎯 DIRECT PERSISTENT STORAGE APPROACH: Get ALL context from persistent storage
            # No file dependencies - uses database storage that persists on Render
            if current_document_id:
                try:
                    from services.persistent_storage import get_ai_context_for_standard
                    
                    # Get complete AI context with metadata, financial statements, and standard chunks
                    complete_context = asyncio.run(get_ai_context_for_standard(
                        document_id=current_document_id,
                        standard_id=standard_id or "UNKNOWN", 
                        question=question
                    ))
                    
                    if complete_context and not complete_context.startswith("ERROR:"):
                        # Use the complete persistent context
                        context = complete_context
                        logger.info(f"🎯 USING PERSISTENT STORAGE CONTEXT: {len(complete_context)} characters of complete document context")
                        logger.info(f"📋 Context includes: metadata, financial statements, standard-specific chunks")
                    else:
                        logger.warning(f"⚠️ No persistent context available - falling back to vector search context")
                        # Keep existing vector search context as fallback
                        
                except Exception as e:
                    logger.error(f"❌ Failed to get persistent context: {str(e)}", exc_info=True)
                    # Keep existing vector search context as fallback

            # Construct the prompt for the AI using the prompts library
            prompt = ai_prompts.get_full_compliance_analysis_prompt(
                question=question, context=context, enhanced_evidence=enhanced_evidence
            )

            # 📊 EXCLUSIVE PIPELINE LOGGING 
            logger.info("=" * 80)
            logger.info("🎯 EXCLUSIVE PIPELINE - FINANCIAL DETECTOR + STANDARD IDENTIFIER ONLY")
            logger.info("=" * 80)
            logger.info(f"📄 DOCUMENT_ID: {current_document_id}")
            logger.info(f"📋 QUESTION: {question}")
            logger.info(f"🎯 STANDARD_ID: {standard_id}")
            logger.info(f"📝 TOTAL_CONTEXT_LENGTH: {len(context)} characters")
            
            # 🎯 EXCLUSIVE CONTENT SOURCE LOGGING
            logger.info(f"🏦 FINANCIAL_CONTENT_TYPE: {financial_content.content_type}")
            logger.info(f"📊 FINANCIAL_CONFIDENCE: {financial_content.total_confidence:.1f}%")
            logger.info(f"📋 STATEMENTS_DETECTED: {[stmt.statement_type for stmt in financial_content.statements]}")
            logger.info(f"🧠 MODEL: {self.deployment_name}")
            logger.info(f"⚡ PIPELINE_MODE: EXCLUSIVE (No vector fallbacks)")
            
            # Log context structure for debugging
            if "STANDARD-SPECIFIC NOTES" in context:
                fs_length = context.find("=== STANDARD-SPECIFIC NOTES") 
                standard_length = len(context) - fs_length
                logger.info(f"📖 CONTEXT_STRUCTURE: Financial Statements ({fs_length} chars) + Standard Notes ({standard_length} chars)")
            else:
                logger.info(f"📖 CONTEXT_STRUCTURE: Financial Statements only ({len(context)} chars)")
            
            # Log excerpt for verification
            context_excerpt = context[:500] + "..." if len(context) > 500 else context
            logger.info(f"📖 CONTEXT_EXCERPT: {context_excerpt}")

            # Enhanced rate limiting with retries and circuit breaker
            max_api_retries = 3
            content = None
            for api_retry in range(max_api_retries):
                try:
                    # RATE LIMITING REMOVED - Process all 217 questions without limits
                    # check_rate_limit_with_backoff(tokens=estimated_tokens)

                    # Get AI response
                    response = self.openai_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {
                                "role": "system",
                                "content": ai_prompts.get_compliance_analysis_system_prompt(),
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )

                    # If we get here, the API call was successful
                    reset_circuit_breaker()  # Reset circuit breaker on success
                    content = response.choices[0].message.content
                    
                    # 📊 PRODUCTION LOGGING - Log the AI response
                    logger.info(f"✅ AI_RESPONSE_RECEIVED: {len(content) if content else 0} characters")
                    if content:
                        content_excerpt = content[:300] + "..." if len(content) > 300 else content
                        logger.info(f"💬 RESPONSE_EXCERPT: {content_excerpt}")
                    logger.info("=" * 80)
                    break

                except Exception as api_error:
                    error_message = str(api_error).lower()

                    if "429" in error_message or "rate limit" in error_message:
                        record_failure()
                        if api_retry < max_api_retries - 1:
                            backoff_time = EXPONENTIAL_BACKOFF_BASE ** (api_retry + 1)
                            logger.warning(
                                f"Rate limit hit, retrying in {backoff_time} seconds (attempt {api_retry + 1}/{max_api_retries})"
                            )
                            time.sleep(backoff_time)
                            continue
                        else:
                            logger.error(
                                "Max API retries exceeded due to rate limiting"
                            )
                            return {
                                "status": "N/A",
                                "confidence": 0.0,
                                "explanation": "Analysis could not be completed due to persistent API rate limits.",
                                "evidence": "Multiple rate limit errors occurred.",
                                "suggestion": "Please retry the analysis later when API rate limits reset.",
                            }

                    elif "timeout" in error_message or "connection" in error_message:
                        record_failure()
                        if api_retry < max_api_retries - 1:
                            backoff_time = EXPONENTIAL_BACKOFF_BASE**api_retry
                            logger.warning(
                                f"Connection error, retrying in {backoff_time} seconds (attempt {api_retry + 1}/{max_api_retries})"
                            )
                            time.sleep(backoff_time)
                            continue
                        else:
                            logger.error(
                                "Max API retries exceeded due to connection issues"
                            )
                            return {
                                "status": "N/A",
                                "confidence": 0.0,
                                "explanation": f"Analysis failed due to connection issues: {str(api_error)}",
                                "evidence": "Connection error occurred.",
                                "suggestion": "Please check network connectivity and retry the analysis.",
                            }

                    else:
                        # Other API error - don't retry
                        record_failure()
                        logger.error(f"Non-retryable API error: {str(api_error)}")
                        return {
                            "status": "N/A",
                            "confidence": 0.0,
                            "explanation": f"Analysis failed due to API error: {str(api_error)}",
                            "evidence": "API error occurred.",
                            "suggestion": "Please retry the analysis or check system logs for details.",
                        }
            else:
                # This should not happen due to the break statement, but just in case
                logger.error("Unexpected exit from retry loop")
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": "Unexpected error during API retry loop",
                    "evidence": "",
                    "suggestion": "Please retry the analysis.",
                }

            # Console log the API response for each question
            logger.info("=" * 80)
            logger.info("📋 CHECKLIST QUESTION API RESPONSE")
            logger.info("=" * 80)
            logger.info(
                f"🔍 Question: {question[:100]}{'...' if len(question) > 100 else ''}"
            )
            logger.info(
                f"📄 Standard ID: {standard_id or 'Not specified'} (from analyze_chunk)"
            )
            logger.info(f"🎯 Document ID: {current_document_id}")
            logger.info("-" * 40)
            logger.info("📤 RAW API RESPONSE:")
            logger.info(content)
            logger.info("-" * 40)

            # Process and format the AI response
            result = self._process_ai_response(content, enhanced_evidence, relevant_chunks=None)
            
            # Log the processed result
            self._log_processed_result(result)
            
            return result

        except Exception as e:
            logger.error(f"Error in analyze_chunk: {str(e)}")
            return {
                "status": "N/A",
                "confidence": 0.0,
                "explanation": f"Error during analysis: {str(e)}",
                "evidence": "",
                "suggestion": "Consider adding explicit disclosure addressing this requirement.",
            }

    def _process_ai_response(self, content: str, enhanced_evidence = None, relevant_chunks = None) -> dict:
        """
        Process and format the AI response into structured result.
        """
        # Handle potential None content
        if content is None:
            logger.error("❌ OpenAI API returned None content")
            return {
                "status": "Error",
                "confidence": 0.0,
                "explanation": "OpenAI API returned empty response",
                "evidence": "",
                "suggestion": "Please retry the analysis.",
            }

        # Try to parse JSON from the response
        try:
            # First, try to parse as JSON directly
            result = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            logger.warning(
                f"Response is not valid JSON, attempting to extract structured data: {content[:100]}..."
            )

            # Extract using regex-like approach for key fields
            result = {}

            # Extract status
            if "YES" in content or "Yes" in content:
                result["status"] = "YES"
            elif "NO" in content or "No" in content:
                result["status"] = "NO"
            else:
                result["status"] = "N/A"

            # Extract confidence (look for number between 0 and 1)
            confidence_match = re.search(r"[Cc]onfidence:?\s*(\d+\.\d+)", content)
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1))
            else:
                result["confidence"] = 0.5

            # Extract explanation
            explanation_match = re.search(
                r"[Ee]xplanation:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", content, re.DOTALL
            )
            if explanation_match:
                result["explanation"] = explanation_match.group(1).strip()
            else:
                result["explanation"] = "No detailed explanation provided."

            # Extract evidence
            if "|" in content:
                evidence_lines = re.findall(
                    r"IAS \d+\.\d+\(?\w?\)?\s*\|.+", content
                )
                if evidence_lines:
                    result["evidence"] = evidence_lines
                else:
                    result["evidence"] = ["No structured evidence provided."]
            else:
                result["evidence"] = [
                    "Evidence not provided in the required format."
                ]

            # Extract suggestion if status is NO
            if result["status"] == "NO":
                suggestion_match = re.search(
                    r"[Ss]uggestion:?:?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                    content,
                    re.DOTALL,
                )
                if suggestion_match:
                    result["suggestion"] = suggestion_match.group(1).strip()
                else:
                    result["suggestion"] = (
                        "Provide a concrete, practical sample disclosure that would satisfy this IAS 40 requirement. For example: 'The entity should disclose [required information] in accordance with IAS 40.[paragraph]'."
                    )

            # Extract content_analysis
            content_analysis_match = re.search(
                r"[Cc]ontent_analysis:?\s*[\"']?(.+?)[\"']?(?:\n\n|\n[A-Z]|,\s*\"|\",|$)",
                content,
                re.DOTALL,
            )
            if content_analysis_match:
                result["content_analysis"] = content_analysis_match.group(1).strip()
            else:
                result["content_analysis"] = "No detailed content analysis provided."

            # Extract disclosure_recommendations
            disclosure_recommendations = []
            # Look for array format ["recommendation1", "recommendation2"]
            disclosure_array_match = re.search(
                r"[Dd]isclosure_recommendations:?\s*\[(.*?)\]",
                content,
                re.DOTALL,
            )
            if disclosure_array_match:
                # Extract individual recommendations from the array
                array_content = disclosure_array_match.group(1)
                recommendations = re.findall(r'["\']([^"\']+)["\']', array_content)
                disclosure_recommendations.extend(recommendations)
            
            if not disclosure_recommendations:
                # Look for single recommendation format
                single_disclosure_match = re.search(
                    r"[Dd]isclosure_recommendations?:?\s*[\"']?(.+?)[\"']?(?:\n\n|\n[A-Z]|,\s*\"|\",|$)",
                    content,
                    re.DOTALL,
                )
                if single_disclosure_match:
                    disclosure_recommendations.append(single_disclosure_match.group(1).strip())
            
            result["disclosure_recommendations"] = disclosure_recommendations if disclosure_recommendations else [
                "Consider enhancing the disclosure to provide more comprehensive information addressing this requirement."
            ]

        # Validate and clean the response
        if "status" not in result or result["status"] not in ["YES", "NO", "N/A"]:
            result["status"] = "N/A"
        if "confidence" not in result:
            result["confidence"] = 0.5
        if "explanation" not in result:
            result["explanation"] = "No explanation provided"
        if "evidence" not in result:
            result["evidence"] = ""
        
        # Validate new enhanced fields
        if "content_analysis" not in result:
            result["content_analysis"] = "No detailed content analysis provided."
        if "disclosure_recommendations" not in result:
            result["disclosure_recommendations"] = [
                "Consider enhancing the disclosure to provide more comprehensive information addressing this requirement."
            ]
        elif not isinstance(result["disclosure_recommendations"], list):
            # Convert single string to list if needed
            result["disclosure_recommendations"] = [str(result["disclosure_recommendations"])]

        # Ensure evidence is structured and meaningful
        if result["evidence"] == "" or all(
            e == "N/A | N/A | N/A | N/A" for e in result["evidence"]
        ):
            result["evidence"] = ["No relevant evidence found in the document."]

        # Add a suggestion when status is "NO" and no suggestion provided
        if result["status"] == "NO" and "suggestion" not in result:
            result["suggestion"] = (
                "Provide a concrete, practical sample disclosure that would satisfy this IAS 40 requirement. For example: 'The entity should disclose [required information] in accordance with IAS 40.[paragraph]'."
            )

        # Add enhanced evidence metadata if available
        if enhanced_evidence:
            result["enhanced_analysis"] = {
                "evidence_quality_score": enhanced_evidence.get(
                    "evidence_quality_assessment", {}
                ).get("overall_quality", 0),
                "confidence_level": enhanced_evidence.get(
                    "evidence_quality_assessment", {}
                ).get("confidence_level", 0.0),
                "source_type": enhanced_evidence.get(
                    "evidence_quality_assessment", {}
                ).get("source_type", "unknown"),
                "is_policy_based": enhanced_evidence.get(
                    "evidence_quality_assessment", {}
                ).get("is_policy_based", True),
                "evidence_source": enhanced_evidence.get(
                    "evidence_quality_assessment", {}
                ).get("evidence_source", "Unknown"),
                "recommendation": enhanced_evidence.get("analysis_summary", {}).get(
                    "recommendation", "Manual review recommended"
                ),
            }

        # Add page number information from vector search results
        if relevant_chunks:
            page_numbers = []
            document_sources = []
            document_extracts = []
            for chunk_data in relevant_chunks:
                if chunk_data.get("page_number") and chunk_data["page_number"] > 0:
                    page_numbers.append(chunk_data["page_number"])
                if chunk_data.get("chunk_index") is not None:
                    document_sources.append({
                        "page": chunk_data.get("page_number", 0),
                        "chunk_index": chunk_data.get("chunk_index", 0),
                        "chunk_type": chunk_data.get("chunk_type", "text")
                    })
                # Include document extracts with page references
                if chunk_data.get("text"):
                    extract_text = chunk_data["text"][:500] + "..." if len(chunk_data["text"]) > 500 else chunk_data["text"]
                    document_extracts.append({
                        "text": extract_text,
                        "page": chunk_data.get("page_number", 0),
                        "chunk_index": chunk_data.get("chunk_index", 0),
                        "relevance_score": chunk_data.get("score", 0.0)
                    })
            
            if page_numbers:
                result["source_pages"] = sorted(list(set(page_numbers)))
            if document_sources:
                result["document_sources"] = document_sources
            if document_extracts:
                result["document_extracts"] = document_extracts

        # Include structured financial data if available
        if hasattr(self, 'structured_financial_data') and self.structured_financial_data:
            result["structured_financial_data"] = self.structured_financial_data
            logger.info(f"📊 Added structured financial data to compliance result: {len(self.structured_financial_data.get('financial_data', []))} statements")

        return result

    def _log_processed_result(self, result: dict):
        """
        Log the processed analysis result for monitoring.
        """
        logger.info("✅ PROCESSED RESULT:")
        logger.info(f"   Status: {result.get('status', 'N/A')}")
        logger.info(f"   Confidence: {result.get('confidence', 0.0):.2f}")
        logger.info(
            f"   Explanation: {result.get('explanation', 'N/A')[:150]}{'...' if len(str(result.get('explanation', ''))) > 150 else ''}"
        )
        evidence = result.get('evidence', [])
        evidence_count = len(evidence) if isinstance(evidence, list) else 1
        logger.info(
            f"   Evidence Count: {evidence_count}"
        )
        if result.get("content_analysis"):
            logger.info(
                f"   Content Analysis: {result.get('content_analysis', 'N/A')[:100]}{'...' if len(str(result.get('content_analysis', ''))) > 100 else ''}"
            )
        if result.get("disclosure_recommendations"):
            logger.info(
                f"   Disclosure Recommendations: {len(result.get('disclosure_recommendations', []))} suggestions"
            )
        if result.get("source_pages"):
            logger.info(f"   Source Pages: {result.get('source_pages')}")
        if result.get("document_sources"):
            source_info = [f"p{s['page']}" for s in result.get('document_sources', [])]
            logger.info(f"   Document Sources: {', '.join(source_info)}")
        if result.get("document_extracts"):
            extract_count = len(result.get('document_extracts', []))
            avg_score = sum(e.get('relevance_score', 0) for e in result.get('document_extracts', [])) / extract_count if extract_count > 0 else 0
            logger.info(f"   Document Extracts: {extract_count} chunks, avg relevance: {avg_score:.3f}")
        if result.get("status") == "NO" and result.get("suggestion"):
            logger.info(
                f"   Suggestion: {result.get('suggestion', 'N/A')[:100]}{'...' if len(str(result.get('suggestion', ''))) > 100 else ''}"
            )
        if result.get("enhanced_analysis"):
            quality_score = result.get('enhanced_analysis', {}).get('evidence_quality_score', 0)
            logger.info(
                f"   Enhanced Analysis: Quality Score {quality_score}/100"
            )
        logger.info("=" * 80)

    def calculate_adequacy(self, confidence: float, has_evidence: bool, status: str) -> str:
        """
        Calculate adequacy score based on confidence, evidence, and status.
        """
        if status == "N/A":
            return "low"
        if confidence >= 0.8 and has_evidence:
            return "high"
        elif confidence >= 0.6 or has_evidence:
            return "medium"
        else:
            return "low"
    
    def _get_content_for_compliance_analysis(self, financial_content) -> str:
        """Extract validated financial statement content for compliance analysis"""
        try:
            if not financial_content or not financial_content.statements:
                return "ERROR: No financial statements found"
            
            # Combine all statement content with metadata
            content_parts = []
            content_parts.append(f"Financial Statement Analysis Summary:")
            content_parts.append(f"Total Confidence: {financial_content.total_confidence:.1f}%")
            content_parts.append(f"Content Type: {financial_content.content_type}")
            content_parts.append(f"Validation Summary: {financial_content.validation_summary}")
            content_parts.append("")
            
            for i, stmt in enumerate(financial_content.statements, 1):
                content_parts.append(f"Statement {i}: {stmt.statement_type}")
                content_parts.append(f"Confidence: {stmt.confidence_score:.1f}%")
                if hasattr(stmt, 'strategy_used'):
                    content_parts.append(f"Strategy: {stmt.strategy_used.value}")
                if hasattr(stmt, 'content'):
                    # Truncate content to reasonable length for analysis
                    content_preview = stmt.content[:2000] + "..." if len(stmt.content) > 2000 else stmt.content
                    content_parts.append(f"Content Preview: {content_preview}")
                content_parts.append("-" * 50)
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error extracting content for compliance analysis: {e}")
            return f"ERROR: Failed to extract content - {str(e)}"
    
    def _extract_structured_financial_data(self, financial_content) -> dict:
        """Extract structured financial data for JSON output"""
        try:
            if not financial_content or not financial_content.statements:
                return {"financial_data": [], "extraction_status": "no_statements"}
            
            structured_data = {
                "financial_data": [],
                "extraction_status": "success",
                "total_confidence": financial_content.total_confidence,
                "content_type": financial_content.content_type,
                "validation_summary": financial_content.validation_summary
            }
            
            # Add hybrid-specific metadata if available
            if hasattr(financial_content, 'strategy_breakdown'):
                structured_data["strategy_breakdown"] = financial_content.strategy_breakdown
            
            if hasattr(financial_content, 'processing_metrics'):
                structured_data["processing_metrics"] = financial_content.processing_metrics
            
            for stmt in financial_content.statements:
                stmt_data = {
                    "statement_type": stmt.statement_type,
                    "confidence_score": stmt.confidence_score if hasattr(stmt, 'confidence_score') else 0.0
                }
                
                # Add hybrid-specific data
                if hasattr(stmt, 'strategy_used'):
                    stmt_data["strategy_used"] = stmt.strategy_used.value
                if hasattr(stmt, 'pattern_confidence'):
                    stmt_data["pattern_confidence"] = stmt.pattern_confidence
                if hasattr(stmt, 'ai_confidence'):
                    stmt_data["ai_confidence"] = stmt.ai_confidence
                if hasattr(stmt, 'validation_markers'):
                    stmt_data["validation_markers"] = stmt.validation_markers
                
                structured_data["financial_data"].append(stmt_data)
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error extracting structured financial data: {e}")
            return {
                "financial_data": [],
                "extraction_status": "extraction_failed",
                "error": str(e)
            }