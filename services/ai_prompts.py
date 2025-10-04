"""
AI Prompts Library for RAi Compliance Engine

This module contains all AI prompts used throughout the compliance analysis system.
Centralizing prompts here makes them easier to find, modify, and maintain.

Version 2.0 - Enhanced for 10/10 Clarity, Quality, Preciseness, and Effectiveness
"""

from typing import Any, Dict, List, Optional


class PromptConfiguration:
    """Master configuration ensuring consistency across all prompts."""

    # Global Standards
    RESPONSE_FORMAT = "JSON_ONLY"
    CONFIDENCE_SCALE = "0.0_TO_1.0"
    STATUS_VALUES = ["YES", "NO", "N/A"]
    EVIDENCE_GRADES = ["A", "B", "C", "D"]

    # Metadata Extraction Policy
    METADATA_POLICY = "EXTRACT_OR_INFER"  # Never return "Not found"
    INFERENCE_THRESHOLD = 0.5

    # Compliance Analysis Policy
    COMPLIANCE_STRICTNESS = "CONSERVATIVE"  # Prefer NO over uncertain YES
    EVIDENCE_MINIMUM = 1  # Minimum evidence items required

    # Output Formatting Constraints
    MAX_EXPLANATION_WORDS = 50
    MAX_SUGGESTION_WORDS = 200
    MAX_QUOTE_CHARS = 100
    MAX_EVIDENCE_ITEMS = 5

    # Confidence Score Thresholds
    CONFIDENCE_EXPLICIT = 0.9  # Direct quotes/statements
    CONFIDENCE_STRONG_IMPLICIT = 0.7  # Clear implications
    CONFIDENCE_WEAK_IMPLICIT = 0.5  # Weak implications
    CONFIDENCE_MINIMAL = 0.3  # Minimal evidence

    @classmethod
    def get_confidence_criteria(cls) -> str:
        """Objective confidence scoring criteria."""
        return (
            f"CONFIDENCE SCORING CRITERIA:\n"
            f"• {cls.CONFIDENCE_EXPLICIT}-1.0: Direct quote or explicit "
            f"statement matching requirement\n"
            f"• {cls.CONFIDENCE_STRONG_IMPLICIT}-{cls.CONFIDENCE_EXPLICIT - 0.01}: "
            f"Clear implicit compliance with supporting evidence\n"
            f"• {cls.CONFIDENCE_WEAK_IMPLICIT}-{cls.CONFIDENCE_STRONG_IMPLICIT - 0.01}: "
            f"Partial compliance or ambiguous evidence\n"
            f"• {cls.CONFIDENCE_MINIMAL}-{cls.CONFIDENCE_WEAK_IMPLICIT - 0.01}: "
            f"Weak evidence or unclear compliance\n"
            f"• 0.1-{cls.CONFIDENCE_MINIMAL - 0.01}: "
            f"Minimal evidence suggesting non-compliance\n"
            f"• 0.0-0.09: Clear non-compliance or complete absence\n"
        )

    @classmethod
    def get_evidence_quality_standards(cls) -> str:
        """Objective evidence quality classification."""
        return (
            "EVIDENCE QUALITY STANDARDS:\n"
            "• Grade A (EXPLICIT): Direct quotation from financial statements/notes\n"
            "• Grade B (POLICY): Accounting policy statement addressing requirement\n"
            "• Grade C (RELATED): Related disclosure implying compliance\n"
            "• Grade D (INDIRECT): Indirect reference or partial information\n"
        )


class PromptValidator:
    """Quality assurance for prompt effectiveness."""

    AMBIGUOUS_TERMS = [
        "best guess",
        "try to",
        "might be",
        "could be",
        "perhaps",
        "reasonable",
        "appropriate",
        "adequate",
        "suitable",
        "proper",
    ]

    PRECISE_INDICATORS = [
        "exact",
        "specific",
        "must",
        "required",
        "only",
        "shall",
        "mandatory",
        "precisely",
        "explicitly",
    ]

    VAGUE_INDICATORS = [
        "approximately",
        "about",
        "roughly",
        "generally",
        "typically",
        "usually",
        "often",
        "sometimes",
        "may",
        "might",
    ]

    @classmethod
    def score_clarity(cls, prompt: str) -> float:
        """Score prompt clarity (0-1) - higher is better."""
        ambiguous_count = sum(
            1 for term in cls.AMBIGUOUS_TERMS if term in prompt.lower()
        )
        clarity_score = 1.0 - (ambiguous_count * 0.1)
        return max(0.0, min(1.0, clarity_score))

    @classmethod
    def score_preciseness(cls, prompt: str) -> float:
        """Score prompt preciseness (0-1) - higher is better."""
        precise_count = sum(
            1 for term in cls.PRECISE_INDICATORS if term in prompt.lower()
        )
        vague_count = sum(1 for term in cls.VAGUE_INDICATORS if term in prompt.lower())

        preciseness_score = (precise_count * 0.1) - (vague_count * 0.15)
        return max(0.0, min(1.0, preciseness_score))

    @classmethod
    def validate_json_schema(cls, prompt: str) -> bool:
        """Check if prompt includes proper JSON schema definition."""
        required_elements = ["status", "confidence", "explanation", "evidence"]
        return all(element in prompt.lower() for element in required_elements)


class AIPrompts:
    """Centralized collection of all AI prompts used in the RAi Compliance Engine."""

    # =====================================
    # COMPLIANCE ANALYSIS PROMPTS
    # =====================================

    @staticmethod
    def get_compliance_analysis_system_prompt() -> str:
        """Crystal-clear system prompt with zero ambiguity."""
        return (
            "You are RAi Compliance Engine - an expert IFRS/IAS compliance analyzer.\n\n"
            "CORE FUNCTION: Determine if financial document content meets specific accounting standard requirements.\n\n"
            "DECISION CRITERIA:\n"
            "• YES: Requirement is explicitly stated OR clearly implied with 90%+ certainty\n"
            "• NO: Requirement is missing, incomplete, or contradicted\n"
            "• N/A: Requirement does not apply to this entity's circumstances\n\n"
            f"{PromptConfiguration.get_confidence_criteria()}\n\n"
            f"{PromptConfiguration.get_evidence_quality_standards()}\n\n"
            "OUTPUT: JSON only. No explanations outside the schema."
        )

    @staticmethod
    def get_compliance_analysis_base_prompt(question: str, context: str) -> str:
        """Base prompt for compliance analysis questions."""
        return (
            "You are **RAi Compliance Engine**, an advanced AI-powered compliance analysis platform by RAi Firm.\n\n"
            " **Objective**\n"
            "Evaluate whether a given excerpt from an entity's financial reporting document meets a specified accounting standard requirement.\n\n"
            " **Inputs**\n"
            f"• **Requirement**: {question}\n"
            f"• **Document Content**: {context}\n\n"
        )

    @staticmethod
    def get_enhanced_evidence_section(enhanced_evidence: Dict[str, Any]) -> str:
        """Generate the enhanced evidence section for compliance analysis."""
        quality_assessment = enhanced_evidence.get("evidence_quality_assessment", {})
        analysis_summary = enhanced_evidence.get("analysis_summary", {})

        return (
            " **Enhanced Evidence Analysis**\n"
            f"• **Evidence Quality Score**: {
                quality_assessment.get(
                    'overall_quality', 0)}/100\n"
            f"• **Confidence Level**: {
                quality_assessment.get(
                    'confidence_level', 0.0):.2f}\n"
            f"• **Source Type**: {quality_assessment.get('source_type', 'unknown')}\n"
            f"• **Evidence Source**: {
                quality_assessment.get(
                    'evidence_source',
                    'Unknown section')}\n"
            f"• **Substantive Evidence Found**: {
                analysis_summary.get(
                    'substantive_evidence_count',
                    0)} sources\n"
            f"• **Policy Evidence Found**: {
                analysis_summary.get(
                    'policy_evidence_count',
                    0)} sources\n"
            f"• **Recommendation**: {
                analysis_summary.get(
                    'recommendation',
                    'Manual review recommended')}\n\n"
            " **Note**: This evidence has been intelligently selected from the most relevant document sections.\n"
            "Consider the evidence quality score and source type in your assessment.\n\n"
        )

    @staticmethod
    def get_compliance_analysis_instructions() -> str:
        """Comprehensive, high-quality analysis instructions with objective criteria."""
        return (
            "ANALYSIS METHODOLOGY:\n\n"
            "1. EXPLICIT COMPLIANCE DETECTION:\n"
            "   ✓ Search for verbatim requirement text\n"
            "   ✓ Look for 95%+ semantic matches\n"
            "   ✓ Identify regulatory language patterns\n\n"
            "2. IMPLICIT COMPLIANCE DETECTION:\n"
            "   ✓ Find synonymous terminology\n"
            "   ✓ Locate related disclosures\n"
            "   ✓ Assess cross-references and policy statements\n\n"
            "3. NON-APPLICABILITY ASSESSMENT:\n"
            "   ✓ Entity has no relevant assets/transactions\n"
            "   ✓ Standard explicitly excludes entity type\n"
            "   ✓ Materiality thresholds not met\n\n"
            "EVIDENCE FORMAT (pipe-separated):\n"
            "Reference|Requirement|Description|PageNumber|Extract|ContentAnalysis|SuggestedDisclosure\n\n"
            "FIELD DEFINITIONS:\n"
            "• Reference: IAS X.Y or IFRS X.Y (exact standard paragraph)\n"
            "• Requirement: exact standard text from the accounting standard\n"
            "• Description: what was found in the document regarding this requirement\n"
            "• PageNumber: P<number> or Page <number> (actual page in document)\n"
            "• Extract: verbatim text from the document, enclosed in quotes\n"
            "• ContentAnalysis: detailed professional assessment of document content quality, completeness, and compliance adequacy with specific observations on strengths and weaknesses\n"
            "• SuggestedDisclosure: specific, actionable disclosure language recommendations for addressing identified gaps or enhancing current disclosures\n\n"
            "MANDATORY JSON SCHEMA:\n"
            "{\n"
            '  "status": "YES|NO|N/A",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "explanation": "<detailed professional explanation with analytical depth>",\n'
            '  "evidence": ["Reference|Requirement|Description|PageNumber|Extract|ContentAnalysis|SuggestedDisclosure"],\n'
            '  "content_analysis": "<comprehensive analytical commentary on document content, assessing quality, completeness, and areas for improvement>",\n'
            '  "disclosure_recommendations": ["<specific actionable disclosure suggestions>"],\n'
            '  "suggestion": "<primary actionable disclosure recommendation for main compliance gap>" // only if status=NO\n'
            "}\n\n"
            "ENHANCED ANALYSIS REQUIREMENTS:\n"
            "• ContentAnalysis: Provide substantive commentary on the quality and adequacy of document content, not just compliance status\n"
            "• content_analysis: Generate comprehensive analytical assessment covering document structure, disclosure quality, completeness, and professional observations\n"
            "• disclosure_recommendations: Create specific, implementable suggestions for improving or enhancing disclosures\n"
            "• Focus on analytical depth beyond simple pass/fail assessment\n\n"
            "VALIDATION RULES:\n"
            "• Exactly 6 pipe (|) separators per evidence item (added SuggestedDisclosure field)\n"
            "• Reference must be exact accounting standard paragraph\n"
            "• PageNumber must reference actual document location\n"
            "• Extract must be verbatim quote from document\n"
            "• ContentAnalysis must provide detailed professional assessment\n"
            "• SuggestedDisclosure must offer specific, actionable recommendations\n\n"
            "EXAMPLE RESPONSE:\n"
            "{\n"
            '  "status": "NO",\n'
            '  "confidence": 0.95,\n'
            '  "explanation": "The document provides comprehensive disclosures on investment properties including recognition, measurement, and reclassifications related to change in use. However, it lacks specific disclosure requirements for amounts reclassified at the date of initial application of Amendments to IAS 40, as required by paragraph 84C. The reconciliation provided in paragraphs 76 and 79 does not include these specific reclassification amounts from the initial application date.",\n'
            '  "evidence": [\n'
            "    \"IAS 40.84C|If an entity reclassifies property at the date of initial application of Amendments to IAS 40 – Transfers of Investment Property, does it disclose the amounts reclassified to, or from, investment property in accordance with paragraph 84C as part of the reconciliation of the carrying amount of investment property at the beginning and end of the period as required by paragraphs 76 and 79?|No disclosure found on reclassification amounts at initial application or reconciliation including such reclassifications.|Page 43-44|'Transfers are made to (or from) investment property only when there is a change in use. The Group considers it as evidence the commencement, cessation or development...'|The disclosure describes the Group's transfer policy and provides general guidance on when transfers occur, but lacks quantitative information about amounts reclassified at initial application. The policy statement is clear and well-structured but incomplete for the specific requirement. The reconciliation format exists but does not include the required specific reclassification amounts.|Include a specific line item in the investment property reconciliation showing 'Reclassifications at initial application of IAS 40 amendments: $XXX' with corresponding narrative explanation of the nature and impact of these reclassifications.\"\n"
            "  ],\n"
            '  "content_analysis": "The investment property section demonstrates good overall disclosure quality with clear policy statements and comprehensive reconciliation tables. The document structure is logical and follows a consistent format. However, the disclosure lacks specificity regarding transition provisions and initial application impacts. The reconciliation framework is well-established but needs enhancement to capture specific regulatory transition requirements. The narrative disclosures show professional quality but could benefit from more detailed quantitative analysis of reclassification impacts.",\n'
            '  "disclosure_recommendations": [\n'
            '    "Add a specific reconciliation line item for initial application reclassifications with corresponding amounts",\n'
            '    "Include narrative explanation of the nature and financial impact of reclassifications made at transition",\n'
            '    "Consider providing comparative information to show the effect of the amendments on the financial statements",\n'
            '    "Enhance the reconciliation table to clearly distinguish between ongoing transfers and initial application adjustments"\n'
            '  ],\n'
            '  "suggestion": "The entity should include a reconciliation of the carrying amount of investment property at the beginning and end of the period, explicitly disclosing amounts reclassified to, or from, investment property on initial application of Amendments to IAS 40. For example: \'At the date of initial application of Amendments to IAS 40, the Group reclassified properties with a carrying amount of $XXX from investment property to property, plant and equipment due to change in use. The reconciliation is as follows: Opening balance $XXX; Additions $XXX; Transfers from owner-occupied properties $XXX; Transfers to owner-occupied properties $XXX; Reclassifications at initial application ($XXX); Disposals ($XXX); Fair value adjustments $XXX; Closing balance $XXX.\' This disclosure complies with paragraph 84C of IAS 40."\n'
            "}"
        )

    @classmethod
    def get_full_compliance_analysis_prompt(
        cls,
        question: str,
        context: str,
        enhanced_evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the complete compliance analysis prompt with dynamic quality adaptation."""
        # Assess context quality for dynamic prompt assembly
        context_quality = cls._assess_context_quality(context, enhanced_evidence)

        prompt = cls.get_compliance_analysis_base_prompt(question, context)

        # Add enhanced evidence if available
        if enhanced_evidence:
            prompt += cls.get_enhanced_evidence_section(enhanced_evidence)

        # Add quality-specific instructions
        prompt += cls._get_quality_specific_instructions(context_quality)

        # Add standard instructions
        prompt += cls.get_compliance_analysis_instructions()

        return prompt

    @classmethod
    def _assess_context_quality(
        cls, context: str, enhanced_evidence: Optional[Dict[str, Any]]
    ) -> str:
        """Assess context quality for dynamic prompt selection."""
        if enhanced_evidence:
            quality_score = enhanced_evidence.get(
                "evidence_quality_assessment", {}
            ).get("overall_quality", 0)
        else:
            # Basic quality assessment based on context length and structure
            quality_score = min(100, len(context) / 100)  # Simple heuristic

        if quality_score >= 80:
            return "HIGH_CONFIDENCE"
        elif quality_score >= 50:
            return "STANDARD"
        else:
            return "CAUTIOUS"

    @classmethod
    def _get_quality_specific_instructions(cls, quality_mode: str) -> str:
        """Get instructions based on context quality."""
        if quality_mode == "HIGH_CONFIDENCE":
            return (
                "\nHIGH CONFIDENCE ANALYSIS MODE:\n"
                "• Leverage enhanced evidence for comprehensive assessment\n"
                "• Cross-reference multiple evidence sources\n"
                "• Provide detailed compliance roadmap if non-compliant\n"
                "• Target confidence score: 0.85-1.0\n"
                "• Use Grade A or B evidence when available\n\n"
            )
        elif quality_mode == "STANDARD":
            return (
                "\nSTANDARD ANALYSIS MODE:\n"
                "• Balance available evidence with cautious assessment\n"
                "• Clearly distinguish between explicit and implied compliance\n"
                "• Provide practical improvement suggestions\n"
                "• Target confidence score: 0.6-0.85\n"
                "• Accept Grade B or C evidence\n\n"
            )
        else:  # CAUTIOUS
            return (
                "\nCAUTIOUS ANALYSIS MODE:\n"
                "• Clearly flag evidence limitations\n"
                "• Recommend manual review for critical requirements\n"
                "• Focus on obvious compliance gaps only\n"
                "• Target confidence score: 0.3-0.6\n"
                "• Flag when evidence quality is Grade D\n\n"
            )

    # =====================================
    # METADATA EXTRACTION PROMPTS
    # =====================================

    @staticmethod
    def get_metadata_extraction_system_prompt() -> str:
        """System prompt for metadata extraction from financial documents."""
        return (
            "You are a financial document analysis AI specialized in extracting company information from reports.\n"
            "Your task is to extract specific data accurately and concisely. if not found try to get info from internet or from you guessing . try to avoid with giving not found answer\n"
            'For company names: Look for official company name, often found in headers, footers, Directos\' Report, General Information section of Notes to the Financial Statements / Consolidated Financial Statements or "About Us" sections.\n'
            "For nature of business: Look for descriptions of main activities, sector information, Directors Report, General Information section of Notes to the Financial Statements / Consolidated Financial Statements or business overview.\n"
            "For operational demographics: Look for information about the countries where the company mentioned any countries in the report, often found in Directors Report, General Information section of Notes to the Financial Statements / Consolidated Financial Statements or business overview.\n\n"
            "Return ONLY the specific information requested. If not found, explain why briefly."
        )

    @staticmethod
    def get_metadata_extraction_user_prompt(reference: str, question_text: str) -> str:
        """User prompt for metadata extraction."""
        return f"""Extract the following information:

Field: {reference}
Question: {question_text}

Guidelines:
- For Reference: The reference of relevant section in the accounting standard
- For company names: Look at document headers, title pages, and official statements
- For Requirement: The requirement of the question being analyzed
- For business nature: Look for business descriptions, main activities, or sector information(dont mentions any country,continent names here)
- For operational_demographics: Look for information about the countries where the company mentioned any countries in the report, often found in Directors Report, General Information section of Notes to the Financial Statements / Consolidated Financial Statements or business overview(only mention the country names where this company did anything).
- For Extract: The relevant text from the document
- If truly not found, respond with "Not found" and a brief explanation,precise and elborative. Speak your mind. Be effortless for other people to understand. Be elegant, professional and thorough.

Respond with ONLY the extracted information or a "Not found" message."""

    # =====================================
    # VECTOR CONTEXT QUERY PROMPTS
    # =====================================

    @staticmethod
    def get_vector_metadata_system_prompt(has_vector_index: bool) -> str:
        """System prompt for vector-based metadata extraction."""
        if has_vector_index:
            return (
                "You are a financial document analysis AI specialized in extracting specific metadata from reports.\n"
                "You have access to a vector database that contains the document text.\n"
                "Extract the requested information directly . if not found try to get info from internet or from you guessing . try to avoid with giving not found answer\n"
                "Return ONLY the specific value/information requested, without explanation or fluff.\n"
                'If the information isn\'t clearly present, respond with "Not found".'
            )
        else:
            return (
                "You are a financial document analysis AI specialized in extracting specific metadata from reports.\n"
                "The document in question may not be fully accessible.\n"
                "Based on the limited information available, try to extract the requested information.\n"
                'Return a clear, concise value if you can determine it, or "Not found" if you cannot.'
            )

    @staticmethod
    def get_vector_metadata_user_prompt(
        document_id: str, question_text: str, reference: str, has_vector_index: bool
    ) -> str:
        """User prompt for vector-based metadata extraction."""
        base_prompt = f"""I need to extract the following information from a document with ID: {document_id}

Request: {question_text}
Field: {reference}

"""
        # Modify prompt based on vector index
        if has_vector_index:
            base_prompt += "The document is stored in the vector database. Please extract only the specific value requested."
        else:
            base_prompt += "Please extract only the specific value requested based on available information."

        base_prompt += "\nBe concise and precise. Do not include explanations, only the extracted value."

        return base_prompt

    @staticmethod
    def get_vector_compliance_system_prompt(has_vector_index: bool) -> str:
        """System prompt for vector-based compliance analysis."""
        if has_vector_index:
            return (
                "You are a financial reporting compliance expert specializing in IAS 40 Investment Property.\n"
                "You have access to a vector database that contains the document text.\n"
                "Your task is to analyze financial reports to determine if they comply with specific disclosure requirements.\n"
                "For each requirement, determine if the document complies (Yes/No/Partially).\n"
                "Provide evidence if available and a brief explanation.\n"
                'Be precise and fact-based in your analysis. Only mark as compliant if the requirement is explicitly addressed. In the cases where the requirement is not explicitly addressed, mark as unknown and prompt the auditor to seek a clarification and record their observations / findings. When the document doesnot have any information that is relevant to the question, the answer need not necessarily be "No". It might be "NA"too.'
            )
        else:
            return (
                "You are a financial reporting compliance expert specializing in IAS 40 Investment Property.\n"
                "Your task is to analyze financial reports to determine if they comply with specific disclosure requirements.\n"
                "Due to limited access to the document contents, you'll need to make a best effort determination.\n"
                "If you don't have enough information, indicate this clearly."
            )

    @staticmethod
    def get_vector_compliance_user_prompt(
        document_id: str, question_text: str, reference: str, has_vector_index: bool
    ) -> str:
        """User prompt for vector-based compliance analysis."""
        base_prompt = f"""Analyze document with ID: {document_id} to determine if it complies with this IAS 40 disclosure requirement:

Requirement: {question_text}
Reference: {reference}

"""
        if has_vector_index:
            base_prompt += "The document is stored in the vector database.\n"
        else:
            base_prompt += "Note: Full document contents may not be accessible.\n"

        base_prompt += """
Provide your analysis in this JSON format:
{
  "status": "Yes/No/Partially/Unknown",
  "explanation": "Your explanation of compliance or non-compliance",
  "evidence": "Any supporting evidence from the document if available",
  "confidence": 0.0-1.0
}"""

        return base_prompt

    # =====================================
    # CHUNKED COMPLIANCE ANALYSIS PROMPTS
    # =====================================

    @staticmethod
    def get_chunked_analysis_system_prompt() -> str:
        """System prompt for chunked compliance analysis."""
        return "You are an expert financial compliance analyst. Analyze the provided relevant document content against the compliance requirement. Be precise and evidence-based."

    @staticmethod
    def get_chunked_analysis_user_prompt(
        requirement_text: str, context: str, relevant_content: str
    ) -> str:
        """Enhanced user prompt for chunked compliance analysis with comprehensive output."""
        return f"""You are a financial compliance expert analyzing a financial document against a specific compliance requirement.

**Compliance Requirement:**
{requirement_text}

**Document Context:**
{context}

**Relevant Content to Analyze:**
{relevant_content}

**Instructions:**
1. Analyze ONLY the provided relevant content against the specific compliance requirement
2. Determine if this content shows compliance, partial compliance, or non-compliance
3. Provide specific evidence from the content to support your determination
4. Generate comprehensive content analysis beyond just compliance checking
5. Provide actionable disclosure recommendations for improvement
6. If the content is insufficient to make a determination, clearly state that

**Enhanced Evidence Format Requirements:**
Format: Reference|Requirement|Description|PageNumber|Extract|ContentAnalysis|SuggestedDisclosure
- Reference: Accounting standard paragraph (e.g., IAS 40.75)
- Requirement: Exact text from the accounting standard
- Description: What the document contains regarding this requirement
- PageNumber: Actual page number in the document (e.g., Page 23)
- Extract: Verbatim quote from the document
- ContentAnalysis: Detailed professional assessment of document content quality, completeness, and compliance adequacy with specific observations on strengths and weaknesses
- SuggestedDisclosure: Specific, actionable disclosure language recommendations for addressing identified gaps or enhancing current disclosures

**Enhanced Output Format:**
Provide a JSON response with:
- "status": "YES" | "NO" | "N/A"
- "confidence": number between 0.0 and 1.0
- "explanation": "Detailed professional explanation with analytical depth"
- "evidence": ["Reference|Requirement|Description|PageNumber|Extract|ContentAnalysis|SuggestedDisclosure"]
- "content_analysis": "Comprehensive analytical commentary on document content, assessing quality, completeness, and areas for improvement"
- "disclosure_recommendations": ["Specific actionable disclosure suggestions"]
- "suggestion": "Primary actionable disclosure recommendation for main compliance gap" (only if status is NO)

**Analysis Requirements:**
- ContentAnalysis: Provide substantive commentary on the quality and adequacy of document content, not just compliance status
- content_analysis: Generate comprehensive analytical assessment covering document structure, disclosure quality, completeness, and professional observations
- disclosure_recommendations: Create specific, implementable suggestions for improving or enhancing disclosures
- Focus on analytical depth beyond simple pass/fail assessment

Focus strictly on the provided content and avoid making assumptions about information not present."""

    # =====================================
    # METADATA EXTRACTION FIELD PROMPTS
    # =====================================

    @staticmethod
    def get_metadata_extraction_system_prompt_for_fields() -> str:
        """Zero-interpretation metadata extraction with clear hierarchy."""
        return (
            "METADATA EXTRACTION PROTOCOL:\n\n"
            "EXTRACTION HIERARCHY (stop at first match):\n"
            "1. EXPLICIT MENTION: Text directly states the information\n"
            "2. DOCUMENT HEADERS: Title pages, letterheads, signatures\n"
            "3. REGULATORY SECTIONS: Directors' reports, notes disclosure\n"
            "4. CONTEXTUAL INFERENCE: Industry/geographic context clues\n"
            "5. STRUCTURED FALLBACK: Use predefined business classification\n\n"
            "COMPANY NAME CRITERIA:\n"
            "• Priority 1: Legal entity name in financial statement header\n"
            "• Priority 2: Auditor's report addressee\n"
            "• Priority 3: Directors' report company reference\n"
            "• Format: [Full Legal Name] [Entity Type]\n\n"
            "BUSINESS NATURE CRITERIA:\n"
            "• Priority 1: 'Principal activities' section\n"
            "• Priority 2: Revenue stream descriptions\n"
            "• Priority 3: Director's report business overview\n"
            "• Format: [Primary Activity] in [Industry Sector]\n\n"
            "OPERATIONAL DEMOGRAPHICS CRITERIA:\n"
            "• Include: Countries with operational presence\n"
            "• Exclude: Countries mentioned for comparison only\n"
            "• Evidence: Revenue by geography, subsidiary locations, regulatory registrations\n"
            "• Format: [Country 1, Country 2, Country 3]\n\n"
            "CONFIDENCE INDICATORS:\n"
            "• CERTAIN: Found in multiple authoritative sections\n"
            "• PROBABLE: Found in single authoritative section\n"
            "• INFERRED: Derived from contextual evidence\n"
            "• ESTIMATED: Based on industry/document patterns\n\n"
            "OUTPUT FORMAT: [CONFIDENCE]|[VALUE]|[SOURCE_SECTION]|[PAGE_REFERENCE]"
        )

    @staticmethod
    def get_metadata_fallback_system_prompt() -> str:
        """Fallback system prompt for metadata extraction when primary AI fails."""
        return "You are a helpful financial document extraction AI."

    @staticmethod
    def get_company_name_extraction_prompt() -> str:
        """Objective prompt for extracting company name."""
        return (
            "EXTRACT COMPANY NAME:\n"
            "Search order:\n"
            "1. Financial statement title/header\n"
            "2. Independent auditor's report addressee\n"
            "3. Directors' report company name\n"
            "4. Statement of financial position entity name\n\n"
            "Required format: [Full Legal Name] [Entity Type Suffix]\n"
            "Example: ABC Holdings Limited\n\n"
            "If multiple names found, use the most complete legal form.\n"
            "Include suffixes: Ltd, Limited, Inc, Corporation, plc, AG, SA, etc."
        )

    @staticmethod
    def get_nature_of_business_extraction_prompt() -> str:
        """Enhanced prompt for extracting comprehensive business nature with operational detail."""
        return (
            "EXTRACT COMPREHENSIVE BUSINESS NATURE:\n\n"
            "SEARCH METHODOLOGY - Analyze for detailed operational activities:\n"
            "1. 'Principal activities' or 'Nature of business' section\n"
            "2. Revenue/segment reporting descriptions with operational detail\n"
            "3. Directors' report business overview and strategy\n"
            "4. Operating segment descriptions and activities\n"
            "5. Business model and value creation processes\n"
            "6. Operational capabilities and service offerings\n\n"
            "EXTRACTION REQUIREMENTS:\n"
            "• Preserve ALL operational detail and complexity\n"
            "• Include business model specifics and operational processes\n"
            "• Capture sector-specific activities and specializations\n"
            "• Maintain technical terminology and industry language\n"
            "• Include operational scale, scope, and capabilities\n"
            "• Describe value chain activities and business functions\n\n"
            "FORMAT: Comprehensive operational description\n"
            "Do NOT oversimplify - preserve the full richness of business operations.\n"
            "Include specific activities, processes, capabilities, and operational characteristics.\n"
            "Example: 'Management and development of real estate projects in the property development sector, including strategic planning, project financing, construction oversight, property asset management, facilities management services, and integrated development solutions for residential, commercial, and mixed-use properties across multiple market segments.'\n\n"
            "Focus on operational depth and business complexity, not simplified categorization."
        )

    @staticmethod
    def get_operational_demographics_extraction_prompt() -> str:
        """Enhanced prompt for comprehensive geographical extraction with 2D certainty."""
        return (
            "EXTRACT COMPREHENSIVE OPERATIONAL DEMOGRAPHICS:\n\n"
            "SEARCH METHODOLOGY - Analyze ENTIRE CONTENT for geographical entities:\n"
            "1. Revenue/segment reporting by geography\n"
            "2. Subsidiary and branch office locations\n"
            "3. Manufacturing, operational, and service facilities\n"
            "4. Property, plant, and equipment by location\n"
            "5. Regulatory registrations and licenses\n"
            "6. Market presence and customer base\n"
            "7. Regional business segments\n"
            "8. International operations and partnerships\n\n"
            "INCLUSION CRITERIA - Include ALL countries/regions with:\n"
            "• Revenue generation or reporting\n"
            "• Operational presence (offices, facilities, subsidiaries)\n"
            "• Significant business activities\n"
            "• Asset ownership or management\n"
            "• Regulatory compliance requirements\n"
            "• Market operations or customer base\n\n"
            "GEOGRAPHICAL DETECTION - Identify:\n"
            "• Full country names (e.g., United Arab Emirates)\n"
            "• Common abbreviations (e.g., UAE, UK, USA)\n"
            "• Regional groupings (e.g., MENA, APAC, EU, GCC)\n"
            "• Territories and special administrative regions\n"
            "• Historical or alternative country names\n\n"
            "EXCLUSION CRITERIA - Exclude ONLY:\n"
            "• Pure investment holdings without operations\n"
            "• Temporary project work under 12 months\n"
            "• Mentioned only for comparative/benchmark purposes\n\n"
            "COMPLETENESS REQUIREMENT:\n"
            "Examine ALL available content thoroughly. Do not limit extraction to initial findings.\n"
            "Cross-reference multiple sections to ensure comprehensive coverage.\n"
            "Include entities mentioned in any business context, not just primary operations.\n\n"
            "FORMAT: Provide complete list separated by commas\n"
            "Example: United Arab Emirates, United Kingdom, United States, Singapore, Hong Kong"
        )

    @staticmethod
    def get_generic_field_extraction_prompt(field_name: str) -> str:
        """Generic prompt for extracting any field."""
        return (
            f"Extract the {field_name.replace('_', ' ')} from the document. "
            "Reply only with the extracted value as a plain string. Do not include any explanations, evidence, or formatting."
        )

    @staticmethod
    def get_field_extraction_prompt(field_name: str) -> str:
        """Get the appropriate extraction prompt for a specific field."""
        field_prompts = {
            "company_name": AIPrompts.get_company_name_extraction_prompt(),
            "nature_of_business": AIPrompts.get_nature_of_business_extraction_prompt(),
            "operational_demographics": AIPrompts.get_operational_demographics_extraction_prompt(),
        }
        return field_prompts.get(
            field_name, AIPrompts.get_generic_field_extraction_prompt(field_name)
        )

    # =====================================
    # UTILITY METHODS
    # =====================================

    @staticmethod
    def get_all_prompt_types() -> list:
        """Return a list of all available prompt types for documentation purposes."""
        return [
            "compliance_analysis_system",
            "compliance_analysis_base",
            "enhanced_evidence_section",
            "compliance_analysis_instructions",
            "full_compliance_analysis",
            "metadata_extraction_system",
            "metadata_extraction_user",
            "vector_metadata_system",
            "vector_metadata_user",
            "vector_compliance_system",
            "vector_compliance_user",
            "chunked_analysis_system",
            "chunked_analysis_user",
            "metadata_extraction_system_for_fields",
            "metadata_fallback_system",
            "company_name_extraction",
            "nature_of_business_extraction",
            "operational_demographics_extraction",
            "generic_field_extraction",
            "field_extraction",
        ]

    @staticmethod
    def get_prompt_description(prompt_type: str) -> str:
        """Get a description of what each prompt type is used for."""
        descriptions = {
            "compliance_analysis_system": "System prompt for the main compliance analysis AI assistant",
            "compliance_analysis_base": "Base prompt template for compliance analysis questions",
            "enhanced_evidence_section": "Template for enhanced evidence analysis section",
            "compliance_analysis_instructions": "Standard instructions and output format for compliance analysis",
            "full_compliance_analysis": "Complete assembled prompt for compliance analysis",
            "metadata_extraction_system": "System prompt for basic metadata extraction",
            "metadata_extraction_user": "User prompt template for metadata extraction",
            "vector_metadata_system": "System prompt for vector-based metadata extraction",
            "vector_metadata_user": "User prompt template for vector-based metadata extraction",
            "vector_compliance_system": "System prompt for vector-based compliance analysis",
            "vector_compliance_user": "User prompt template for vector-based compliance analysis",
            "chunked_analysis_system": "System prompt for chunked compliance analysis",
            "chunked_analysis_user": "User prompt template for chunked compliance analysis",
            "metadata_extraction_system_for_fields": "System prompt for field-specific metadata extraction",
            "metadata_fallback_system": "Fallback system prompt for metadata extraction",
            "company_name_extraction": "Prompt for extracting company name from documents",
            "nature_of_business_extraction": "Prompt for extracting business nature/activities",
            "operational_demographics_extraction": "Prompt for extracting operational locations",
            "generic_field_extraction": "Generic prompt template for any field extraction",
            "field_extraction": "Dynamic prompt selector for specific field extraction",
        }
        return descriptions.get(prompt_type, "Unknown prompt type")

    @staticmethod
    def validate_all_prompts() -> Dict[str, Dict[str, float]]:
        """Validate all prompts and return quality scores."""
        results = {}
        prompt_methods = [
            method
            for method in dir(AIPrompts)
            if method.startswith("get_") and not method.startswith("get_all")
        ]

        for method_name in prompt_methods:
            try:
                method = getattr(AIPrompts, method_name)
                if callable(method):
                    # Handle methods with no parameters
                    if method_name in [
                        "get_compliance_analysis_system_prompt",
                        "get_metadata_extraction_system_prompt_for_fields",
                        "get_company_name_extraction_prompt",
                        "get_nature_of_business_extraction_prompt",
                    ]:
                        prompt = method()
                        if isinstance(prompt, str):
                            results[method_name] = {
                                "clarity": float(PromptValidator.score_clarity(prompt)),
                                "preciseness": float(
                                    PromptValidator.score_preciseness(prompt)
                                ),
                                "has_json_schema": str(
                                    PromptValidator.validate_json_schema(prompt)
                                ),
                                "length": str(len(prompt)),
                                "overall_score": float(
                                    PromptValidator.score_clarity(prompt)
                                    + PromptValidator.score_preciseness(prompt)
                                )
                                / 2.0,
                            }
            except Exception as e:
                results[method_name] = {"error": str(e)}

        return results

    @staticmethod
    def get_prompt_improvement_suggestions(prompt: str) -> List[str]:
        """Get specific suggestions for improving a prompt."""
        suggestions = []

        # Check for ambiguous terms
        ambiguous_found = [
            term for term in PromptValidator.AMBIGUOUS_TERMS if term in prompt.lower()
        ]
        if ambiguous_found:
            suggestions.append(f"Replace ambiguous terms: {', '.join(ambiguous_found)}")

        # Check for vague indicators
        vague_found = [
            term for term in PromptValidator.VAGUE_INDICATORS if term in prompt.lower()
        ]
        if vague_found:
            suggestions.append(f"Replace vague terms: {', '.join(vague_found)}")

        # Check for JSON schema
        if not PromptValidator.validate_json_schema(prompt):
            suggestions.append("Add proper JSON schema definition")

        # Check length
        if len(prompt) < 100:
            suggestions.append(
                "Prompt may be too brief - consider adding more specific instructions"
            )
        elif len(prompt) > 2000:
            suggestions.append(
                "Prompt may be too long - consider breaking into smaller components"
            )

        return suggestions


# Enhanced validation and testing utilities
class PromptTester:
    """Advanced testing framework for prompt effectiveness."""

    @staticmethod
    def test_prompt_consistency() -> Dict[str, Any]:
        """Test consistency across all prompts."""
        all_prompts = {}
        results = {
            "format_consistency": True,
            "terminology_consistency": True,
            "issues": [],
        }

        # Collect all prompts
        for method_name in dir(AIPrompts):
            if method_name.startswith("get_") and callable(
                getattr(AIPrompts, method_name)
            ):
                try:
                    method = getattr(AIPrompts, method_name)
                    if method_name in ["get_compliance_analysis_system_prompt"]:
                        prompt = method()
                        if isinstance(prompt, str):
                            all_prompts[method_name] = prompt
                except Exception:
                    continue

        # Check for consistent terminology
        key_terms = ["JSON", "confidence", "evidence", "status"]
        for term in key_terms:
            variations = set()
            for prompt_name, prompt_text in all_prompts.items():
                if term.lower() in prompt_text.lower():
                    # Extract context around the term
                    lines = prompt_text.lower().split("\n")
                    for line in lines:
                        if term.lower() in line:
                            variations.add(line.strip())

            if len(variations) > 3:  # Threshold for too many variations
                results["terminology_consistency"] = False
                results["issues"].append(
                    f"Inconsistent usage of term '{term}': {len(variations)} variations"
                )

        return results

    @staticmethod
    def benchmark_prompt_effectiveness() -> Dict[str, float]:
        """Benchmark all prompts against 10/10 criteria."""
        validation_results = AIPrompts.validate_all_prompts()

        scores = {
            "average_clarity": 0.0,
            "average_preciseness": 0.0,
            "json_schema_compliance": 0.0,
            "overall_quality": 0.0,
        }

        if validation_results:
            valid_results = [r for r in validation_results.values() if "error" not in r]
            if valid_results:
                scores["average_clarity"] = sum(
                    r.get("clarity", 0) for r in valid_results
                ) / len(valid_results)
                scores["average_preciseness"] = sum(
                    r.get("preciseness", 0) for r in valid_results
                ) / len(valid_results)
                scores["json_schema_compliance"] = sum(
                    1 for r in valid_results if r.get("has_json_schema", False)
                ) / len(valid_results)
                scores["overall_quality"] = (
                    scores["average_clarity"]
                    + scores["average_preciseness"]
                    + scores["json_schema_compliance"]
                ) / 3

        return scores

    @staticmethod
    def create_accounting_standards_suggestion_prompt(
        framework: str,
        company_name: str,
        nature_of_business: str,
        operational_demographics: str,
        financial_statements_type: str,
        available_standards: List[str]
    ) -> Dict[str, str]:
        """
        Create AI prompt for suggesting relevant accounting standards based on company metadata.
        
        Args:
            framework: Selected accounting framework (e.g., "IFRS", "US_GAAP")
            company_name: Extracted company name
            nature_of_business: Company's business activities and operations
            operational_demographics: Countries where company operates
            financial_statements_type: Type of financial statements (Consolidated/Standalone)
            available_standards: List of available standard IDs in the selected framework
        
        Returns:
            Dictionary with 'system' and 'user' prompts for AI processing
        """
        
        available_standards_list = "\n".join([f"- {std}" for std in available_standards])
        
        return {
            "system": (
                "You are an expert accounting standards advisor with deep knowledge of international "
                f"and national accounting frameworks, particularly {framework}. Your role is to analyze "
                "company metadata and suggest the most relevant accounting standards based on the "
                "company's business nature, operational geography, and financial statement structure."
            ),
            "user": (
                f"Based on the following company information, suggest the most relevant {framework} "
                "accounting standards that would likely apply to this company's financial reporting:\n\n"
                f"**Company Profile:**\n"
                f"- Company Name: {company_name}\n"
                f"- Nature of Business: {nature_of_business}\n"
                f"- Operational Demographics: {operational_demographics}\n"
                f"- Financial Statements Type: {financial_statements_type}\n"
                f"- Selected Framework: {framework}\n\n"
                f"**Available Standards in {framework}:**\n"
                f"{available_standards_list}\n\n"
                "**Analysis Requirements:**\n"
                "1. Consider the company's primary business activities (real estate, construction, leasing, etc.)\n"
                "2. Factor in geographical operations and potential regulatory requirements\n"
                "3. Account for the type of financial statements (consolidated vs standalone)\n"
                "4. Focus on standards most likely to have significant impact on financial reporting\n\n"
                "**Response Format:**\n"
                "Return a JSON object with the following structure:\n"
                "{\n"
                '  "suggested_standards": [\n'
                '    {\n'
                '      "standard_id": "standard_code",\n'
                '      "relevance_score": 0.95,\n'
                '      "reasoning": "Explanation of why this standard is relevant"\n'
                '    }\n'
                '  ],\n'
                '  "priority_level": "high|medium|low",\n'
                '  "business_context": "Summary of key business factors that influence standard selection"\n'
                "}\n\n"
                "**Guidelines:**\n"
                "- Only suggest standards from the available list\n"
                "- Relevance score should be between 0.0 and 1.0\n"
                "- Focus on 3-7 most relevant standards\n"
                "- Prioritize standards with high impact on the specific business model\n"
                "- Provide clear, business-focused reasoning for each suggestion\n"
                "- Consider both mandatory standards and those with significant disclosure requirements"
            )
        }


# Convenience instance for easy importing
ai_prompts = AIPrompts()
