"""
Hybrid Financial Statement Detector - Best of Both Worlds

Combines reliable pattern-based detection with intelligent AI enhancement
for maximum consistency AND flexibility across all document types.
"""

import logging
import re
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DetectionStrategy(Enum):
    """Detection strategy used for each result"""
    PATTERN_ONLY = "pattern_based"
    AI_ENHANCED = "ai_enhanced" 
    HYBRID_CONSENSUS = "hybrid_consensus"
    FALLBACK_RECOVERY = "fallback_recovery"


@dataclass 
class HybridDetectionResult:
    """Enhanced detection result with strategy tracking"""
    statement_type: str
    content: str
    confidence_score: float
    strategy_used: DetectionStrategy
    pattern_confidence: float
    ai_confidence: Optional[float] = None
    consensus_factors: Dict[str, float] = None
    validation_markers: List[str] = None
    page_numbers: List[int] = None
    
    def __post_init__(self):
        if self.page_numbers is None:
            self.page_numbers = []
    
    def to_dict(self):
        return {
            "statement_type": self.statement_type,
            "content_length": len(self.content),
            "confidence_score": self.confidence_score,
            "strategy_used": self.strategy_used.value,
            "pattern_confidence": self.pattern_confidence,
            "ai_confidence": self.ai_confidence,
            "consensus_factors": self.consensus_factors or {},
            "validation_markers": self.validation_markers or [],
            "page_numbers": self.page_numbers or []
        }


@dataclass
class HybridFinancialContent:
    """Hybrid results with strategy insights and API compatibility"""
    statements: List[HybridDetectionResult]
    total_confidence: float
    strategy_breakdown: Dict[str, int]
    processing_metrics: Dict[str, float]
    fallback_activations: int = 0
    # API compatibility fields
    validation_summary: str = ""
    content_type: str = "hybrid_financial_content"
    pattern_consensus: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pattern_consensus is None:
            self.pattern_consensus = {}
        if not self.validation_summary:
            statement_types = [s.statement_type for s in self.statements]
            unique_types = list(dict.fromkeys(statement_types))
            self.validation_summary = f"Hybrid detection found {len(self.statements)} financial statements: {', '.join(unique_types)}" if self.statements else "No financial statements detected"
        # Set content type based on results
        if len(self.statements) >= 3:
            self.content_type = "complete_financial_statements"
        elif len(self.statements) >= 1:
            self.content_type = "financial_statements"
        else:
            self.content_type = "no_financial_content"
    
    def to_dict(self):
        return {
            "statements": [stmt.to_dict() for stmt in self.statements],
            "total_confidence": self.total_confidence,
            "strategy_breakdown": self.strategy_breakdown,
            "processing_metrics": self.processing_metrics,
            "fallback_activations": self.fallback_activations,
            "validation_summary": self.validation_summary,
            "content_type": self.content_type,
            "pattern_consensus": self.pattern_consensus
        }


class EnhancedPatternManager:
    """
    Enhanced pattern-based detection with expanded terminology coverage
    """
    
    def __init__(self):
        logger.info("🌟 Enhanced Pattern Manager: Expanded pattern coverage")
        
        # Core patterns (original)
        self.core_statement_patterns = [
            'consolidated statement', 'balance sheet', 'income statement',
            'statement of financial position', 'profit and loss',
            'comprehensive income', 'cash flow', 'changes in equity'
        ]
        
        # Extended patterns for better coverage
        self.extended_patterns = [
            # Income statement variations
            'statement of operations', 'statement of earnings', 'statement of income',
            'profit and loss statement', 'p&l statement', 'p & l',
            'consolidated statement of operations', 'statement of comprehensive income',
            
            # Balance sheet variations  
            'statement of financial position', 'position statement', 'financial position',
            'consolidated balance sheet', 'balance sheet', 'statement of assets',
            
            # Cash flow variations
            'statement of cash flows', 'cash flow statement', 'cashflow statement',
            'statement of cash flow', 'consolidated cash flows',
            
            # International variations
            'bilan', 'compte de résultat', 'gewinn- und verlustrechnung',
            'bilanz', 'estado de resultados', 'balanço patrimonial'
        ]
        
        # All patterns combined
        self.all_patterns = self.core_statement_patterns + self.extended_patterns
        
        # Financial indicators
        self.financial_indicators = [
            'assets', 'liabilities', 'equity', 'revenue', 'expenses',
            'current assets', 'non-current', 'total assets', 'shareholders',
            'net income', 'gross profit', 'operating income', 'ebitda',
            'working capital', 'retained earnings', 'depreciation'
        ]
        
        # Performance tracking
        self.cache = {}
        self.pattern_hits = {}
        
    def enhanced_classify(self, text: str) -> Dict[str, Any]:
        """Enhanced classification with detailed pattern analysis"""
        if not text or not isinstance(text, str):
            return {"confidence": 0.0, "patterns_found": [], "strategy": "none"}
            
        text_lower = text.lower()
        
        # Core pattern scoring
        core_score = 0.0
        found_core_patterns = []
        
        for pattern in self.core_statement_patterns:
            if pattern in text_lower:
                core_score += 0.3
                found_core_patterns.append(pattern)
                
        # Extended pattern scoring
        extended_score = 0.0
        found_extended_patterns = []
        
        for pattern in self.extended_patterns:
            if pattern in text_lower:
                extended_score += 0.2
                found_extended_patterns.append(pattern)
                
        # Financial indicator scoring
        indicator_score = 0.0
        found_indicators = []
        
        for indicator in self.financial_indicators:
            if indicator in text_lower:
                indicator_score += 0.1
                found_indicators.append(indicator)
                
        # Calculate total confidence
        total_confidence = min(1.0, core_score + extended_score + indicator_score)
        
        return {
            "confidence": total_confidence,
            "core_patterns": found_core_patterns,
            "extended_patterns": found_extended_patterns, 
            "financial_indicators": found_indicators,
            "pattern_scores": {
                "core": core_score,
                "extended": extended_score,
                "indicators": indicator_score
            },
            "strategy": "enhanced_pattern"
        }


class LightweightAIEnhancer:
    """
    Lightweight AI enhancement for pattern-based detection gaps
    """
    
    def __init__(self):
        logger.info("🤖 Lightweight AI Enhancer: Semantic gap filling")
        
        # Simple semantic rules (no external AI dependencies for now)
        # This would be where you'd integrate your preferred AI model
        self.semantic_rules = {
            "financial_context_indicators": [
                r'\$[\d,]+', r'€[\d,]+', r'£[\d,]+',  # Currency amounts
                r'revenue|income|profit|loss|expense',  # Financial terms
                r'fiscal year|quarter|annual|monthly',  # Time periods
                r'shareholders|investors|stakeholders',  # Entities
                r'assets exceed|total assets|net worth'  # Financial concepts
            ],
            
            "statement_structure_indicators": [
                r'for the year ended|as at|as of',  # Date contexts
                r'consolidated|subsidiary|parent',   # Corporate structure
                r'audited|unaudited|reviewed',      # Audit status
                r'in thousands|in millions|\(in \w+\)'  # Scale indicators
            ]
        }
        
    def ai_enhance_detection(self, text: str, pattern_result: Dict) -> Dict[str, Any]:
        """AI enhancement when pattern detection needs help"""
        
        # Simulate AI semantic analysis with rule-based enhancement
        semantic_score = 0.0
        semantic_indicators = []
        
        # Check for financial context
        for indicator_type, patterns in self.semantic_rules.items():
            for pattern in patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    semantic_score += 0.1 * len(matches)
                    semantic_indicators.append(f"{indicator_type}: {len(matches)} matches")
        
        # AI confidence boost calculation
        ai_confidence = min(0.8, semantic_score)  # Cap at 80% for rule-based AI
        
        # Determine if AI should enhance pattern result
        should_enhance = (
            pattern_result["confidence"] < 0.5 and  # Low pattern confidence
            ai_confidence > 0.3  # But AI sees financial indicators
        )
        
        return {
            "ai_confidence": ai_confidence,
            "semantic_indicators": semantic_indicators,
            "should_enhance": should_enhance,
            "enhancement_reason": "semantic_context" if should_enhance else None
        }


class HybridFinancialDetector:
    """
    Hybrid detector combining pattern-based reliability with AI flexibility
    """
    
    def __init__(self):
        logger.info("🌟 Hybrid Financial Detector: Best of both worlds")
        
        self.pattern_manager = EnhancedPatternManager()
        self.ai_enhancer = LightweightAIEnhancer()
        
        # Hybrid configuration
        self.pattern_confidence_threshold = 0.6  # High confidence = pattern only
        self.ai_activation_threshold = 0.3       # Low confidence = activate AI
        self.consensus_weight = 0.7              # Pattern vs AI weighting
        
        # Performance metrics
        self.metrics = {
            "pattern_only_count": 0,
            "ai_enhanced_count": 0,
            "hybrid_consensus_count": 0,
            "fallback_activations": 0
        }
        
        logger.info("✅ Hybrid system ready: Pattern-first with AI enhancement")
        
    def detect_with_hybrid_strategy(self, document_text: str, document_id: str = None) -> HybridFinancialContent:
        """
        Main hybrid detection with intelligent strategy selection
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Hybrid detection starting for {document_id or 'document'}")
            
            # Phase 1: Enhanced pattern detection
            pattern_result = self.pattern_manager.enhanced_classify(document_text)
            pattern_confidence = pattern_result["confidence"]
            
            # Phase 2: Strategy decision
            if pattern_confidence >= self.pattern_confidence_threshold:
                # High pattern confidence - use pattern only
                strategy = DetectionStrategy.PATTERN_ONLY
                final_confidence = pattern_confidence
                ai_result = None
                self.metrics["pattern_only_count"] += 1
                
            elif pattern_confidence >= self.ai_activation_threshold:
                # Medium confidence - use hybrid consensus
                strategy = DetectionStrategy.HYBRID_CONSENSUS
                ai_result = self.ai_enhancer.ai_enhance_detection(document_text, pattern_result)
                
                # Blend confidences
                final_confidence = (
                    pattern_confidence * self.consensus_weight +
                    ai_result["ai_confidence"] * (1 - self.consensus_weight)
                )
                self.metrics["hybrid_consensus_count"] += 1
                
            else:
                # Low pattern confidence - try AI enhancement
                ai_result = self.ai_enhancer.ai_enhance_detection(document_text, pattern_result)
                
                if ai_result["should_enhance"]:
                    strategy = DetectionStrategy.AI_ENHANCED
                    final_confidence = max(pattern_confidence, ai_result["ai_confidence"])
                    self.metrics["ai_enhanced_count"] += 1
                else:
                    strategy = DetectionStrategy.FALLBACK_RECOVERY
                    final_confidence = pattern_confidence  # Keep original
                    self.metrics["fallback_activations"] += 1
            
            # Phase 3: Generate results
            statements = self._generate_hybrid_statements(
                document_text, pattern_result, ai_result, strategy, final_confidence
            )
            
            # Phase 4: Calculate metrics
            processing_time = time.time() - start_time
            strategy_breakdown = {
                "pattern_only": self.metrics["pattern_only_count"],
                "ai_enhanced": self.metrics["ai_enhanced_count"], 
                "hybrid_consensus": self.metrics["hybrid_consensus_count"],
                "fallback_recovery": self.metrics["fallback_activations"]
            }
            
            processing_metrics = {
                "total_time": processing_time,
                "pattern_confidence": pattern_confidence,
                "final_confidence": final_confidence,
                "strategy": strategy.value
            }
            
            logger.info(f"✅ Hybrid detection completed: {len(statements)} statements, {final_confidence:.1%} confidence, {strategy.value}")
            
            return HybridFinancialContent(
                statements=statements,
                total_confidence=final_confidence * 100,  # Convert to percentage
                strategy_breakdown=strategy_breakdown,
                processing_metrics=processing_metrics,
                fallback_activations=self.metrics["fallback_activations"]
            )
            
        except Exception as e:
            logger.error(f"❌ Hybrid detection failed: {e}")
            return self._create_empty_hybrid_result(str(e))
            
    def _generate_hybrid_statements(self, text: str, pattern_result: Dict, 
                                  ai_result: Optional[Dict], strategy: DetectionStrategy, 
                                  confidence: float) -> List[HybridDetectionResult]:
        """Generate statements based on hybrid analysis"""
        
        statements = []
        
        # Determine statement types from patterns
        statement_types = []
        
        # From core patterns
        for pattern in pattern_result.get("core_patterns", []):
            if "balance sheet" in pattern or "financial position" in pattern:
                statement_types.append("Balance Sheet")
            elif "income" in pattern or "profit" in pattern or "operations" in pattern:
                statement_types.append("Statement of Comprehensive Income") 
            elif "cash flow" in pattern:
                statement_types.append("Statement of Cashflows")
            elif "equity" in pattern or "changes" in pattern:
                statement_types.append("Statement of Changes in Equity")
                
        # From extended patterns  
        for pattern in pattern_result.get("extended_patterns", []):
            if "operations" in pattern or "earnings" in pattern:
                statement_types.append("Statement of Operations")
            elif "bilan" in pattern:
                statement_types.append("Balance Sheet")
            elif "résultat" in pattern or "gewinn" in pattern:
                statement_types.append("Income Statement")
                
        # If no specific types found but high financial confidence, create generic
        if not statement_types and confidence > 0.5:
            statement_types = ["Financial Statement"]
            
        # Create hybrid results
        for stmt_type in statement_types:
            consensus_factors = {
                "pattern_score": pattern_result.get("pattern_scores", {}).get("core", 0),
                "extended_score": pattern_result.get("pattern_scores", {}).get("extended", 0), 
                "indicator_score": pattern_result.get("pattern_scores", {}).get("indicators", 0)
            }
            
            if ai_result:
                consensus_factors["ai_confidence"] = ai_result.get("ai_confidence", 0)
                
            statement = HybridDetectionResult(
                statement_type=stmt_type,
                content=text[:5000],  # Truncate for storage
                confidence_score=confidence,
                strategy_used=strategy,
                pattern_confidence=pattern_result["confidence"],
                ai_confidence=ai_result.get("ai_confidence") if ai_result else None,
                consensus_factors=consensus_factors,
                validation_markers=pattern_result.get("core_patterns", []),
                page_numbers=[]  # Initialize empty page numbers for hybrid detection
            )
            
            statements.append(statement)
            
        return statements
        
    def _create_empty_hybrid_result(self, reason: str) -> HybridFinancialContent:
        """Create empty result for error cases"""
        return HybridFinancialContent(
            statements=[],
            total_confidence=0.0,
            strategy_breakdown={"error": 1},
            processing_metrics={"error": reason, "total_time": 0.0}
        )


# Convenience functions for backward compatibility
def detect_financial_statements_hybrid(document_text: str, document_id: Optional[str] = None) -> HybridFinancialContent:
    """
    Hybrid financial statement detection - best of both worlds
    """
    detector = HybridFinancialDetector()
    return detector.detect_with_hybrid_strategy(document_text, document_id)


if __name__ == "__main__":
    # Quick test
    test_doc = """
    CONSOLIDATED STATEMENT OF OPERATIONS
    For the year ended December 31, 2024
    
    Revenue: $1,000,000
    Operating expenses: $800,000
    Net income: $200,000
    """
    
    result = detect_financial_statements_hybrid(test_doc, "test")
    print(f"Hybrid Detection: {len(result.statements)} statements, {result.total_confidence:.1f}% confidence")
    print(f"Strategy used: {result.processing_metrics.get('strategy')}")