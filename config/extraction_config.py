"""
Smart Extraction Configuration

Configuration settings for optimized metadata extraction with token limits,
confidence thresholds, and performance tuning parameters.
"""

import os
from typing import Dict, List, Optional


class SmartExtractionConfig:
    """Configuration for optimized metadata extraction"""
    
    # Token limits for AI calls
    MAX_TOKENS_PER_FIELD = int(os.getenv("SMART_EXTRACTION_MAX_TOKENS_PER_FIELD", "1000"))
    MAX_TOTAL_TOKENS_PER_DOCUMENT = int(os.getenv("SMART_EXTRACTION_MAX_TOTAL_TOKENS", "15000"))
    
    # Confidence thresholds
    PATTERN_CONFIDENCE_THRESHOLD = float(os.getenv("PATTERN_CONFIDENCE_THRESHOLD", "0.7"))
    SEMANTIC_CONFIDENCE_THRESHOLD = float(os.getenv("SEMANTIC_CONFIDENCE_THRESHOLD", "0.8"))
    AI_VALIDATION_THRESHOLD = float(os.getenv("AI_VALIDATION_THRESHOLD", "0.6"))
    
    # Semantic search configuration
    MAX_CHUNKS_PER_SEARCH = int(os.getenv("MAX_CHUNKS_PER_SEARCH", "3"))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    
    # Geographical detection settings
    GEOGRAPHICAL_CONFIDENCE_THRESHOLD = float(os.getenv("GEOGRAPHICAL_CONFIDENCE_THRESHOLD", "0.75"))
    
    # Pattern matching weights
    PATTERN_WEIGHTS = {
        'company_name': float(os.getenv("COMPANY_NAME_PATTERN_WEIGHT", "0.8")),
        'nature_of_business': float(os.getenv("BUSINESS_TYPE_PATTERN_WEIGHT", "0.6")),
        'operational_demographics': float(os.getenv("GEOGRAPHICAL_PATTERN_WEIGHT", "0.9"))
    }
    
    # Company name patterns (can be extended via config)
    COMPANY_SUFFIXES = [
        "Ltd", "Inc", "Corp", "LLC", "Pty", "GmbH", "SA", "AG", "NV", "BV",
        "SPA", "SRL", "AB", "AS", "OY", "ApS", "Limited", "Corporation", 
        "Company", "Incorporated", "Proprietary", "Holdings", "Group", 
        "Industries", "Enterprises", "International", "Bank", "Banking"
    ]
    
    # Business type categories with keywords
    BUSINESS_CATEGORIES = {
        'financial_services': [
            'bank', 'banking', 'finance', 'financial', 'investment', 'insurance', 
            'credit', 'lending', 'mortgage', 'asset management', 'wealth management',
            'fund management', 'private equity', 'venture capital', 'securities',
            'trading', 'brokerage', 'financial services', 'capital markets'
        ],
        'technology': [
            'software', 'technology', 'IT', 'digital', 'tech', 'computer',
            'telecommunications', 'internet', 'cloud', 'data', 'analytics',
            'artificial intelligence', 'machine learning', 'cybersecurity',
            'information technology', 'software development', 'programming'
        ],
        'manufacturing': [
            'manufacturing', 'production', 'factory', 'industrial', 'automotive',
            'aerospace', 'chemicals', 'pharmaceuticals', 'textiles', 'machinery',
            'equipment', 'steel', 'metals', 'electronics', 'assembly', 'fabrication'
        ],
        'healthcare': [
            'hospital', 'medical', 'healthcare', 'health', 'pharmaceutical',
            'biotechnology', 'clinical', 'diagnostics', 'therapy', 'surgery',
            'nursing', 'dental', 'veterinary', 'medical devices', 'life sciences'
        ],
        'retail': [
            'retail', 'store', 'shop', 'commerce', 'supermarket', 'department store',
            'fashion', 'clothing', 'grocery', 'wholesale', 'distribution', 'logistics',
            'e-commerce', 'online retail', 'consumer goods'
        ],
        'energy': [
            'energy', 'oil', 'gas', 'petroleum', 'renewable', 'solar', 'wind',
            'nuclear', 'utilities', 'power', 'electricity', 'coal', 'mining',
            'hydroelectric', 'geothermal', 'biofuel'
        ],
        'real_estate': [
            'real estate', 'property', 'construction', 'development', 'housing',
            'commercial property', 'residential', 'land development', 'property management',
            'building', 'architecture', 'engineering'
        ],
        'education': [
            'education', 'school', 'university', 'college', 'training', 'learning',
            'academic', 'research', 'institute', 'educational services', 'teaching'
        ],
        'hospitality': [
            'hotel', 'hospitality', 'restaurant', 'tourism', 'travel', 'airline',
            'entertainment', 'leisure', 'casino', 'resort', 'catering', 'food service'
        ],
        'transportation': [
            'transportation', 'shipping', 'logistics', 'freight', 'delivery',
            'aviation', 'railway', 'trucking', 'maritime', 'courier', 'cargo'
        ]
    }
    
    # AI model configuration
    AI_MODEL_TEMPERATURE = float(os.getenv("AI_MODEL_TEMPERATURE", "0.1"))  # Low temperature for consistency
    AI_MAX_RESPONSE_TOKENS = int(os.getenv("AI_MAX_RESPONSE_TOKENS", "100"))
    
    # Performance monitoring
    ENABLE_PERFORMANCE_TRACKING = os.getenv("ENABLE_PERFORMANCE_TRACKING", "true").lower() == "true"
    LOG_TOKEN_USAGE = os.getenv("LOG_TOKEN_USAGE", "true").lower() == "true"
    
    # Fallback configuration - DISABLED for zero-fallback operation
    ENABLE_FALLBACK_TO_LEGACY = False  # Strict extraction only
    FALLBACK_ON_LOW_CONFIDENCE = False  # No confidence-based fallbacks
    MINIMUM_CONFIDENCE_FOR_SUCCESS = float(os.getenv("MINIMUM_CONFIDENCE_FOR_SUCCESS", "0.7"))  # Higher threshold
    
    @classmethod
    def get_company_pattern_regex(cls) -> List[str]:
        """Generate regex patterns for company name detection"""
        suffixes = "|".join(cls.COMPANY_SUFFIXES)
        return [
            # Match company names with legal suffixes (more restrictive)
            rf'\b([A-Z][a-zA-Z\s&\-\.]*(?:{suffixes})\.?)\b',
            # Match well-known business entity patterns
            r'\b([A-Z][a-zA-Z\s&\-\.]*(?:Limited|Corporation|Company|Incorporated|Proprietary))\b',
            # Match holding companies and groups (shorter patterns)
            r'\b([A-Z][a-zA-Z\s&\-\.]{1,30}(?:Holdings|Group|Industries|Enterprises|International))\b',
            # Match PJSC, PLLC, and other regional formats
            r'\b([A-Z][a-zA-Z\s&\-\.]{1,30}(?:PJSC|PLLC|JSC|PLC))\b'
        ]
    
    @classmethod
    def get_business_keywords(cls, category: Optional[str] = None) -> Dict[str, List[str]]:
        """Get business keywords for specific category or all categories"""
        if category:
            return {category: cls.BUSINESS_CATEGORIES.get(category, [])}
        return cls.BUSINESS_CATEGORIES.copy()
    
    @classmethod
    def should_use_ai_validation(cls, confidence: float) -> bool:
        """Determine if AI validation should be used based on confidence"""
        return confidence < cls.AI_VALIDATION_THRESHOLD
    
    @classmethod
    def should_use_semantic_search(cls, confidence: float) -> bool:
        """Determine if semantic search should be used based on confidence"""
        return confidence < cls.SEMANTIC_CONFIDENCE_THRESHOLD
    
    @classmethod
    def get_token_budget_per_field(cls) -> int:
        """Calculate token budget per field based on total limit"""
        return cls.MAX_TOTAL_TOKENS_PER_DOCUMENT // 3  # Divide by 3 fields
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration settings"""
        validations = {
            "token_limits_valid": cls.MAX_TOKENS_PER_FIELD > 0 and cls.MAX_TOTAL_TOKENS_PER_DOCUMENT > 0,
            "confidence_thresholds_valid": all(0.0 <= threshold <= 1.0 for threshold in [
                cls.PATTERN_CONFIDENCE_THRESHOLD,
                cls.SEMANTIC_CONFIDENCE_THRESHOLD,
                cls.AI_VALIDATION_THRESHOLD,
                cls.GEOGRAPHICAL_CONFIDENCE_THRESHOLD
            ]),
            "pattern_weights_valid": all(0.0 <= weight <= 1.0 for weight in cls.PATTERN_WEIGHTS.values()),
            "search_config_valid": cls.MAX_CHUNKS_PER_SEARCH > 0 and cls.CHUNK_OVERLAP_TOKENS >= 0
        }
        return validations


# Environment-specific configurations
class DevelopmentConfig(SmartExtractionConfig):
    """Development environment configuration"""
    LOG_TOKEN_USAGE = True
    ENABLE_PERFORMANCE_TRACKING = True
    AI_VALIDATION_THRESHOLD = 0.5  # Lower threshold for more AI usage in dev


class ProductionConfig(SmartExtractionConfig):
    """Production environment configuration"""
    LOG_TOKEN_USAGE = False
    ENABLE_PERFORMANCE_TRACKING = True
    AI_VALIDATION_THRESHOLD = 0.7  # Higher threshold to save tokens in production


class TestingConfig(SmartExtractionConfig):
    """Testing environment configuration"""
    MAX_TOKENS_PER_FIELD = 500
    MAX_TOTAL_TOKENS_PER_DOCUMENT = 5000
    ENABLE_FALLBACK_TO_LEGACY = False  # Force smart extraction in tests


def get_config() -> SmartExtractionConfig:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()