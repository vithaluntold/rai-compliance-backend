"""
Metadata Models

Pydantic v2 models for structured metadata responses with optimization metrics
and consistent field validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal, Union

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    field_serializer,
    ValidationInfo,
    ConfigDict,
)


# ----------------------------------------------------------------------
# Core field models
# ----------------------------------------------------------------------

class MetadataField(BaseModel):
    """Base metadata field with confidence and extraction method tracking"""

    value: str = Field(..., description="Extracted field value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    extraction_method: Literal[
        "pattern", "semantic", "ai_validation", "geographical_service", "legacy", "none"
    ] = Field(..., description="Method used to extract this field")
    source_chunk_id: Optional[str] = Field(None, description="ID of the chunk where this field was found")

    @field_validator("value", mode="before")
    def validate_value(cls, v):
        """Ensure value is properly cleaned - strict validation, no fallbacks"""
        if v is None:
            raise ValueError("Field value cannot be None")
        cleaned_value = str(v).strip()
        if not cleaned_value:
            raise ValueError("Field value cannot be empty")
        return cleaned_value


class CompanyNameField(MetadataField):
    """Company name field with additional legal structure information"""

    legal_structure: Optional[str] = Field(None, description="Legal structure (Ltd, Inc, Corp, etc.)")
    full_name: Optional[str] = Field(None, description="Full company name if different from value")

    @field_validator("legal_structure", mode="after")
    def extract_legal_structure(cls, v, info: ValidationInfo):
        """Extract legal structure from company name if not provided"""
        if v is None:
            company_name = (info.data or {}).get("value") or ""
            legal_suffixes = ["Ltd", "Inc", "Corp", "LLC", "Pty", "GmbH", "SA", "AG", "Limited", "Corporation"]
            for suffix in legal_suffixes:
                if suffix.lower() in company_name.lower():
                    return suffix
        return v


class NatureOfBusinessField(MetadataField):
    """Nature of business field with industry classification"""

    industry_classification: Optional[str] = Field(None, description="Primary industry classification")
    business_type: Optional[str] = Field(None, description="Specific business type or sector")
    keywords_matched: Optional[List[str]] = Field(None, description="Keywords that matched for classification")


class OperationalDemographicsField(MetadataField):
    """Operational demographics with geographical entity details"""

    geographical_entities: Optional[List[Dict[str, Any]]] = Field(
        None, description="Detected geographical entities with details"
    )
    regions_detected: Optional[List[str]] = Field(None, description="List of detected regions/countries")
    primary_location: Optional[str] = Field(None, description="Primary operational location")

    @property
    def geography_of_operations(self) -> str:
        """Extract geographical operations information for result page display."""
        geography_parts = []
        
        # Add primary location if available
        if self.primary_location and self.primary_location.strip():
            geography_parts.append(self.primary_location.strip())
        
        # Add detected regions, avoiding duplicates
        if self.regions_detected:
            for region in self.regions_detected:
                if region and region.strip() and region.strip() not in geography_parts:
                    geography_parts.append(region.strip())
        
        # Add geographical entities information
        if self.geographical_entities:
            for entity in self.geographical_entities:
                if isinstance(entity, dict):
                    # Extract location name from various possible keys
                    location_name = (
                        entity.get("name") or 
                        entity.get("location") or 
                        entity.get("place") or 
                        entity.get("country") or 
                        entity.get("region")
                    )
                    if location_name and str(location_name).strip() and str(location_name).strip() not in geography_parts:
                        geography_parts.append(str(location_name).strip())
        
        # Return consolidated geography information
        if geography_parts:
            # Remove duplicates while preserving order
            unique_parts = []
            for part in geography_parts:
                if part not in unique_parts:
                    unique_parts.append(part)
            return ", ".join(unique_parts)
        
        # Fallback to base value if available
        if self.value and self.value.strip():
            return self.value.strip()
        
        return "Geographic operations not specified"


# ----------------------------------------------------------------------
# Optimization metrics
# ----------------------------------------------------------------------

class OptimizationMetrics(BaseModel):
    """Metrics tracking the optimization performance"""

    tokens_used: int = Field(..., ge=0, description="Total tokens used in extraction")
    tokens_saved: int = Field(0, ge=0, description="Tokens saved compared to legacy method")
    cost_saved_usd: float = Field(0.0, ge=0.0, description="Cost saved in USD")
    processing_time_seconds: float = Field(..., gt=0.0, description="Total processing time")
    extraction_methods_used: List[str] = Field(..., description="List of extraction methods used")
    pattern_match_count: int = Field(0, ge=0, description="Number of pattern matches found")
    semantic_search_count: int = Field(0, ge=0, description="Number of semantic searches performed")
    ai_validation_count: int = Field(0, ge=0, description="Number of AI validation calls made")
    fallback_used: bool = Field(False, description="Whether fallback to legacy extraction was used")


# ----------------------------------------------------------------------
# Optimized metadata
# ----------------------------------------------------------------------

class OptimizedMetadata(BaseModel):
    """Complete optimized metadata response structure"""

    model_config = ConfigDict()

    company_name: CompanyNameField
    nature_of_business: NatureOfBusinessField
    operational_demographics: OperationalDemographicsField
    optimization_metrics: OptimizationMetrics
    extracted_at: datetime = Field(default_factory=datetime.now, description="Timestamp of extraction")
    document_id: Optional[str] = Field(None, description="Document ID this metadata belongs to")

    @field_serializer("extracted_at")
    def serialize_extracted_at(self, v: datetime):
        return v.isoformat()


# ----------------------------------------------------------------------
# Validation result
# ----------------------------------------------------------------------

class MetadataValidationResult(BaseModel):
    """Result of metadata validation"""

    is_valid: bool = Field(..., description="Whether the metadata is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Completeness score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")


# ----------------------------------------------------------------------
# Extraction request/response
# ----------------------------------------------------------------------

class MetadataExtractionRequest(BaseModel):
    """Request model for metadata extraction"""

    document_id: str = Field(..., description="Unique document identifier")
    chunks: List[str] = Field(..., min_length=1, description="Document chunks to process")
    force_ai_validation: bool = Field(False, description="Force AI validation even for high-confidence patterns")
    custom_patterns: Optional[Dict[str, List[str]]] = Field(None, description="Custom extraction patterns")


class MetadataExtractionResponse(BaseModel):
    """Response model for metadata extraction"""

    success: bool = Field(..., description="Whether extraction was successful")
    metadata: Optional[OptimizedMetadata] = Field(None, description="Extracted metadata")
    validation_result: Optional[MetadataValidationResult] = Field(None, description="Validation results")
    error_message: Optional[str] = Field(None, description="Error message if extraction failed")
    processing_time: float = Field(..., description="Total processing time in seconds")


# ----------------------------------------------------------------------
# Legacy models
# ----------------------------------------------------------------------

class LegacyMetadataField(BaseModel):
    """Legacy metadata field format for backward compatibility"""

    value: str
    confidence: float = 0.85


class LegacyMetadata(BaseModel):
    """Legacy metadata format for backward compatibility"""

    company_name: Union[LegacyMetadataField, str]
    nature_of_business: Union[LegacyMetadataField, str]
    operational_demographics: Union[LegacyMetadataField, str]


# ----------------------------------------------------------------------
# Validation functions
# ----------------------------------------------------------------------

def validate_metadata_response(metadata: Dict[str, Any]) -> MetadataValidationResult:
    """Validate metadata response structure and content"""
    errors: List[str] = []
    warnings: List[str] = []

    required_fields = ["company_name", "nature_of_business", "operational_demographics"]

    # Check required fields
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(metadata[field], dict):
            errors.append(f"Field {field} must be an object")
        elif "value" not in metadata[field]:
            errors.append(f"Field {field} must have a 'value' property")
        elif "confidence" not in metadata[field]:
            warnings.append(f"Field {field} missing confidence score")

    # Calculate comple
    completeness_score = 0.0
    valid_fields = 0
    for field in required_fields:
        if field in metadata and isinstance(metadata[field], dict):
            field_value = metadata[field].get("value", "")
            if field_value and field_value.strip():
                valid_fields += 1
                completeness_score += 1.0 / len(required_fields)

    # Calculate overall confidence score
    confidence_scores: List[float] = []
    for field in required_fields:
        if isinstance(metadata.get(field), dict) and "confidence" in metadata[field]:
            confidence_scores.append(metadata[field]["confidence"])
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    # Check for low confidence warnings
    for field in required_fields:
        if isinstance(metadata.get(field), dict) and metadata[field].get("confidence", 0) < 0.5:
            warnings.append(f"Low confidence for field {field}: {metadata[field]['confidence']}")

    return MetadataValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        completeness_score=completeness_score,
        confidence_score=overall_confidence,
    )


# ----------------------------------------------------------------------
# Conversion functions
# ----------------------------------------------------------------------

def convert_legacy_to_optimized(legacy_metadata: Dict[str, Any]) -> OptimizedMetadata:
    """Convert legacy metadata format to optimized format - strict validation, no fallbacks"""

    def convert_company_field(field_data) -> CompanyNameField:
        if isinstance(field_data, str):
            if not field_data.strip():
                raise ValueError("Company name cannot be empty")
            return CompanyNameField(value=field_data, confidence=0.85, extraction_method="legacy")
        elif isinstance(field_data, dict):
            value = field_data.get("value", "")
            if not value.strip():
                raise ValueError("Company name value cannot be empty")
            return CompanyNameField(
                value=value,
                confidence=field_data.get("confidence", 0.85),
                extraction_method=field_data.get("extraction_method", "legacy"),
                source_chunk_id=field_data.get("source_chunk_id"),
                legal_structure=field_data.get("legal_structure"),
                full_name=field_data.get("full_name"),
            )
        raise ValueError("Company name field is required and cannot be None")

    def convert_business_field(field_data) -> NatureOfBusinessField:
        if isinstance(field_data, str):
            if not field_data.strip():
                raise ValueError("Nature of business cannot be empty")
            return NatureOfBusinessField(value=field_data, confidence=0.85, extraction_method="legacy")
        elif isinstance(field_data, dict):
            value = field_data.get("value", "")
            if not value.strip():
                raise ValueError("Nature of business value cannot be empty")
            return NatureOfBusinessField(
                value=value,
                confidence=field_data.get("confidence", 0.85),
                extraction_method=field_data.get("extraction_method", "legacy"),
                source_chunk_id=field_data.get("source_chunk_id"),
                industry_classification=field_data.get("industry_classification"),
                business_type=field_data.get("business_type"),
                keywords_matched=field_data.get("keywords_matched"),
            )
        raise ValueError("Nature of business field is required and cannot be None")

    def convert_demographics_field(field_data) -> OperationalDemographicsField:
        if isinstance(field_data, str):
            if not field_data.strip():
                raise ValueError("Operational demographics cannot be empty")
            return OperationalDemographicsField(value=field_data, confidence=0.85, extraction_method="legacy")
        elif isinstance(field_data, dict):
            value = field_data.get("value", "")
            if not value.strip():
                raise ValueError("Operational demographics value cannot be empty")
            return OperationalDemographicsField(
                value=value,
                confidence=field_data.get("confidence", 0.85),
                extraction_method=field_data.get("extraction_method", "legacy"),
                source_chunk_id=field_data.get("source_chunk_id"),
                geographical_entities=field_data.get("geographical_entities"),
                regions_detected=field_data.get("regions_detected"),
                primary_location=field_data.get("primary_location"),
            )
        raise ValueError("Operational demographics field is required and cannot be None")

    # Validate required fields exist
    if "company_name" not in legacy_metadata or not legacy_metadata["company_name"]:
        raise ValueError("Missing required field: company_name")
    if "nature_of_business" not in legacy_metadata or not legacy_metadata["nature_of_business"]:
        raise ValueError("Missing required field: nature_of_business") 
    if "operational_demographics" not in legacy_metadata or not legacy_metadata["operational_demographics"]:
        raise ValueError("Missing required field: operational_demographics")

    return OptimizedMetadata(
        company_name=convert_company_field(legacy_metadata["company_name"]),
        nature_of_business=convert_business_field(legacy_metadata["nature_of_business"]),
        operational_demographics=convert_demographics_field(legacy_metadata["operational_demographics"]),
        optimization_metrics=OptimizationMetrics(
            tokens_used=legacy_metadata.get("tokens_used", 0),
            tokens_saved=0,
            cost_saved_usd=0.0,
            processing_time_seconds=legacy_metadata.get("processing_time", 0.0),
            extraction_methods_used=["legacy"],
            pattern_match_count=0,
            semantic_search_count=0,
            ai_validation_count=0,
            fallback_used=False,
        ),
        document_id=legacy_metadata.get("document_id"),
    )
