#!/usr/bin/env python3
"""
Test script to validate JSON serialization fixes for GeographicalEntity objects.
"""

import json
import sys
from pathlib import Path

# Add the render-backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from services.geographical_service import GeographicalEntity, GeographicalDetectionService
from routes.analysis_routes import _ensure_json_serializable


def test_geographical_entity_serialization():
    """Test that GeographicalEntity objects can be serialized to JSON."""
    print("🧪 Testing GeographicalEntity JSON serialization...")
    
    # Create a sample GeographicalEntity
    entity = GeographicalEntity(
        name="United Arab Emirates",
        type="country",
        iso_code="AE",
        region="Middle East",
        confidence=0.95,
        accuracy=0.9,
        completeness=0.85,
        source_text="Operations in the United Arab Emirates",
        page_reference="page_1"
    )
    
    # Test direct dictionary conversion
    entity_dict = entity.to_dict()
    print(f"✅ Entity to_dict(): {entity_dict}")
    
    # Test JSON serialization of dictionary
    json_str = json.dumps(entity_dict, indent=2)
    print(f"✅ JSON serialization successful: {len(json_str)} characters")
    
    # Test the defensive serialization function
    test_data = {
        "metadata": {
            "operational_demographics": {
                "value": "United Arab Emirates",
                "confidence": 0.9,
                "geographical_entities": [entity]  # Raw entity object
            }
        },
        "optimization_metrics": {
            "tokens_used": 1500,
            "extraction_methods_used": {"pattern", "semantic"}  # Set object
        }
    }
    
    print("🔧 Testing defensive serialization with mixed data types...")
    serializable_data = _ensure_json_serializable(test_data)
    
    # Verify that geographical_entities contains dicts, not objects
    geo_entities = serializable_data["metadata"]["operational_demographics"]["geographical_entities"]
    assert isinstance(geo_entities[0], dict), "GeographicalEntity should be converted to dict"
    assert geo_entities[0]["name"] == "United Arab Emirates", "Data should be preserved"
    
    # Verify that sets are converted to lists
    methods_used = serializable_data["optimization_metrics"]["extraction_methods_used"]
    assert isinstance(methods_used, list), "Set should be converted to list"
    
    # Test final JSON serialization
    final_json = json.dumps(serializable_data, indent=2)
    print(f"✅ Final JSON serialization successful: {len(final_json)} characters")
    
    print("🎉 All serialization tests passed!")
    return True


def test_geographical_service():
    """Test the geographical service detection."""
    print("\n🌍 Testing GeographicalDetectionService...")
    
    service = GeographicalDetectionService()
    
    # Test with sample text
    sample_text = "The company operates primarily in the United Arab Emirates and Saudi Arabia."
    entities = service.detect_geographical_entities(sample_text)
    
    print(f"📍 Detected {len(entities)} geographical entities")
    for entity in entities:
        print(f"  - {entity.name} ({entity.type}, confidence: {entity.confidence:.2f})")
        
        # Test that each entity can be converted to dict
        entity_dict = entity.to_dict()
        json.dumps(entity_dict)  # Should not raise an exception
    
    print("✅ GeographicalDetectionService test passed!")
    return True


if __name__ == "__main__":
    try:
        test_geographical_entity_serialization()
        test_geographical_service()
        print("\n🚀 All tests completed successfully!")
        print("✅ JSON serialization fixes are working properly")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)