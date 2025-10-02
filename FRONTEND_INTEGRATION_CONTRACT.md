# DEFINITIVE BACKEND API CONTRACT

## Production API Endpoint
```
GET https://rai-compliance-backend.onrender.com/api/v1/analysis/documents/{document_id}
```

## Response Format (CONFIRMED FROM PRODUCTION TESTING)

### When Metadata Extraction Complete:
```json
{
  "document_id": "RAI-02102025-E1JZP-HV77U",
  "status": "COMPLETED",
  "metadata_extraction": "COMPLETED", 
  "compliance_analysis": "PENDING",
  "processing_mode": "smart",
  "smart_categorization": {
    "total_categories": 0,
    "content_chunks": 0,
    "categorization_complete": false,
    "categories_found": []
  },
  "metadata": {
    "company_name": "The Shareholders of Phoenix Group",
    "nature_of_business": "The Group is engaged in various businesses primarily...",
    "operational_demographics": "United Arab Emirates", 
    "financial_statements_type": "Consolidated"
  },
  "sections": [],
  "progress": {},
  "framework": null,
  "standards": [],
  "specialInstructions": null,
  "extensiveSearch": false,
  "message": "Metadata extraction completed successfully. Ready for framework selection."
}
```

## FRONTEND INTEGRATION REQUIREMENTS

### 1. Polling Logic
```javascript
// Frontend must poll this endpoint until:
if (response.status === "COMPLETED" && response.metadata_extraction === "COMPLETED") {
  // Metadata is ready - display framework selection
}
```

### 2. Field Mapping (CRITICAL)
```javascript
// Backend uses snake_case - Frontend must read:
const metadata = response.metadata;
const companyName = metadata.company_name;                    // ← EXACT FIELD NAME
const natureOfBusiness = metadata.nature_of_business;         // ← EXACT FIELD NAME  
const demographics = metadata.operational_demographics;       // ← EXACT FIELD NAME
const statementsType = metadata.financial_statements_type;    // ← EXACT FIELD NAME
```

### 3. Status Checking
```javascript
// Frontend trigger conditions:
const isMetadataReady = (
  response.status === "COMPLETED" && 
  response.metadata_extraction === "COMPLETED" &&
  Object.keys(response.metadata).length > 0
);

if (isMetadataReady) {
  // Show metadata fields and enable framework selection
}
```

### 4. Error Handling
```javascript
if (response.status === "FAILED") {
  // Show error: response.message
}
```

## FRONTEND CHANGES REQUIRED

The frontend must be updated to:

1. **Use snake_case field names** when reading metadata
2. **Check both status AND metadata_extraction** for completion  
3. **Handle the nested metadata object structure**
4. **Poll the correct endpoint** with proper error handling

## BACKEND STATUS: ✅ WORKING PERFECTLY
The backend API is consistent and returns the correct data format.
Frontend needs to conform to this API contract.