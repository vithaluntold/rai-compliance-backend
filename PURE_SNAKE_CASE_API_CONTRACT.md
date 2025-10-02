# BACKEND API CONTRACT - PURE SNAKE_CASE
## Final Integration Specification

**Backend maintains PURE snake_case standards. Frontend MUST adapt.**

### API Endpoint Response Format
```json
{
  "document_id": "RAI-02102025-xxxxx",
  "status": "COMPLETED",
  "metadata_extraction": "COMPLETED", 
  "metadata": {
    "company_name": "Phoenix Group PLC",
    "nature_of_business": "Insurance and Financial Services",
    "operational_demographics": "UK-based multinational insurance company", 
    "financial_statements_type": "Consolidated Financial Statements"
  },
  "message": "Analysis complete"
}
```

### Frontend Integration Requirements

**CRITICAL: Frontend JavaScript must use snake_case field names:**

```javascript
// ✅ CORRECT - Use snake_case as backend provides
const companyName = response.metadata.company_name;
const businessNature = response.metadata.nature_of_business;
const demographics = response.metadata.operational_demographics;
const statementType = response.metadata.financial_statements_type;

// ❌ WRONG - camelCase will NOT work
const companyName = response.metadata.companyName; // UNDEFINED!
```

### Status Checking Logic
```javascript
// Frontend polling logic
if (response.status === 'COMPLETED' && response.metadata_extraction === 'COMPLETED') {
    // Metadata is ready - display framework selection
    displayFrameworkSelection(response.metadata);
}
```

### Field Mapping for Display
```javascript
// Map backend fields to user-friendly labels
const fieldLabels = {
    'company_name': 'Company Name',
    'nature_of_business': 'Nature of Business', 
    'operational_demographics': 'Operational Demographics',
    'financial_statements_type': 'Financial Statements Type'
};
```

**NO BACKEND CHANGES NEEDED - Frontend must conform to backend API!**