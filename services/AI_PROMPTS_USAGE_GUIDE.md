# AI Prompts Library Usage Guide

## Overview

All AI prompts used in the RAi Compliance Engine have been centralized in the `ai_prompts.py` library file. This makes it easier to find, modify, and maintain prompts across the system.

## File Location

```
backend/services/ai_prompts.py
```

## Quick Usage

### Import the prompts library:
```python
from services.ai_prompts import ai_prompts
```

### Use prompts in your code:
```python
# Main compliance analysis prompt
prompt = ai_prompts.get_full_compliance_analysis_prompt(
    question="Does the entity disclose fair value methodology?",
    context="Document content here...",
    enhanced_evidence=enhanced_data  # Optional
)

# System prompt for compliance analysis
system_prompt = ai_prompts.get_compliance_analysis_system_prompt()

# Metadata extraction prompts
system_prompt = ai_prompts.get_metadata_extraction_system_prompt()
user_prompt = ai_prompts.get_metadata_extraction_user_prompt(
    reference="company_name", 
    question_text="What is the company name?"
)
```

## Available Prompt Types

### 1. Compliance Analysis Prompts
- `get_compliance_analysis_system_prompt()` - System prompt for AI assistant
- `get_compliance_analysis_base_prompt(question, context)` - Base prompt template  
- `get_enhanced_evidence_section(enhanced_evidence)` - Enhanced evidence section
- `get_compliance_analysis_instructions()` - Standard instructions and format
- `get_full_compliance_analysis_prompt(question, context, enhanced_evidence)` - Complete assembled prompt

### 2. Metadata Extraction Prompts
- `get_metadata_extraction_system_prompt()` - System prompt for metadata extraction
- `get_metadata_extraction_user_prompt(reference, question_text)` - User prompt template

### 3. Vector Context Query Prompts
- `get_vector_metadata_system_prompt(has_vector_index)` - Vector-based metadata extraction
- `get_vector_metadata_user_prompt(document_id, question_text, reference, has_vector_index)` - Vector metadata user prompt
- `get_vector_compliance_system_prompt(has_vector_index)` - Vector-based compliance analysis
- `get_vector_compliance_user_prompt(document_id, question_text, reference, has_vector_index)` - Vector compliance user prompt

## How to Modify Prompts

### 1. Find the prompt you want to modify:
```python
# List all available prompt types
prompt_types = ai_prompts.get_all_prompt_types()
print(prompt_types)

# Get description of a specific prompt
description = ai_prompts.get_prompt_description("compliance_analysis_system")
print(description)
```

### 2. Edit the prompt in `ai_prompts.py`:
- Open `backend/services/ai_prompts.py`
- Find the method corresponding to your prompt type
- Modify the prompt text as needed
- Save the file

### 3. Changes take effect immediately:
- No need to restart the server for prompt changes
- All code using the prompts library will automatically use the updated prompts

## Benefits of Centralized Prompts

### ✅ **Easy to Find**
- All prompts in one location
- Clear naming conventions
- Comprehensive documentation

### ✅ **Easy to Modify** 
- Change once, updates everywhere
- No need to search through multiple files
- Version control for prompt changes

### ✅ **Consistent Formatting**
- Standardized prompt structure
- Reusable components
- Reduced duplication

### ✅ **Better Maintenance**
- Clear separation of concerns
- Easier testing and validation
- Documentation of prompt purposes

## Examples

### Adding a New Prompt Type

1. Add a new static method to the `AIPrompts` class:
```python
@staticmethod
def get_my_new_prompt(parameter1: str, parameter2: str) -> str:
    """Description of what this prompt does."""
    return f"""
    Your new prompt template here.
    Parameter 1: {parameter1}
    Parameter 2: {parameter2}
    """
```

2. Add it to the documentation lists:
```python
# In get_all_prompt_types()
return [
    # ... existing types ...
    "my_new_prompt"
]

# In get_prompt_description()
descriptions = {
    # ... existing descriptions ...
    "my_new_prompt": "Description of your new prompt"
}
```

3. Use it in your code:
```python
prompt = ai_prompts.get_my_new_prompt("value1", "value2")
```

### Modifying Existing Prompts

To modify the compliance analysis instructions:

1. Open `ai_prompts.py`
2. Find `get_compliance_analysis_instructions()`
3. Edit the returned string
4. Save the file

All code using this prompt will immediately use the updated version.

## Migration Status

✅ **Completed:**
- Main compliance analysis prompt (`analyze_chunk` method)
- Metadata extraction prompts
- Vector context query prompts
- System prompts for all AI interactions

✅ **Files Updated:**
- `backend/services/ai.py` - Now imports and uses `ai_prompts`
- `backend/services/ai_prompts.py` - New centralized prompts library

## Best Practices

1. **Use descriptive method names** that clearly indicate the prompt's purpose
2. **Include docstrings** explaining what each prompt does
3. **Use parameters** for dynamic content instead of hardcoding values
4. **Test prompts thoroughly** after making changes
5. **Document any special formatting requirements** in the prompt comments
6. **Keep related prompts grouped together** in the class

## Support

If you need to add new prompt types or modify existing ones, refer to the patterns established in `ai_prompts.py` or consult the development team.