#!/usr/bin/env python3
"""
Test token usage for optimized prompts
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.ai_prompts import AIPrompts

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token"""
    return len(text) // 4

def test_prompt_optimization():
    """Test the optimized prompt token usage"""
    
    # Sample question and context
    question = "Does the financial statement comply with IAS 1 requirements for presentation of financial statements?"
    context = "Financial Statements for ABC Company\n" + "Sample context text that would normally be much longer..." * 50
    context = context[:4000]  # Ensure 4000 char limit
    
    # Generate the full prompt
    full_prompt = AIPrompts.get_full_compliance_analysis_prompt(question, context)
    
    # Estimate tokens
    total_chars = len(full_prompt)
    estimated_tokens = estimate_tokens(full_prompt)
    
    print("=== OPTIMIZED PROMPT TOKEN ANALYSIS ===")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"Azure limit: 8,192 tokens")
    print(f"Tokens remaining: {8192 - estimated_tokens:,}")
    
    if estimated_tokens < 8192:
        print("✅ UNDER TOKEN LIMIT - Should work!")
    else:
        print("❌ OVER TOKEN LIMIT - Still needs optimization")
    
    # Component breakdown
    print("\n=== COMPONENT BREAKDOWN ===")
    system_prompt = AIPrompts.get_compliance_analysis_system_prompt()
    base_prompt = AIPrompts.get_compliance_analysis_base_prompt(question, context)
    instructions = AIPrompts.get_compliance_analysis_instructions()
    
    print(f"System prompt: {estimate_tokens(system_prompt):,} tokens ({len(system_prompt)} chars)")
    print(f"Base prompt: {estimate_tokens(base_prompt):,} tokens ({len(base_prompt)} chars)")
    print(f"Instructions: {estimate_tokens(instructions):,} tokens ({len(instructions)} chars)")
    print(f"Total: {estimate_tokens(full_prompt):,} tokens")
    
    # Show first 500 chars of optimized prompt
    print("\n=== SAMPLE PROMPT (first 500 chars) ===")
    print(full_prompt[:500] + "...")

if __name__ == "__main__":
    test_prompt_optimization()