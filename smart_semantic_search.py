#!/usr/bin/env python3
"""
Semantic Search Enhancement Module
Uses categorized questions to provide intelligent document retrieval for compliance analysis.
"""

import json
import os
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class SmartComplianceRetriever:
    def __init__(self, categorized_questions_file: str):
        """Initialize the smart retrieval system."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality model
        
        # Load categorized questions
        with open(categorized_questions_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.questions = self.data['questions']
        self.question_embeddings = None
        self.nlp = spacy.load("en_core_web_sm")
        
        # Build category mappings
        self.build_category_mappings()
        
        # Precompute question embeddings for fast retrieval
        self.precompute_embeddings()

    def build_category_mappings(self):
        """Build fast lookup mappings for categories."""
        self.category_map = {}
        self.topic_map = {}
        self.requirement_map = {}
        
        for q in self.questions:
            # Category mapping
            cat = q['category']
            if cat not in self.category_map:
                self.category_map[cat] = []
            self.category_map[cat].append(q)
            
            # Topic mapping
            topic = q['topic']
            if topic not in self.topic_map:
                self.topic_map[topic] = []
            self.topic_map[topic].append(q)
            
            # Requirement mapping
            req = q['requirement_type']
            if req not in self.requirement_map:
                self.requirement_map[req] = []
            self.requirement_map[req].append(q)

    def precompute_embeddings(self):
        """Precompute embeddings for all questions for fast similarity search."""
        question_texts = [q['question'] for q in self.questions]
        print(f"Computing embeddings for {len(question_texts)} questions...")
        self.question_embeddings = self.model.encode(question_texts)
        print("Embeddings computed successfully.")

    def classify_user_question(self, question: str) -> Tuple[str, str, str, List[str]]:
        """Classify an incoming user question to determine search strategy."""
        question_lower = question.lower()
        
        # Determine likely category
        category_scores = {
            'MEASUREMENT': sum([1 for kw in ['fair value', 'measure', 'valuation', 'cost', 'amount'] if kw in question_lower]),
            'RECOGNITION': sum([1 for kw in ['recognize', 'record', 'when', 'criteria'] if kw in question_lower]),
            'DISCLOSURE': sum([1 for kw in ['disclose', 'information', 'note', 'explain', 'describe'] if kw in question_lower]),
            'PRESENTATION': sum([1 for kw in ['present', 'classify', 'separate', 'statement', 'line item'] if kw in question_lower])
        }
        category = max(category_scores, key=category_scores.get) if any(category_scores.values()) else 'GENERAL'
        
        # Determine topic
        topic_scores = {
            'FINANCIAL_INSTRUMENTS': sum([1 for kw in ['financial instrument', 'derivative', 'hedge'] if kw in question_lower]),
            'PROPERTY_ASSETS': sum([1 for kw in ['investment property', 'property', 'asset', 'intangible'] if kw in question_lower]),
            'REVENUE_PERFORMANCE': sum([1 for kw in ['revenue', 'business combination', 'share-based'] if kw in question_lower]),
            'IMPAIRMENT': sum([1 for kw in ['impairment', 'recoverable'] if kw in question_lower]),
            'TAX': sum([1 for kw in ['tax', 'deferred tax'] if kw in question_lower]),
            'LEASES': sum([1 for kw in ['lease', 'lessee', 'lessor'] if kw in question_lower])
        }
        topic = max(topic_scores, key=topic_scores.get) if any(topic_scores.values()) else 'GENERAL'
        
        # Extract search terms using NLP
        doc = self.nlp(question)
        search_terms = []
        
        # Key phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                search_terms.append(chunk.text.lower())
        
        # Important keywords
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 2):
                search_terms.append(token.lemma_.lower())
        
        search_terms = list(set(search_terms))[:8]  # Top 8 terms
        
        return category, topic, 'MANDATORY', search_terms

    def find_relevant_questions(self, user_question: str, max_questions: int = 10) -> List[Dict]:
        """Find most relevant questions using semantic similarity."""
        # Get user question embedding
        user_embedding = self.model.encode([user_question])
        
        # Compute similarities
        similarities = cosine_similarity(user_embedding, self.question_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:max_questions]
        
        relevant_questions = []
        for idx in top_indices:
            question_data = self.questions[idx].copy()
            question_data['similarity_score'] = float(similarities[idx])
            relevant_questions.append(question_data)
        
        return relevant_questions

    def get_targeted_search_terms(self, user_question: str) -> Dict[str, List[str]]:
        """Get targeted search terms based on question classification."""
        category, topic, _, search_terms = self.classify_user_question(user_question)
        
        # Find similar questions
        relevant_questions = self.find_relevant_questions(user_question, max_questions=5)
        
        # Aggregate search terms from similar questions
        aggregated_terms = set(search_terms)
        for q in relevant_questions:
            if q['similarity_score'] > 0.6:  # High similarity threshold
                aggregated_terms.update(q.get('search_terms', []))
        
        # Category-specific terms
        category_terms = {
            'MEASUREMENT': ['fair value', 'measurement', 'valuation', 'amortized cost', 'carrying amount'],
            'DISCLOSURE': ['disclose', 'information', 'note', 'explanation', 'description'],
            'PRESENTATION': ['present', 'classify', 'separate', 'line item', 'statement'],
            'RECOGNITION': ['recognize', 'recognition', 'criteria', 'when to record']
        }
        
        if category in category_terms:
            aggregated_terms.update(category_terms[category])
        
        return {
            'category': category,
            'topic': topic,
            'search_terms': list(aggregated_terms)[:15],  # Limit to 15 terms
            'relevant_questions': relevant_questions[:3]  # Top 3 for context
        }

    def generate_document_search_strategy(self, user_question: str) -> Dict[str, Any]:
        """Generate a comprehensive search strategy for document retrieval."""
        search_data = self.get_targeted_search_terms(user_question)
        
        # Generate section-specific search patterns
        section_patterns = {
            'DISCLOSURE': ['Note', 'Notes to', 'Disclosure', 'Additional information'],
            'MEASUREMENT': ['Fair value', 'Measurement', 'Valuation', 'Policy'],
            'PRESENTATION': ['Statement of', 'Balance Sheet', 'Income Statement', 'Classification'],
            'RECOGNITION': ['Accounting Policy', 'Recognition', 'Criteria']
        }
        
        strategy = {
            'primary_search_terms': search_data['search_terms'],
            'category': search_data['category'],
            'topic': search_data['topic'],
            'section_patterns': section_patterns.get(search_data['category'], []),
            'relevant_questions': search_data['relevant_questions'],
            'search_priority': self._get_search_priority(search_data['category']),
            'expected_content_length': self._get_expected_content_length(search_data['category'])
        }
        
        return strategy

    def _get_search_priority(self, category: str) -> List[str]:
        """Get search priority based on category."""
        priority_map = {
            'DISCLOSURE': ['notes', 'disclosures', 'explanations', 'additional information'],
            'MEASUREMENT': ['accounting policies', 'fair value', 'measurement basis', 'valuation'],
            'PRESENTATION': ['financial statements', 'balance sheet', 'income statement', 'classification'],
            'RECOGNITION': ['accounting policies', 'recognition criteria', 'when to recognize']
        }
        return priority_map.get(category, ['general information'])

    def _get_expected_content_length(self, category: str) -> int:
        """Get expected content length for different categories."""
        length_map = {
            'DISCLOSURE': 800,  # Longer for disclosure requirements
            'MEASUREMENT': 600,  # Medium for measurement details
            'PRESENTATION': 400,  # Shorter for presentation requirements
            'RECOGNITION': 500   # Medium for recognition criteria
        }
        return length_map.get(category, 600)

    def create_smart_search_config(self, output_file: str):
        """Create configuration file for smart search implementation."""
        config = {
            'category_weights': {
                'DISCLOSURE': 0.4,
                'PRESENTATION': 0.25,
                'MEASUREMENT': 0.25,
                'RECOGNITION': 0.1
            },
            'topic_priorities': {
                'FINANCIAL_INSTRUMENTS': 1.0,
                'PROPERTY_ASSETS': 0.9,
                'REVENUE_PERFORMANCE': 0.8,
                'IMPAIRMENT': 0.7,
                'TAX': 0.6,
                'LEASES': 0.6,
                'GENERAL': 0.3
            },
            'search_strategies': {
                'DISCLOSURE': {
                    'chunk_size': 800,
                    'overlap': 100,
                    'focus_sections': ['notes', 'disclosures', 'additional information'],
                    'keywords_weight': 0.7,
                    'semantic_weight': 0.3
                },
                'MEASUREMENT': {
                    'chunk_size': 600,
                    'overlap': 80,
                    'focus_sections': ['accounting policies', 'fair value', 'measurement'],
                    'keywords_weight': 0.6,
                    'semantic_weight': 0.4
                },
                'PRESENTATION': {
                    'chunk_size': 400,
                    'overlap': 60,
                    'focus_sections': ['financial statements', 'classification'],
                    'keywords_weight': 0.8,
                    'semantic_weight': 0.2
                },
                'RECOGNITION': {
                    'chunk_size': 500,
                    'overlap': 70,
                    'focus_sections': ['accounting policies', 'recognition criteria'],
                    'keywords_weight': 0.7,
                    'semantic_weight': 0.3
                }
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Smart search configuration saved to {output_file}")


def main():
    """Test the smart retrieval system."""
    categorized_file = "categorized_questions/categorized_questions.json"
    
    if not os.path.exists(categorized_file):
        print(f"Error: {categorized_file} not found. Run create_question_directory.py first.")
        return
    
    # Initialize retriever
    retriever = SmartComplianceRetriever(categorized_file)
    
    # Test questions
    test_questions = [
        "Does the entity disclose fair value of investment property?",
        "How should the entity recognize impairment losses?",
        "What information should be presented about lease liabilities?",
        "Does the entity classify financial instruments correctly?"
    ]
    
    print("=== SMART RETRIEVAL TESTING ===")
    for question in test_questions:
        print(f"\nUser Question: {question}")
        strategy = retriever.generate_document_search_strategy(question)
        
        print(f"Category: {strategy['category']}")
        print(f"Topic: {strategy['topic']}")
        print(f"Search Terms: {', '.join(strategy['primary_search_terms'][:8])}")
        print(f"Expected Content Length: {strategy['expected_content_length']} chars")
        print(f"Relevant Questions Found: {len(strategy['relevant_questions'])}")
    
    # Create configuration
    retriever.create_smart_search_config("categorized_questions/smart_search_config.json")
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed {len(retriever.questions)} categorized questions")
    print(f"Categories: {list(retriever.category_map.keys())}")
    print(f"Topics: {list(retriever.topic_map.keys())}")


if __name__ == "__main__":
    main()