#!/usr/bin/env python3
"""
Question Category Mapper
Analyzes all checklist questions and maps them to semantic categories for intelligent retrieval.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Installing spaCy English model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class QuestionCategoryMapper:
    def __init__(self, frameworks_dir: str):
        self.frameworks_dir = Path(frameworks_dir)
        self.questions = []
        self.categorized_questions = []
        
        # Define category patterns based on our analysis
        self.category_patterns = {
            'MEASUREMENT': {
                'keywords': ['fair value', 'measure', 'measurement', 'amortized cost', 'present value', 
                           'carrying amount', 'valuation', 'revaluation', 'cost model', 'fair value model'],
                'verbs': ['measure', 'valued', 'valued at', 'determines', 'calculated']
            },
            'RECOGNITION': {
                'keywords': ['recognize', 'recognition', 'derecognize', 'derecognition', 'initial recognition',
                           'when to recognize', 'criteria for recognition'],
                'verbs': ['recognize', 'derecognize', 'record', 'account for']
            },
            'DISCLOSURE': {
                'keywords': ['disclose', 'disclosure', 'information', 'note', 'explain', 'describe',
                           'provide information', 'additional information', 'reconciliation'],
                'verbs': ['disclose', 'provide', 'explain', 'describe', 'reconcile']
            },
            'PRESENTATION': {
                'keywords': ['present', 'presentation', 'classify', 'classification', 'separate',
                           'line item', 'statement', 'current', 'non-current'],
                'verbs': ['present', 'classify', 'separate', 'include', 'exclude']
            }
        }
        
        # Define topic patterns
        self.topic_patterns = {
            'FINANCIAL_INSTRUMENTS': ['financial instrument', 'financial asset', 'financial liability', 
                                    'derivative', 'hedge', 'fair value through profit'],
            'PROPERTY_ASSETS': ['investment property', 'property plant equipment', 'intangible asset',
                              'biological asset', 'right-of-use asset'],
            'REVENUE_PERFORMANCE': ['revenue', 'business combination', 'share-based payment',
                                  'employee benefit', 'earnings per share'],
            'FINANCIAL_POSITION': ['inventory', 'provision', 'contingent', 'related party',
                                 'segment', 'cash flow', 'going concern'],
            'IMPAIRMENT': ['impairment', 'impairment loss', 'recoverable amount', 'value in use'],
            'TAX': ['tax', 'deferred tax', 'current tax', 'income tax'],
            'FOREIGN_CURRENCY': ['foreign currency', 'exchange rate', 'functional currency',
                                'presentation currency'],
            'LEASES': ['lease', 'lessee', 'lessor', 'lease liability', 'right-of-use']
        }
        
        # Define requirement type patterns
        self.requirement_patterns = {
            'MANDATORY': ['does the entity', 'shall the entity', 'must the entity', 'is required'],
            'CONDITIONAL': ['if', 'when', 'unless', 'except', 'provided that'],
            'TRANSITIONAL': ['initial application', 'first-time adoption', 'transition', 'comparative'],
            'OPTIONAL': ['may', 'can', 'is permitted', 'option', 'choice']
        }

    def load_all_questions(self) -> List[Dict]:
        """Load all questions from all framework JSON files."""
        questions = []
        
        for framework_dir in self.frameworks_dir.iterdir():
            if framework_dir.is_dir():
                for json_file in framework_dir.glob('*.json'):
                    try:
                        with open(json_file, 'r', encoding='utf-8-sig') as f:
                            data = json.load(f)
                            
                        if 'sections' in data:
                            for section in data['sections']:
                                if 'items' in section:
                                    for item in section['items']:
                                        question_data = {
                                            'id': item.get('id', ''),
                                            'question': item.get('question', item.get('requirement', '')),
                                            'reference': item.get('reference', item.get('paragraph', '')),
                                            'section': item.get('section', section.get('section', '')),
                                            'framework': framework_dir.name,
                                            'standard': json_file.stem,
                                            'source_file': str(json_file)
                                        }
                                        if question_data['question']:
                                            questions.append(question_data)
                                            
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error reading {json_file}: {e}")
                        continue
        
        print(f"Loaded {len(questions)} questions from all frameworks")
        return questions

    def categorize_question(self, question_text: str) -> Tuple[str, str, str, List[str]]:
        """Categorize a single question using NLP and pattern matching."""
        question_lower = question_text.lower()
        
        # Determine Category
        category_scores = {}
        for category, patterns in self.category_patterns.items():
            score = 0
            for keyword in patterns['keywords']:
                if keyword in question_lower:
                    score += 2
            for verb in patterns['verbs']:
                if verb in question_lower:
                    score += 1
            category_scores[category] = score
        
        category = max(category_scores, key=category_scores.get) if any(category_scores.values()) else 'OTHER'
        
        # Determine Topic
        topic_scores = {}
        for topic, keywords in self.topic_patterns.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            topic_scores[topic] = score
        
        topic = max(topic_scores, key=topic_scores.get) if any(topic_scores.values()) else 'GENERAL'
        
        # Determine Requirement Type
        requirement_scores = {}
        for req_type, patterns in self.requirement_patterns.items():
            score = sum(1 for pattern in patterns if pattern in question_lower)
            requirement_scores[req_type] = score
        
        requirement = max(requirement_scores, key=requirement_scores.get) if any(requirement_scores.values()) else 'MANDATORY'
        
        # Extract semantic search terms using spaCy
        doc = nlp(question_text)
        search_terms = []
        
        # Extract key phrases and entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep short phrases
                search_terms.append(chunk.text.lower())
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'LAW']:  # Relevant entity types
                search_terms.append(ent.text.lower())
        
        # Extract important keywords
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Combine and deduplicate search terms
        search_terms.extend(keywords)
        search_terms = list(set(search_terms))[:10]  # Limit to top 10 terms
        
        return category, topic, requirement, search_terms

    def process_all_questions(self):
        """Process and categorize all questions."""
        self.questions = self.load_all_questions()
        
        print("Categorizing questions...")
        for i, question_data in enumerate(self.questions):
            if i % 100 == 0:
                print(f"Processed {i}/{len(self.questions)} questions")
            
            category, topic, requirement, search_terms = self.categorize_question(question_data['question'])
            
            categorized_question = {
                **question_data,
                'category': category,
                'topic': topic,
                'requirement_type': requirement,
                'search_terms': search_terms,
                'question_length': len(question_data['question']),
                'word_count': len(question_data['question'].split())
            }
            
            self.categorized_questions.append(categorized_question)
        
        print(f"Categorized {len(self.categorized_questions)} questions")

    def generate_statistics(self) -> Dict:
        """Generate statistics about categorized questions."""
        df = pd.DataFrame(self.categorized_questions)
        
        stats = {
            'total_questions': len(df),
            'category_distribution': df['category'].value_counts().to_dict(),
            'topic_distribution': df['topic'].value_counts().to_dict(),
            'requirement_distribution': df['requirement_type'].value_counts().to_dict(),
            'framework_distribution': df['framework'].value_counts().to_dict(),
            'avg_question_length': df['question_length'].mean(),
            'avg_word_count': df['word_count'].mean()
        }
        
        return stats

    def export_categorized_questions(self, output_file: str):
        """Export categorized questions to JSON."""
        output_data = {
            'metadata': {
                'total_questions': len(self.categorized_questions),
                'generation_date': pd.Timestamp.now().isoformat(),
                'categories': list(self.category_patterns.keys()),
                'topics': list(self.topic_patterns.keys()),
                'requirement_types': list(self.requirement_patterns.keys())
            },
            'statistics': self.generate_statistics(),
            'questions': self.categorized_questions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported categorized questions to {output_file}")

    def create_category_lookup(self, output_file: str):
        """Create a fast lookup table for semantic search."""
        lookup_table = {}
        
        for question in self.categorized_questions:
            key = f"{question['category']}_{question['topic']}"
            if key not in lookup_table:
                lookup_table[key] = []
            
            lookup_table[key].append({
                'id': question['id'],
                'question': question['question'],
                'search_terms': question['search_terms'],
                'reference': question['reference'],
                'standard': question['standard']
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(lookup_table, f, indent=2, ensure_ascii=False)
        
        print(f"Created category lookup table: {output_file}")


def main():
    """Main execution function."""
    frameworks_dir = "checklist_data/frameworks"
    
    if not os.path.exists(frameworks_dir):
        print(f"Error: Directory {frameworks_dir} not found")
        return
    
    # Create output directory
    output_dir = "categorized_questions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize mapper and process questions
    mapper = QuestionCategoryMapper(frameworks_dir)
    mapper.process_all_questions()
    
    # Export results
    mapper.export_categorized_questions(f"{output_dir}/categorized_questions.json")
    mapper.create_category_lookup(f"{output_dir}/category_lookup.json")
    
    # Display statistics
    stats = mapper.generate_statistics()
    print("\n=== CATEGORIZATION STATISTICS ===")
    print(f"Total Questions: {stats['total_questions']}")
    print(f"\nCategory Distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count} ({count/stats['total_questions']*100:.1f}%)")
    
    print(f"\nTop Topics:")
    for topic, count in list(stats['topic_distribution'].items())[:10]:
        print(f"  {topic}: {count}")
    
    print(f"\nRequirement Types:")
    for req_type, count in stats['requirement_distribution'].items():
        print(f"  {req_type}: {count}")


if __name__ == "__main__":
    main()