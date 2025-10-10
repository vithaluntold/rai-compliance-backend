"""
Content Filter Service - Reject audit reports, accept only financial statements
"""
import logging
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ContentFilter:
    """Filter to reject audit reports and accept only financial statements content."""
    
    def __init__(self):
        # Audit report rejection patterns (case-insensitive)
        self.audit_rejection_patterns = [
            r'independent\s+auditor[\'s]*\s+report',
            r'report\s+on\s+the\s+audit',
            r'auditor[\'s]*\s+responsibility',
            r'management[\'s]*\s+responsibility',
            r'basis\s+for\s+opinion',
            r'key\s+audit\s+matters',
            r'material\s+uncertainty\s+related\s+to\s+going\s+concern',
            r'other\s+information',
            r'responsibilities\s+of\s+management',
            r'auditor[\'s]*\s+responsibilities\s+for\s+the\s+audit',
            r'we\s+have\s+audited',
            r'in\s+our\s+opinion',
            r'basis\s+for\s+qualified\s+opinion',
            r'emphasis\s+of\s+matter',
            r'other\s+matter',
            r'report\s+on\s+other\s+legal',
            r'ISA\s+\d+',  # International Standards on Auditing references
            r'audit\s+procedures',
            r'reasonable\s+assurance',
            r'material\s+misstatement',
            r'audit\s+evidence',
            r'professional\s+skepticism',
            r'PCAOB\s+standards',  # Public Company Accounting Oversight Board
        ]
        
        # Financial statement acceptance patterns (case-insensitive) 
        self.financial_statement_patterns = [
            r'consolidated\s+statement\s+of\s+financial\s+position',
            r'consolidated\s+balance\s+sheet',
            r'statement\s+of\s+financial\s+position',
            r'balance\s+sheet',
            r'consolidated\s+statement\s+of\s+profit\s+or\s+loss',
            r'consolidated\s+income\s+statement',
            r'statement\s+of\s+profit\s+or\s+loss',
            r'income\s+statement',
            r'consolidated\s+statement\s+of\s+comprehensive\s+income',
            r'statement\s+of\s+comprehensive\s+income',
            r'consolidated\s+statement\s+of\s+cash\s+flows',
            r'statement\s+of\s+cash\s+flows',
            r'cash\s+flow\s+statement',
            r'consolidated\s+statement\s+of\s+changes\s+in\s+equity',
            r'statement\s+of\s+changes\s+in\s+equity',
            r'statement\s+of\s+stockholders[\']*\s+equity',
            r'notes\s+to\s+the\s+consolidated\s+financial\s+statements',
            r'notes\s+to\s+the\s+financial\s+statements',
            r'notes\s+to\s+consolidated\s+accounts',
            r'notes\s+to\s+accounts',
            r'director[s\']*\s+report',
            r'chairman[\'s]*\s+statement',
            r'chief\s+executive[\'s]*\s+review',
            r'management\s+discussion\s+and\s+analysis',
            r'accounting\s+policies',
            r'significant\s+accounting\s+policies',
            r'basis\s+of\s+preparation',
            r'note\s+\d+',  # Note 1, Note 2, etc.
        ]
        
        # Strong indicators of audit report content (instant rejection)
        self.audit_rejection_strong = [
            'we have audited the consolidated financial statements',
            'in our opinion, the consolidated financial statements',
            'we conducted our audit in accordance with',
            'those standards require that we plan and perform',
            'reasonable assurance about whether the financial statements',
            'an audit involves performing procedures to obtain audit evidence',
            'the procedures selected depend on the auditor\'s judgment',
            'we believe that the audit evidence we have obtained',
        ]
        
    def analyze_content_type(self, text: str, document_id: str = "") -> Dict[str, any]:
        """
        Analyze content and determine if it should be processed or rejected.
        
        Returns:
        {
            'should_process': bool,
            'content_type': str,
            'rejection_reason': str or None,
            'confidence': float,
            'patterns_matched': List[str]
        }
        """
        if not text or len(text.strip()) < 50:
            return {
                'should_process': False,
                'content_type': 'insufficient_content',
                'rejection_reason': 'Content too short for analysis',
                'confidence': 1.0,
                'patterns_matched': []
            }
        
        text_lower = text.lower()
        
        # Check for strong audit report indicators (instant rejection)
        for strong_pattern in self.audit_rejection_strong:
            if strong_pattern.lower() in text_lower:
                return {
                    'should_process': False,
                    'content_type': 'audit_report',
                    'rejection_reason': f'Strong audit report indicator: {strong_pattern}',
                    'confidence': 0.95,
                    'patterns_matched': [strong_pattern]
                }
        
        # Count audit rejection patterns
        audit_matches = []
        for pattern in self.audit_rejection_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                audit_matches.extend([pattern] * len(matches))
        
        # Count financial statement patterns
        financial_matches = []
        for pattern in self.financial_statement_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                financial_matches.extend([pattern] * len(matches))
        
        # Calculate scores
        audit_score = len(audit_matches)
        financial_score = len(financial_matches)
        total_words = len(text.split())
        
        # Scoring logic
        if audit_score > 3 and financial_score == 0:
            # Strong audit report indicators with no financial statement content
            return {
                'should_process': False,
                'content_type': 'audit_report',
                'rejection_reason': f'High audit content score ({audit_score}) with no financial statement indicators',
                'confidence': min(0.9, 0.6 + (audit_score * 0.05)),
                'patterns_matched': audit_matches[:5]  # Limit for readability
            }
        
        elif audit_score > financial_score and audit_score > 2:
            # More audit content than financial content
            return {
                'should_process': False,
                'content_type': 'audit_report',
                'rejection_reason': f'Audit score ({audit_score}) exceeds financial score ({financial_score})',
                'confidence': min(0.85, 0.5 + ((audit_score - financial_score) * 0.05)),
                'patterns_matched': audit_matches[:3]
            }
        
        elif financial_score > 0:
            # Has financial statement content
            return {
                'should_process': True,
                'content_type': 'financial_statements',
                'rejection_reason': None,
                'confidence': min(0.9, 0.5 + (financial_score * 0.08)),
                'patterns_matched': financial_matches[:5]
            }
        
        elif audit_score == 0 and financial_score == 0:
            # No clear indicators - could be notes or other content, allow processing
            return {
                'should_process': True,
                'content_type': 'unknown_financial_content',
                'rejection_reason': None,
                'confidence': 0.3,
                'patterns_matched': []
            }
        
        else:
            # Default to rejection if unclear but has some audit indicators
            return {
                'should_process': False,
                'content_type': 'mixed_content',
                'rejection_reason': 'Mixed content with audit indicators',
                'confidence': 0.6,
                'patterns_matched': audit_matches
            }
    
    def filter_chunks(self, chunks: List[Dict], document_id: str = "") -> Tuple[List[Dict], List[Dict]]:
        """
        Filter chunks into accepted and rejected lists.
        
        Returns: (accepted_chunks, rejected_chunks)
        """
        accepted_chunks = []
        rejected_chunks = []
        
        logger.info(f"🔍 CONTENT FILTER: Analyzing {len(chunks)} chunks for document {document_id}")
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            chunk_id = chunk.get('id', f'chunk_{i}')
            
            analysis = self.analyze_content_type(chunk_text, document_id)
            
            # Add analysis metadata to chunk
            chunk_with_analysis = chunk.copy()
            chunk_with_analysis['content_analysis'] = analysis
            
            if analysis['should_process']:
                accepted_chunks.append(chunk_with_analysis)
                logger.info(f"✅ ACCEPTED: {chunk_id} - {analysis['content_type']} (confidence: {analysis['confidence']:.2f})")
            else:
                rejected_chunks.append(chunk_with_analysis)
                logger.warning(f"❌ REJECTED: {chunk_id} - {analysis['rejection_reason']} (confidence: {analysis['confidence']:.2f})")
        
        # Summary logging
        acceptance_rate = len(accepted_chunks) / len(chunks) * 100 if chunks else 0
        logger.info(f"📊 CONTENT FILTER SUMMARY: {len(accepted_chunks)}/{len(chunks)} chunks accepted ({acceptance_rate:.1f}%)")
        
        if rejected_chunks:
            rejection_reasons = {}
            for chunk in rejected_chunks:
                reason = chunk['content_analysis']['rejection_reason']
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            logger.info("🚫 REJECTION BREAKDOWN:")
            for reason, count in rejection_reasons.items():
                logger.info(f"   - {reason}: {count} chunks")
        
        return accepted_chunks, rejected_chunks
    
    def get_filter_stats(self, chunks: List[Dict]) -> Dict[str, any]:
        """Get detailed statistics about content filtering."""
        stats = {
            'total_chunks': len(chunks),
            'accepted_chunks': 0,
            'rejected_chunks': 0,
            'content_types': {},
            'rejection_reasons': {},
            'average_confidence': 0.0,
            'financial_content_percentage': 0.0
        }
        
        total_confidence = 0.0
        financial_content_chars = 0
        total_chars = 0
        
        for chunk in chunks:
            if 'content_analysis' not in chunk:
                continue
                
            analysis = chunk['content_analysis']
            content_type = analysis['content_type']
            should_process = analysis['should_process']
            
            # Count by content type
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
            
            # Count accepted/rejected
            if should_process:
                stats['accepted_chunks'] += 1
                financial_content_chars += len(chunk.get('text', ''))
            else:
                stats['rejected_chunks'] += 1
                rejection_reason = analysis['rejection_reason']
                stats['rejection_reasons'][rejection_reason] = stats['rejection_reasons'].get(rejection_reason, 0) + 1
            
            total_confidence += analysis['confidence']
            total_chars += len(chunk.get('text', ''))
        
        # Calculate averages and percentages
        if len(chunks) > 0:
            stats['average_confidence'] = total_confidence / len(chunks)
            stats['financial_content_percentage'] = (financial_content_chars / total_chars * 100) if total_chars > 0 else 0
        
        return stats


def get_content_filter() -> ContentFilter:
    """Get singleton content filter instance."""
    if not hasattr(get_content_filter, '_instance'):
        get_content_filter._instance = ContentFilter()
    return get_content_filter._instance