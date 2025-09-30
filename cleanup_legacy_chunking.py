#!/usr/bin/env python3
"""
Legacy Code Cleanup Script - Remove Obsolete Chunking Logic
Identifies and removes old document chunking files that have been replaced
by the hierarchical mega-chunk system in the enhanced structure parser.
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

class LegacyChunkingCleanup:
    """Cleanup manager for obsolete chunking logic"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.backup_dir = self.base_path / "legacy_backup"
        self.cleanup_log = []
        
        # Files to be removed (obsolete chunking logic)
        self.obsolete_files = [
            # Old chunking implementations
            "check-backend/document_chunker.py",
            "check-backend/chunked_compliance_analyzer.py", 
            "check-backend/install_chunking_libraries.py",
            "render-backend/services/document_chunker.py",
            
            # Enhanced chunking experiments (replaced by mega-chunks)
            "enhanced_document_chunker.py",
            "enhanced_document_chunker_corrected.py", 
            "enhanced_document_chunker_demo.py",
            "document_chunker_integration.py",
            "topic_aware_chunk_accumulator.py",
            
            # Old test files for obsolete chunking
            "test_document_chunking.py",
            "test_chunk_integration.py"
        ]
        
        # Files to be retained (current implementation)
        self.current_files = [
            "nlp_tools/enhanced_structure_parser.py",  # Contains hierarchical mega-chunks
            "nlp_tools/ai_content_classifier.py",      # Uses mega-chunks
            "nlp_tools/complete_nlp_validation_pipeline.py",  # Integrates mega-chunks
            "nlp_tools/intelligent_content_question_mapper.py"  # Uses processed segments
        ]
        
    def analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze which files import the obsolete chunking modules"""
        
        dependencies = {}
        
        # Search for imports of obsolete modules
        obsolete_module_names = [
            "document_chunker", 
            "chunked_compliance_analyzer",
            "enhanced_document_chunker",
            "topic_aware_chunk_accumulator"
        ]
        
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        for module_name in obsolete_module_names:
                            if f"import {module_name}" in content or f"from {module_name}" in content:
                                if str(file_path) not in dependencies:
                                    dependencies[str(file_path)] = []
                                dependencies[str(file_path)].append(module_name)
                                
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        return dependencies
    
    def create_backup(self) -> bool:
        """Create backup of files before deletion"""
        
        try:
            if not self.backup_dir.exists():
                self.backup_dir.mkdir(parents=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_session_dir = self.backup_dir / f"chunking_cleanup_{timestamp}"
            backup_session_dir.mkdir()
            
            backed_up_files = []
            
            for file_path in self.obsolete_files:
                source_path = self.base_path / file_path
                if source_path.exists():
                    # Preserve directory structure in backup
                    backup_path = backup_session_dir / file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(source_path, backup_path)
                    backed_up_files.append(str(source_path))
                    
            self.cleanup_log.append({
                'action': 'backup_created',
                'backup_dir': str(backup_session_dir),
                'files_backed_up': len(backed_up_files),
                'timestamp': timestamp
            })
            
            print(f"✅ Backup created: {backup_session_dir}")
            print(f"   Files backed up: {len(backed_up_files)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def remove_obsolete_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """Remove obsolete chunking files"""
        
        results = {
            'files_found': 0,
            'files_removed': 0,
            'files_not_found': 0,
            'errors': [],
            'removed_files': [],
            'missing_files': []
        }
        
        for file_path in self.obsolete_files:
            full_path = self.base_path / file_path
            results['files_found'] += 1
            
            if full_path.exists():
                if not dry_run:
                    try:
                        full_path.unlink()
                        results['files_removed'] += 1
                        results['removed_files'].append(str(full_path))
                        print(f"🗑️  Removed: {file_path}")
                    except Exception as e:
                        results['errors'].append(f"Error removing {file_path}: {e}")
                        print(f"❌ Error removing {file_path}: {e}")
                else:
                    print(f"🔍 Would remove: {file_path}")
                    results['removed_files'].append(str(full_path))
            else:
                results['files_not_found'] += 1
                results['missing_files'].append(str(full_path))
                print(f"⚠️  Not found: {file_path}")
        
        return results
    
    def update_imports(self, dry_run: bool = True) -> Dict[str, Any]:
        """Update import statements that reference obsolete modules"""
        
        update_results = {
            'files_checked': 0,
            'files_updated': 0,
            'updates_made': []
        }
        
        dependencies = self.analyze_dependencies()
        
        for file_path, obsolete_imports in dependencies.items():
            update_results['files_checked'] += 1
            
            if not dry_run:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Comment out obsolete imports
                    for module_name in obsolete_imports:
                        import_pattern = f"from {module_name} import"
                        import_pattern_2 = f"import {module_name}"
                        
                        content = re.sub(
                            f"^(\\s*)(from {module_name} import .+)$",
                            r"\1# REMOVED: \2  # Obsolete chunking logic", 
                            content,
                            flags=re.MULTILINE
                        )
                        
                        content = re.sub(
                            f"^(\\s*)(import {module_name}(?:\\.\\w+)?)$",
                            r"\1# REMOVED: \2  # Obsolete chunking logic",
                            content,
                            flags=re.MULTILINE  
                        )
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        update_results['files_updated'] += 1
                        update_results['updates_made'].append(file_path)
                        print(f"🔄 Updated imports in: {file_path}")
                        
                except Exception as e:
                    print(f"❌ Error updating {file_path}: {e}")
            else:
                print(f"🔍 Would update imports in: {file_path}")
                update_results['updates_made'].append(file_path)
        
        return update_results
    
    def generate_cleanup_report(self) -> str:
        """Generate comprehensive cleanup report"""
        
        report = f"""
# Legacy Chunking Cleanup Report
Generated: {datetime.now().isoformat()}

## Summary
This cleanup removes obsolete document chunking logic that has been replaced by:
- Enhanced Structure Parser with Hierarchical Mega-Chunks
- AI Content Classification Engine 
- Complete NLP Validation Pipeline

## Obsolete Files Identified ({len(self.obsolete_files)} total):
"""
        
        for file_path in self.obsolete_files:
            full_path = self.base_path / file_path
            status = "EXISTS" if full_path.exists() else "MISSING"
            size = ""
            if full_path.exists():
                try:
                    size = f" ({full_path.stat().st_size:,} bytes)"
                except:
                    pass
            report += f"- {file_path} - {status}{size}\\n"
        
        report += f"""

## Current Implementation (RETAINED):
"""
        for file_path in self.current_files:
            full_path = self.base_path / file_path
            status = "EXISTS" if full_path.exists() else "MISSING"
            report += f"- {file_path} - {status}\\n"
        
        # Analyze dependencies
        dependencies = self.analyze_dependencies()
        if dependencies:
            report += f"""

## Files with Obsolete Imports ({len(dependencies)} files):
"""
            for file_path, imports in dependencies.items():
                report += f"- {file_path}: {', '.join(imports)}\\n"
        
        report += f"""

## Replacement Mapping:
- document_chunker.py → nlp_tools/enhanced_structure_parser.py (hierarchical mega-chunks)
- chunked_compliance_analyzer.py → nlp_tools/complete_nlp_validation_pipeline.py 
- enhanced_document_chunker*.py → Integrated into enhanced_structure_parser.py
- services/document_chunker.py → Replaced by NLP pipeline components

## Benefits of Cleanup:
1. Removes ~2,000+ lines of obsolete code
2. Eliminates dependency confusion  
3. Simplifies maintenance and debugging
4. Reduces codebase complexity
5. Improves performance (modern pipeline is faster)

## Backup Location:
All removed files will be backed up to: {self.backup_dir}
"""
        
        return report
    
    def run_cleanup(self, dry_run: bool = True, create_backup: bool = True) -> Dict[str, Any]:
        """Run complete cleanup process"""
        
        print("🚀 Legacy Chunking Cleanup Process")
        print("=" * 50)
        
        # Generate report
        report = self.generate_cleanup_report()
        print(report)
        
        if dry_run:
            print("\\n🔍 DRY RUN MODE - No files will be modified")
        else:
            print("\\n⚠️  LIVE MODE - Files will be modified/removed!")
        
        results = {
            'backup_success': False,
            'removal_results': {},
            'import_update_results': {},
            'cleanup_log': []
        }
        
        # Create backup if requested
        if create_backup and not dry_run:
            results['backup_success'] = self.create_backup()
            if not results['backup_success']:
                print("❌ Backup failed - aborting cleanup for safety")
                return results
        
        # Remove obsolete files
        print("\\n🗑️  Removing obsolete files...")
        results['removal_results'] = self.remove_obsolete_files(dry_run)
        
        # Update imports
        print("\\n🔄 Updating import statements...")
        results['import_update_results'] = self.update_imports(dry_run)
        
        # Summary
        if dry_run:
            print("\\n📋 Dry Run Summary:")
            print(f"   Files to remove: {len([f for f in self.obsolete_files if (self.base_path / f).exists()])}")
            print(f"   Files with obsolete imports: {len(self.analyze_dependencies())}")
        else:
            print("\\n✅ Cleanup Complete:")
            print(f"   Files removed: {results['removal_results']['files_removed']}")
            print(f"   Import files updated: {results['import_update_results']['files_updated']}")
        
        return results

def main():
    """Main cleanup execution"""
    
    # Initialize cleanup manager
    cleanup = LegacyChunkingCleanup()
    
    # Run dry run first
    print("Running dry run to analyze cleanup...")
    dry_results = cleanup.run_cleanup(dry_run=True)
    
    # Ask for confirmation 
    print("\\n" + "=" * 50)
    response = input("Proceed with actual cleanup? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\\nProceeding with cleanup...")
        live_results = cleanup.run_cleanup(dry_run=False, create_backup=True)
        
        if live_results['backup_success']:
            print("\\n🎯 Cleanup completed successfully!")
            print("   Old chunking logic removed")
            print("   Modern pipeline preserved")
            print("   Backup created for safety")
        else:
            print("\\n❌ Cleanup aborted due to backup failure")
    else:
        print("\\n🛑 Cleanup cancelled by user")

if __name__ == "__main__":
    main()