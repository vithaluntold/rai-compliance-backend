with open('services/intelligent_document_analyzer.py', 'r') as f:
    content = f.read()
    
count = content.count('"""')
print(f'intelligent_document_analyzer.py Triple quotes: {count}, Even: {count % 2 == 0}')

with open('services/progress_tracker.py', 'r') as f:
    content = f.read()
    
count = content.count('"""')
print(f'progress_tracker.py Triple quotes: {count}, Even: {count % 2 == 0}')