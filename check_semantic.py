with open('routes/services/semantic_processor.py', 'r') as f:
    content = f.read()
    
count = content.count('"""')
print(f'Triple quotes: {count}, Even: {count % 2 == 0}')

# Find positions
import re
positions = []
for match in re.finditer('"""', content):
    line_num = content[:match.start()].count('\n') + 1
    positions.append(line_num)

print('Triple quote positions:')
for i, line in enumerate(positions):
    print(f'{i+1}: Line {line}')