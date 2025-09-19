with open('models/metadata_models.py', 'r') as f:
    content = f.read()

triple_quotes = content.count('"""')
print(f'Triple quote count: {triple_quotes}')
print(f'Should be even: {triple_quotes % 2 == 0}')

import re
positions = []
for match in re.finditer('"""', content):
    line_num = content[:match.start()].count('\n') + 1
    positions.append(line_num)

print('Triple quote positions:')
for i, line in enumerate(positions):
    print(f'{i+1}: Line {line}')