import re

# Read the analysis_routes.py file
with open('routes/analysis_routes.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the line that has 'import psutil' and replace it with optional import
lines = content.split('\n')
new_lines = []

found_imports = 0
for i, line in enumerate(lines):
    if 'import psutil' in line and not line.strip().startswith('#') and not 'try:' in lines[i-1]:
        print(f'Found psutil import at line {i+1}: {line}')
        found_imports += 1
        # Replace with optional import
        indent = line[:len(line) - len(line.lstrip())]  # Get indentation
        new_lines.append(f'{indent}# Optional psutil import for system monitoring')
        new_lines.append(f'{indent}try:')
        new_lines.append(f'{indent}    import psutil')
        new_lines.append(f'{indent}    PSUTIL_AVAILABLE = True')
        new_lines.append(f'{indent}except ImportError:')
        new_lines.append(f'{indent}    psutil = None')
        new_lines.append(f'{indent}    PSUTIL_AVAILABLE = False')
        new_lines.append(f'{indent}    print("Warning: psutil not available, system monitoring disabled")')
    else:
        new_lines.append(line)

# Also make psutil usage optional
new_content = '\n'.join(new_lines)

# Replace psutil usage patterns
new_content = re.sub(r'psutil\.virtual_memory\(\)', 'psutil.virtual_memory() if psutil else type("obj", (), {"percent": 0, "total": 0})()', new_content)
new_content = re.sub(r'psutil\.disk_usage\(([^)]+)\)', r'psutil.disk_usage(\1) if psutil else type("obj", (), {"percent": 0, "total": 0, "free": 0})()', new_content)

# Write back the modified content
with open('routes/analysis_routes.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f'Made {found_imports} psutil imports optional and usage safe')