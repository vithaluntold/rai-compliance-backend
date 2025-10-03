import json

with open('checklist_data/frameworks/IFRS/IAS 1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total_questions = 0
for section in data['sections']:
    if 'items' in section:
        questions_in_section = len(section['items'])
        total_questions += questions_in_section
        print(f'Section: {section.get("title", "Unknown")} - {questions_in_section} questions')

print(f'\nTotal questions in IAS 1: {total_questions}')

# Show some sample questions
print('\nFirst 5 questions:')
section = data['sections'][0]
for i, item in enumerate(section['items'][:5]):
    print(f'{i+1}. {item["question"]}')