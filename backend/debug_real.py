import json

with open('candidates_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('debug_real_resumes.txt', 'w', encoding='utf-8') as out:
    for b in data['batches']:
        for c in b['candidates']:
            name = c.get('name', '').lower()
            if 'uzair' in name or 'vishv' in name:
                out.write(f"\n============================\n--- {c.get('name')} ---\n============================\n")
                out.write(c.get('raw_text', ''))
