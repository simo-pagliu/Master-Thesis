import os, re, csv
ROOT = os.path.join(os.path.dirname(__file__), '..', '02 - Value Functions')
vf_re = re.compile(r'^value_functions(?:_(\d+))?\.csv$', flags=re.IGNORECASE)
rows = []
for folder in os.listdir(ROOT):
    folderp = os.path.join(ROOT, folder)
    if not os.path.isdir(folderp):
        continue
    for fn in os.listdir(folderp):
        if not vf_re.match(fn):
            continue
        path = os.path.join(folderp, fn)
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    name = (r.get('name') or '').strip()
                    group = (r.get('group') or '').strip()
                    assigned = None
                    if group and '-' in group:
                        assigned = group.split('-')[-1].strip().upper()
                    else:
                        m = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', name)
                        if m:
                            assigned = m.group('ctry').upper()
                    rows.append((folder, fn, name, group, assigned))
        except Exception as e:
            print('ERR reading', path, e)

# Summarize
from collections import defaultdict
by_country = defaultdict(list)
for folder, fn, name, group, assigned in rows:
    by_country[assigned].append((folder, fn, name, group))

for k in sorted(by_country.keys(), key=lambda x: str(x)):
    print('==== COUNTRY', k)
    for item in by_country[k]:
        print(item)
print('\nTotal rows scanned:', len(rows))
