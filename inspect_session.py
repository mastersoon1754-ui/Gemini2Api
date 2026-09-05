import sqlite3, json, os
db_path = os.path.expanduser("~/.local/share/opencode/opencode.db")
db = sqlite3.connect(db_path)
cur = db.cursor()
cur.execute('SELECT id FROM session ORDER BY time_created DESC LIMIT 1')
sid = cur.fetchone()[0]
cur2 = db.cursor()
cur2.execute("""SELECT p.data FROM part p JOIN message m ON p.message_id=m.id
                WHERE p.session_id=? ORDER BY p.time_created""", (sid,))
for (data,) in cur2.fetchall():
    try:
        d = json.loads(data)
    except Exception:
        continue
    if d.get('type') != 'text':
        continue
    t = d.get('text', '')
    if len(t) > 1000:
        print('FULL TEXT len:', len(t))
        for kw in ('tool_call', '```html', '```', 'filePath', 'pagoda.html', 'write'):
            print('  %r: %d occurrences' % (kw, t.count(kw)))
        i = t.find('tool_call')
        print('  tool_call context:', repr(t[max(0, i - 80):i + 150]) if i >= 0 else 'NONE')
        i2 = t.find('```html')
        print('  html fence context:', repr(t[max(0, i2 - 80):i2 + 120]) if i2 >= 0 else 'NONE')
        i3 = t.find('pagoda.html')
        print('  pagoda.html context:', repr(t[max(0, i3 - 100):i3 + 100]) if i3 >= 0 else 'NONE')
