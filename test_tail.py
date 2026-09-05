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
    if d.get('type') == 'text' and len(d.get('text', '')) > 1000:
        t = d['text']
        print("TAIL (last 400 chars):")
        print(repr(t[-400:]))
        print()
        i = t.find('```tool_call')
        print("tool_call starts at:", i)
        print("after tool_call start, text len:", len(t) - i)
        break
