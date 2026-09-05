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
        with open("text_dump.txt", "w", encoding="utf-8") as f:
            f.write("=== 900-1500 (block start) ===\n")
            f.write(t[900:1500])
            f.write("\n\n=== last 1500 chars ===\n")
            f.write(t[-1500:])
            f.write("\n\n=== search closing fence variants ===\n")
            for pat in ("\n```\n", "\n```", "}\n```", "arguments", 'filePath', 'file_path', '"content"'):
                f.write("%r count: %d\n" % (pat, t.count(pat)))
        print("dumped")
        break
