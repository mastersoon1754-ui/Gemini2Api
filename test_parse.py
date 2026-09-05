import sqlite3, json, os, re
db_path = os.path.expanduser("~/.local/share/opencode/opencode.db")
db = sqlite3.connect(db_path)
cur = db.cursor()
cur.execute('SELECT id FROM session ORDER BY time_created DESC LIMIT 1')
sid = cur.fetchone()[0]
cur2 = db.cursor()
cur2.execute("""SELECT p.data FROM part p JOIN message m ON p.message_id=m.id
                WHERE p.session_id=? ORDER BY p.time_created""", (sid,))
text = ""
for (data,) in cur2.fetchall():
    try:
        d = json.loads(data)
    except Exception:
        continue
    if d.get('type') == 'text' and len(d.get('text', '')) > 1000:
        text = d['text']
        break

pattern = r'```tool_call\s*\n(.*?)\n```'
m = re.search(pattern, text, re.DOTALL)
if not m:
    print("NO MATCH for regex")
else:
    block = m.group(1).strip()
    print("block len:", len(block))
    print("block head:", repr(block[:200]))
    print("block tail:", repr(block[-200:]))
    try:
        data = json.loads(block)
        print("json.loads: OK, name =", data.get("name"))
        args = data.get("arguments", {})
        print("content len:", len(args.get("content", "")) if isinstance(args, dict) else "args-not-dict")
    except json.JSONDecodeError as e:
        print("json.loads FAILED at:", e)
        # show context around the error
        pos = e.pos
        print("  context:", repr(block[max(0, pos - 100):pos + 100]))
        # find first unescaped newline inside a string
        m2 = re.search(r'"content": "([^"]*(?:\\.[^"]*)*)', block, re.DOTALL)
        if m2:
            inner = m2.group(1)
            # check for raw newlines inside the JSON string
            nl = [i for i, ch in enumerate(inner) if ch == '\n']
            print("  raw newlines in content string:", len(nl), "first at:", nl[:5])
