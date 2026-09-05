import sqlite3, json, os
import importlib.util
spec = importlib.util.spec_from_file_location("gw", "gemini_web2api.py")
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)

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

clean, calls = gw.parse_tool_calls(text)
print("tool_calls found:", len(calls))
for c in calls:
    print("  name:", c["function"]["name"])
    args = json.loads(c["function"]["arguments"])
    print("  args keys:", list(args.keys()))
    print("  filePath:", args.get("filePath"))
    print("  content len:", len(args.get("content", "")))
print("clean text len:", len(clean))
print("clean tail:", repr(clean[-120:]))
