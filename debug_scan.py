import sqlite3, json, os, re
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

ms = list(re.finditer(r'```tool_call', text))
print("matches:", len(ms))
if ms:
    m = ms[0]
    rest = text[m.end():]
    stripped = len(rest) - len(rest.lstrip("\r\n "))
    body = rest[stripped:]
    print("body starts with '{'?", body.startswith("{"))
    print("body head:", repr(body[:150]))
    json_str, end = gw._scan_json_object(body, 0)
    print("scan result:", (json_str is not None), "end:", end)
    if json_str:
        print("json_str head:", repr(json_str[:120]))
        print("json_str len:", len(json_str))
        try:
            data = json.loads(json_str)
            print("json.loads OK, name:", data.get("name"))
        except json.JSONDecodeError as e:
            print("json.loads FAILED:", e)
            print("  context:", repr(json_str[max(0, e.pos-80):e.pos+80]))
    else:
        # find why: print the last 200 chars of body
        print("body tail:", repr(body[-200:]))
