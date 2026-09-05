"""Example tool executor (webhook) for gemini-web2api agent mode.

Run this in one terminal:

    python agent_executor_example.py

Then the proxy will POST tool calls here whenever you include
"tool_executor_url": "http://127.0.0.1:3000/exec" in a request.

It exposes two example tools:
  - get_weather(city)   -> returns a canned weather result
  - get_time()          -> returns the current UTC time

Add your own tools by adding a branch in ``handle_call``.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "127.0.0.1"
PORT = 3000


def handle_call(name: str, arguments: dict):
    """Implement your tools here. Return anything JSON-serializable."""
    if name == "get_weather":
        city = (arguments or {}).get("city", "unknown")
        return {
            "city": city,
            "weather": "heavy rain",
            "temp_c": 24,
            "advice": "bring an umbrella",
        }
    if name == "get_time":
        return {"time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    raise ValueError(f"unknown tool: {name}")


class ExecutorHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        print(f"[executor] tool call: {request.get('name')} {request.get('arguments')}", flush=True)

        try:
            result = handle_call(request.get("name"), request.get("arguments"))
            body = json.dumps({"result": result}).encode("utf-8")
            status = 200
        except Exception as e:
            body = json.dumps({"result": f"error: {e}"}).encode("utf-8")
            status = 200  # return the error to the model so it can recover

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Tool executor listening on http://{HOST}:{PORT}/exec")
    HTTPServer((HOST, PORT), ExecutorHandler).serve_forever()
