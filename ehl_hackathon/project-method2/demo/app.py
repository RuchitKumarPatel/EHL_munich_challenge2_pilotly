from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method2.cost import pricing
from method2.data import Call, Trajectory, load_trajectories, split
from method2.labels import observed_outcome
from method2.reward import CalibratedRewardModel
from method2.router import SafeRouter


PAGE = """<!doctype html><html><head><meta charset='utf-8'><title>Method 2</title><style>body{font-family:system-ui;margin:40px;max-width:920px;color:#172033}textarea,select{width:100%;box-sizing:border-box;padding:12px;margin:6px 0}textarea{height:150px}button{padding:10px 16px;background:#6748fd;color:white;border:0;border-radius:6px}pre{background:#f4f4f7;padding:16px;white-space:pre-wrap}</style></head><body><h1>Method 2 Safe Router</h1><p>Enter a task and select the currently logged model. Method 2 only changes it when a cheaper model has calibrated support.</p><select id='logged'></select><textarea id='task'>Summarize a technical report and identify implementation risks.</textarea><button onclick='routeTask()'>Route task</button><pre id='result'>Ready.</pre><script>async function setup(){const r=await fetch('/models');const d=await r.json();document.getElementById('logged').innerHTML=d.models.map(x=>`<option>${x}</option>`).join('')}async function routeTask(){const response=await fetch('/route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:document.getElementById('task').value,logged_model:document.getElementById('logged').value})});document.getElementById('result').textContent=JSON.stringify(await response.json(),null,2)}setup()</script></body></html>"""


class Service:
    def __init__(self, source: str) -> None:
        trajectories = load_trajectories(source)
        parts = split(trajectories)
        models = sorted({item.model for item in trajectories if item.model != "mixed"})
        labels = {item.key: observed_outcome(item) for item in trajectories}
        self.reward = CalibratedRewardModel(models).fit(parts["train"] or trajectories, labels, parts["calibration"])
        self.router = SafeRouter(self.reward, pricing(models))

    def route(self, task: str, logged_model: str) -> dict[str, object]:
        call = Call(logged_model, [{"role": "user", "content": [{"type": "input_text", "text": task}]}], [], "demo", 1)
        decision = self.router.route(Trajectory("demo", [call]), logged_model)
        return {"selected_model": decision.model, "reason": decision.reason, "estimated_cost": decision.cost, "quality_mean": decision.estimate.mean, "quality_lower_bound": decision.estimate.lower, "uncertainty": decision.estimate.uncertainty, "support": decision.estimate.support, "candidates": decision.candidates}


def handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/models":
                self.send_json(200, {"models": service.reward.models})
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                task = str(payload["task"]).strip()
                model = str(payload["logged_model"])
                if not task or model not in service.reward.models:
                    raise ValueError("valid task and logged model are required")
                self.send_json(200, service.route(task, model))
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self.send_json(400, {"error": str(error)})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    source = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8767
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(Service(source)))
    print(f"demo=http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
