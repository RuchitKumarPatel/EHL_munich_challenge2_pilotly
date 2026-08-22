from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method1.data.loader import load_trajectories
from method1.data.schema import RequestRecord, Trajectory
from method1.pricing import CostModel, ensure_models, load_pricing
from method1.quality.outcome_model import composite_outcome
from method1.routing.cache_aware_router import CacheAwareRouter
from method1.routing.difficulty_router import DifficultyRouter


PAGE = """<!doctype html><html><head><meta charset='utf-8'><title>Project Method 1</title><style>body{font-family:system-ui;margin:40px;max-width:900px;color:#172033}textarea{width:100%;height:160px;padding:12px}button{padding:10px 18px;background:#6748fd;color:white;border:0;border-radius:6px;cursor:pointer}pre{background:#f3f4f6;padding:16px;border-radius:6px;white-space:pre-wrap}</style></head><body><h1>Project Method 1</h1><p>Cache-aware trajectory router</p><textarea id='task'>Analyze this task and select the most cost-effective model.</textarea><p><button onclick='routeTask()'>Route task</button></p><pre id='result'>Ready.</pre><script>async function routeTask(){const task=document.getElementById('task').value;const response=await fetch('/route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task})});document.getElementById('result').textContent=JSON.stringify(await response.json(),null,2)}</script></body></html>"""


class RouterService:
    def __init__(self, source: str) -> None:
        trajectories = load_trajectories(source)
        models = sorted({item.logged_model for item in trajectories if item.logged_model != "mixed"})
        pricing = ensure_models(load_pricing(), models)
        self.cost_model = CostModel(pricing)
        self.predictor = DifficultyRouter(models, self.cost_model).fit(trajectories, {item.key: composite_outcome(item) for item in trajectories})
        self.router = CacheAwareRouter(self.predictor, cost_weight=0.15)

    def route(self, task: str) -> dict[str, object]:
        call = RequestRecord("demo", [{"role": "user", "content": [{"type": "input_text", "text": task}]}], [], "demo", 1)
        trajectory = Trajectory("demo", [call])
        decision = self.router.route(trajectory)
        return {"model": decision.model, "estimated_cost": decision.expected_cost, "expected_quality": decision.expected_quality, "uncertainty": decision.uncertainty, "scores": decision.scores, "costs": decision.costs, "cacheable_tokens": 0}


def make_handler(service: RouterService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: str, content_type: str) -> None:
            encoded = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self) -> None:
            if self.path != "/route":
                self._send(404, json.dumps({"error": "not found"}), "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                task = str(body.get("task", ""))
                if not task.strip():
                    raise ValueError("task is required")
                self._send(200, json.dumps(service.route(task)), "application/json")
            except (ValueError, json.JSONDecodeError) as error:
                self._send(400, json.dumps({"error": str(error)}), "application/json")

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "viktor-tumai-starter" / "viktor-tumai-starter" / "export")
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
    service = RouterService(source)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"demo=http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

