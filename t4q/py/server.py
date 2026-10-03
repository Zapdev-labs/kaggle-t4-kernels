"""Minimal OpenAI-compatible HTTP server for t4q batched decode (greedy, stdlib http.server only).

  python server.py --model Qwen3.8-27B-Q4_0.gguf [--port 8000] [--slots 16] [--slot_ctx 4096] [--state_f16 1]

Endpoints: GET /v1/models, POST /v1/chat/completions, POST /v1/completions (both with "stream": true|false).
Sampling is greedy only (temperature / top_p are accepted and ignored). Stops on <|im_end|> / <|endoftext|>; "stop"
strings are applied to the decoded text. Concurrent requests share batched decode steps (py/batch.py).
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch import Request, Scheduler  # noqa: E402
from t4q import EOS_IDS  # noqa: E402

MODEL_NAME = "qwen3.8-27b-t4q"


class Detok:
    """incremental detokenizer: emits only complete UTF-8 text (holds back a trailing replacement char)"""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], ""

    def push(self, t):
        if t in EOS_IDS:
            return ""
        self.ids.append(int(t))
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        new = text[len(self.sent):]
        self.sent = text
        return new


def make_handler(sched, tok):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # quiet
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model",
                                                                    "owned_by": "t4q", "created": 0}]})
            if self.path.rstrip("/") in ("/health", ""):
                return self._json(200, {"ok": True, "active": len(sched.active)})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            path = self.path.rstrip("/")
            if path not in ("/v1/chat/completions", "/v1/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:  # noqa: BLE001
                return self._json(400, {"error": {"message": f"bad json: {e}"}})
            chat = path.endswith("chat/completions")
            if chat:
                msgs = body.get("messages") or []
                for m in msgs:  # content parts -> text
                    if isinstance(m.get("content"), list):
                        m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
                text = tok.chat_messages(msgs)
            else:
                p = body.get("prompt", "")
                text = p[0] if isinstance(p, list) and p and isinstance(p[0], str) else p
            ids = tok.encode(text)
            max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
            stops = body.get("stop") or []
            if isinstance(stops, str):
                stops = [stops]
            stream = bool(body.get("stream"))
            q = queue.Queue()
            req = Request(ids, max_new=max_new, ignore_eos=bool(body.get("ignore_eos", False)),
                          on_token=lambda r, t: q.put(("tok", t)), on_done=lambda r: q.put(("done", None)))
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            created = int(time.time())
            sched.submit(req)
            detok = Detok(tok)
            text_out = ""
            stopped_by_str = False

            def chunk(delta_text, finish=None):
                if chat:
                    ch = {"index": 0, "delta": ({"content": delta_text} if delta_text else {}), "finish_reason": finish}
                    return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                            "choices": [ch]}
                return {"id": rid, "object": "text_completion", "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0, "text": delta_text, "finish_reason": finish}]}

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                if chat:
                    first = chunk("")
                    first["choices"][0]["delta"] = {"role": "assistant", "content": ""}
                    self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
            while True:
                kind, t = q.get()
                if kind == "done":
                    break
                if stopped_by_str:
                    continue
                piece = detok.push(t)
                if not piece:
                    continue
                cand = text_out + piece
                cut = min((cand.find(s) for s in stops if s and s in cand), default=-1)
                if cut >= 0:
                    piece = cand[len(text_out):cut] if cut > len(text_out) else ""
                    if cut < len(text_out):
                        text_out = text_out[:cut]
                    stopped_by_str = True
                    req.max_new = 0  # the scheduler finishes the request at its next token
                text_out += piece
                if stream and piece:
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk(piece))}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        req.max_new = 0  # client gone: let the scheduler finish it at the next token
            finish = "stop" if (stopped_by_str or req.finish_reason == "stop") else (req.finish_reason or "length")
            if req.error:
                if stream:
                    self.wfile.write(f"data: {json.dumps({'error': {'message': req.error}})}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                return self._json(400, {"error": {"message": req.error}})
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(req.out),
                     "total_tokens": len(ids) + len(req.out)}
            if stream:
                last = chunk("", finish)
                last["usage"] = usage
                self.wfile.write(f"data: {json.dumps(last)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": text_out}, "finish_reason": finish}
                obj = "chat.completion"
            else:
                choice = {"index": 0, "text": text_out, "finish_reason": finish}
                obj = "text_completion"
            self._json(200, {"id": rid, "object": obj, "created": created, "model": MODEL_NAME, "choices": [choice],
                             "usage": usage})

    return H


def start(sched, tok, host="0.0.0.0", port=8000):
    """engine loop thread + HTTP server thread; returns (httpd, engine_thread)"""
    et = threading.Thread(target=sched.serve_forever, daemon=True)
    et.start()
    httpd = ThreadingHTTPServer((host, port), make_handler(sched, tok))
    httpd.daemon_threads = True
    st = threading.Thread(target=httpd.serve_forever, daemon=True)
    st.start()
    return httpd, et


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--slot_ctx", type=int, default=4096)
    ap.add_argument("--state_f16", type=int, default=1)
    ap.add_argument("--max_ctx", type=int, default=4096, help="single-stream context (longest prompt)")
    a = ap.parse_args()
    from t4q import T4Q, Tokenizer
    eng = T4Q(a.model, lib=a.lib, max_ctx=a.max_ctx, tp=1)
    tok = Tokenizer()
    sched = Scheduler(eng, a.slots, a.slot_ctx, a.state_f16)
    httpd, _ = start(sched, tok, a.host, a.port)
    print(f"t4q server on http://{a.host}:{a.port}/v1 ({a.slots} slots x {a.slot_ctx})", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        httpd.shutdown()
        sched.shutdown()


if __name__ == "__main__":
    main()
