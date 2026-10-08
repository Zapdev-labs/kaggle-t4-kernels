"""Continuous-batching scheduler over the t4q batched decode API (greedy).

Requests join and leave between steps: each iteration admits up to `prefill_per_iter` waiting requests (one
single-stream batched prefill each, copied into a free slot), then runs one batched decode step over every active
slot. A request finishes on a stop token, after max_new tokens, or when its slot context is full; its slot is freed
for the next waiting request. submit() is thread-safe (the HTTP server calls it from handler threads); step() and
run() must be called from one engine thread.
"""
import itertools
import queue
import threading
import time

from t4q import EOS_IDS


class Request:
    _ids = itertools.count()

    def __init__(self, ids, max_new=512, stop=EOS_IDS, ignore_eos=False, on_token=None, on_done=None):
        self.id = next(Request._ids)
        self.ids = [int(x) for x in ids]
        self.max_new = int(max_new)
        self.stop = set() if ignore_eos else set(int(x) for x in stop)
        self.on_token = on_token      # callback(req, token) from the engine thread
        self.on_done = on_done        # callback(req)
        self.out = []
        self.slot = -1
        self.finish_reason = None
        self.error = None
        self.t_submit = time.time()
        self.t_first = None
        self.t_done = None
        self.done = threading.Event()


class Scheduler:
    def __init__(self, eng, n_slots, slot_ctx, state_f16=1, prefill_per_iter=4, init=True):
        self.eng = eng
        self.n_slots, self.slot_ctx = n_slots, slot_ctx
        if init:
            eng.batch_init(n_slots, slot_ctx, state_f16)
        self.prefill_per_iter = prefill_per_iter
        self.waiting = queue.Queue()
        self.active = {}  # slot -> Request
        self.free = list(range(n_slots))[::-1]
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stats = {"prefills": 0, "prefill_s": 0.0, "prefill_tokens": 0, "steps": 0, "step_s": 0.0,
                      "decode_tokens": 0, "max_batch": 0}
        self._stop = False
        self._fatal = None

    def _fail(self, req, msg):
        req.error = msg
        req.finish_reason = "error"
        req.done.set()
        if req.on_done:
            req.on_done(req)

    def submit(self, req):
        if self._fatal is not None:
            self._fail(req, f"engine dead: {self._fatal}")
            return req
        if len(req.ids) + 2 > self.slot_ctx:
            self._fail(req, f"prompt of {len(req.ids)} tokens does not fit the slot context ({self.slot_ctx})")
            return req
        self.waiting.put(req)
        self.wake.set()
        return req

    def busy(self):
        return bool(self.active) or not self.waiting.empty()

    def _emit(self, req, tok):
        req.out.append(tok)
        if req.t_first is None:
            req.t_first = time.time()
        if req.on_token:
            req.on_token(req, tok)
        if tok in req.stop:
            req.finish_reason = "stop"
        elif len(req.out) >= req.max_new:
            req.finish_reason = "length"
        elif len(req.ids) + len(req.out) >= self.slot_ctx:
            req.finish_reason = "length"
        return req.finish_reason is not None

    def _finish(self, req):
        req.t_done = time.time()
        if req.slot in self.active:
            del self.active[req.slot]
            self.free.append(req.slot)
        req.done.set()
        if req.on_done:
            req.on_done(req)

    def step(self):
        """one scheduler iteration; returns the number of tokens produced"""
        n = 0
        admitted = 0
        while self.free and admitted < self.prefill_per_iter:
            try:
                req = self.waiting.get_nowait()
            except queue.Empty:
                break
            slot = self.free.pop()
            req.slot = slot
            t = time.time()
            try:
                first = self.eng.batch_prefill(slot, req.ids)
            except Exception as e:  # noqa: BLE001
                self.free.append(slot)
                self._fail(req, str(e))
                continue
            self.stats["prefills"] += 1
            self.stats["prefill_s"] += time.time() - t
            self.stats["prefill_tokens"] += len(req.ids)
            admitted += 1
            self.active[slot] = req
            n += 1
            if self._emit(req, int(first)):
                self._finish(req)
        if self.active:
            slots = sorted(self.active)
            t = time.time()
            out = self.eng.batch_step(slots)
            self.stats["steps"] += 1
            self.stats["step_s"] += time.time() - t
            self.stats["decode_tokens"] += len(slots)
            self.stats["max_batch"] = max(self.stats["max_batch"], len(slots))
            n += len(slots)
            for s, tok in zip(slots, out):
                req = self.active[s]
                if self._emit(req, int(tok)):
                    self._finish(req)
        return n

    def run_until_idle(self):
        while self.busy():
            self.step()

    def serve_forever(self):
        """engine loop for the HTTP server (call from a dedicated thread)"""
        while not self._stop:
            if not self.busy():
                self.wake.wait(0.05)
                self.wake.clear()
                continue
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                # a failed engine call must not strand clients: report the error to every active
                # and waiting request (on_done releases their HTTP handlers), then stop the loop
                self._fatal = f"{type(e).__name__}: {e}"
                for req in list(self.active.values()):
                    self._fail(req, self._fatal)
                while True:
                    try:
                        req = self.waiting.get_nowait()
                    except queue.Empty:
                        break
                    self._fail(req, self._fatal)
                return

    def shutdown(self):
        self._stop = True
        self.wake.set()
