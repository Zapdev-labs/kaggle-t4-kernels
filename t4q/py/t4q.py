"""ctypes driver for libt4q.so plus the HF tokenizer (Qwen/Qwen3.8-27B).

Token ids are the interface: prompts are tokenized once here and the same int32 id files go to t4q and to
tools/oracle_dump (llama.cpp), so tokenizer differences cannot contaminate validation.
"""
import ctypes as C
import os

import numpy as np

EOS_IDS = (248046, 248044)  # <|im_end|> (GGUF eos) and <|endoftext|> (HF config eos); stop on both


class Params(C.Structure):
    _fields_ = [("n_gpu", C.c_int), ("tp", C.c_int), ("max_ctx", C.c_int), ("kv_q8", C.c_int), ("spec_k", C.c_int),
                ("draft_vocab", C.c_int), ("verbose", C.c_int)]


class Sampling(C.Structure):
    _fields_ = [("temp", C.c_float), ("top_k", C.c_int), ("top_p", C.c_float), ("seed", C.c_uint64)]


class T4Q:
    def __init__(self, gguf, lib=None, max_ctx=4096, verbose=1):
        lib = lib or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libt4q.so")
        self.lib = L = C.CDLL(lib)
        L.t4q_load.restype = C.c_void_p
        L.t4q_load.argtypes = [C.c_char_p, C.POINTER(Params)]
        L.t4q_logits.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.POINTER(C.c_float)]
        L.t4q_prefill.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_int]
        L.t4q_generate.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.POINTER(Sampling),
                                   C.POINTER(C.c_int32), C.c_int]
        L.t4q_dump.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_float), C.c_size_t]
        L.t4q_set_dump.argtypes = [C.c_void_p, C.c_int]
        L.t4q_set_option.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
        L.t4q_dump_keys.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
        L.t4q_stats.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
        L.t4q_reset.argtypes = [C.c_void_p]
        L.t4q_free.argtypes = [C.c_void_p]
        L.t4q_n_vocab.argtypes = [C.c_void_p]
        L.t4q_pos.argtypes = [C.c_void_p]
        L.t4q_last_error.restype = C.c_char_p
        p = Params(2, 0, max_ctx, 0, 0, 0, verbose)
        self.ctx = L.t4q_load(gguf.encode(), C.byref(p))
        if not self.ctx:
            raise RuntimeError("t4q_load failed: " + L.t4q_last_error().decode())
        self.n_vocab = L.t4q_n_vocab(self.ctx)

    def _err(self, rc, what):
        if rc < 0:
            raise RuntimeError(f"{what} failed: " + self.lib.t4q_last_error().decode())
        return rc

    def reset(self):
        self.lib.t4q_reset(self.ctx)

    @property
    def pos(self):
        return self.lib.t4q_pos(self.ctx)

    def logits(self, ids):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        out = np.empty((len(ids), self.n_vocab), dtype=np.float32)
        self._err(self.lib.t4q_logits(self.ctx, ids.ctypes.data_as(C.POINTER(C.c_int32)), len(ids),
                                      out.ctypes.data_as(C.POINTER(C.c_float))), "t4q_logits")
        return out

    def prefill(self, ids):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        self._err(self.lib.t4q_prefill(self.ctx, ids.ctypes.data_as(C.POINTER(C.c_int32)), len(ids)), "t4q_prefill")

    def generate(self, max_new, stop=()):
        out = np.zeros(max_new, dtype=np.int32)
        st = np.ascontiguousarray(stop, dtype=np.int32) if len(stop) else np.zeros(1, np.int32)
        smp = Sampling(0.0, 1, 1.0, 0)
        n = self._err(self.lib.t4q_generate(self.ctx, out.ctypes.data_as(C.POINTER(C.c_int32)), max_new, C.byref(smp),
                                            st.ctypes.data_as(C.POINTER(C.c_int32)), len(stop)), "t4q_generate")
        return out[:n]

    def set_option(self, key, value):
        self._err(self.lib.t4q_set_option(self.ctx, key.encode(), int(value)), "t4q_set_option")

    def set_dump(self, on):
        self.lib.t4q_set_dump(self.ctx, 1 if on else 0)

    def dump(self, name, layer=-1):
        n = self.lib.t4q_dump(self.ctx, name.encode(), layer, None, 0)
        if n < 0:
            return None
        out = np.empty(n, dtype=np.float32)
        self.lib.t4q_dump(self.ctx, name.encode(), layer, out.ctypes.data_as(C.POINTER(C.c_float)), n)
        return out

    def dump_all(self):
        n = self.lib.t4q_dump_keys(self.ctx, None, 0)
        buf = C.create_string_buffer(n + 1)
        self.lib.t4q_dump_keys(self.ctx, buf, n + 1)
        res = {}
        for k in buf.value.decode().split("\n"):
            if not k:
                continue
            nm, _, ly = k.rpartition("-")
            if nm and ly.isdigit():
                res[k] = self.dump(nm, int(ly))
            else:
                res[k] = self.dump(k, -1)
        return res

    def stats(self):
        import json
        buf = C.create_string_buffer(4096)
        self.lib.t4q_stats(self.ctx, buf, 4096)
        return json.loads(buf.value.decode())

    def close(self):
        if self.ctx:
            self.lib.t4q_free(self.ctx)
            self.ctx = None


class Tokenizer:
    """HF tokenizer for Qwen/Qwen3.8-27B (public). Chat prompts use enable_thinking=False."""

    def __init__(self, repo="Qwen/Qwen3.8-27B"):
        self.tok = None
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(repo)
            self.kind = "transformers"
        except Exception as e:  # noqa: BLE001
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer as HT
            self.tok = HT.from_file(hf_hub_download(repo, "tokenizer.json"))
            self.kind = f"tokenizers ({e.__class__.__name__})"

    def chat_text(self, user):
        if self.kind == "transformers":
            try:
                return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
            except Exception:  # noqa: BLE001
                pass
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def encode(self, text):
        if self.kind == "transformers":
            return list(self.tok(text, add_special_tokens=False)["input_ids"])
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids):
        ids = [int(i) for i in ids]
        if self.kind == "transformers":
            return self.tok.decode(ids, skip_special_tokens=False)
        return self.tok.decode(ids, skip_special_tokens=False)
