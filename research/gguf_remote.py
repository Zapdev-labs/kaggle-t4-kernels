#!/usr/bin/env python3
"""Hand-rolled GGUF v3 header parser over HTTP Range requests (no deps)."""
import struct, sys, urllib.request, json
GGML_TYPES = {0:("F32",1,4),1:("F16",1,2),2:("Q4_0",32,18),3:("Q4_1",32,20),6:("Q5_0",32,22),7:("Q5_1",32,24),
 8:("Q8_0",32,34),9:("Q8_1",32,36),10:("Q2_K",256,84),11:("Q3_K",256,110),12:("Q4_K",256,144),13:("Q5_K",256,176),
 14:("Q6_K",256,210),15:("Q8_K",256,292),16:("IQ2_XXS",256,66),17:("IQ2_XS",256,74),18:("IQ3_XXS",256,98),
 19:("IQ1_S",256,50),20:("IQ4_NL",32,18),21:("IQ3_S",256,110),22:("IQ2_S",256,82),23:("IQ4_XS",256,136),
 24:("I8",1,1),25:("I16",1,2),26:("I32",1,4),27:("I64",1,8),28:("F64",1,8),29:("IQ1_M",256,56),30:("BF16",1,2),
 34:("TQ1_0",256,54),35:("TQ2_0",256,66),39:("MXFP4",32,17)}
class R:
    def __init__(s,url,chunk=8<<20): s.url=url;s.buf=b"";s.base=0;s.pos=0;s.chunk=chunk
    def _need(s,n):
        while s.pos+n > s.base+len(s.buf):
            start=s.base+len(s.buf)
            req=urllib.request.Request(s.url,headers={"Range":f"bytes={start}-{start+s.chunk-1}"})
            s.buf+=urllib.request.urlopen(req).read()
    def read(s,n):
        s._need(n); o=s.pos-s.base; s.pos+=n; return s.buf[o:o+n]
    def u(s,f): return struct.unpack("<"+f,s.read(struct.calcsize("<"+f)))[0]
    def str(s): n=s.u("Q"); return s.read(n).decode("utf-8","replace")
SC={0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
def val(r,t):
    if t in SC: return r.u(SC[t])
    if t==8: return r.str()
    if t==9:
        et=r.u("I"); n=r.u("Q")
        if et in SC:
            sz=struct.calcsize(SC[et]); raw=r.read(sz*n)
            return ("ARR",et,n, list(struct.unpack(f"<{min(n,8)}{SC[et]}",raw[:sz*min(n,8)])))
        return ("ARR",et,n,[val(r,et) for _ in range(n)][:8])
    raise ValueError(t)
def parse(url):
    r=R(url)
    assert r.read(4)==b"GGUF"; ver=r.u("I"); nt=r.u("Q"); nkv=r.u("Q")
    kv={}
    for _ in range(nkv):
        k=r.str(); t=r.u("I"); kv[k]=val(r,t)
    align=kv.get("general.alignment",32)
    tens=[]
    for _ in range(nt):
        name=r.str(); nd=r.u("I"); dims=[r.u("Q") for _ in range(nd)]; ty=r.u("I"); off=r.u("Q")
        tens.append((name,dims,ty,off))
    hdr_end=r.pos; data_start=(hdr_end+align-1)//align*align
    return ver,kv,tens,hdr_end,data_start,align
def nbytes(dims,ty):
    n=1
    for d in dims: n*=d
    _,bs,ts=GGML_TYPES[ty]; return n//bs*ts
if __name__=="__main__":
    url=sys.argv[1]; total=int(sys.argv[2]); out=sys.argv[3]
    ver,kv,tens,he,ds,al=parse(url)
    from collections import Counter, defaultdict
    L=[f"# {url}",f"# gguf version {ver}, n_tensors {len(tens)}, n_kv {len(kv)}, alignment {al}",
       f"# header(kv+tensor infos) ends at {he}, tensor data starts at {ds}, file size {total}"]
    L.append("## KV metadata")
    for k,v in kv.items():
        if isinstance(v,tuple): L.append(f"{k} = array<type {v[1]}>[{v[2]}] first={v[3]!r}"[:400])
        else: L.append(f"{k} = {v!r}"[:400])
    L.append("## Tensors: name | type | shape(ne0,ne1,..) | offset_rel | bytes")
    by=Counter(); cnt=Counter(); tot=0
    for name,dims,ty,off in tens:
        b=nbytes(dims,ty); tn=GGML_TYPES[ty][0]; by[tn]+=b; cnt[tn]+=1; tot+=b
        L.append(f"{name} | {tn} | {dims} | {off} | {b}")
    L.append("## Summary by type: type count bytes GiB")
    for tn in by: L.append(f"{tn} {cnt[tn]} {by[tn]} {by[tn]/2**30:.3f}")
    L.append(f"total tensor bytes {tot} ({tot/2**30:.3f} GiB); data_start+tensor_bytes={ds+tot} vs file {total}")
    open(out,"w").write("\n".join(L)+"\n")
    print("\n".join(L[:3]+L[-12:]))
