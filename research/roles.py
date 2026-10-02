import re,sys,collections
for v in sys.argv[1:]:
    rows=[l.split(' | ') for l in open(f'gguf_tensors_{v}.txt') if l.count(' | ')==4 and not l.startswith('##')]
    roles=collections.defaultdict(collections.Counter); rb=collections.Counter()
    dec=0; mtp=0; emb=0; tot=0
    for n,t,s,o,b in rows:
        b=int(b); tot+=b
        m=re.match(r'blk\.(\d+)\.(.*)',n)
        if m:
            L=int(m.group(1)); r=m.group(2)
            kind='mtp' if L==64 else ('attn' if L%4==3 else 'dn')
            key=f'{kind}:{r}'
        else: key=n
        roles[key][t]+=1; rb[key]+=b
        if n=='token_embd.weight': emb+=b
        elif m and int(m.group(1))==64: mtp+=b
        else: dec+=b
    print(f'=== {v}: total {tot/1e9:.3f} GB; per-token decode weight read (main 64 layers + output head, excl embd gather & MTP) {dec/1e9:.3f} GB; token_embd {emb/1e9:.3f} GB; MTP blk.64 {mtp/1e9:.3f} GB')
    for k in sorted(roles): print(f'  {k:42s} {dict(roles[k])}  {rb[k]/1e6:.1f} MB')
