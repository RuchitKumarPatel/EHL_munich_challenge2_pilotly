"""Frozen ModernBERT-base embeddings of the PRE-TREATMENT slice. Scratch, not part of the repo."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0,'/home/frans/Projekte/EHL_munich_challenge2_pilotly')
from transformers import AutoTokenizer, AutoModel
from router.io import iter_lines
from router.features import item_text
SP='/tmp/claude-1000/-home-frans-Projekte-EHL-munich-challenge2-pilotly/9b068e6c-8b32-41b5-987d-3924a4317d1b/scratchpad'
MD=f'{SP}/models/modernbert-base'
L=int(os.environ.get('MAXLEN','1024')); B=int(os.environ.get('BS','8'))
torch.set_num_threads(12)
tk=AutoTokenizer.from_pretrained(MD)
m=AutoModel.from_pretrained(MD,dtype=torch.float32,attn_implementation='sdpa'); m.eval()

def routing_view(req):
    """ADR-001 legal: tools + input[0] + input[1], nothing else."""
    tools=','.join(sorted({t.get('name') for t in (req.get('tools') or []) if isinstance(t,dict)} - {None}))
    u=item_text(req['input'][1])
    return f"TOOLS: {tools}\n{u}"

texts={}
for idx,req in iter_lines(): texts[idx]=routing_view(req)
json.dump({str(k):len(v) for k,v in texts.items()}, open(f'{SP}/tc/textlen.json','w'))

# head+tail truncation: the task text often sits at the END of the user message
def encode_ids(t):
    ids=tk(t, add_special_tokens=False)['input_ids']
    if len(ids)>L-2:
        h=(L-2)//2; ids=ids[:h]+ids[-(L-2-h):]
    return [tk.cls_token_id]+ids+[tk.sep_token_id]

order=sorted(texts)
E=np.zeros((len(order),768),dtype=np.float32)
t0=time.time()
for s in range(0,len(order),B):
    chunk=[encode_ids(texts[i]) for i in order[s:s+B]]
    mx=max(len(c) for c in chunk)
    ids=torch.full((len(chunk),mx),tk.pad_token_id,dtype=torch.long)
    am=torch.zeros((len(chunk),mx),dtype=torch.long)
    for j,c in enumerate(chunk): ids[j,:len(c)]=torch.tensor(c); am[j,:len(c)]=1
    with torch.no_grad():
        h=m(input_ids=ids,attention_mask=am).last_hidden_state
    mask=am.unsqueeze(-1).float()
    E[s:s+len(chunk)]=((h*mask).sum(1)/mask.sum(1)).numpy()
    if (s//B)%10==0:
        el=time.time()-t0; done=s+len(chunk)
        print(f'  {done}/{len(order)}  {el/60:.1f}min elapsed, ETA {(el/max(done,1)*(len(order)-done))/60:.1f}min',flush=True)
np.save(f'{SP}/tc/emb_modernbert_L{L}.npy',E)
print('DONE',E.shape,f'{(time.time()-t0)/60:.1f}min')
