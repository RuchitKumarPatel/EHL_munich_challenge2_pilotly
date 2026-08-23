"""Fine-tune ModernBERT-base on the PRE-TREATMENT slice: one trunk, two heads.

  spend head    h            -> log1p(effective billed tokens)      [arm-FREE: ADR-001, the arm
                                                                     is the treatment and is
                                                                     unknown at routing time]
  friction head [h, emb(arm)] -> y_fric                              [arm-CONDITIONED, so that
                                                                     Delta_hat = p(sonnet)-p(logged)
                                                                     can be read off at inference]

Train/test split is GROUPED BY CLUSTER (literal cron path): a job never straddles the boundary.
Population: claude lane AND cron-triggered (558 rows, both arms present).
"""
import sys, os, json, time
from _scratch import scratch_dir, on_path
SP=scratch_dir(); on_path(SP, f'{SP}/tc')   # resolved BEFORE transformers: see _scratch.py
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from router.io import iter_lines
from router.features import item_text
from evalpool import (POOL, TRAIN_ALL, YLOG, CLUSTER, PANEL, ARMS, AI, LOGGED, N, auc)

MD=f'{SP}/models/modernbert-base'
L      = int(os.environ.get('MAXLEN','512'))
EPOCHS = int(os.environ.get('EPOCHS','2'))
BS     = int(os.environ.get('BS','4'))
LR     = float(os.environ.get('LR','2e-5'))
SPLIT  = int(os.environ.get('SPLIT','0'))       # 0 => train on half A, test on B; 1 => mirrored
SEED   = int(os.environ.get('SEED','4'))
torch.set_num_threads(12); torch.manual_seed(SEED); np.random.seed(SEED)

tk=AutoTokenizer.from_pretrained(MD)
def routing_view(req):
    tools=','.join(sorted({t.get('name') for t in (req.get('tools') or []) if isinstance(t,dict)} - {None}))
    return f"TOOLS: {tools}\n{item_text(req['input'][1])}"
TXT={}
for idx,req in iter_lines(): TXT[idx]=routing_view(req)
def enc(i):
    ids=tk(TXT[i], add_special_tokens=False)['input_ids']
    if len(ids)>L-2:
        h=(L-2)//2; ids=ids[:h]+ids[-(L-2-h):]
    return [tk.cls_token_id]+ids+[tk.sep_token_id]
IDS={i:enc(i) for i in TRAIN_ALL}
print(f'tokenised {len(IDS)} rows at L={L}',flush=True)

class Net(nn.Module):
    """Friction logit = text main effect + ARM main effect + explicit text x arm interaction.

    The arm gets its OWN unmixed path. In the first revision the arm was 16 of 784 input dims
    to a 128-unit MLP and the model ignored it outright (max|p_sonnet - p_opus| = 1.6e-4)."""
    def __init__(s,nA,d=16):
        super().__init__()
        s.trunk=AutoModel.from_pretrained(MD,dtype=torch.float32,attn_implementation='sdpa')
        s.spend=nn.Linear(768,1)
        s.text =nn.Sequential(nn.Linear(768,128), nn.GELU(), nn.Linear(128,1))   # main effect
        s.proj =nn.Linear(768,d)                                                 # interaction basis
        s.u    =nn.Embedding(nA,d); nn.init.normal_(s.u.weight,0,0.5)            # per-arm loading
        s.b    =nn.Embedding(nA,1); nn.init.zeros_(s.b.weight)                   # per-arm intercept
    def pooled(s,ids,am):
        h=s.trunk(input_ids=ids,attention_mask=am).last_hidden_state
        m=am.unsqueeze(-1).float()
        return (h*m).sum(1)/m.sum(1)
    def fric_logit(s,h,arm):
        return s.text(h).squeeze(-1) + s.b(arm).squeeze(-1) + (s.proj(h)*s.u(arm)).sum(-1)
    def forward(s,ids,am,arm):
        h=s.pooled(ids,am)
        return s.spend(h).squeeze(-1), s.fric_logit(h,arm), h

def batchify(rows):
    ch=[IDS[i] for i in rows]; mx=max(len(c) for c in ch)
    ids=torch.full((len(ch),mx),tk.pad_token_id,dtype=torch.long)
    am=torch.zeros((len(ch),mx),dtype=torch.long)
    for j,c in enumerate(ch): ids[j,:len(c)]=torch.tensor(c); am[j,:len(c)]=1
    return ids,am

# ---- grouped split ------------------------------------------------------------
rng=np.random.default_rng(SEED); uc=np.unique(CLUSTER); perm=rng.permutation(uc)
half=set(perm[:len(uc)//2].tolist())
inHalf=np.array([c in half for c in CLUSTER])
trmask = inHalf if SPLIT==0 else ~inHalf
TR=[i for i in TRAIN_ALL if trmask[i]]; TE=[i for i in TRAIN_ALL if not trmask[i]]
print(f'SPLIT={SPLIT}  train {len(TR)}  test {len(TE)}  train y+ {int(PANEL.y[TR].sum())}  test y+ {int(PANEL.y[TE].sum())}',flush=True)

net=Net(len(ARMS))
HEAD_LR=float(os.environ.get('HEAD_LR','1e-3'))
head_p=[p for n,p in net.named_parameters() if not n.startswith('trunk')]
trunk_p=[p for n,p in net.named_parameters() if n.startswith('trunk')]
opt=torch.optim.AdamW([{'params':trunk_p,'lr':LR},{'params':head_p,'lr':HEAD_LR}],weight_decay=0.01)
ymu,ysd=float(np.mean(YLOG[TR])),float(np.std(YLOG[TR]))
steps=EPOCHS*((len(TR)+BS-1)//BS)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[LR,HEAD_LR],total_steps=steps,pct_start=0.15)
bce=nn.BCEWithLogitsLoss(); mse=nn.MSELoss()
t0=time.time(); step=0
for ep in range(EPOCHS):
    net.train(); order=np.random.permutation(TR)
    for s in range(0,len(order),BS):
        rows=order[s:s+BS]; ids,am=batchify(rows)
        arm=torch.tensor([int(LOGGED[i]) for i in rows])
        ys=torch.tensor([(YLOG[i]-ymu)/ysd for i in rows],dtype=torch.float32)
        yf=torch.tensor([PANEL.y[i] for i in rows],dtype=torch.float32)
        ps,pf,_=net(ids,am,arm)
        loss=mse(ps,ys)+bce(pf,yf)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step(); sched.step(); step+=1
        if step%10==0:
            el=time.time()-t0
            print(f'  ep{ep} step {step}/{steps} loss {loss.item():.4f}  {el/60:.1f}min  ETA {(el/step*(steps-step))/60:.1f}min',flush=True)

# ---- inference on the HELD-OUT half, both arms -------------------------------
net.eval()
OPU=AI['claude-opus-5']; SON=AI['claude-sonnet-5']
out={}
with torch.no_grad():
    for s in range(0,len(TE),8):
        rows=TE[s:s+8]; ids,am=batchify(rows)
        h=net.pooled(ids,am)
        ps=(net.spend(h).squeeze(-1)*ysd+ymu).numpy()
        pf={}
        for a in (OPU,SON):
            arm=torch.full((len(rows),),a,dtype=torch.long)
            pf[a]=torch.sigmoid(net.fric_logit(h,arm)).numpy()
        for j,i in enumerate(rows):
            out[int(i)]={'spend':float(ps[j]),'p_opus':float(pf[OPU][j]),'p_sonnet':float(pf[SON][j])}
json.dump(out,open(f'{SP}/tc/ft_split{SPLIT}_L{L}.json','w'))
te=np.array(TE)
print(f'HELD-OUT  spend R2 = {1-((YLOG[te]-np.array([out[int(i)]["spend"] for i in te]))**2).sum()/((YLOG[te]-YLOG[te].mean())**2).sum():+.3f}',flush=True)
pl=np.array([out[int(i)]['p_opus' if LOGGED[i]==OPU else 'p_sonnet'] if LOGGED[i] in (OPU,SON) else out[int(i)]['p_opus'] for i in te])
print(f'HELD-OUT  friction AUC (logged arm) = {auc(pl,PANEL.y[te]):.3f}',flush=True)
d=np.array([out[int(i)]['p_sonnet']-out[int(i)]['p_opus'] for i in te])
print(f'HELD-OUT  Delta_hat(sonnet-opus): mean {d.mean():+.4f} sd {d.std():.4f}',flush=True)
print(f'total {(time.time()-t0)/60:.1f}min')
