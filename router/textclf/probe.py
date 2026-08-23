"""Frozen-embedding probe: does ModernBERT beat the hashed n-grams, on cost and on risk?
Everything grouped-CV by cluster. Nothing here is fine-tuned; this is the floor the
fine-tune has to beat."""
import sys, json, numpy as np
from _scratch import scratch_dir, on_path
SP=scratch_dir(); on_path(SP, f'{SP}/tc')
from evalpool import *

E   = np.load(f'{SP}/tc/emb_modernbert_L1024.npy').astype(np.float64)
E   = (E-E.mean(0))/np.where(E.std(0)>1e-9,E.std(0),1.0)
XT  = np.load(f'{SP}/Xtext.npy').astype(np.float64)          # hashed n-grams (current champion)
XB  = np.load(f'{SP}/X.npy'); COLSB=json.load(open(f'{SP}/cols.json'))
XB  = XB[:,[j for j,c in enumerate(COLSB) if c!='job_run_index']]

def ridge_oof(Z, tgt, folds, lam):
    p=np.zeros(N)
    for f in np.unique(folds):
        tr=folds!=f; te=~tr
        mu=Z[tr].mean(0); sg=np.where(Z[tr].std(0)>1e-9,Z[tr].std(0),1.)
        A=np.hstack([(Z[tr]-mu)/sg,np.ones((tr.sum(),1))])
        B=np.hstack([np.clip((Z[te]-mu)/sg,-5,5),np.ones((te.sum(),1))])
        w=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@tgt[tr])
        p[te]=np.clip(B@w,tgt[tr].min(),tgt[tr].max())
    return p
def kernel_oof(K, tgt, folds, lam):
    p=np.zeros(N)
    for f in np.unique(folds):
        tr=np.where(folds!=f)[0]; te=np.where(folds==f)[0]
        m=tgt[tr].mean()
        a=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),tgt[tr]-m)
        p[te]=np.clip(K[np.ix_(te,tr)]@a+m,tgt[tr].min(),tgt[tr].max())
    return p

KE = E@E.T/E.shape[1]
KT = XT@XT.T
Y  = PANEL.y

print('=== A. SPEND head: rank quality on the CRON POOL (287 rows) ===')
print(f"{'representation':38s} {'OOF R2':>7s} {'$@60':>7s} {'cap60':>6s} {'$@100':>7s} {'cap100':>6s} {'$@150':>7s} {'cap150':>6s}")
preds={}
preds['tabular 35 cols']        = ridge_oof(XB, YLOG, FOLD, 1.0)
preds['hashed n-grams (kernel)']= kernel_oof(KT, YLOG, FOLD, 0.3)
for lam in (0.03,0.1,0.3,1.0,3.0):
    preds[f'ModernBERT frozen L1024 lam={lam}'] = kernel_oof(KE, YLOG, FOLD, lam)
for nm,p in preds.items():
    r2=1-((YLOG-p)**2).sum()/((YLOG-YLOG.mean())**2).sum()
    row=''.join(f'{capture(p,k)[0]:8.2f} {capture(p,k)[1]:5.1f}%' for k in (60,100,150))
    print(f'{nm:38s} {r2:+7.3f} {row}')
best_bert=max((k for k in preds if k.startswith('ModernBERT')), key=lambda k: capture(preds[k],100)[1])
print(f'\nbest frozen-BERT spend: {best_bert}')
print('\n=== blends with the hashed n-gram predictor ===')
hb=preds['hashed n-grams (kernel)']; bb=preds[best_bert]
for w in (0,.25,.5,.75,1.0):
    p=(1-w)*hb+w*bb
    print(f'  w_bert={w:4.2f}  cap60 {capture(p,60)[1]:5.1f}%  cap100 {capture(p,100)[1]:5.1f}%  cap150 {capture(p,150)[1]:5.1f}%')
np.save(f'{SP}/tc/spend_bert.npy', bb); np.save(f'{SP}/tc/spend_hash.npy', hb)

print('\n=== B. FRICTION head: OOF AUC on the claude-cron population (558 rows, 99 events) ===')
tr_all=np.array(TRAIN_ALL)
def auc_on(p): return auc(p[tr_all], Y[tr_all])
print(f"  tabular 35 cols            {auc_on(ridge_oof(XB,Y,FOLD,1.0)):.3f}")
print(f"  hashed n-grams  lam=1.0    {auc_on(kernel_oof(KT,Y,FOLD,1.0)):.3f}")
for lam in (0.1,0.3,1.0,3.0,10.0):
    print(f"  ModernBERT frozen lam={lam:<5} {auc_on(kernel_oof(KE,Y,FOLD,lam)):.3f}")
