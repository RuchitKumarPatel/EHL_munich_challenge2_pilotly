"""Cron-restricted routing evaluation. Same friction estimand as every earlier run (score2/hi_ct)."""
import sys, json, numpy as np
from _scratch import scratch_dir, on_path
SP=scratch_dir(); on_path(SP)
from contrast import *          # PANEL, ARMS, AI, CAND, C, score2, show2, contrast, boot_contrast, FOLD, ...
from router import pricing

SON = AI['claude-sonnet-5']
X   = np.load(f'{SP}/X.npy'); COLS = json.load(open(f'{SP}/cols.json'))
TRIG_CRON = X[:, COLS.index('trig_cron')].astype(bool)
CLAUDE    = np.array([PANEL.family[i]=='claude' for i in range(N)])
POOL      = [i for i in range(N) if CLAUDE[i] and TRIG_CRON[i] and any(d['t']==SON for d in CAND[i])]
TRAIN_ALL = [i for i in range(N) if CLAUDE[i] and TRIG_CRON[i]]      # both arms present here
D    = C['assumed_default']
RATE = np.array([pricing.rate_in(ARMS[LOGGED[i]],'assumed_default') for i in range(N)])
EFF  = D[R,LOGGED]/RATE*1e6
YLOG = np.log1p(EFF)
TRUE_SAV = D[R,LOGGED]-D[R,SON]
CLUSTER = PANEL.cluster_code

def policy_topk(score, k, pool=None):
    pool = POOL if pool is None else pool
    o = sorted(pool, key=lambda i: -score[i])[:k]
    t = LOGGED.copy()
    for i in o: t[i] = SON
    return t

def oracle(k, pool=None):
    pool = POOL if pool is None else pool
    return sum(sorted((TRUE_SAV[i] for i in pool), reverse=True)[:k])

def capture(score, k, pool=None):
    pool = POOL if pool is None else pool
    o = sorted(pool, key=lambda i: -score[i])[:k]
    got = sum(TRUE_SAV[i] for i in o)
    return got, 100*got/oracle(k,pool)

def grouped_splits(n_splits=8, frac=0.5, seed=4):
    """train/test split by CLUSTER (literal cron path) — never split a job across the boundary."""
    rng = np.random.default_rng(seed); uc = np.unique(CLUSTER)
    for _ in range(n_splits):
        perm = rng.permutation(uc); h = int(len(uc)*frac)
        tr = set(perm[:h].tolist())
        m = np.array([c in tr for c in CLUSTER])
        yield np.where(m)[0], np.where(~m)[0]

def auc(s, y):
    r = np.argsort(np.argsort(s)); p = y==1; n = y==0
    if p.sum()==0 or n.sum()==0: return float('nan')
    return (r[p].mean()-r[n].mean())/len(s)+0.5

nA, nS = len(ARMS), len(PANEL.strata)
def cells_on(rows):
    ns=np.zeros((nS,nA)); ys=np.zeros((nS,nA))
    np.add.at(ns,(S_[rows],PANEL.arm_code[rows]),1.); np.add.at(ys,(S_[rows],PANEL.arm_code[rows]),PANEL.y[rows])
    g=PANEL.y[rows].mean(); na=ns.sum(0); ya=ys.sum(0); ar=(ya+20.*g)/(na+20.)
    return (ys+5.*ar[None,:])/(ns+5.), ns

def oos_risk_and_money(sel, test_rows, cr, ns):
    """same modelled-contrast risk as contrast(), restricted to the test half, count-weighted."""
    t = LOGGED.copy()
    for i in sel: t[i] = SON
    w = np.zeros(N); w[test_rows] = 1.0; w /= w.sum()
    sw = t != PANEL.arm_code; sup = ns[S_,t] >= 3
    risk = float((w*sw*sup*(cr[S_,t]-cr[S_,PANEL.arm_code])).sum()) \
         + float((w*sw*(~sup)*(1-cr[S_,PANEL.arm_code])).sum())
    return risk, float(sum(TRUE_SAV[i] for i in sel))
