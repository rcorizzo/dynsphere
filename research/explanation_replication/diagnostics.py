"""Exploratory checks using saved attribution evidence only."""
import numpy as np
from research.explanation_study import core


def budget_curve(part):
    phi=np.asarray(part['attributions'],float)
    order=np.argsort(-np.maximum(phi,0),kind='stable');out={}
    for budget in (1,2,4):
        kept=np.zeros_like(phi);indices=order[:min(budget,len(phi))]
        kept[indices]=np.maximum(phi[indices],0)
        out[str(budget)]=core.infidelity(kept,part['heldout_masks'],part['heldout_changes'])
    out['all']=core.infidelity(np.maximum(phi,0),part['heldout_masks'],part['heldout_changes'])
    out['signed_full']=core.infidelity(phi,part['heldout_masks'],part['heldout_changes'])
    return out


def cross_seed(values):
    if set(values)!={37,38,39}:return None
    keys=set(values[37])
    if not keys or any(set(v)!=keys for v in values.values()):return None
    agreements=[core.rank_agreement(values[a][k],values[b][k]) for k in sorted(keys)
                for a,b in ((37,38),(37,39),(38,39))]
    return None if any(v is None for v in agreements) else float(np.mean(agreements))
