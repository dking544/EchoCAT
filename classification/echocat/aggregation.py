import math
import numpy as np


def aggregate(q, diagnosis_scores, task):
    q=np.asarray(q,dtype=float)
    scores=np.asarray(diagnosis_scores,dtype=float)
    k={'binary':2,'six':6}.get(task)
    if k is None or q.ndim!=1 or not len(q) or scores.shape!=(len(q),k):
        raise ValueError('Expected nonempty [N] 4CH probabilities and [N,C] diagnostic probabilities')
    if not np.isfinite(q).all() or not np.isfinite(scores).all():raise ValueError('Nonfinite probabilities')
    if (q<0).any() or (q>1).any() or (scores<0).any() or (scores>1).any() or not np.allclose(scores.sum(1),1,atol=1e-5):
        raise ValueError('Inputs must be single-softmax probabilities, not logits')
    candidates=np.flatnonzero(q>=.5)
    fallback=not len(candidates)
    if fallback:
        selected=np.asarray([int(np.argmax(q))])
    else:
        ranked=candidates[np.argsort(-q[candidates],kind='stable')]
        selected=ranked[:math.ceil(len(ranked)/2)]
    if task=='binary':
        pooled=scores[selected].mean(0)
        prediction=int(pooled[1]>=.5)
    else:
        weights=q[selected]

        pooled=np.average(scores[selected],axis=0,weights=weights) if weights.sum()>0 else scores[selected].mean(0)
        prediction=int(pooled.argmax())
    return {'scores':pooled.tolist(),'prediction':prediction,
            'selected_indices':selected.tolist(),'used_fallback':fallback}
