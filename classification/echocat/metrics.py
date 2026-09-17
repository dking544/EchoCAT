import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score)


def classification_metrics(scores, targets, predictions=None):
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=np.int64)
    if scores.ndim != 2 or len(scores) != len(targets) or len(targets) == 0:
        raise ValueError('Expected nonempty [N,C] scores and [N] labels')
    k = scores.shape[1]
    if not np.isfinite(scores).all() or np.any(targets < 0) or np.any(targets >= k):
        raise ValueError('Invalid scores or labels')
    predictions = scores.argmax(1) if predictions is None else np.asarray(predictions,dtype=np.int64)
    if predictions.shape!=targets.shape or (predictions<0).any() or (predictions>=k).any():
        raise ValueError('Invalid predicted labels')
    p, r, f, support = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(k), zero_division=0)
    matrix = confusion_matrix(targets, predictions, labels=np.arange(k))
    auc, ap, npv, specificity = [], [], [], []
    for c in range(k):
        positive = targets == c
        auc.append(float(roc_auc_score(positive, scores[:, c])) if len(np.unique(positive)) == 2 else None)
        ap.append(float(average_precision_score(positive, scores[:, c])) if positive.any() else None)
        tn = int(matrix.sum() - matrix[c].sum() - matrix[:, c].sum() + matrix[c, c])
        fn = int(matrix[c].sum() - matrix[c, c])
        fp = int(matrix[:, c].sum() - matrix[c, c])
        npv.append(tn / (tn + fn) if tn + fn else None)
        specificity.append(tn/(tn+fp) if tn+fp else None)
    out = dict(accuracy=float(accuracy_score(targets, predictions)),
               precision_macro=float(p.mean()), recall_macro=float(r.mean()),
               f1_macro=float(f.mean()), precision_per_class=p.tolist(),
               recall_per_class=r.tolist(), f1_per_class=f.tolist(),
               support_per_class=support.tolist(), npv_per_class=npv,
               specificity_per_class=specificity, balanced_accuracy=float(r.mean()),
               auroc_per_class=auc, auprc_per_class=ap,
               confusion_matrix=matrix.tolist(), total_samples=len(targets),
               class_distribution={i: int(n) for i,n in enumerate(support)})
    out.update(precision=out['precision_macro'], recall=out['recall_macro'],
               f1_score=out['f1_macro'])
    out['macro_auroc'] = float(np.mean(auc)) if all(v is not None for v in auc) else None
    out['macro_auprc'] = float(np.mean(ap)) if all(v is not None for v in ap) else None
    out['top5_accuracy']=float(np.any(np.argsort(-scores,axis=1,kind='stable')[:,:min(5,k)]==targets[:,None],axis=1).mean())
    return out


def training_metrics(scores, targets):
    return classification_metrics(scores.detach().cpu().numpy(), targets.detach().cpu().numpy())


def selection_key(metrics, task, mode='recipe'):
    recalls = metrics['recall_per_class']
    if any(n == 0 for n in metrics['support_per_class']):
        raise ValueError('Validation set must contain every class for model selection')
    loss = metrics['avg_loss']
    if mode == 'macro_f1':
        return (metrics['f1_macro'], metrics['recall_macro'], metrics['accuracy'], -loss)
    if mode == 'accuracy' or (mode == 'recipe' and task == 'six'):
        return (metrics['accuracy'], -loss)
    if mode != 'recipe' or len(recalls) != 2:
        raise ValueError('Unsupported validation selection mode')


    floor = .9 if task == 'view' else .8
    worst, bacc = min(recalls), sum(recalls)/2
    eligible = worst >= floor
    return ((1, bacc, worst, metrics['accuracy'], -loss) if eligible else
            (0, worst, bacc, metrics['accuracy'], -loss))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores, targets, total_loss = [], [], 0.
    for images, labels, _ in loader:
        probabilities, losses = model(images.to(device), targets=labels.to(device),
                                      return_loss=True, train_statu=True)
        scores.append(probabilities.cpu().numpy())
        targets.append(labels.numpy())
        total_loss += float(losses['loss'])
    if not scores:
        raise ValueError('Validation loader has no batches')
    result = classification_metrics(np.concatenate(scores), np.concatenate(targets))

    result['avg_loss'] = total_loss / len(loader)
    return result
