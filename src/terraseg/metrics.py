import torch


def confusion_counts(logits, targets, threshold=.5):
    preds=torch.sigmoid(logits) >= threshold; truth=targets >= .5
    dims=tuple(range(1,preds.ndim))
    tp=(preds & truth).sum(dims).double(); fp=(preds & ~truth).sum(dims).double(); fn=(~preds & truth).sum(dims).double(); tn=(~preds & ~truth).sum(dims).double()
    return tp,fp,fn,tn


def metrics_from_counts(tp, fp, fn, tn):
    def safe(num, den): return torch.where(den==0, torch.ones_like(num), num/den)
    per={"iou":safe(tp,tp+fp+fn).mean(), "dice":safe(2*tp,2*tp+fp+fn).mean(), "precision":safe(tp,tp+fp).mean(), "recall":safe(tp,tp+fn).mean()}
    TP,FP,FN=tp.sum(),fp.sum(),fn.sum()
    global_m={"iou":safe(TP,TP+FP+FN), "dice":safe(2*TP,2*TP+FP+FN), "precision":safe(TP,TP+FP), "recall":safe(TP,TP+FN)}
    return {f"per_image_{k}":float(v.item()) for k,v in per.items()} | {f"global_{k}":float(v.item()) for k,v in global_m.items()}


def binary_metrics(logits, targets, threshold=.5):
    return metrics_from_counts(*confusion_counts(logits,targets,threshold))
