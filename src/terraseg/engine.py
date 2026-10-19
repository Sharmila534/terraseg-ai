import time
import torch
from .metrics import confusion_counts, metrics_from_counts


def run_epoch(model, loader, criterion, device, optimizer=None, threshold=.5):
    training=optimizer is not None
    model.train(training); total_loss=0.; seen=0; count_lists=[[],[],[],[]]
    start=time.perf_counter()
    context=torch.enable_grad if training else torch.no_grad
    with context():
        for images,masks in loader:
            images,masks=images.to(device),masks.to(device)
            logits=model(images)
            if logits.shape != masks.shape: raise AssertionError(f"Output/mask shape mismatch: {logits.shape} vs {masks.shape}")
            loss=criterion(logits,masks)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            batch=len(images); total_loss+=loss.item()*batch; seen+=batch
            for store,value in zip(count_lists,confusion_counts(logits.detach(),masks,threshold)):
                store.append(value)
    if not seen: raise ValueError("Loader has no samples")
    counts=[torch.cat(x) for x in count_lists]
    return {"loss":total_loss/seen, **metrics_from_counts(*counts), "seconds":time.perf_counter()-start}
