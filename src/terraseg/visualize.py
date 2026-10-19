from pathlib import Path
import matplotlib.pyplot as plt
import torch


def save_predictions(model, loader, device, output, count=4, normalized=False):
    model.eval(); images,masks=next(iter(loader)); images=images[:count].to(device); masks=masks[:count]
    with torch.no_grad(): probs=torch.sigmoid(model(images)).cpu()
    imgs=images.cpu()
    if normalized:
        mean=torch.tensor([.485,.456,.406])[:,None,None]; std=torch.tensor([.229,.224,.225])[:,None,None]
        imgs=imgs*std+mean
    n=len(imgs); fig,axes=plt.subplots(n,3,figsize=(9,3*n),squeeze=False)
    for i in range(n):
        axes[i,0].imshow(imgs[i].permute(1,2,0).clamp(0,1)); axes[i,0].set_title("RGB image")
        axes[i,1].imshow(masks[i,0],cmap="gray",vmin=0,vmax=1); axes[i,1].set_title("Target mask")
        axes[i,2].imshow(probs[i,0],cmap="magma",vmin=0,vmax=1); axes[i,2].set_title("Predicted probability")
        for ax in axes[i]: ax.axis("off")
    Path(output).parent.mkdir(parents=True,exist_ok=True); fig.tight_layout(); fig.savefig(output,dpi=140); plt.close(fig)
