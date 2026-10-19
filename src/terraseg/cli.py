import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .data import audit_pairs, matched_pairs, make_splits, save_splits, load_splits, BijieDataset
from .models import build_model
from .losses import build_loss
from .engine import run_epoch
from .visualize import save_predictions


def _loader(args, ids, augment=False, shuffle=False):
    ds=BijieDataset(args.images,args.masks,ids,image_size=args.image_size,augment=augment,imagenet_normalization=args.imagenet_normalization)
    return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_workers)


def _common(p):
    p.add_argument("--images",required=True); p.add_argument("--masks",required=True)


def main(argv=None):
    p=argparse.ArgumentParser(prog="terraseg"); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("audit"); _common(a)
    t=sub.add_parser("train"); _common(t); t.add_argument("--model",choices=["custom_cnn","unet","resnet34_unet"],required=True); t.add_argument("--config",default="configs/default.json"); t.add_argument("--seed",type=int,default=42); t.add_argument("--splits",default="results/splits.json"); t.add_argument("--augment",action="store_true"); t.add_argument("--imagenet-normalization",action="store_true"); t.add_argument("--pretrained",action="store_true"); t.add_argument("--output",default=None); t.add_argument("--device",default="auto")
    e=sub.add_parser("evaluate"); _common(e); e.add_argument("--model",choices=["custom_cnn","unet","resnet34_unet"],required=True); e.add_argument("--checkpoint",required=True); e.add_argument("--splits",default="results/splits.json"); e.add_argument("--partition",choices=["validation","test"],default="test"); e.add_argument("--threshold",type=float,default=.5); e.add_argument("--image-size",type=int,default=256); e.add_argument("--batch-size",type=int,default=8); e.add_argument("--num-workers",type=int,default=0); e.add_argument("--imagenet-normalization",action="store_true"); e.add_argument("--output",default=None); e.add_argument("--device",default="auto")
    s=sub.add_parser("smoke"); s.add_argument("--device",default="cpu")
    args=p.parse_args(argv)
    if args.command=="audit":
        report=audit_pairs(args.images,args.masks); print(json.dumps(report,indent=2))
        if report["duplicate_images"] or report["duplicate_masks"] or report["unmatched_images"] or report["unmatched_masks"]: raise SystemExit(2)
        return
    if args.command=="smoke":
        dev=torch.device(args.device); x=torch.rand(2,3,256,256,device=dev)
        for name in ("custom_cnn","unet"):
            y=build_model(name).to(dev)(x); assert y.shape==(2,1,256,256), (name,y.shape)
        print("Synthetic CPU/GPU shape smoke passed for custom_cnn and unet")
        return
    config={}
    if args.command=="train":
        config=json.loads(Path(args.config).read_text(encoding="utf-8")); args.image_size=config["image_size"]; args.batch_size=config["batch_size"]; args.num_workers=config.get("num_workers",0); args.threshold=config.get("threshold",.5)
        args.imagenet_normalization=args.imagenet_normalization or config.get("imagenet_normalization",False)
        pairs=matched_pairs(args.images,args.masks,strict=True); ids=[p[0] for p in pairs]
        if Path(args.splits).exists(): splits=load_splits(args.splits)
        else:
            splits=make_splits(ids,args.seed,config["train_fraction"],config["validation_fraction"]); save_splits(splits,args.splits,args.seed)
        split_seed=json.loads(Path(args.splits).read_text(encoding="utf-8")).get("seed")
        flattened=sum(splits.values(),[])
        if set(splits) != {"train","validation","test"} or len(flattened) != len(set(flattened)) or set(flattened) != set(ids): raise ValueError("Saved splits do not match current paired dataset")
        if not splits["train"] or not splits["validation"] or not splits["test"]: raise ValueError("Train, validation, and test partitions must all be non-empty")
        model_name=args.model; epochs=config["epochs"]; patience=config.get("patience",10); lr=config["learning_rate"]; loss_name=config["loss"]; pos=config.get("positive_weight",15.)
        augment=args.augment or config.get("augmentation",False)
        run_config={**config,"model":model_name,"seed":args.seed,"split_seed":split_seed,"augmentation":augment,"imagenet_normalization":args.imagenet_normalization,"splits":str(Path(args.splits))}
        output=Path(args.output or f"results/{model_name}/{Path(args.config).stem}_seed{args.seed}")
    else:
        splits=load_splits(args.splits); model_name=args.model; output=Path(args.output or f"results/evaluation_{model_name}_{args.partition}"); augment=False
    random.seed(getattr(args,"seed",42)); np.random.seed(getattr(args,"seed",42)); torch.manual_seed(getattr(args,"seed",42))
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    if args.command=="train":
        if (output/"best.pt").exists(): raise FileExistsError(f"Run already exists: {output}. Choose a new --output directory to preserve it.")
        train_loader=_loader(args,splits["train"],augment,True); val_loader=_loader(args,splits["validation"])
        model=build_model(model_name,args.pretrained or model_name=="resnet34_unet").to(device); criterion=build_loss(loss_name,pos,device); optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        output.mkdir(parents=True,exist_ok=True); (output/"config.json").write_text(json.dumps(run_config,indent=2),encoding="utf-8")
        best=float("inf"); wait=0; log=[]; start=time.perf_counter()
        for epoch in range(1,epochs+1):
            tr=run_epoch(model,train_loader,criterion,device,optimizer,args.threshold); va=run_epoch(model,val_loader,criterion,device,None,args.threshold)
            row={"epoch":epoch,"train":tr,"validation":va}; log.append(row); print(json.dumps(row))
            if va["loss"] < best:
                best=va["loss"]; wait=0; torch.save({"model":model.state_dict(),"model_name":model_name,"epoch":epoch,"config":run_config},output/"best.pt")
            else:
                wait+=1
                if wait>=patience: break
        (output/"epochs.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
        ck=torch.load(output/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"])
        metrics=run_epoch(model,val_loader,criterion,device,None,args.threshold); metrics.update({"best_epoch":ck["epoch"],"runtime_seconds":time.perf_counter()-start,"partition":"validation"})
        (output/"validation_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        save_predictions(model,val_loader,device,output/"validation_predictions.png",normalized=args.imagenet_normalization)
    else:
        ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
        saved_config=ck.get("config",{}) if isinstance(ck,dict) else {}
        args.imagenet_normalization=saved_config.get("imagenet_normalization",args.imagenet_normalization)
        loader=_loader(args,splits[args.partition]); model=build_model(model_name,False).to(device)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        criterion=build_loss(saved_config.get("loss","bce"),saved_config.get("positive_weight",15.),device)
        metrics=run_epoch(model,loader,criterion,device,None,args.threshold)
        metrics.update({"partition":args.partition,"threshold":args.threshold,"checkpoint":str(args.checkpoint)})
        output.mkdir(parents=True,exist_ok=True); (output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        save_predictions(model,loader,device,output/"predictions.png",normalized=args.imagenet_normalization)
        print(json.dumps(metrics,indent=2))


if __name__=="__main__": main()
