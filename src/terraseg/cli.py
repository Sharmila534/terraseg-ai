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


def _validate_data_paths(args, parser):
    missing=[(name,value) for name,value in (("image",args.images),("mask",args.masks)) if not Path(value).is_dir()]
    if missing:
        details="; ".join(f"{name} directory does not exist: {value}" for name,value in missing)
        parser.error(f"{details}. Replace the example paths with the actual extracted image and mask folder paths.")


def main(argv=None):
    p=argparse.ArgumentParser(prog="terraseg"); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("audit"); _common(a)
    t=sub.add_parser("train"); _common(t); t.add_argument("--model",choices=["custom_cnn","unet","resnet34_unet"],required=True); t.add_argument("--config",default="configs/default.json"); t.add_argument("--seed",type=int,default=42); t.add_argument("--splits",default="results/splits.json"); t.add_argument("--augment",action="store_true"); t.add_argument("--imagenet-normalization",action="store_true"); t.add_argument("--pretrained",action="store_true"); t.add_argument("--output",default=None); t.add_argument("--device",default="auto")
    e=sub.add_parser("evaluate"); _common(e); e.add_argument("--model",choices=["custom_cnn","unet","resnet34_unet"],required=True); e.add_argument("--checkpoint",required=True); e.add_argument("--splits",default="results/splits.json"); e.add_argument("--partition",choices=["validation","test"],default="test"); e.add_argument("--threshold",type=float,default=.5); e.add_argument("--image-size",type=int,default=256); e.add_argument("--batch-size",type=int,default=8); e.add_argument("--num-workers",type=int,default=0); e.add_argument("--imagenet-normalization",action="store_true"); e.add_argument("--output",default=None); e.add_argument("--device",default="auto")
    s=sub.add_parser("smoke"); s.add_argument("--device",default="cpu")
    args=p.parse_args(argv)
    if args.command in {"audit","train","evaluate"}:
        _validate_data_paths(args,p)
    if args.command=="audit":
        report=audit_pairs(args.images,args.masks); print(json.dumps(report,indent=2))
        if not report["paired_count"]:
            print("No image-mask pairs found. Put matching files in the selected folders or correct the folder paths.")
            raise SystemExit(2)
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
        try:
            config=json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            p.error(f"Could not read config file {args.config}: {exc}")
        if not isinstance(config, dict):
            p.error(f"Config file {args.config} must contain a JSON object.")
        try:
            args.image_size=int(config["image_size"]); args.batch_size=int(config["batch_size"]); args.num_workers=int(config.get("num_workers",0)); args.threshold=float(config.get("threshold",.5))
            if args.image_size < 1 or args.batch_size < 1 or args.num_workers < 0 or not 0 < args.threshold < 1:
                raise ValueError("image_size/batch_size must be positive, num_workers nonnegative, and threshold between 0 and 1")
            train_fraction=float(config["train_fraction"]); validation_fraction=float(config["validation_fraction"])
            test_fraction=float(config.get("test_fraction",1-train_fraction-validation_fraction))
            if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-6:
                raise ValueError("train_fraction, validation_fraction, and test_fraction must sum to 1")
        except (KeyError, TypeError, ValueError) as exc:
            p.error(f"Invalid training config {args.config}: {exc}")
        args.imagenet_normalization=args.imagenet_normalization or config.get("imagenet_normalization",False)
        try:
            pairs=matched_pairs(args.images,args.masks,strict=True)
        except ValueError as exc:
            p.error(f"Dataset pairing failed: {exc}. Run `terraseg audit` to inspect the files.")
        ids=[p[0] for p in pairs]
        if not ids:
            p.error("No matched image-mask pairs found. Run `terraseg audit` and place corresponding files directly in the selected folders.")
        if len(ids) < 3:
            p.error(f"Only {len(ids)} matched image-mask pair(s) found. At least 3 are required to create non-empty train, validation, and test partitions.")
        split_path=Path(args.splits)
        if split_path.exists():
            try:
                splits=load_splits(split_path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                p.error(f"Could not read split file {split_path}: {exc}. Choose a new path with --splits.")
            if isinstance(splits,dict) and set(splits)=={"train","validation","test"} and all(isinstance(v,list) for v in splits.values()) and not any(splits.values()):
                print(f"Existing split file {split_path} is empty; regenerating it from the {len(ids)} matched pairs.")
                splits=make_splits(ids,args.seed,train_fraction,validation_fraction)
                save_splits(splits,split_path,args.seed)
            else:
                if not isinstance(splits,dict) or set(splits) != {"train","validation","test"} or not all(isinstance(v,list) and all(isinstance(i,str) for i in v) for v in splits.values()):
                    p.error(f"Split file {split_path} must contain train, validation, and test lists of sample IDs.")
                flattened=sum(splits.values(),[])
                if len(flattened) != len(set(flattened)) or set(flattened) != set(ids):
                    p.error(f"Split file {split_path} does not match the current dataset. Use a new path with --splits or remove the stale split file after reviewing it.")
        else:
            try:
                splits=make_splits(ids,args.seed,train_fraction,validation_fraction)
            except ValueError as exc:
                p.error(str(exc))
            save_splits(splits,split_path,args.seed)
        try:
            split_metadata=json.loads(split_path.read_text(encoding="utf-8"))
            split_seed=split_metadata.get("seed") if isinstance(split_metadata,dict) else None
        except (OSError, json.JSONDecodeError) as exc:
            p.error(f"Could not read split metadata from {split_path}: {exc}")
        flattened=sum(splits.values(),[])
        if not splits["train"] or not splits["validation"] or not splits["test"]:
            p.error(f"Split file {split_path} has an empty partition. Remove it after review or pass a new --splits path.")
        model_name=args.model; epochs=config["epochs"]; patience=config.get("patience",10); lr=config["learning_rate"]; loss_name=config["loss"]; pos=config.get("positive_weight",15.)
        augment=args.augment or config.get("augmentation",False)
        run_config={**config,"model":model_name,"seed":args.seed,"split_seed":split_seed,"augmentation":augment,"imagenet_normalization":args.imagenet_normalization,"splits":str(Path(args.splits))}
        output=Path(args.output or f"results/{model_name}/{Path(args.config).stem}_seed{args.seed}")
    else:
        try:
            splits=load_splits(args.splits)
        except (OSError, ValueError, KeyError) as exc:
            p.error(f"Could not read split file {args.splits}: {exc}")
        if not isinstance(splits,dict) or set(splits)!={"train","validation","test"} or not all(isinstance(v,list) for v in splits.values()):
            p.error(f"Split file {args.splits} must contain train, validation, and test lists.")
        if not splits[args.partition]:
            p.error(f"The {args.partition} partition in {args.splits} is empty.")
        try:
            pairs=matched_pairs(args.images,args.masks,strict=True)
        except ValueError as exc:
            p.error(f"Dataset pairing failed: {exc}. Run `terraseg audit` to inspect the files.")
        if set(sum(splits.values(),[])) != {pair[0] for pair in pairs}:
            p.error(f"Split file {args.splits} does not match the current dataset.")
        model_name=args.model; output=Path(args.output or f"results/evaluation_{model_name}_{args.partition}"); augment=False
    random.seed(getattr(args,"seed",42)); np.random.seed(getattr(args,"seed",42)); torch.manual_seed(getattr(args,"seed",42))
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    if args.command=="train":
        if (output/"best.pt").exists(): p.error(f"Run already exists: {output}. Choose a new --output directory to preserve it.")
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
