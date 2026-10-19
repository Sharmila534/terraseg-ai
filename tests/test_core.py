import numpy as np
import pytest
import torch
from PIL import Image

from terraseg.data import audit_pairs, matched_pairs, make_splits, BijieDataset
from terraseg.metrics import binary_metrics
from terraseg.models import build_model


def _pair_dirs(tmp_path):
    im=tmp_path/"images"; ma=tmp_path/"masks"; im.mkdir(); ma.mkdir()
    Image.fromarray(np.zeros((12,10,3),dtype=np.uint8)).save(im/"tile_1.png")
    mask=np.zeros((6,10),dtype=np.uint8); mask[1:4,2:7]=255
    Image.fromarray(mask).save(ma/"TILE_1.png")
    return im,ma


def test_filename_matching_and_casefold(tmp_path):
    im,ma=_pair_dirs(tmp_path); assert len(matched_pairs(im,ma))==1


def test_unmatched_files_reported(tmp_path):
    im,ma=_pair_dirs(tmp_path); Image.new("RGB",(2,2)).save(im/"orphan.png")
    report=audit_pairs(im,ma); assert report["unmatched_images"]==["orphan.png"]
    with pytest.raises(ValueError): matched_pairs(im,ma)
    (im/"orphan.png").unlink(); Image.new("L",(2,2)).save(ma/"orphan_mask.png")
    report=audit_pairs(im,ma); assert report["unmatched_masks"]==["orphan_mask.png"]


def test_missing_directory_is_reported(tmp_path):
    im,ma=_pair_dirs(tmp_path)
    with pytest.raises(FileNotFoundError): audit_pairs(im,ma/"missing")


def test_duplicate_stems_rejected(tmp_path):
    im,ma=_pair_dirs(tmp_path); Image.new("RGB",(2,2)).save(im/"tile_1.jpg")
    with pytest.raises(ValueError): matched_pairs(im,ma)


def test_dataset_shapes_ranges_and_deterministic_eval(tmp_path, monkeypatch):
    im,ma=_pair_dirs(tmp_path); ds=BijieDataset(im,ma,image_size=16)
    x,y=ds[0]; assert x.shape==(3,16,16) and y.shape==(1,16,16)
    assert x.min()>=0 and x.max()<=1 and set(y.unique().tolist()) <= {0.,1.}
    import random
    monkeypatch.setattr(random,"random",lambda: (_ for _ in ()).throw(AssertionError("validation transform sampled randomness")))
    assert torch.equal(ds[0][0],ds[0][0]) and torch.equal(ds[0][1],ds[0][1])


def test_train_augmentation_is_seeded_and_changes_dataset(tmp_path):
    im,ma=_pair_dirs(tmp_path); ds=BijieDataset(im,ma,image_size=16,augment=True)
    torch.manual_seed(1); import random; random.seed(1); a=ds[0]
    random.seed(9); b=ds[0]
    assert a[0].shape==b[0].shape and a[1].shape==b[1].shape
    assert not torch.equal(a[1],b[1])


def test_hand_computable_metrics_and_empty_case():
    # One TP, one FP, one FN, one TN.
    logits=torch.tensor([[[[8.,8.],[-8.,-8.]]]])
    target=torch.tensor([[[[1.,0.],[1.,0.]]]])
    m=binary_metrics(logits,target)
    assert m["global_iou"]==pytest.approx(1/3) and m["global_dice"]==pytest.approx(.5)
    assert m["global_precision"]==pytest.approx(.5) and m["global_recall"]==pytest.approx(.5)
    empty=binary_metrics(torch.full((1,1,2,2),-10.),torch.zeros(1,1,2,2))
    assert empty["global_iou"]==1 and empty["per_image_dice"]==1


def test_per_image_macro_and_global_pixel_iou_are_distinct():
    logits=torch.tensor([[[[8.,-8.],[-8.,-8.]]], [[[8.,8.],[-8.,8.]]]])
    target=torch.tensor([[[[1.,0.],[0.,0.]]], [[[1.,1.],[1.,0.]]]])
    m=binary_metrics(logits,target)
    assert m["per_image_iou"]==pytest.approx(.75)
    assert m["global_iou"]==pytest.approx(.6)


def test_reproducible_splits():
    ids=[f"{i}" for i in range(20)]; a=make_splits(ids,11); b=make_splits(ids,11)
    assert a==b and len(set(sum(a.values(),[])))==20
    assert len(a["train"])==14 and len(a["validation"])==3 and len(a["test"])==3
    assert a != make_splits(ids,12)


@pytest.mark.parametrize("name",["custom_cnn","unet"])
def test_model_output_shape(name):
    y=build_model(name)(torch.rand(1,3,64,64)); assert y.shape==(1,1,64,64)


def test_resnet34_model_output_shape():
    pytest.importorskip("segmentation_models_pytorch")
    y=build_model("resnet34_unet",pretrained=False)(torch.rand(1,3,64,64))
    assert y.shape==(1,1,64,64)
