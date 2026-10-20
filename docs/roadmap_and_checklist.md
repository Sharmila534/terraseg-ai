# TerraSeg AI roadmap and checklist

This checklist records what is supported by the current project files and local run artifacts as of 2026-10-21. It is a project plan, not a claim that unfinished work has been completed.

## Current verified state

- [x] Project is in the C-drive workspace and its Git remote is configured as `https://github.com/Sharmila534/terraseg-ai.git`.
- [x] Dataset audit previously reported 770 image/mask pairs, with no unmatched or duplicate names. The archive's 2,003 non-landslide images do not have masks and are not included in this paired segmentation experiment.
- [x] A seed-42 split is saved locally in `results/splits.json`: 539 train, 115 validation, and 116 test IDs. All four validation runs use the same split.
- [x] Four training runs have saved checkpoints, epoch logs, configs, validation metrics, and prediction plots locally.
- [x] A test evaluation has been run once for Advanced U-Net at threshold 0.5. It produced 29.97% global IoU and 34.61% mean per-image IoU. The test predictions have been inspected and visibly overpredict in several examples.
- [x] The current highest validation global IoU is ResNet-34 U-Net: 79.32% global and 78.21% mean per-image IoU at epoch 31. No test result exists for this model.
- [x] Dataset files, checkpoints, plots, and full run outputs are ignored by Git; a compact metric summary is tracked separately.

## Results snapshot

| Configuration | Validation global IoU | Validation mean per-image IoU | Best epoch | Test status |
|---|---:|---:|---:|---|
| Custom CNN, default | 23.48% | 26.55% | 8 | Not evaluated |
| U-Net, default | 26.92% | 29.60% | 9 | Not evaluated |
| U-Net, advanced | 33.34% | 36.45% | 7 | Evaluated once: 29.97% global / 34.61% mean per-image IoU |
| ResNet-34 U-Net | 79.32% | 78.21% | 31 | Not evaluated |

These are local results from one seed and one random image-level partition. The original presentation's approximate 78.01% IoU has not been reproduced under a verified matching protocol. Do not compare those values as if they came from the same evaluation method.

## Roadmap

### Stage 1 — Clean verification and provenance

- [ ] In VS Code PowerShell, activate the project's `.venv`, run `python -m pytest -q tests`, and save the full output. Earlier targeted tests passed, but the full test invocation had Windows temporary-folder permission errors; a clean full run is still needed.
- [ ] Record Python, PyTorch, torchvision, segmentation-models-pytorch, Pillow, and CUDA versions plus CPU/GPU model for each future run.
- [ ] Preserve the current seed-42 run artifacts locally and keep the exact config/split associated with every result.
- [ ] Confirm dataset terms and citation requirements with the legitimate provider before redistributing the dataset, prediction imagery, or derived artifacts.

### Stage 2 — Choose a candidate using validation only

- [ ] Review validation metrics and plots, without using test scores. ResNet-34 U-Net currently leads validation, but its much higher score than the other models deserves a careful data/split review.
- [ ] Check that saved split IDs are disjoint, exhaustive over the paired samples, and stable across model comparisons.
- [ ] If threshold calibration is needed, choose the threshold using validation only and record it before any future test evaluation.
- [ ] Avoid rerunning a completed configuration into its existing output folder; use a new output name so the saved checkpoint and logs remain intact.

### Stage 3 — Strengthen generalization evidence

- [ ] Investigate spatial/group leakage in the current random image split. If location metadata or acquisition grouping is available, build a group-aware partition so nearby/related image patches do not cross partitions.
- [ ] Because the Advanced U-Net test partition was already evaluated and inspected, do not call it a pristine untouched benchmark. For a stronger final claim, reserve a new group-held-out test set or obtain independent data.
- [ ] Repeat the selected experiment with multiple fixed seeds or group folds and report variation, not only a single score.
- [ ] After freezing model, preprocessing, and threshold from training/validation evidence, evaluate once on the new held-out set and archive its IDs and metrics.

### Stage 4 — Presentation and release

- [ ] Update the result summary and README only from saved metric files; identify split, seed, model, threshold, and aggregation for every reported score.
- [ ] Add a small prediction figure only if data terms allow sharing it. Keep original dataset files and large checkpoints out of Git.
- [ ] Replace the `LICENSE` placeholder only after deciding which license the project owner has authority to apply. A code license does not grant rights to the dataset.
- [ ] Verify the pushed commit and repository page after publishing; retain the commit hash with the experiment record.

## Repeatable commands (Windows PowerShell)

Run from the project root in VS Code PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$images = ".\data\bijie\raw\Bijie-landslide-dataset\landslide\image"
$masks = ".\data\bijie\raw\Bijie-landslide-dataset\landslide\mask"
python -m terraseg audit --images $images --masks $masks
python -m terraseg smoke --device cpu
python -m pytest -q tests
```

Training commands and model-specific output names are documented in the README. Full-dataset CPU training was very slow in the recorded runs; use a compatible CUDA PyTorch setup or Colab GPU for practical repeat experiments. Evaluation commands must point to a checkpoint and the saved split file. Do not evaluate a candidate on test until the model and threshold are fixed.

## Release safety checklist

- [x] `data/` and `results/` run artifacts are excluded from Git, except `results/summary/*.json`.
- [ ] Review `git status --ignored --short` and staged changes before each commit/push.
- [ ] Confirm no dataset archives, local paths, credentials, checkpoints, or large prediction plots are staged.
- [ ] Verify README numbers against `results/summary/current_results.json` and the underlying local JSON metrics.
- [ ] Before a formal release or inviting reuse, make the code license choice and dataset attribution/data-use notes clear.
