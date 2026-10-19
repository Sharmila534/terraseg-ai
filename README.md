# TerraSeg AI

TerraSeg AI is a PyTorch project for binary landslide-boundary segmentation from RGB satellite imagery. It packages the supplied Colab prototype into a local command-line workflow with three retained model approaches, auditable image/mask pairing, reproducible partitions, and explicit evaluation metrics.

## 1. Problem and motivation

Landslide boundary maps can support hazard assessment and post-event analysis. This project explores pixel-level segmentation of landslide masks from satellite image patches. It is an educational and experimental project; it is not validated for operational hazard response.

## 2. What is implemented

- Custom CNN, baseline U-Net, and ResNet-34 encoder U-Net model options based on the supplied code.
- Image/mask matching by case-insensitive filename stem, with reporting for unmatched and duplicate files.
- RGB and binary-mask loading, resizing, shape/value checks, and optional horizontal/vertical flips for training samples.
- Configurable train/validation/test partitions saved by sample ID and reused across models.
- BCE, Dice, and combined BCE/Dice loss functions.
- Per-image macro and global pixel metrics: IoU, Dice/F1, precision, and recall.
- Training, validation, checkpointing, machine-readable epoch/results logs, evaluation, and prediction plots.
- Synthetic tests that need no dataset download and do not train a large model.

The source architectures are retained. Corrected data pairing, mask interpolation, encoder preprocessing, split design, and metric aggregation change parts of the original experiment, so scores may differ. No result equality or improvement is claimed without a controlled run.

## 3. Model architectures

| Model name | Description | Source training recipe represented |
|---|---|---|
| `custom_cnn` | Two pooling encoder stages and transposed-convolution decoder | Adam, learning rate 0.001, 10 epochs, weighted BCE |
| `unet` | Two-level encoder/decoder with skip connections | Baseline: Adam 0.001, 10 epochs, weighted BCE. Advanced: Adam 0.001, up to 50 epochs, Dice loss, patience 10, flips |
| `resnet34_unet` | `segmentation-models-pytorch` U-Net with ImageNet-pretrained ResNet-34 encoder | Adam 0.0005, up to 50 epochs, BCE+Dice, patience 10, flips |

Presets are in `configs/`. The ResNet preset enables ImageNet mean/std normalization. That corrects preprocessing for the pretrained encoder and differs from the source run, which did not normalize inputs. The first use of pretrained weights may need internet access. To compare against source-like preprocessing, run a separate config with `imagenet_normalization` set to `false` and record it as a distinct experiment.

## 4. Dataset and data-use notes

No dataset or model checkpoint is included. The [Wuhan University GPCV Bijie dataset page](https://gpcv.whu.edu.cn/data/Bijie_pages.html) is the upstream page and describes the dataset and download. The supplied script refers to the Kaggle mirror `ksncik/bijie-landslide-dataset`. The upstream page does not state a reuse license, and the mirror's terms have not been verified. Public availability is not proof of redistribution permission; check the current terms or ask the provider before reuse or redistribution. Do not commit downloaded data.

The upstream description distinguishes landslide samples with masks from non-landslide samples. Do not create masks for unmatched files by assumption. Inspect the extracted files, audit pairs, and decide which matched subset is appropriate for the segmentation experiment. Cite Ji et al. (2020) and follow the provider's attribution requirements.

## 5. Pipeline and evaluation

1. Inspect the dataset directories and run `audit`.
2. Pair images and masks by filename stem. Duplicate stems are errors; unmatched names are reported and training stops rather than silently pairing by sorted index.
3. Create and save a seeded train/validation/test partition. Reuse the same split file for model comparisons.
4. Apply stochastic flips only when the training dataset is constructed with augmentation enabled. Validation/test transforms are deterministic. Images use bilinear resizing; categorical masks use nearest-neighbor resizing before binary thresholding.
5. Select checkpoints and stop training using validation loss only. Do not use the test partition for model selection or threshold tuning.
6. Evaluate once on the held-out test partition after the experiment is selected.

The default probability threshold is 0.5. Macro metrics calculate the metric for each image and average those image scores. Global metrics aggregate pixel counts across the full partition before calculating the score. Empty prediction and empty target score 1 for IoU/Dice; precision or recall with a zero denominator is defined as 1. These conventions are documented in `src/terraseg/metrics.py`.

## 6. Installation and setup

Python 3.10–3.12 is the documented target. Create a virtual environment in Windows PowerShell and install a PyTorch/torchvision build selected for your hardware from the [official PyTorch installer](https://pytorch.org/get-started/locally/). Run the selector's generated command; CPU and CUDA commands differ.

```powershell
cd "PATH\TO\terraseg-ai"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install torch and torchvision using the command from pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell window, then activate again. Git Bash uses Unix-style paths and activation: `source .venv/Scripts/activate`; the Python commands are otherwise the same.

### Google Colab

Select a GPU runtime if one is available, upload or clone this project into the runtime, and install dependencies. Colab generally provides a compatible PyTorch build; do not replace it with a CUDA wheel unless needed.

```python
!python -m pip install -r requirements.txt
!python -m pip install -e ".[test]"
```

If data is in Google Drive, mount it and pass the actual extracted image and mask paths to the commands below. GPU is recommended for training, particularly ResNet-34; audit, tests, and the synthetic smoke command can run on CPU. Training/evaluation can run on CPU too, but may be slow.

## 7. Dataset configuration and audit

Keep the project, virtual environment, local dataset folders, and results under the C-drive project folder. The `data` directory is ignored by Git. The code does not assume the archive's folder layout, so inspect the extracted archive first. To make project-local destinations, run this from the project root:

```powershell
New-Item -ItemType Directory -Force -Path .\data\bijie\raw, .\data\bijie\images, .\data\bijie\masks | Out-Null
Get-ChildItem .\data\bijie -Recurse -Directory
```

Download the dataset from the [upstream source page](https://gpcv.whu.edu.cn/data/Bijie_pages.html) and extract it under `data\bijie\raw`. Inspect its real structure, then copy the corresponding image and mask files directly into the `images` and `masks` folders above (not into nested subfolders), preserving matching filename stems. Do not invent masks for unmatched files. Or, if the archive already has clean matching folders, set `$images` and `$masks` to those actual folder paths instead. Then run:

```powershell
$images = ".\data\bijie\images"
$masks = ".\data\bijie\masks"
if (!(Test-Path -LiteralPath $images) -or !(Test-Path -LiteralPath $masks)) { throw "Create the image and mask folders above first." }
Test-Path -LiteralPath $images
Test-Path -LiteralPath $masks
Get-ChildItem -LiteralPath $images -File | Select-Object -First 5 Name
Get-ChildItem -LiteralPath $masks -File | Select-Object -First 5 Name
python -m terraseg audit --images $images --masks $masks
```

Supported file extensions are PNG, JPG/JPEG, and TIFF. The audit prints counts and unmatched/duplicate names; it exits with an error status if the pairs are not clean or no pairs are found. Resolve discrepancies before training. Files under `data` are ignored by Git and will stay local.

## 8. Train, smoke-test, and evaluate

Run from the project root after installation. The first training run writes `results/splits.json`; all comparisons should reuse this file. Choose the config that matches the experiment:

```powershell
# Synthetic model-shape smoke check; no dataset needed
python -m terraseg smoke --device cpu

# Set these once per PowerShell window. Change them only if your actual folders differ.
$images = ".\data\bijie\images"
$masks = ".\data\bijie\masks"
if (!(Test-Path -LiteralPath $images) -or !(Test-Path -LiteralPath $masks)) { throw "Create the image and mask folders in the dataset configuration section first." }

# Initial baseline settings
python -m terraseg train --images $images --masks $masks --model unet --config configs/default.json --seed 42

# Advanced U-Net and pretrained ResNet-34 U-Net presets
python -m terraseg train --images $images --masks $masks --model unet --config configs/advanced_unet.json --seed 42
python -m terraseg train --images $images --masks $masks --model resnet34_unet --config configs/resnet34.json --seed 42

# Evaluate the selected checkpoint on the held-out test partition
python -m terraseg evaluate --images $images --masks $masks --model unet --checkpoint results/unet/advanced_unet_seed42/best.pt --splits results/splits.json --partition test
```

Use `--model custom_cnn` to train the custom CNN. A run is written to `results/<model>/<config>_seed<seed>/`; an existing checkpoint is not overwritten. Pass a distinct `--output` when repeating the same config/seed. Training writes `config.json`, `epochs.json`, `best.pt`, `validation_metrics.json`, and a validation plot. Evaluation writes `metrics.json` and a plot. Run logs include per-epoch loss/metrics and elapsed times. Do not tune thresholds or choose models from test results.

If using Git Bash, the equivalent project-local paths are `./data/bijie/images` and `./data/bijie/masks`.

## 9. Tests and reproducibility

Install test dependencies using the setup commands, then run:

```powershell
python -m pytest -q tests
```

Tests create tiny synthetic images/masks and cover matching, integrity errors, shapes/ranges, augmentation, deterministic evaluation preprocessing, metrics and edge cases, model output dimensions, and seeded splits. No Bijie download or full training is required. The CPU smoke check passed. In this restricted Windows environment, six tests passed but seven tests could not create pytest temporary folders (Windows access denied); the full suite remains unverified. Run `python -m pytest -q tests` in your project environment.

Seeds, split IDs, configurations, epoch records, and metrics are saved. Exact bitwise reproducibility can vary by hardware, software versions, and GPU kernels. The saved test partition must remain out of model selection and threshold tuning.

## 10. Results

| Result source | Status |
|---|---|
| Original supplied Python script | No numeric final metric is included in the supplied source file. |
| Original project presentation | Approximately 78.01% IoU was reported; this is presentation-reported and has not been independently reproduced. |
| Reproduced after refactoring | Not run; no dataset/weights or compatible local ML dependencies were available. |
| Corrected evaluation protocol | Not run; no validation result is available. |
| Final held-out test | Not available; no test evaluation has been run. |

Do not describe the presentation figure as a reproduced result or compare it directly with a new score until the original split and metric aggregation are established.

## 11. Visual predictions

No verified prediction image is available yet. Training/evaluation commands save plots in their run output directory once run on the dataset.

## 12. Limitations

- No reproduced score or held-out test score is available.
- Dataset license/reuse rights need confirmation from the provider.
- A random image split may not prevent geographic or temporal leakage.
- The exact source data layout and mask coverage need to be audited before training.
- ImageNet normalization, explicit mask interpolation, changed partitions, loss aggregation, and corrected metrics can change performance relative to the original prototype.
- There is no deployment, field validation, or evidence of production readiness.

## 13. Planned work

Verify the dataset terms and sample layout, run the tests and audit, establish fixed splits, compare the three architectures under documented configurations, and report final held-out test metrics. Investigate spatially grouped splits. DEM/slope/SAR fusion is future work only; it is not implemented.

## 14. Citation and attribution

The upstream dataset page requests attribution to the Bijie dataset work:

Ji, S. et al. (2020). “Landslide detection from an open satellite imagery and digital elevation model dataset using attention boosted convolutional neural networks.” *Landslides*.

Dataset page: [Wuhan University GPCV — Bijie Landslide Dataset](https://gpcv.whu.edu.cn/data/Bijie_pages.html). Check the source page for complete bibliographic details and current data-use terms before publication.

The code is based on the supplied project script. Document your own contributions accurately and credit teammates for work they completed.

## 15. License

No project license has been selected. `LICENSE` is an explicit placeholder and grants no reuse rights. Choose a license you have the authority to apply before publishing the repository. The dataset may have separate terms regardless of the code license.

## 16. Publish to GitHub

The supplied GitHub repository has now been populated with this project. To publish a separate copy under another account, replace `<YOUR_GITHUB_USERNAME>` below with that account's username.

1. The supplied destination already exists; skip repository creation when updating it. For a different account, sign in to GitHub and choose **New repository**.
2. Name a new repository `terraseg-ai`. Leave “Add a README,” `.gitignore`, and license unchecked because those files already exist locally. Choose visibility intentionally.
3. From PowerShell, switch to the project directory and review the local commit and working tree:

   ```powershell
   cd "PATH\TO\terraseg-ai"
   git status --short
   git status --ignored --short
   git ls-files --others --exclude-standard
   git diff --cached
   ```

4. Confirm no dataset, checkpoint, credential, or unrelated file is staged. The supplied local project already has its commits. After making later changes, stage and commit only the reviewed files:

   ```powershell
   git add .gitignore LICENSE README.md pyproject.toml requirements.txt configs docs src tests
   git status --short
   git diff --cached --stat
   git diff --cached
   git commit -m "Describe your change"
   ```

5. For the supplied project, `origin` already points to its GitHub repository. For a new destination, replace `<YOUR_GITHUB_USERNAME>` and add or update the remote, then push `main`:

   ```powershell
   # Run one of these: add for a new origin, or set-url for an existing origin.
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/terraseg-ai.git
   # git remote set-url origin https://github.com/<YOUR_GITHUB_USERNAME>/terraseg-ai.git
   git remote -v
   git push -u origin main
   ```

6. Visit `https://github.com/<YOUR_GITHUB_USERNAME>/terraseg-ai` and verify the README and files. The supplied destination is already populated; copy its verified repository URL for your resume.

Git Bash uses the same Git commands, but path syntax differs (for example, `/c/path/to/terraseg-ai`). Review the staged files before pushing. `LICENSE` is currently a placeholder, so the repository grants no reuse rights; choose a license before inviting others to reuse the code. The dataset may have separate terms.
