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

If data is in Google Drive, mount it and pass the actual extracted image and mask paths to the commands below. GPU is recommended for training, particularly ResNet-34; audit, tests, and the synthetic smoke command can run on CPU. Training/evaluation can run on CPU too, but may be slow. On some Windows CPU-only installations, importing PyTorch takes tens of seconds; the CLI prints a startup message while loading it. A 256×256 U-Net run over the full dataset can take many minutes per epoch on a CPU, so use a compatible CUDA build or Colab GPU for practical training.

## 7. Dataset configuration and audit

Keep the project, virtual environment, local dataset folders, and results under the C-drive project folder. The `data` directory is ignored by Git. The code does not assume the archive's folder layout, so inspect the extracted archive first. To make project-local destinations, run this from the project root:

```powershell
New-Item -ItemType Directory -Force -Path .\data\bijie\raw, .\data\bijie\images, .\data\bijie\masks | Out-Null
Get-ChildItem .\data\bijie -Recurse -Directory
```

Download the dataset from the [upstream source page](https://gpcv.whu.edu.cn/data/Bijie_pages.html). In the current Windows workspace the downloaded archive is `data\bijie dataset` (it has no `.zip` suffix). Extract it with the project virtual environment's Python:

```powershell
python -c "import zipfile; z=zipfile.ZipFile(r'.\data\bijie dataset'); z.extractall(r'.\data\bijie\raw')"
```

The archive extracts to `data\bijie\raw\Bijie-landslide-dataset`. It has 770 landslide images and corresponding masks; the 2,003 non-landslide images have no masks and are not part of this paired binary-segmentation run. Use the actual extracted paths (do not use the empty `data\bijie\images` and `data\bijie\masks` placeholders):

```powershell
$images = ".\data\bijie\raw\Bijie-landslide-dataset\landslide\image"
$masks = ".\data\bijie\raw\Bijie-landslide-dataset\landslide\mask"
Get-ChildItem -LiteralPath $images -File | Select-Object -First 5 Name
Get-ChildItem -LiteralPath $masks -File | Select-Object -First 5 Name
python -m terraseg audit --images $images --masks $masks
```

The verified archive yields 770 pairs with no unmatched or duplicate names. Supported file extensions are PNG, JPG/JPEG, and TIFF. Stop if your audit reports zero pairs or unmatched/duplicate files. Files under `data` are ignored by Git and stay local.

## 8. Train, smoke-test, and evaluate

Run these commands in Windows PowerShell from the current project folder. The dataset is already extracted locally; do not use the empty `data\bijie\images` and `data\bijie\masks` placeholders. If the project virtual environment is set up, activate it and set the real data paths:

```powershell
cd "C:\Users\saish\Documents\New project\terraseg-ai"
.\.venv\Scripts\Activate.ps1
$images = ".\data\bijie\raw\Bijie-landslide-dataset\landslide\image"
$masks = ".\data\bijie\raw\Bijie-landslide-dataset\landslide\mask"
```

Audit the dataset and run the synthetic model-shape smoke check:

```powershell
python -m terraseg audit --images $images --masks $masks
python -m terraseg smoke --device cpu
```

The audit should report 770 matched pairs and zero unmatched or duplicate files. The original code reports validation IoU for four runs. To see the corresponding results here, run these commands **one at a time** in the VS Code PowerShell terminal, keeping the terminal open until each completes. The first PyTorch import can take tens of seconds; full-data CPU training can take many minutes per epoch. This environment has CPU-only PyTorch.

```powershell
# Original Model 1: CNN
python -u -m terraseg train --images $images --masks $masks --model custom_cnn --config configs/default.json --seed 42 --device cpu

# Original Model 2: baseline U-Net
python -u -m terraseg train --images $images --masks $masks --model unet --config configs/default.json --seed 42 --device cpu

# Advanced U-Net
python -u -m terraseg train --images $images --masks $masks --model unet --config configs/advanced_unet.json --seed 42 --device cpu

# ResNet-34 U-Net; first use may need to download pretrained ImageNet weights
python -u -m terraseg train --images $images --masks $masks --model resnet34_unet --config configs/resnet34.json --seed 42 --device cpu
```

Each epoch prints a JSON line with train and validation metrics. After a run finishes, the terminal prints a `Final Validation IoU (global)` percentage and a per-image mean IoU percentage, similar to the final IoU lines in the original code. The machine-readable validation metrics are saved at `results\<model>\<config>_seed42\validation_metrics.json`; checkpoints are `best.pt` in the same run folder. All four runs use the same `results\splits.json` partition (539 train, 115 validation, 116 test).

The original code reports one batch-averaged validation IoU. This project reports global pixel IoU and mean per-image IoU separately, and uses deterministic validation data, so the numbers will not be exactly identical. Compare models using validation results only. After selecting one model, evaluate it once on the held-out test partition. For example, to evaluate the baseline U-Net after its training has completed and `best.pt` exists:

```powershell
python -u -m terraseg evaluate --images $images --masks $masks --model unet --checkpoint results\unet\default_seed42\best.pt --splits results\splits.json --partition test
```

The test IoU is printed as a percentage; full metrics are printed as JSON and saved at `results\evaluation_unet_test\metrics.json`. Do not tune thresholds or choose models based on test results. Use a new `--output` path when repeating a completed run so its checkpoint is preserved. Run tests with `python -m pytest -q tests`.

For Git Bash, activate using `source .venv/Scripts/activate`; the image and mask paths become `./data/bijie/raw/Bijie-landslide-dataset/landslide/image` and `./data/bijie/raw/Bijie-landslide-dataset/landslide/mask`.

## 9. Tests and reproducibility

Install test dependencies using the setup commands, then run:

```powershell
python -m pytest -q tests
```

Tests create tiny synthetic images/masks and cover matching, integrity errors, shapes/ranges, augmentation, deterministic evaluation preprocessing, metrics and edge cases, model output dimensions, and seeded splits. No Bijie download or full training is required. The CPU smoke check passed. A targeted run passed 12 tests. The full run had 12 passed and 7 test setup errors because Windows denied access to pytest temporary folders; the full suite remains unverified in this environment. Run `python -m pytest -q tests` in your project environment. Training presets cap PyTorch CPU threads at four by default; change `num_threads` in the config if your CPU benefits from a different setting. Training requires at least three clean matched pairs so every partition has a sample; with tiny datasets, split sizes are adjusted to preserve non-empty train, validation, and test partitions.

Seeds, split IDs, configurations, epoch records, and metrics are saved. Exact bitwise reproducibility can vary by hardware, software versions, and GPU kernels. The saved test partition must remain out of model selection and threshold tuning.

## 10. Results

| Result source | Status |
|---|---|
| Original supplied Python script | No numeric final metric is included in the supplied source file. |
| Original project presentation | Approximately 78.01% IoU was reported; this is presentation-reported and has not been independently reproduced. |
| Reproduced after refactoring | Dataset extracted and 770 pairs audited. CPU-only training was started but stopped before the first epoch completed; no trained result is available yet. |
| Corrected evaluation protocol | Not run; training has not produced a checkpoint. |
| Final held-out test | Not available; no test evaluation has been run. |

The archived Colab source pairs independently sorted image and mask lists by position, applies random flips through a dataset also used for validation, creates only a train/validation split, and averages batch-level IoU values. The corrected project pairs by filename, confines augmentation to training, reserves a test split, and reports per-image and global metrics. These protocol changes mean the presentation figure is not directly comparable to a corrected score. Do not describe it as reproduced or promise the same or higher IoU until the original data, partition, and aggregation are established and both runs are controlled.

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
