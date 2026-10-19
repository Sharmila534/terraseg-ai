# TerraSeg AI — resume evidence

## Truthful two-line description

Organized a PyTorch semantic-segmentation prototype for landslide boundaries from RGB satellite imagery into a reproducible local project.<br>
The project retains custom CNN, U-Net, and ResNet-34 U-Net approaches and adds auditable pairing, saved data partitions, metrics, and command-line workflows.

## Resume bullet options

- Refactored a Colab-based landslide segmentation prototype into a modular PyTorch project with custom CNN, U-Net, and ResNet-34 U-Net model options.
- Implemented filename-stem image/mask matching, deterministic preprocessing, configurable seeded data splits, and training/validation/test command-line workflows.
- Added binary segmentation metrics for per-image and global IoU, Dice, precision, and recall, plus synthetic tests for data integrity and model output shapes.

Use only the bullets that match your personal contribution. Replace “refactored” or describe collaboration accurately if teammates completed part of the work.

## Supported ATS keywords

Python, PyTorch, semantic segmentation, computer vision, U-Net, transfer learning (ResNet-34/ImageNet encoder option), image preprocessing, binary masks, model evaluation, reproducibility, dataset auditing.

## Evidence for claims

- Architectures: `src/terraseg/models.py`.
- Dataset matching and preprocessing: `src/terraseg/data.py`.
- Metrics and training loop: `src/terraseg/metrics.py`, `src/terraseg/engine.py`, `src/terraseg/cli.py`.
- Tests: `tests/test_core.py` covers the listed synthetic cases. A targeted local run passed 12 tests. The full run had seven setup errors because Windows denied access to pytest temporary folders; the full suite is not verified yet. The CPU synthetic model-shape smoke check did pass.

## Claims not supported yet

- 78.01% reproduced IoU, improved performance, or a held-out test score.
- Deployment, production readiness, state-of-the-art results, publication, or patent.
- DEM/slope/SAR fusion, a verified dataset license, dataset size, or independent ownership of teammates' work.

## GitHub publishing guide (PowerShell)

1. Review files and the license status. The current `LICENSE` is a placeholder; choose a license before inviting others to reuse the project.
2. The supplied destination repository already contains this project. For another account, create an **empty** repository named `terraseg-ai` (do not initialize it with a README, since one already exists locally). Choose public visibility only if you want the resume link to be publicly viewable.
3. In PowerShell, `cd` to this project directory and run `git status --short` and `git status --ignored --short`. A local Git repository has already been initialized here; do not run `git init` again unless you intentionally start over.
4. Inspect the candidate file list with `git ls-files --others --exclude-standard`. Confirm no data, weights, credentials, or unrelated files are present.
5. If you are following this guide in a different copy without Git metadata, run `git init -b main`. Stage the intended project files, including `pyproject.toml`: `git add README.md LICENSE .gitignore requirements.txt pyproject.toml configs src tests docs`.
6. Review `git status`, then `git diff --cached --stat` and `git diff --cached`. For a new uncommitted copy, create the initial commit with `git commit -m "Initial TerraSeg AI project"`.
7. For the supplied repository, `origin` is already configured. For a different account, replace `<YOUR_GITHUB_USERNAME>` and use these commands (use `git remote set-url` if `origin` already exists):

   ```powershell
   git remote add origin "https://github.com/<YOUR_GITHUB_USERNAME>/terraseg-ai.git"
   # If origin already exists, run this instead:
   # git remote set-url origin "https://github.com/<YOUR_GITHUB_USERNAME>/terraseg-ai.git"
   git remote -v
   git push -u origin main
   ```

8. Open `https://github.com/<YOUR_GITHUB_USERNAME>/terraseg-ai`, verify the README and tracked files, and copy that repository address for your resume. The supplied destination is already populated and verified.

Git Bash uses the same Git commands; use `/c/path/to/...` path syntax there. Review the staged files before pushing. The `LICENSE` file remains a placeholder and grants no reuse rights; select a project license before inviting reuse. Dataset terms are separate.
