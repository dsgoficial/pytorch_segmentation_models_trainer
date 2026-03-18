# Run flake8

Run flake8 linting on the project, excluding venv and build artifacts.

## Usage

```
/flake8 [args]
```

Where `[args]` is an optional path or extra flags:
- `/flake8` — runs full lint check (errors + warnings)
- `/flake8 --errors-only` — runs only fatal errors (E9, F63, F7, F82)
- `/flake8 pytorch_segmentation_models_trainer/custom_losses/` — runs on a specific path

## Instructions

Run flake8 in two passes using the project's venv at `.venv/`.

**Pass 1 — fatal errors only (same as CI):**
```bash
cd /Users/philipeborba/github_repos/pytorch_segmentation_models_trainer && .venv/bin/python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,build 2>&1
```

**Pass 2 — all warnings (exit-zero so it never blocks):**
```bash
cd /Users/philipeborba/github_repos/pytorch_segmentation_models_trainer && .venv/bin/python -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=.venv,build 2>&1
```

If `$ARGUMENTS` contains `--errors-only`, run only Pass 1.

After running, report:
1. Number of fatal errors (Pass 1). If 0, confirm CI will pass.
2. Top warning categories from Pass 2 (if run), summarised by error code.
3. Any files with a high concentration of issues worth fixing.
