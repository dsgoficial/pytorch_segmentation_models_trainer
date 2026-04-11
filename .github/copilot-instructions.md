## Quick orientation

This repository is a PyTorch + PyTorch Lightning training framework that uses Hydra for configuration composition.
Primary entry points:
- CLI console script: `pytorch-smt` (defined in `setup.py` -> `pytorch_segmentation_models_trainer.main:entry`).
- Module entry: `pytorch_segmentation_models_trainer.main` (see `main.py`).

Key directories/files to inspect when changing behavior:
- `pytorch_segmentation_models_trainer/train.py` — training orchestration (builds model, trainer, callbacks).
- `pytorch_segmentation_models_trainer/main.py` — mode switcher for train/predict/convert/evaluate.
- `pytorch_segmentation_models_trainer/model_loader/` — model factory and Lightning wrappers (see `model.py`, `frame_field_model.py`).
- `pytorch_segmentation_models_trainer/dataset_loader/` — dataset and dataloader construction.
- `conf/` — Hydra configuration files (default config used by the CLI).
- `config_definitions/` — typed config fragments and examples.

How configuration is used (important patterns)
- Hydra is the primary config system. Typical invocation:
  `pytorch-smt --config-dir /path/to/config --config-name mycfg +mode=train`
  The code often uses `instantiate(cfg.xxx, _recursive_=False)` or `import_module_from_cfg(cfg.pl_model)`.
- Trainer, logger, dataset, model, optimizer, loss and callbacks are supplied via Hydra config entries. Inspect `conf/` and `config_definitions/` when changing runtime wiring.
- To resume training, include `hyperparameters.resume_from_checkpoint` in the config; `train.py` will call `load_from_checkpoint`.

Developer workflows and commands
- Run locally (editable install):
  1) pip install -e .
  2) Run the CLI: `pytorch-smt --config-dir path/to/conf --config-name my.yaml +mode=train`
- Run tests: `pytest -q` (tests live in `tests/`).
- Build/publish (project uses `setup.py`): building/upload helpers exist in `setup.py` (`python setup.py upload`). For local releases prefer `pip wheel` or `twine` as usual.
- Docker image: a published image (`phborba/pytorch_segmentation_models_trainer`) exists and is mentioned in `README.md`.

Project-specific conventions (do not assume defaults)
- Hydra config composition is heavily used — configs frequently refer to `_target_` keys to instantiate classes. When editing code that is constructed by config, check `config_definitions/*` and `conf/` for expected keys and shapes.
- Callbacks are listed in config and instantiated dynamically; adding a callback typically requires adding it to the config rather than hard-coding in `train.py`.
- The framework supports multiple modes (train, predict, convert-dataset, evaluate-experiments). Add new modes by extending `main.py`'s mode dispatcher.
- **Dataset splits**: three dataset keys are supported. `train_dataset` feeds `training_step` (gradient updates); `val_dataset` feeds `validation_step` (called at the end of every epoch during `trainer.fit()`, used for early stopping and LR scheduling); `test_dataset` feeds `test_step` (called once via `trainer.test()` after `trainer.fit()`, used for final held-out evaluation). All three are optional — absent splits return `None` from their respective dataloader methods and are silently skipped by Lightning. Metrics are logged with `train/`, `val/`, and `test/` prefixes respectively.

Important integration points & dependencies
- PyTorch Lightning Trainer is created with `cfg.pl_trainer` in `train.py` — changes to Lightning arguments should be reflected in the config (`conf/` files).
- Models are built through `model_loader` and may be either plain `Model(cfg)` or a Lightning module referenced by `pl_model` in the config.
- Data access and geospatial libs: the project depends on geopandas, rasterio, fiona, shapely — be careful when running tests on CI if these are not available or require system libs.

Files to open first when debugging
- `main.py` — to see mode routing and Hydra config versioning.
- `train.py` — to understand model instantiation, callback handling, and Trainer creation.
- `model_loader/model.py` and `dataset_loader/*` — where model and data pipelines live.
- `conf/config.yaml` (and files under `conf/`) — canonical runtime config.

If you change public API or behavior
- Add/adjust a config in `conf/` and a corresponding definition in `config_definitions/` if the new option must be validated or typed.
- Update tests in `tests/` (many focused unit tests exist — run them locally).

Examples (copy-paste friendly)
- Train with local config folder `configs/` and `train.yaml`:
  `pytorch-smt --config-dir configs --config-name train +mode=train`
- Resume from checkpoint by adding to your config:
  ```yaml
  hyperparameters:
    resume_from_checkpoint: /path/to/checkpoint.ckpt
  ```

Questions or missing pieces? Ask me to add more examples (e.g., sample config snippets, common debug patterns, or CI details).
