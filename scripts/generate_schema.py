"""
generate_schema.py
------------------
Introspects the installed Python libraries and generates web/src/assets/schema.json.
Run during GitHub Actions build before `npm run build`.

Usage:
    python scripts/generate_schema.py
"""

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "web" / "src" / "assets" / "schema.json"


# ── segmentation_models_pytorch ───────────────────────────────────────────────


def extract_smp():
    try:
        import segmentation_models_pytorch as smp
        from segmentation_models_pytorch.encoders import encoders as _encoders

        architectures = [
            name
            for name in dir(smp)
            if inspect.isclass(getattr(smp, name, None))
            and issubclass(getattr(smp, name), smp.base.SegmentationModel)
        ]

        encoder_info = {}
        for name, enc in _encoders.items():
            pretrained = list(enc.get("pretrained_settings", {}).keys())
            encoder_info[name] = {"pretrained_settings": pretrained}

        return {
            "version": smp.__version__,
            "architectures": sorted(architectures),
            "encoders": encoder_info,
        }
    except ImportError as e:
        print(f"[WARN] segmentation_models_pytorch not available: {e}", file=sys.stderr)
        return None


# ── albumentations ────────────────────────────────────────────────────────────

# Transforms que funcionam tanto na imagem quanto na máscara (DualTransform).
# Excluímos transforms abstratos, internos e os que requerem targets especiais.
_ALBUMENTATIONS_EXCLUDE = {
    "BasicTransform",
    "DualTransform",
    "ImageOnlyTransform",
    "NoOp",
    "Compose",
    "ReplayCompose",
    "OneOf",
    "SomeOf",
    "Sequential",
    "BaseCompose",
    "SelectiveChannelTransform",
}


def _get_param_default(param):
    if param.default is inspect.Parameter.empty:
        return None
    val = param.default
    # Garante que o valor é serializável em JSON
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return (
            list(val)
            if all(isinstance(x, (int, float, str, bool)) for x in val)
            else None
        )
    return None


def extract_albumentations():
    try:
        import albumentations as A

        result = {}
        for name in sorted(dir(A)):
            if name.startswith("_") or name in _ALBUMENTATIONS_EXCLUDE:
                continue
            cls = getattr(A, name, None)
            if not (inspect.isclass(cls) and issubclass(cls, A.BasicTransform)):
                continue
            if cls in (A.BasicTransform, A.DualTransform, A.ImageOnlyTransform):
                continue

            try:
                sig = inspect.signature(cls.__init__)
            except (ValueError, TypeError):
                continue

            params = {}
            for k, v in sig.parameters.items():
                if k in ("self", "always_apply", "p", "args", "kwargs"):
                    continue
                params[k] = {"default": _get_param_default(v)}

            result[name] = {
                "module": f"albumentations.{name}",
                "params": params,
            }

        return {"version": A.__version__, "transforms": result}
    except ImportError as e:
        print(f"[WARN] albumentations not available: {e}", file=sys.stderr)
        return None


# ── torchmetrics ──────────────────────────────────────────────────────────────

_TORCHMETRICS_SEGMENTATION = [
    "Accuracy",
    "JaccardIndex",
    "F1Score",
    "Precision",
    "Recall",
    "Specificity",
    "AUROC",
    "AveragePrecision",
    "CohenKappa",
    "MatthewsCorrCoef",
    "ConfusionMatrix",
]


def extract_torchmetrics():
    try:
        import torchmetrics

        available = [m for m in _TORCHMETRICS_SEGMENTATION if hasattr(torchmetrics, m)]
        return {"version": torchmetrics.__version__, "metrics": available}
    except ImportError as e:
        print(f"[WARN] torchmetrics not available: {e}", file=sys.stderr)
        return None


# ── torch optimizers & schedulers ────────────────────────────────────────────


def extract_torch():
    try:
        import torch
        import torch.optim as optim
        import torch.optim.lr_scheduler as sched

        optimizers = sorted(
            [
                name
                for name in dir(optim)
                if inspect.isclass(getattr(optim, name, None))
                and issubclass(getattr(optim, name), optim.Optimizer)
                and name != "Optimizer"
            ]
        )

        schedulers = sorted(
            [
                name
                for name in dir(sched)
                if inspect.isclass(getattr(sched, name, None))
                and issubclass(getattr(sched, name), sched.LRScheduler)
                and not name.startswith("_")
                and name not in ("LRScheduler", "_LRScheduler")
            ]
        )

        return {
            "version": torch.__version__,
            "optimizers": optimizers,
            "schedulers": schedulers,
        }
    except ImportError as e:
        print(f"[WARN] torch not available: {e}", file=sys.stderr)
        return None


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("Generating schema.json...")

    smp_data = extract_smp()
    albu_data = extract_albumentations()
    torchm_data = extract_torchmetrics()
    torch_data = extract_torch()

    schema = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "library_versions": {
            "segmentation_models_pytorch": smp_data["version"] if smp_data else None,
            "albumentations": albu_data["version"] if albu_data else None,
            "torchmetrics": torchm_data["version"] if torchm_data else None,
            "torch": torch_data["version"] if torch_data else None,
        },
        "smp": smp_data or {},
        "albumentations": albu_data or {},
        "torchmetrics": torchm_data or {},
        "torch": torch_data or {},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False))

    libs_ok = sum(1 for v in [smp_data, albu_data, torchm_data, torch_data] if v)
    print(f"Done: {OUTPUT_PATH}  ({libs_ok}/4 libraries available)")

    if libs_ok == 0:
        print(
            "[ERROR] No libraries found. Install requirements first.", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
