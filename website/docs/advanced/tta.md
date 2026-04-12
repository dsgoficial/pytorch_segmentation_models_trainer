---
sidebar_position: 3
title: Test Time Augmentation (TTA)
---

# Test Time Augmentation (TTA)

Test Time Augmentation (TTA) melhora a qualidade das predições sem retraining: o modelo é executado múltiplas vezes sobre versões transformadas do mesmo input, os resultados são desfeitos de volta à orientação original e em seguida são calculada sua média.

---

## Como funciona

Para cada augmentação configurada, o pipeline executa os seguintes passos:

```
entrada → augmentação → modelo → augmentação inversa → predição de-augmentada
                                                              ↓
                                                 média de todas as predições
                                                              ↓
                                               resultado final consolidado
```

Na inferência por **janela deslizante**, o TTA é aplicado por tile — a média já consolidada é entregue ao `TileMerger`, que então faz a junção espacial normal. O código da janela deslizante em si não é alterado.

No **test step** do modelo (avaliação com `trainer.test()`), o TTA é aplicado a cada batch completo de imagens.

---

## Augmentações disponíveis

As oito augmentações formam o **grupo diedral D4** — todas as simetrias de um quadrado. Cada uma possui uma inversa exata, garantindo que a de-augmentação não introduza artefatos.

| Nome no config   | Transformação                                  | Inversa              |
|------------------|------------------------------------------------|----------------------|
| `rot0`           | Identidade (imagem original)                   | `rot0`               |
| `rot90`          | Rotação 90° no sentido anti-horário            | `rot270`             |
| `rot180`         | Rotação 180°                                   | `rot180`             |
| `rot270`         | Rotação 270° no sentido anti-horário           | `rot90`              |
| `flip_h`         | Espelhamento horizontal (esquerda-direita)     | `flip_h`             |
| `flip_v`         | Espelhamento vertical (cima-baixo)             | `flip_v`             |
| `flip_h_rot90`   | Espelhamento horizontal + rotação 90° anti-horária | `rot270` + `flip_h` |
| `flip_v_rot90`   | Espelhamento vertical + rotação 90° anti-horária   | `rot270` + `flip_v` |

:::tip Presets recomendados
- **Rotações apenas (padrão):** `rot0`, `rot90`, `rot180`, `rot270` — 4× forward passes, excelente custo-benefício.
- **D4 completo:** todos os 8 — 8× forward passes, máxima cobertura de simetrias.
- **Mínimo:** `rot0`, `rot180` — 2× forward passes, útil para ortofotos sem orientação predominante.
:::

---

## Usando TTA na inferência (`predict`)

Adicione `use_tta` e `tta_augmentations` ao bloco `inference_processor` do seu YAML de predict:

```yaml title="configs/predict_with_tta.yaml"
checkpoint_path: /checkpoints/unet_best.ckpt
device: cuda:0

pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model

hyperparameters:
  batch_size: 8

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /data/test_images/
  image_extension: tif

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  # ── TTA ──────────────────────────────────────────────────────────────────
  use_tta: true
  tta_augmentations:
    - rot0          # imagem original
    - rot90         # 90° anti-horário
    - rot180        # 180°
    - rot270        # 270° anti-horário
    - flip_h        # espelhamento horizontal
    - flip_v        # espelhamento vertical
    - flip_h_rot90  # espelhamento horizontal + 90° anti-horário
    - flip_v_rot90  # espelhamento vertical + 90° anti-horário
  # ─────────────────────────────────────────────────────────────────────────

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: /output/prediction_tta.tif

inference_threshold: 0.5
```

O mesmo parâmetro funciona para `MultiClassInferenceProcessor`:

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor
  model_input_shape: [512, 512]
  step_shape: [256, 256]
  num_classes: 5
  use_tta: true
  tta_augmentations:
    - rot0
    - rot90
    - rot180
    - rot270
```

---

## Usando TTA no test step (`trainer.test()`)

Para ativar o TTA durante a avaliação do modelo, adicione `use_tta` diretamente no config de treino/avaliação (mesma raiz que o modelo):

```yaml title="configs/train_with_tta_eval.yaml"
# ... demais configurações de treino ...

# ── TTA para o test_step ────────────────────────────────────────────────────
use_tta: true
tta_augmentations:
  - rot0          # imagem original
  - rot90         # 90° anti-horário
  - rot180        # 180°
  - rot270        # 270° anti-horário
```

Quando `use_tta: true`, o método `test_step` aplica automaticamente as augmentações ao batch, faz a média das predições de-augmentadas e usa o resultado no cálculo de perda e métricas.

:::note
O TTA no test step **não afeta** o treino (`training_step`) nem a validação (`validation_step`) — é ativado exclusivamente durante `trainer.test()`.
:::

---

## TTA com modelos Frame Field (`SingleImageFromFrameFieldProcessor`)

O processador frame field produz dois tensores de saída: `seg` (máscara de segmentação) e `crossfield` (campo vetorial de tangentes).

O `crossfield` codifica ângulos tangenciais. Corrigir os valores de ângulo corretamente durante a de-augmentação exigiria transformações no espaço de Fourier/complexos, o que vai além do escopo da implementação atual.

**Comportamento com TTA ativado:**

| Saída        | Tratamento com TTA                                               |
|--------------|------------------------------------------------------------------|
| `seg`        | De-augmentado e médio sobre todas as augmentações               |
| `crossfield` | Tomado diretamente da passagem de identidade (`rot0`); se `rot0` não estiver na lista, usa a primeira augmentação |

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageFromFrameFieldProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  mask_bands: 1
  use_tta: true
  tta_augmentations:
    - rot0      # necessário para crossfield correto
    - rot90
    - rot180
    - rot270
```

:::caution
Para `SingleImageFromFrameFieldProcessor` inclua sempre `rot0` na lista de augmentações quando `use_tta: true`, para garantir que o `crossfield` seja tomado da passagem sem transformação.
:::

---

## Custo computacional

O TTA aumenta o número de forward passes proporcionalmente ao tamanho da lista de augmentações. Com `batch_size` e `step_shape` equivalentes:

| Augmentações | Forward passes | Custo relativo |
|---|---|---|
| Nenhum (sem TTA) | 1× | 1× |
| 4 rotações (`ROTATION_AUGMENTATIONS`) | 4× | ~4× |
| D4 completo (8 augmentações) | 8× | ~8× |

O custo de memória por inferência não aumenta significativamente — as augmentações são aplicadas e acumuladas em sequência por batch.

---

## API Python

```python
from pytorch_segmentation_models_trainer.tools.tta.tta import (
    apply_tta,
    ROTATION_AUGMENTATIONS,
    D4_AUGMENTATIONS,
    ROT0, ROT90, ROT180, ROT270,
    FLIP_H, FLIP_V, FLIP_H_ROT90, FLIP_V_ROT90,
)

# Uso direto com qualquer callable
pred = apply_tta(
    model_fn=model,
    batch=tiles_batch,          # torch.Tensor [B, C, H, W]
    augmentations=["rot0", "rot90", "rot180", "rot270"],
)

# Com skip_keys (para modelos com saída dict onde alguns tensores não devem
# ser de-augmentados):
pred = apply_tta(
    model_fn=model,
    batch=tiles_batch,
    augmentations=ROTATION_AUGMENTATIONS,
    skip_keys=frozenset({"crossfield"}),
)
```
