# Plano de Refatoração Arquitetural — Revisão Claude

## Contexto e Premissas

Este plano parte da análise do REFACTOR.md existente, incorpora seu diagnóstico correto, corrige seus pontos fracos e adiciona problemas não identificados. O requisito primordial é preservado: **YAML continua sendo o contrato público, `_target_` continua sendo o mecanismo de extensão, nenhum campo novo é exigido.**

### O que muda em relação ao plano anterior

1. Validators chegam **antes** das factories — protegem a refatoração, não documentam depois.
2. Factories são **funções em módulo**, não classes — menos overhead, mais simples.
3. DataModules são **externos ao `Model`** — criados em `train.py`, não como `self._data_module` privado.
4. `_shared_step` é decomposto em **métodos privados**, não em Protocol/StepHandler — menos indireção.
5. `DomainAdaptationModel` usa **mixins**, não herança de nova classe base.
6. IO em `setup()` (`_compute_steps_from_config` com `pd.read_csv`) é **removido do Model**.
7. CLI permanece em `main.py` como dict inline — sem pacote `cli/`.
8. Dataset split tem **mapeamento de dependências** antes de mover qualquer arquivo.

### Requisitos Absolutos (imutáveis)

1. Todo YAML existente continua funcionando sem alteração.
2. Nenhuma nova chave YAML é exigida pela refatoração.
3. Nenhuma chave existente muda de nome, lugar, tipo público ou semântica.
4. Coverage global não diminui em nenhuma fase.
5. Testes antigos passam sem alteração relevante.
6. Imports antigos (ex: `from dataset_loader.dataset import X`) continuam funcionando.

---

## Diagnóstico Atualizado

### `model_loader/model.py` — Problemas Reais

**Confirmado no código:**

- 1225 linhas, 17+ responsabilidades (concordância com plano anterior).
- `_compute_steps_from_config` (linhas 199–308): faz `pd.read_csv(csv_path)` dentro de `setup()`. IO síncrono de arquivo em inicialização de modelo — trava quando CSV não existe, impede uso em inferência pura, duplica trabalho do DataLoader.
- `train_dataloader / val_dataloader / test_dataloader` (linhas 662–758): 97 linhas quasi-idênticas com pequenas diferenças de defaults. `DomainAdaptationModel._make_dataloader` já resolveu isso corretamente — `Model` não aprendeu com o filho.
- `_shared_step` (linhas 830–1060): template method com hooks implícitos via `hasattr`. Correto no diagnóstico do plano anterior.

**Problema ignorado pelo plano anterior — mais grave:**

`_compute_steps_from_config` é chamado em `setup()` e em `configure_optimizers()`. Isso significa que ao usar `OneCycleLR`, o framework lê o CSV duas vezes (uma no setup, uma ao construir o scheduler). Além disso, qualquer erro no CSV (arquivo ausente, formato errado) causa falha tardia — durante treino, não na validação do config.

### `model_loader/domain_adaptation_model.py` — Problema Real

`domain_adaptation_model.py:91` — `pl.LightningModule.__init__(self)` chamado diretamente. O comentário explica o motivo: `Model.__init__` acessa `cfg.train_dataset` e `cfg.val_dataset` incondicionalmente no bloco de GPU augmentations. Mas a raiz do problema é que `DomainAdaptationModel` precisa de exatamente este subconjunto de `Model`:

- `get_model()` — sim
- `get_loss_function()` — sim
- `_compute_loss()` — sim
- `_prepare_preds_for_metrics()` — sim
- `train_dataloader/val_dataloader` — não (tem os próprios)
- `_compute_steps_from_config` — não
- GPU augmentations setup — não

Isso não é herança "is-a". É reutilização de comportamento. Mixins resolvem isso sem o bypass.

### `config_definitions/train_config.py` — Bug Real

`train_config.py:120`:
```python
optimizer: List[OptimizerConfig] = field(default_factory=lambda: [OptimizerConfig()])
```

Runtime em `model.py:379–400` trata `self.cfg.optimizer` como objeto único com `._target_`, `.lr`, `.weight_decay`. A dataclass declara lista, o runtime usa objeto. Isso é uma inconsistência real que quebra validação estática e confunde leitores.

### `dataset_loader/dataset.py` — 3034 Linhas

Ao analisar imports internos do arquivo, `dataset.py` importa de:
- `utils/polygonrnn_utils`
- `utils/object_detection_utils`
- `utils/dataframe_utils`

E esses utils **não importam de `dataset_loader`**, então o split não gera circular imports neste caso específico. Mas é necessário verificar antes de qualquer movimento de código.

### `main.py` — 98 Linhas, 9 Modos

O if/elif chain é legível e funcional. Criar pacote `cli/` com `registry.py` para substituir 9 entradas é overengineering. Um dict inline resolve com menos arquivos.

---

## Arquitetura Alvo

### Princípios

1. `Model` delega para funções de factories e métodos privados pequenos.
2. DataModules são externos — criados em `train.py`, não dentro de `Model`.
3. `DomainAdaptationModel` compartilha comportamento via mixins, não herança de base.
4. IO de CSV nunca acontece dentro de `Model` — fica no DataModule ou na factory de scheduler.
5. `_shared_step` é decomposto em métodos privados com responsabilidade única.
6. `main.py` usa dict literal para dispatch, sem pacote `cli/`.

### Fluxo Alvo

```text
YAML Hydra
  -> ConfigValidator (valida semântica cross-field)
  -> train.py
     -> build_model(cfg) -> nn.Module
     -> build_loss(cfg) -> nn.Module
     -> SegmentationDataModule(cfg) -> pl.LightningDataModule
     -> Model(cfg, model, loss_fn) -> pl.LightningModule fino
     -> Trainer(cfg)
     -> trainer.fit(model, datamodule)
```

Para Domain Adaptation:
```text
YAML Hydra
  -> DomainAdaptationConfigValidator
  -> train_da.py (ou train.py com desvio por pl_model._target_)
     -> build_model(cfg)
     -> build_loss(cfg)
     -> DomainAdaptationDataModule(cfg)
     -> DomainAdaptationModel(cfg, model, loss_fn, method)
     -> Trainer
```

### Estrutura de Módulos Novos

```text
pytorch_segmentation_models_trainer/
  factories.py                     # funções: build_model, build_loss, build_optimizer, build_schedulers
  config_validation/
    __init__.py
    train_validator.py
    domain_adaptation_validator.py
  data/
    __init__.py
    datamodules.py                 # SegmentationDataModule, DomainAdaptationDataModule
    dataloader_builder.py          # _make_dataloader, _prefetch_factor, _make_generator
  model_loader/
    mixins.py                      # ModelBuildMixin, LossComputeMixin, MetricsMixin, OptimizerMixin
    model.py                       # Model usando mixins, _shared_step decomposto
    domain_adaptation_model.py     # DomainAdaptationModel usando mixins, sem bypass
  dataset_loader/
    base.py                        # AbstractDataset
    readers.py                     # leitura PIL/rasterio
    augmentations.py               # _sanitize_aug_config, load_augmentation_object
    segmentation.py                # datasets de segmentação
    detection.py                   # datasets de detecção
    windowed.py                    # window/crop/grid datasets
    class_balancing.py             # class balancing
    soft_labels.py                 # soft labels
    collate.py                     # collate_fn
    dataset.py                     # reexports de compat — não deletar
```

---

## Módulos Novos: Especificação Detalhada

### `factories.py`

Arquivo único (não pacote) com funções puras.

```python
# factories.py

def build_model(cfg: DictConfig) -> nn.Module:
    """Instancia modelo via Hydra, aplica replace_activation e fine_tuning."""
    ...

def build_loss(cfg: DictConfig) -> nn.Module:
    """Instancia loss: compound_loss, multi_loss ou cfg.loss simples."""
    ...

def build_optimizer(cfg: DictConfig, params) -> torch.optim.Optimizer:
    """Instancia optimizer. Suporta layer-wise LR decay."""
    ...

def build_schedulers(cfg: DictConfig, optimizer, steps_per_epoch: Optional[int] = None) -> list:
    """Instancia scheduler_list. Recebe steps_per_epoch externamente — sem IO de CSV."""
    ...
```

**Regras:**
- Nenhuma função tem estado — são puras (mesma entrada, mesma saída).
- `build_schedulers` recebe `steps_per_epoch` como parâmetro — quem chama calcula, não a função.
- Toda lógica de `hasattr(cfg, "loss_params")` fica em `build_loss`, não em `Model`.
- Toda lógica de `layer_decay` fica em `build_optimizer`, não em `Model`.

### `config_validation/train_validator.py`

```python
class TrainConfigValidator:
    def validate(self, cfg: DictConfig) -> None:
        """Lança ConfigValidationError com mensagem apontando o campo YAML."""
        self._check_model_target(cfg)
        self._check_loss_config(cfg)
        self._check_optimizer_format(cfg)
        self._check_scheduler_steps(cfg)
        self._check_dataset_targets(cfg)

    def _check_model_target(self, cfg): ...
    def _check_loss_config(self, cfg): ...
    def _check_optimizer_format(self, cfg): ...
    def _check_scheduler_steps(self, cfg): ...
    def _check_dataset_targets(self, cfg): ...
```

**Validações obrigatórias:**
- `cfg.model._target_` existe e é importável.
- Exatamente um formato de loss: `loss`, `loss_params.compound_loss`, ou `loss_params.multi_loss`. Dois ao mesmo tempo = erro claro.
- `cfg.optimizer` tem `_target_` (objeto) ou é lista com exatamente um elemento com `_target_`. Normaliza internamente.
- Se `OneCycleLR` em `scheduler_list`, verifica se `steps_per_epoch` é fornecido explicitamente OU se `train_dataset.input_csv_path` existe E `batch_size` é calculável. Senão, erro antecipado.
- Se `train_dataset` presente, tem `_target_`.

**Não valida:**
- Semântica de augmentations.
- Conteúdo de CSV.
- Compatibilidade de loss com arquitetura do modelo.

### `config_validation/domain_adaptation_validator.py`

```python
class DomainAdaptationConfigValidator:
    def validate(self, cfg: DictConfig) -> None:
        self._check_da_section(cfg)
        self._check_source_target_datasets(cfg)
        self._check_method(cfg)
        self._check_feature_layers_vs_method(cfg)

    def _check_da_section(self, cfg): ...
    def _check_source_target_datasets(self, cfg): ...
    def _check_method(self, cfg): ...
    def _check_feature_layers_vs_method(self, cfg): ...
```

### `data/dataloader_builder.py`

Extrai o que `DomainAdaptationModel._make_dataloader` já implementou corretamente:

```python
def make_dataloader(
    dataset: Dataset,
    data_loader_cfg,
    shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Constrói DataLoader com defaults seguros. data_loader_cfg pode ser None."""
    ...

def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Retorna Generator seeded ou None."""
    ...

def compute_prefetch_factor(data_loader_cfg, num_workers: int) -> Optional[int]:
    """Retorna prefetch_factor ou None quando num_workers=0."""
    ...
```

### `data/datamodules.py`

```python
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        seed = self.cfg.get("seed", None)
        if stage in ("fit", None):
            if "train_dataset" in self.cfg:
                self.train_ds = instantiate(self.cfg.train_dataset, seed=seed, _recursive_=False)
            if "val_dataset" in self.cfg:
                self.val_ds = instantiate(self.cfg.val_dataset, seed=seed, _recursive_=False)
        if stage in ("test", None):
            if "test_dataset" in self.cfg:
                self.test_ds = instantiate(self.cfg.test_dataset, seed=seed, _recursive_=False)

    def compute_steps_per_epoch(self) -> Optional[int]:
        """Calcula steps_per_epoch após setup(). Sem IO — usa len(dataset) se disponível."""
        ...

    def train_dataloader(self): ...
    def val_dataloader(self): ...
    def test_dataloader(self): ...


class DomainAdaptationDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        da_cfg = self.cfg.domain_adaptation
        self.source_train_ds = instantiate(da_cfg.source_dataset, _recursive_=False)
        self.target_train_ds = instantiate(da_cfg.target_dataset, _recursive_=False)
        self.source_val_ds = (
            instantiate(da_cfg.source_val_dataset, _recursive_=False)
            if da_cfg.source_val_dataset is not None else None
        )
        self.target_val_ds = (
            instantiate(da_cfg.target_val_dataset, _recursive_=False)
            if da_cfg.target_val_dataset is not None else None
        )

    def train_dataloader(self) -> CombinedLoader: ...
    def val_dataloader(self) -> Union[DataLoader, CombinedLoader]: ...
```

### `model_loader/mixins.py`

```python
class ModelBuildMixin:
    """Constrói e configura o nn.Module a partir do cfg."""
    def get_model(self) -> nn.Module:
        model = build_model(self.cfg)
        return model

    def set_encoder_trainable(self, trainable: bool) -> None: ...


class LossComputeMixin:
    """Computa loss simples e compound."""
    def get_loss_function(self) -> nn.Module:
        return build_loss(self.cfg)

    def _compute_loss(self, predicted_masks, masks) -> Tuple[Tensor, dict, dict]: ...


class MetricsMixin:
    """Setup e logging de torchmetrics."""
    def _setup_metrics(self) -> None:
        if "metrics" not in self.cfg:
            return
        metrics = torchmetrics.MetricCollection(
            [instantiate(i, _recursive_=False) for i in self.cfg.metrics]
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _setup_per_class_iou(self) -> None: ...
    def _prepare_preds_for_metrics(self, predicted_masks) -> Optional[Tensor]: ...


class OptimizerMixin:
    """Configura optimizer e schedulers via factories."""
    def get_optimizer(self) -> torch.optim.Optimizer:
        return build_optimizer(self.cfg, self.parameters())

    def _build_scheduler_list(self, optimizer, steps_per_epoch=None) -> list:
        return build_schedulers(self.cfg, optimizer, steps_per_epoch)
```

**Regras dos Mixins:**
- Nenhum Mixin tem `__init__`. Dependem de `self.cfg` estar definido antes do uso.
- Nenhum Mixin instancia datasets.
- Nenhum Mixin faz IO.
- Todos são testáveis com `MagicMock` para `self.cfg`.

---

## Módulos que Mudam: Especificação Detalhada

### `model_loader/model.py` — Depois da Refatoração

```python
class Model(ModelBuildMixin, LossComputeMixin, MetricsMixin, OptimizerMixin, pl.LightningModule):
    def __init__(self, cfg, inference_mode=False):
        pl.LightningModule.__init__(self)
        self.cfg = cfg

        seed = cfg.get("seed", None)
        if seed is not None:
            set_training_seed(seed, deterministic_cudnn=cfg.get("deterministic_cudnn", False))

        self.model = self.get_model()
        if inference_mode:
            return

        self.loss_function = self.get_loss_function()
        self._is_dual_head = getattr(self.loss_function, "is_dual_head_loss", False)
        if self._is_dual_head:
            self.loss_function.set_model(self.model)

        self.use_compound_loss = isinstance(self.loss_function, MultiLoss)
        self._setup_metrics()
        self._setup_per_class_iou()
        self._setup_gpu_augmentations()
        self.steps_per_epoch = None

        self.save_hyperparameters(ignore=["model", "loss_function"])

    def _setup_gpu_augmentations(self):
        """Setup de GPU augmentations sem acesso a datasets."""
        self.gpu_train_transform = self._build_gpu_transform("train_dataset")
        self.gpu_val_transform = self._build_gpu_transform("val_dataset")
        self.gpu_test_transform = self._build_gpu_transform("test_dataset")

    def _build_gpu_transform(self, dataset_key: str):
        if dataset_key not in self.cfg:
            return None
        ds_cfg = self.cfg[dataset_key]
        if "gpu_augmentation_list" not in ds_cfg:
            return None
        return torch.nn.Sequential(
            *[instantiate(aug, _recursive_=False) for aug in ds_cfg.gpu_augmentation_list]
        )

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        steps_per_epoch = getattr(self, "steps_per_epoch", None)
        scheduler_list = self._build_scheduler_list(optimizer, steps_per_epoch)
        return [optimizer], scheduler_list

    # DataLoaders — Model ainda expõe esses métodos para compatibilidade
    # mas delega para make_dataloader de data/dataloader_builder.py
    def train_dataloader(self): ...
    def val_dataloader(self): ...
    def test_dataloader(self): ...

    # _shared_step decomposto
    def _shared_step(self, batch, prefix):
        images, masks, hard_masks = self._unpack_and_prepare(batch)
        self._set_dual_head_context(hard_masks, batch)
        raw_output = self(images)
        output_for_loss, output_for_metrics = self._adapt_output(raw_output, prefix)
        loss = self._compute_final_loss(output_for_loss, masks, prefix)
        self._log_step_results(loss, output_for_metrics, hard_masks, prefix)
        self._cleanup_step_state()
        return loss

    def _unpack_and_prepare(self, batch):
        """Extrai imagens, masks e hard_masks do batch."""
        ...

    def _set_dual_head_context(self, hard_masks, batch):
        """Popula self.model.last_hard_mask para dual-head."""
        ...

    def _adapt_output(self, raw_output, prefix):
        """EDL: extrai probs para métricas, loga uncertainty. Outros: passa adiante."""
        ...

    def _compute_final_loss(self, output_for_loss, masks, prefix):
        """Aplica OHEM, dual-head, MoE aux, MEDOE expert loss em sequência."""
        ...

    def _log_step_results(self, loss, output_for_metrics, hard_masks, prefix):
        """Loga total loss, individual losses, extra info, métricas, per-class IoU."""
        ...

    def _cleanup_step_state(self):
        """Zera last_logits_A/B, last_expert_outputs, etc."""
        ...
```

**Ganho:** `_shared_step` de 230 linhas vira 8 linhas + 6 métodos privados com ~20–40 linhas cada. Cada método é testável isoladamente com mock de `self`.

### `model_loader/domain_adaptation_model.py` — Depois

```python
class DomainAdaptationModel(ModelBuildMixin, LossComputeMixin, MetricsMixin, OptimizerMixin, pl.LightningModule):
    def __init__(self, cfg):
        pl.LightningModule.__init__(self)  # Direto, correto — não há Model.__init__ para bypassar
        self.cfg = cfg

        self.model = self.get_model()
        self.loss_function = self.get_loss_function()
        self.use_compound_loss = isinstance(self.loss_function, MultiLoss)
        self._setup_metrics()
        self.check_if_should_normalize()
        self.steps_per_epoch = None

        self.save_hyperparameters(ignore=["model", "loss_function"])

        da_cfg = cfg.domain_adaptation
        self.method: BaseDomainAdaptationMethod = instantiate(da_cfg.method, _recursive_=False)
        self._setup_feature_hook(da_cfg)
        # Datasets ficam no DomainAdaptationDataModule — não aqui

    # configure_optimizers via OptimizerMixin + method.get_extra_parameter_groups()
    def configure_optimizers(self): ...

    # training_step, validation_step — permanecem como estão (já estão bem)
```

**Ganho:** `DomainAdaptationModel.__init__` não bypassa mais `Model.__init__`. Usa `pl.LightningModule.__init__` diretamente — isso é correto porque não há mais herança de `Model`. Os datasets não são instanciados aqui — vivem no `DomainAdaptationDataModule`.

### `train.py` — Depois

```python
def train(cfg: DictConfig) -> None:
    TrainConfigValidator().validate(cfg)

    dm = SegmentationDataModule(cfg)
    dm.setup("fit")  # Instancia datasets aqui, antes do model

    # steps_per_epoch calculado pelo DataModule, não pelo Model
    steps_per_epoch = dm.compute_steps_per_epoch()

    model = instantiate(cfg.pl_model, cfg=cfg, _recursive_=False)
    model.steps_per_epoch = steps_per_epoch  # Injetado externamente

    trainer = pl.Trainer(**OmegaConf.to_container(cfg.pl_trainer, resolve=True))
    trainer.fit(model, datamodule=dm)
```

**Ganho:** IO de CSV sai do `Model`. `Model` nunca lê CSV. `steps_per_epoch` é passado de fora como dado calculado pelo DataModule. `DomainAdaptationModel` recebe `DomainAdaptationDataModule` do mesmo jeito.

### `main.py` — Depois

```python
_COMMANDS = {
    "train": lambda cfg: _import_and_run("train", cfg),
    "predict": lambda cfg: _import_and_run("predict", cfg),
    "predict-from-batch": lambda cfg: _import_and_run("predict_from_batch", cfg),
    "predict-mod-polymapper-from-batch": lambda cfg: _import_and_run("predict_mod_polymapper_from_batch", cfg),
    "validate-config": lambda cfg: _import_and_run("config_utils", cfg, fn="validate_config"),
    "build-mask": lambda cfg: _import_and_run("build_mask", cfg, fn="build_masks"),
    "convert-dataset": lambda cfg: _import_and_run("convert_ds", cfg, fn="convert_dataset"),
    "evaluate-experiments": lambda cfg: _import_and_run("evaluate_experiments", cfg, fn="evaluate"),
    "run-experiments": lambda cfg: ExperimentsRunner(cfg).run(),
}

def main(cfg):
    ...
    handler = _COMMANDS.get(cfg.mode)
    if handler is None:
        raise NotImplementedError(f"Unknown mode: {cfg.mode!r}. Available: {list(_COMMANDS)}")
    return handler(cfg)
```

### `config_definitions/train_config.py` — Correção

```python
# ANTES (bug)
optimizer: List[OptimizerConfig] = field(default_factory=lambda: [OptimizerConfig()])

# DEPOIS (correto — usa Union para aceitar ambos durante transição)
from typing import Union
optimizer: Union[OptimizerConfig, List[OptimizerConfig]] = field(default_factory=OptimizerConfig)
```

O validator normaliza: se vier como lista com um elemento, usa o elemento. Se vier como objeto único, usa direto. Se vier como lista com múltiplos, levanta erro claro.

---

## Plano Faseado

### Fase 0: Baseline e Guard Rails

**Objetivo:** Medir antes de mexer. Estabelecer baseline imutável.

**Passos:**

1. Rodar suite padrão e salvar resultado:
```bash
uv run pytest tests/ -v --tb=short \
  --ignore=tests/test_detection_model.py \
  --ignore=tests/test_inference.py \
  --ignore=tests/test_predict.py \
  --ignore=tests/test_script.py
```

2. Rodar e salvar coverage baseline:
```bash
uv run pytest tests/ -v --tb=short \
  --ignore=tests/test_detection_model.py \
  --ignore=tests/test_inference.py \
  --ignore=tests/test_predict.py \
  --ignore=tests/test_script.py \
  --cov=pytorch_segmentation_models_trainer \
  --cov-report=term-missing \
  --cov-report=xml
```

3. Salvar percentual global e lista de arquivos com <80% cobertura em `coverage_baseline.txt`.

4. Adicionar ao CI: `--cov-fail-under=<percentual_baseline>`.

5. Criar snapshot de log keys de `_shared_step`:
   - Script que instancia `Model` com config mínimo e coleta todos os `self.log(key, ...)` chamados durante um step fake.
   - Salvar em `tests/snapshots/log_keys_shared_step.json`.
   - Teste que verifica esse snapshot não muda após refatoração.

6. Rodar todos os YAMLs de `conf/examples/` em modo de validação e documentar os que passam.

**Testes novos:**
- `tests/test_log_keys_snapshot.py` — verifica que log keys não mudam.
- `tests/test_yaml_examples_loadable.py` — verifica que todos os YAMLs de exemplo carregam sem erro.

**Critério de saída:** CI verde, baseline salvo, snapshots criados.

---

### Fase 0.5: Config Validators

**Objetivo:** Criar rede de segurança semântica antes de mexer em qualquer código de produção.

**Motivo para ser antes das factories:** Se uma factory introduzir regressão de comportamento de config, o validator detecta na mesma PR antes que o problema chegue em runtime de treino.

**Passos:**

1. Criar `config_validation/__init__.py`.

2. Criar `config_validation/exceptions.py`:
```python
class ConfigValidationError(ValueError):
    """Erro de validação de config com caminho YAML."""
    def __init__(self, field_path: str, message: str):
        super().__init__(f"Config error at '{field_path}': {message}")
        self.field_path = field_path
```

3. Criar `config_validation/train_validator.py` com `TrainConfigValidator`:

   **`_check_model_target`:**
   - Verifica `cfg.model` existe.
   - Verifica `cfg.model._target_` existe.
   - Tenta `importlib.import_module` no path base do `_target_`. Se falhar, levanta `ConfigValidationError("model._target_", "cannot import ...")`.
   - Não instancia o modelo — apenas verifica importabilidade.

   **`_check_loss_config`:**
   - Conta quantas fontes de loss estão presentes: `cfg.loss`, `cfg.loss_params.compound_loss`, `cfg.loss_params.multi_loss`.
   - Se zero: `ConfigValidationError("loss", "no loss configuration found. Use cfg.loss, cfg.loss_params.compound_loss, or cfg.loss_params.multi_loss")`.
   - Se dois ou mais: `ConfigValidationError("loss_params", "ambiguous loss: both compound_loss and multi_loss are set")`.

   **`_check_optimizer_format`:**
   - Aceita `cfg.optimizer` como objeto com `_target_`.
   - Aceita `cfg.optimizer` como lista de um elemento com `_target_`.
   - Recusa lista com múltiplos elementos (não suportado em runtime).
   - Emite warning (não erro) se formato for lista — indica dataclass legacy.

   **`_check_scheduler_steps`:**
   - Para cada item em `cfg.scheduler_list`, se `_target_` contém `OneCycleLR`:
     - Se `steps_per_epoch` está explícito e é inteiro positivo: OK.
     - Se `steps_per_epoch` é `None`, `"auto"`, ou ausente:
       - Verifica se `train_dataset.input_csv_path` existe (sem ler).
       - Verifica se `batch_size` é calculável.
       - Se não: levanta `ConfigValidationError("scheduler_list[n].scheduler.steps_per_epoch", "OneCycleLR requires steps_per_epoch ...")`.

   **`_check_dataset_targets`:**
   - Para cada dataset key em `["train_dataset", "val_dataset", "test_dataset"]`:
     - Se presente, verifica `_target_` existe.

4. Criar `config_validation/domain_adaptation_validator.py` com `DomainAdaptationConfigValidator`:

   **`_check_da_section`:** Verifica `cfg.domain_adaptation` existe.
   **`_check_source_target_datasets`:** Verifica `source_dataset._target_` e `target_dataset._target_`.
   **`_check_method`:** Verifica `domain_adaptation.method._target_` importável.
   **`_check_feature_layers_vs_method`:** Se `method.requires_features=True` (se acessível sem instanciar) e `feature_layers` vazio, emite warning (não erro — é aceitável ter vazio com warning).

5. Integrar validators nos pontos de entrada existentes:
   - Em `config_utils.validate_config(cfg)`: chamar `TrainConfigValidator().validate(cfg)`.
   - Ainda não integrar em `train.py` — isso fica para Fase 3 para não acoplar mudanças.

**Testes novos:**

`tests/test_train_config_validator.py`:
- `model._target_` ausente → `ConfigValidationError`.
- `model._target_` não importável → `ConfigValidationError`.
- `cfg.loss` ausente, `loss_params` ausente → `ConfigValidationError`.
- `loss` e `loss_params.compound_loss` ambos presentes → `ConfigValidationError`.
- Optimizer como objeto único → OK.
- Optimizer como lista de um elemento → OK + warning.
- Optimizer como lista de múltiplos → `ConfigValidationError`.
- `OneCycleLR` sem `steps_per_epoch` e sem CSV → `ConfigValidationError`.
- `OneCycleLR` com `steps_per_epoch` explícito → OK.
- `train_dataset` presente sem `_target_` → `ConfigValidationError`.
- Config mínimo válido → nenhum erro.
- Nenhuma chave nova é exigida → testar que config com apenas `model` e `loss` passa.

`tests/test_da_config_validator.py`:
- `domain_adaptation` ausente → `ConfigValidationError`.
- `source_dataset` ausente → `ConfigValidationError`.
- `method._target_` não importável → `ConfigValidationError`.
- Config DA mínimo válido → nenhum erro.

**Critério de saída:** Validators implementados, integrados no `validate_config`, todos os testes passando, coverage dos validators = 100%.

---

### Fase 1: Unificar Dataloader Boilerplate em `Model`

**Objetivo:** Antes de qualquer factory, eliminar o código quasi-idêntico de `train_dataloader / val_dataloader / test_dataloader`. Esta fase é mecânica e sem risco — não muda comportamento, só deduplication.

**Motivo:** `Model` tem 97 linhas de boilerplate de dataloader que `DomainAdaptationModel._make_dataloader` já resolveu. Unificar antes de mover para DataModule evita migrar código ruim.

**Passos:**

1. Criar `data/__init__.py` e `data/dataloader_builder.py`:

```python
# data/dataloader_builder.py

from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset


def compute_prefetch_factor(data_loader_cfg, num_workers: int) -> Optional[int]:
    """Retorna None quando num_workers=0 (requisito do PyTorch)."""
    if num_workers == 0:
        return None
    if data_loader_cfg is not None and hasattr(data_loader_cfg, "prefetch_factor"):
        return data_loader_cfg.prefetch_factor
    return 2


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Retorna Generator seeded ou None."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed) % (2**32))
    return g


def make_dataloader(
    dataset: Dataset,
    data_loader_cfg,
    *,
    batch_size: int,
    shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
    include_worker_init: bool = True,
) -> DataLoader:
    """
    Constrói DataLoader com defaults seguros.

    data_loader_cfg pode ser None — usa defaults conservadores.
    """
    if data_loader_cfg is not None:
        num_workers = getattr(data_loader_cfg, "num_workers", 4)
        pin_memory = getattr(data_loader_cfg, "pin_memory", True)
        drop_last = getattr(data_loader_cfg, "drop_last", False)
        persistent_workers = getattr(data_loader_cfg, "persistent_workers", False)
        cfg_shuffle = getattr(data_loader_cfg, "shuffle", shuffle)
    else:
        num_workers = 4
        pin_memory = True
        drop_last = False
        persistent_workers = False
        cfg_shuffle = shuffle

    kwargs = dict(
        batch_size=batch_size,
        shuffle=cfg_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        generator=generator,
    )

    pf = compute_prefetch_factor(data_loader_cfg, num_workers)
    if pf is not None:
        kwargs["prefetch_factor"] = pf

    collate_fn = getattr(dataset, "collate_fn", None)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn

    if include_worker_init:
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import _worker_init_fn
        kwargs["worker_init_fn"] = _worker_init_fn

    return DataLoader(dataset, **kwargs)
```

2. Reescrever `Model.train_dataloader`, `Model.val_dataloader`, `Model.test_dataloader` para usar `make_dataloader`:

```python
def train_dataloader(self):
    return make_dataloader(
        self.train_ds,
        self.cfg.train_dataset.data_loader,
        batch_size=self.cfg.hyperparameters.batch_size,
        shuffle=self.cfg.train_dataset.data_loader.shuffle,
        generator=make_generator(self.cfg.get("seed")),
    )

def val_dataloader(self):
    if self.val_ds is None:
        return None
    return make_dataloader(
        self.val_ds,
        self.cfg.val_dataset.data_loader,
        batch_size=self.cfg.hyperparameters.batch_size,
        shuffle=False,
        generator=make_generator(self.cfg.get("seed")),
    )

def test_dataloader(self):
    if self.test_ds is None:
        return None
    return make_dataloader(
        self.test_ds,
        self.cfg.test_dataset.data_loader,
        batch_size=self.cfg.hyperparameters.batch_size,
        shuffle=False,
        generator=make_generator(self.cfg.get("seed")),
    )
```

3. Reescrever `DomainAdaptationModel._make_dataloader` para usar a mesma função (eliminar duplicata).

4. Remover `_prefetch_factor` e `_make_dataloader_generator` de `Model` (eram usados internamente, agora em `dataloader_builder`).

**Testes novos:**

`tests/test_dataloader_builder.py`:
- `num_workers=0` → `prefetch_factor` ausente no DataLoader kwargs.
- `num_workers=4` → `prefetch_factor=2` (default).
- `num_workers=4` + `data_loader_cfg.prefetch_factor=4` → `prefetch_factor=4`.
- `pin_memory`, `drop_last`, `persistent_workers` — testados via mock.
- `generator` com seed → determinístico.
- `generator` sem seed → `None`.
- `collate_fn` no dataset → propagado para DataLoader.
- `data_loader_cfg=None` → defaults seguros.
- `shuffle=True` override.

**Testes antigos que protegem:**
- `tests/test_model.py`
- `tests/test_model_training_step.py`
- `tests/test_domain_adaptation_model.py`

**Critério de saída:** Comportamento de dataloader idêntico, testes antigos passando, `test_dataloader_builder.py` com 100% coverage, `_prefetch_factor` e `_make_dataloader_generator` removidos de `Model`.

---

### Fase 2: Factories como Funções

**Objetivo:** Extrair construção de model, loss, optimizer e scheduler para funções puras em `factories.py`. `Model` chama funções. Comportamento preservado.

**Passos:**

1. Criar `factories.py` na raiz do pacote:

```python
# factories.py

import logging
import re
from typing import Optional, List
import torch
import torch.nn as nn
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Instancia modelo via Hydra. Aplica replace_activation e fine_tuning se configurados.

    Args:
        cfg: Config raiz com chaves 'model', opcionalmente 'replace_model_activation'
             e 'fine_tuning'.

    Returns:
        nn.Module instanciado e configurado.
    """
    model = instantiate(cfg.model, _recursive_=False)

    if "replace_model_activation" in cfg:
        from pytorch_segmentation_models_trainer.utils.model_utils import replace_activation
        old_act = instantiate(cfg.replace_model_activation.old_activation, _recursive_=False)
        new_act = instantiate(cfg.replace_model_activation.new_activation, _recursive_=False)
        replace_activation(model, old_act, new_act)

    if "fine_tuning" in cfg:
        from pytorch_segmentation_models_trainer.fine_tuning.lora_utils import apply_fine_tuning_strategy
        model = apply_fine_tuning_strategy(model, cfg.fine_tuning)

    return model


def build_loss(cfg: DictConfig) -> nn.Module:
    """
    Instancia loss a partir do config.

    Prioridade:
    1. loss_params.compound_loss (novo)
    2. loss_params.multi_loss (legado)
    3. cfg.loss (simples)

    Args:
        cfg: Config raiz.

    Returns:
        nn.Module de loss.

    Raises:
        ValueError: Se nenhuma configuração de loss for encontrada.
        ValueError: Se múltiplos formatos conflitantes forem encontrados.
    """
    from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss

    if hasattr(cfg, "loss_params") and hasattr(cfg.loss_params, "compound_loss"):
        if cfg.loss_params.compound_loss is not None:
            from pytorch_segmentation_models_trainer.custom_losses.loss_builder import (
                build_compound_loss_from_config,
            )
            logger.info("Building compound loss from loss_params.compound_loss")
            return build_compound_loss_from_config(cfg.loss_params.compound_loss)

    if hasattr(cfg, "loss_params") and hasattr(cfg.loss_params, "multi_loss"):
        from pytorch_segmentation_models_trainer.custom_losses.loss_builder import (
            build_loss_from_config,
        )
        logger.info("Building loss from legacy multi_loss configuration")
        return build_loss_from_config(cfg)

    if "loss" in cfg:
        logger.info("Building simple loss from cfg.loss")
        return instantiate(cfg.loss, _recursive_=False)

    raise ValueError(
        "No loss configuration found. Please specify one of:\n"
        "  - cfg.loss_params.compound_loss (recommended)\n"
        "  - cfg.loss_params.multi_loss (legacy)\n"
        "  - cfg.loss (simple)"
    )


def build_optimizer(
    cfg: DictConfig,
    params,
) -> torch.optim.Optimizer:
    """
    Instancia optimizer. Suporta layer-wise LR decay via cfg.hyperparameters.layer_decay.

    Args:
        cfg: Config raiz com chave 'optimizer' e opcionalmente 'hyperparameters.layer_decay'.
        params: Parâmetros do modelo (self.parameters() ou param_groups customizados).

    Returns:
        torch.optim.Optimizer.
    """
    layer_decay = None
    if hasattr(cfg, "hyperparameters"):
        layer_decay = getattr(cfg.hyperparameters, "layer_decay", None)

    if layer_decay is not None and layer_decay < 1.0:
        # Constrói param_groups com LLRD — requer acesso a named_parameters via params
        # params aqui é um iterator — quem chama deve passar named_params também
        # Para manter compat: aceita callable que retorna named_parameters
        raise ValueError(
            "build_optimizer com layer_decay requer named_parameters. "
            "Use build_optimizer_with_llrd(cfg, module) diretamente."
        )

    return instantiate(cfg.optimizer, params=params, _recursive_=False)


def build_optimizer_with_llrd(
    cfg: DictConfig,
    module: nn.Module,
) -> torch.optim.Optimizer:
    """
    Instancia optimizer com Layer-wise LR Decay para módulo específico.

    Args:
        cfg: Config raiz.
        module: nn.Module de onde extrair named_parameters.

    Returns:
        torch.optim.Optimizer com param_groups por layer.
    """
    layer_decay = cfg.hyperparameters.layer_decay
    param_groups = _build_llrd_param_groups(cfg, module, layer_decay)
    optimizer_cls = get_class(cfg.optimizer._target_)
    opt_kwargs = OmegaConf.to_container(cfg.optimizer, resolve=True)
    opt_kwargs.pop("_target_")
    opt_kwargs.pop("lr", None)
    opt_kwargs.pop("weight_decay", None)
    return optimizer_cls(param_groups, **opt_kwargs)


def _build_llrd_param_groups(cfg: DictConfig, module: nn.Module, layer_decay: float) -> list:
    """Constrói param_groups com stage-wise LR decay. Lógica movida de Model."""
    base_lr = cfg.optimizer.lr
    weight_decay = getattr(cfg.optimizer, "weight_decay", 0.05)

    max_stage = -1
    encoder_names_sample = []
    for name, _ in module.named_parameters():
        if "encoder" in name:
            if len(encoder_names_sample) < 5:
                encoder_names_sample.append(name)
            match = re.search(r"stages[\.\[](\d+)", name)
            if match:
                max_stage = max(max_stage, int(match.group(1)))

    num_stages = max_stage + 1 if max_stage >= 0 else 0

    param_groups = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" not in name:
            group_lr = base_lr
        else:
            match = re.search(r"stages\.(\d+)", name)
            if match:
                stage_id = int(match.group(1))
                depth = num_stages - stage_id
                group_lr = base_lr * (layer_decay ** depth)
            else:
                group_lr = base_lr * (layer_decay ** (num_stages + 1))
        param_wd = 0.0 if param.ndim <= 1 else weight_decay
        param_groups.append({"params": [param], "lr": group_lr, "weight_decay": param_wd})

    return param_groups


def build_schedulers(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: Optional[int] = None,
) -> list:
    """
    Instancia scheduler_list do config.

    Para OneCycleLR: usa steps_per_epoch fornecido. Se não fornecido e necessário,
    levanta ValueError com mensagem clara (nunca faz IO de CSV).

    Args:
        cfg: Config raiz.
        optimizer: Optimizer já instanciado.
        steps_per_epoch: Calculado externamente pelo DataModule. None se não aplicável.

    Returns:
        Lista de dicts de scheduler para Lightning.
    """
    if "scheduler_list" not in cfg:
        return []

    warmup_epochs = 0
    if hasattr(cfg, "hyperparameters"):
        warmup_epochs = getattr(cfg.hyperparameters, "warmup_epochs", 0)

    scheduler_list = []
    for item in cfg.scheduler_list:
        dict_item = dict(item)
        scheduler_target = item.scheduler.get("_target_", "")
        is_one_cycle = "OneCycleLR" in scheduler_target

        if is_one_cycle:
            scheduler_config = dict(item.scheduler)
            needs_auto_steps = (
                "steps_per_epoch" not in scheduler_config
                or scheduler_config.get("steps_per_epoch") in [None, "auto", -1]
            )
            if needs_auto_steps:
                if steps_per_epoch is None:
                    raise ValueError(
                        "OneCycleLR requires steps_per_epoch. "
                        "Either set it explicitly in scheduler_list or ensure "
                        "the DataModule can compute it from the dataset."
                    )
                scheduler_config["steps_per_epoch"] = steps_per_epoch
            dict_item["scheduler"] = instantiate(
                scheduler_config, optimizer=optimizer, _recursive_=False
            )
        else:
            base_scheduler = instantiate(item.scheduler, optimizer=optimizer, _recursive_=False)
            if warmup_epochs > 0:
                from torch.optim.lr_scheduler import LinearLR, SequentialLR
                if hasattr(base_scheduler, "T_max"):
                    base_scheduler.T_max -= warmup_epochs
                warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
                dict_item["scheduler"] = SequentialLR(
                    optimizer, schedulers=[warmup, base_scheduler], milestones=[warmup_epochs]
                )
            else:
                dict_item["scheduler"] = base_scheduler

        scheduler_list.append(dict_item)

    return scheduler_list
```

2. Alterar `Model.get_model()` para chamar `build_model(self.cfg)`.
3. Alterar `Model.get_loss_function()` para chamar `build_loss(self.cfg)`.
4. Alterar `Model.get_optimizer()` para chamar `build_optimizer` ou `build_optimizer_with_llrd`.
5. Alterar `Model.configure_optimizers()` para chamar `build_schedulers(cfg, optimizer, steps_per_epoch)`.
6. Manter métodos antigos como delegadores com `DeprecationWarning` interno (não público):
   - `Model.get_model()` → `return build_model(self.cfg)` (não deleta o método, apenas delega)
   - `Model.get_loss_function()` → `return build_loss(self.cfg)`

**Testes novos:**

`tests/test_factories.py`:

**`build_model`:**
- Config com `model._target_=torch.nn.Conv2d` → instancia Conv2d.
- Config com `replace_model_activation` → ativações substituídas (mock de `replace_activation`).
- Config com `fine_tuning` → `apply_fine_tuning_strategy` chamado (mock).
- Config sem `replace_model_activation` e sem `fine_tuning` → modelo limpo.

**`build_loss`:**
- Config com `loss._target_=torch.nn.CrossEntropyLoss` → CrossEntropyLoss.
- Config com `loss_params.compound_loss` → `build_compound_loss_from_config` chamado (mock).
- Config com `loss_params.multi_loss` → `build_loss_from_config` chamado (mock).
- Config sem nenhum loss → `ValueError`.
- Config com `compound_loss` e `multi_loss` ambos → `compound_loss` tem prioridade (não erro, mas documentado).

**`build_optimizer`:**
- Optimizer simples com `_target_=torch.optim.Adam` → Adam instanciado.
- `build_optimizer_with_llrd` com módulo de dois estágios → param_groups com LRs diferentes.
- Parâmetro com `requires_grad=False` → não aparece em param_groups.
- Parâmetro 1D (bias) → `weight_decay=0`.

**`build_schedulers`:**
- Sem `scheduler_list` → lista vazia.
- `OneCycleLR` com `steps_per_epoch` explícito → instanciado.
- `OneCycleLR` sem `steps_per_epoch` + `steps_per_epoch=None` → `ValueError`.
- `OneCycleLR` sem `steps_per_epoch` + `steps_per_epoch=100` → usa 100.
- Scheduler não-OneCycleLR com `warmup_epochs=0` → instanciado direto.
- Scheduler não-OneCycleLR com `warmup_epochs=5` → `SequentialLR` com warmup.

**Testes antigos que protegem:**
- `tests/test_model.py`
- `tests/test_model_base_comprehensive.py`
- `tests/test_losses_callbacks_tta.py`
- `tests/test_loss_builder.py`

**Critério de saída:** `factories.py` com 100% coverage nos testes novos. Testes antigos passando. `Model` ainda funciona igual mas delega para funções. Log keys snapshot inalterado.

---

### Fase 3: DataModules Externos e Remoção de IO do Model

**Objetivo:** Criar `SegmentationDataModule` e `DomainAdaptationDataModule` como componentes externos. Remover `_compute_steps_from_config` de `Model`. Integrar validators em `train.py`.

**Passos:**

1. Criar `data/datamodules.py` com `SegmentationDataModule`:

```python
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._seed = cfg.get("seed", None)

    def setup(self, stage=None):
        if stage in ("fit", None):
            if "train_dataset" in self.cfg:
                self.train_ds = instantiate(
                    self.cfg.train_dataset, seed=self._seed, _recursive_=False
                )
            if "val_dataset" in self.cfg:
                self.val_ds = instantiate(
                    self.cfg.val_dataset, seed=self._seed, _recursive_=False
                )
        if stage in ("test", None):
            if "test_dataset" in self.cfg:
                self.test_ds = instantiate(
                    self.cfg.test_dataset, seed=self._seed, _recursive_=False
                )

    def compute_steps_per_epoch(self) -> Optional[int]:
        """
        Calcula steps_per_epoch após setup().

        Usa len(self.train_ds) quando disponível. Fallback para CSV row count
        apenas se dataset não foi instanciado ainda. Nunca chamado dentro de Model.
        """
        if self.train_ds is None:
            return None

        dataset_cfg = self.cfg.get("train_dataset", {})

        if dataset_cfg.get("grid_mode", False):
            return self._compute_from_dataset_len()

        if "samples_per_epoch" in dataset_cfg:
            size = dataset_cfg.samples_per_epoch
            if size <= 0 and hasattr(self.train_ds, "samples_per_epoch"):
                size = self.train_ds.samples_per_epoch
        else:
            size = len(self.train_ds)

        batch_size = self._get_batch_size()
        if batch_size is None:
            return None

        device_count = self._compute_device_count()
        accumulate = self._get_accumulate_grad_batches()
        effective_batch = batch_size * accumulate * device_count

        return size // effective_batch

    def _compute_from_dataset_len(self) -> Optional[int]:
        if self.train_ds is None:
            return None
        batch_size = self._get_batch_size()
        if batch_size is None:
            return None
        return len(self.train_ds) // batch_size

    def _get_batch_size(self) -> Optional[int]:
        """Lê batch_size de múltiplas localizações no config."""
        ds_cfg = self.cfg.get("train_dataset", {})
        dl_cfg = ds_cfg.get("data_loader", {})
        if hasattr(dl_cfg, "batch_size"):
            return dl_cfg.batch_size
        if "hyperparameters" in self.cfg:
            return self.cfg.hyperparameters.get("batch_size")
        return self.cfg.get("batch_size")

    def _compute_device_count(self) -> int:
        """Extrai device count do config. Lógica movida de Model._compute_device_count."""
        # ... mesma lógica de Model, mas sem acesso a self.trainer
        ...

    def _get_accumulate_grad_batches(self) -> int:
        if "hyperparameters" in self.cfg:
            return self.cfg.hyperparameters.get("accumulate_grad_batches", 1)
        if "pl_trainer" in self.cfg:
            return self.cfg.pl_trainer.get("accumulate_grad_batches", 1)
        return 1

    def train_dataloader(self):
        return make_dataloader(
            self.train_ds,
            self.cfg.train_dataset.data_loader,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=self.cfg.train_dataset.data_loader.shuffle,
            generator=make_generator(self._seed),
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return make_dataloader(
            self.val_ds,
            self.cfg.val_dataset.data_loader,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=False,
            generator=make_generator(self._seed),
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return make_dataloader(
            self.test_ds,
            self.cfg.test_dataset.data_loader,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=False,
            generator=make_generator(self._seed),
        )
```

2. Criar `DomainAdaptationDataModule` em `data/datamodules.py` (migrar lógica de `DomainAdaptationModel.train_dataloader/val_dataloader`).

3. Atualizar `train.py`:
```python
def train(cfg: DictConfig) -> None:
    TrainConfigValidator().validate(cfg)

    dm = SegmentationDataModule(cfg)
    dm.setup("fit")
    steps_per_epoch = dm.compute_steps_per_epoch()

    pl_model = instantiate(cfg.pl_model, cfg=cfg, _recursive_=False)
    pl_model.steps_per_epoch = steps_per_epoch

    trainer = instantiate_trainer(cfg)
    trainer.fit(pl_model, datamodule=dm)
```

4. Remover de `Model`:
   - `_compute_steps_from_config` — deletar.
   - `_compute_device_count` — deletar (movido para DataModule).
   - `setup(stage)` — simplificar: apenas chamar `super().setup(stage)`.
   - `train_ds`, `val_ds`, `test_ds` de `__init__` — manter por compatibilidade mas deprecar internamente.

5. `Model.train_dataloader()` passa a checar `self.train_ds` (compatibilidade) e delegar para `make_dataloader`. Se `train_ds is None`, assume que DataModule externo foi usado.

**Testes novos:**

`tests/test_segmentation_datamodule.py`:
- `setup("fit")` instancia `train_ds` e `val_ds`.
- `setup("test")` instancia `test_ds`.
- `val_ds` e `test_ds` ausentes no config → `None`.
- `train_dataloader()` retorna DataLoader com batch_size correto.
- `val_dataloader()` retorna `None` quando `val_ds` é `None`.
- `compute_steps_per_epoch()` sem `setup()` → `None`.
- `compute_steps_per_epoch()` com dataset de 100 items, batch 10 → 10.
- `compute_steps_per_epoch()` com `grid_mode=True` → usa `len(dataset)`.
- `compute_steps_per_epoch()` sem `batch_size` no config → `None`.

`tests/test_da_datamodule.py`:
- `train_dataloader()` retorna `CombinedLoader` com keys `source` e `target`.
- `val_dataloader()` com apenas `source_val_ds` → DataLoader simples.
- `val_dataloader()` com ambos → `CombinedLoader`.
- `val_dataloader()` sem nenhum → comportamento definido (DataLoader vazio ou aviso).

**Testes antigos que protegem:**
- `tests/test_model.py` — Model ainda funciona sem DataModule externo.
- `tests/test_domain_adaptation_model.py`.
- `tests/test_seed_utils.py`.

**Critério de saída:** `_compute_steps_from_config` deletado. `_compute_device_count` deletado. `SegmentationDataModule` e `DomainAdaptationDataModule` com 100% coverage. Testes antigos passando. `train.py` usa DataModule externo. `steps_per_epoch` injetado de fora.

---

### Fase 4: Mixins e Remoção do Bypass em `DomainAdaptationModel`

**Objetivo:** Criar `model_loader/mixins.py` com comportamentos compartilhados. `Model` e `DomainAdaptationModel` herdam de mixins. `DomainAdaptationModel` deixa de chamar `pl.LightningModule.__init__` diretamente como workaround.

**Passos:**

1. Criar `model_loader/mixins.py`:

```python
# mixins.py — sem __init__, apenas métodos que assumem self.cfg existe

class ModelBuildMixin:
    def get_model(self) -> nn.Module:
        return build_model(self.cfg)

    def set_encoder_trainable(self, trainable: bool) -> None:
        # Mover lógica de Model.set_encoder_trainable aqui
        ...


class LossComputeMixin:
    def get_loss_function(self) -> nn.Module:
        return build_loss(self.cfg)

    def _compute_loss(self, predicted_masks, masks):
        # Mover lógica de Model._compute_loss aqui
        ...

    def check_if_should_normalize(self) -> bool:
        # Mover lógica de Model.check_if_should_normalize aqui
        ...


class MetricsMixin:
    def _setup_metrics(self) -> None:
        if "metrics" not in self.cfg:
            return
        metrics = torchmetrics.MetricCollection(
            [instantiate(i, _recursive_=False) for i in self.cfg.metrics]
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _setup_per_class_iou(self) -> None:
        self._per_class_iou = None
        self._class_names = None
        if "class_definitions" not in self.cfg:
            return
        n_cls = len(self.cfg.class_definitions.names)
        self._per_class_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=n_cls, average="none", ignore_index=255
        )
        self._class_names = list(self.cfg.class_definitions.names)

    def _prepare_preds_for_metrics(self, predicted_masks):
        # Mover lógica de Model._prepare_preds_for_metrics aqui
        ...


class OptimizerMixin:
    def get_optimizer(self) -> torch.optim.Optimizer:
        layer_decay = None
        if hasattr(self.cfg, "hyperparameters"):
            layer_decay = getattr(self.cfg.hyperparameters, "layer_decay", None)
        if layer_decay is not None and layer_decay < 1.0:
            return build_optimizer_with_llrd(self.cfg, self)
        return build_optimizer(self.cfg, self.parameters())

    def _build_scheduler_list(self, optimizer, steps_per_epoch=None) -> list:
        return build_schedulers(self.cfg, optimizer, steps_per_epoch)
```

2. Alterar `Model`:
```python
class Model(ModelBuildMixin, LossComputeMixin, MetricsMixin, OptimizerMixin, pl.LightningModule):
    def __init__(self, cfg, inference_mode=False):
        pl.LightningModule.__init__(self)
        self.cfg = cfg
        # setup de seed, model, loss, dual-head, metrics, gpu_transforms
        # NÃO mais instancia datasets aqui (datasets vivem no DataModule)
        ...
```

3. Alterar `DomainAdaptationModel`:
```python
class DomainAdaptationModel(ModelBuildMixin, LossComputeMixin, MetricsMixin, OptimizerMixin, pl.LightningModule):
    def __init__(self, cfg):
        pl.LightningModule.__init__(self)  # Correto — não é workaround, é uso direto
        self.cfg = cfg
        self.model = self.get_model()
        self.loss_function = self.get_loss_function()
        self.use_compound_loss = isinstance(self.loss_function, MultiLoss)
        self._setup_metrics()
        self.check_if_should_normalize()
        self.steps_per_epoch = None
        self.save_hyperparameters(ignore=["model", "loss_function"])

        da_cfg = cfg.domain_adaptation
        self.method = instantiate(da_cfg.method, _recursive_=False)
        self._setup_feature_hook(da_cfg)
        # Sem source_train_ds, target_train_ds, etc — estão no DomainAdaptationDataModule
```

4. Remover de `DomainAdaptationModel`:
   - `source_train_ds`, `target_train_ds`, `source_val_ds`, `target_val_ds` — movidos para DataModule.
   - `train_dataloader()` e `val_dataloader()` — removidos (DataModule provê).
   - `_make_dataloader()` — migrado para `dataloader_builder.py` na Fase 1.

5. Verificar que `DomainAdaptationModel.configure_optimizers()` ainda funciona via `OptimizerMixin` + lógica de extra_groups do method.

**Testes novos:**

`tests/test_mixins.py`:
- `ModelBuildMixin.get_model()` com cfg mínimo → instancia modelo.
- `LossComputeMixin.get_loss_function()` → delega para `build_loss`.
- `LossComputeMixin._compute_loss()` com loss simples → retorna (loss, {}, {}).
- `LossComputeMixin._compute_loss()` com compound loss → retorna (loss, dict, dict).
- `MetricsMixin._setup_metrics()` sem `metrics` no cfg → nenhum atributo criado.
- `MetricsMixin._setup_metrics()` com metrics → `train_metrics`, `val_metrics`, `test_metrics` criados.
- `MetricsMixin._setup_per_class_iou()` sem `class_definitions` → `_per_class_iou` é None.
- `MetricsMixin._prepare_preds_for_metrics()` com tensor (B,1,H,W) → squeeze para (B,H,W).
- `MetricsMixin._prepare_preds_for_metrics()` com não-tensor → None + warning.
- `OptimizerMixin.get_optimizer()` sem layer_decay → `build_optimizer` chamado.
- `OptimizerMixin.get_optimizer()` com layer_decay < 1.0 → `build_optimizer_with_llrd` chamado.

Atualizar `tests/test_domain_adaptation_model.py`:
- Assert que `DomainAdaptationModel.__init__` NÃO acessa `cfg.train_dataset`.
- Assert que `DomainAdaptationModel` NÃO herda de `Model`.
- Assert que comportamento de training/validation é idêntico.

**Testes antigos que protegem:**
- `tests/test_da_model_comprehensive.py`
- `tests/test_dann_method.py`
- `tests/test_base_method.py`
- `tests/test_domain_adaptation_model.py`

**Critério de saída:** `DomainAdaptationModel` não herda `Model`. `Model` e `DomainAdaptationModel` herdam de mixins. Nenhum bypass de `__init__` como workaround. Testes antigos passando.

---

### Fase 5: Decomposição de `_shared_step`

**Objetivo:** Decompor `_shared_step` de 230 linhas em métodos privados com responsabilidade única. Sem Protocol, sem StepHandler, sem StepContext dataclass. Contexto é `self`.

**Motivo para não usar Protocol/StepHandler:** EDL, dual-head, OHEM, MoE, MEDOE têm comportamentos com dependência de ordem e estado mútuo (`current_epoch`, `global_step`, cleanup de tensores). Um Protocol com `before_loss/after_loss/after_metrics` não cobre todos os pontos de extensão sem inflar `StepContext` para o tamanho do próprio `self`. Métodos privados têm acesso direto a `self` — mais simples, igualmente testáveis.

**Passos:**

1. Extrair `_unpack_and_prepare(batch)`:
```python
def _unpack_and_prepare(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Retorna (images, masks, hard_masks).

    - Batch dict: usa image_key e mask_key do cfg.
    - Batch tuple: images=batch[0], masks=batch[1].
    - Soft labels (float, B,C,H,W): hard_masks via argmax.
    - Hard labels (int, B,H,W): hard_masks = masks.
    """
    ...
```

2. Extrair `_set_dual_head_context(hard_masks, batch)`:
```python
def _set_dual_head_context(self, hard_masks: Tensor, batch) -> None:
    """Popula self.model.last_hard_mask para dual-head loss."""
    if not self._is_dual_head:
        return
    self.model.last_hard_mask = batch.get("hard_mask", hard_masks).long()
```

3. Extrair `_adapt_output(raw_output, prefix)`:
```python
def _adapt_output(self, raw_output, prefix: str) -> Tuple[Any, Optional[Tensor]]:
    """
    EDL: extrai 'probs' para métricas, loga uncertainty.
    Outros: output_for_loss == output_for_metrics.

    Returns:
        (output_for_loss, output_for_metrics)
    """
    if isinstance(raw_output, dict) and "alpha" in raw_output:
        is_train = prefix == "train"
        uncertainty = raw_output["uncertainty"].mean()
        self.log(
            f"edl/{prefix}_uncertainty",
            uncertainty,
            on_step=is_train,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return raw_output, raw_output["probs"]
    return raw_output, raw_output
```

4. Extrair `_compute_final_loss(output_for_loss, masks, prefix)`:
```python
def _compute_final_loss(self, output_for_loss, masks: Tensor, prefix: str) -> Tensor:
    """
    Aplica em sequência: dual-head, OHEM, compute_loss base, MoE aux, MEDOE expert.
    Retorna loss total.
    """
    is_train = prefix == "train"

    if self._is_dual_head:
        loss = self.loss_function(output_for_loss, masks, epoch=self.current_epoch)
        self._individual_losses = {}
        self._extra_info = {}
        return loss

    if (
        hasattr(self.model, "compute_ohem_loss")
        and getattr(self.model, "ohem_ratio", 0) > 0
        and is_train
    ):
        loss = self.model.compute_ohem_loss(self.loss_function, output_for_loss, masks)
        self._individual_losses = {}
        self._extra_info = {}
    else:
        loss, self._individual_losses, self._extra_info = self._compute_loss(
            output_for_loss, masks
        )

    loss = self._add_moe_aux_loss(loss, prefix)
    loss = self._add_medoe_expert_loss(loss, masks, prefix)
    return loss

def _add_moe_aux_loss(self, loss: Tensor, prefix: str) -> Tensor:
    if not (hasattr(self.model, "last_aux_loss") and self.model.last_aux_loss is not None):
        return loss
    is_train = prefix == "train"
    moe_aux = self.model.last_aux_loss
    self.log(f"losses/{prefix}_moe_aux", moe_aux, on_step=is_train, on_epoch=True, sync_dist=True)
    return loss + moe_aux

def _add_medoe_expert_loss(self, loss: Tensor, hard_masks: Tensor, prefix: str) -> Tensor:
    if not (
        hasattr(self.model, "compute_expert_loss")
        and self.model.last_expert_outputs is not None
    ):
        return loss
    is_train = prefix == "train"
    expert_loss = self.model.compute_expert_loss(self.model.last_expert_outputs, hard_masks)
    self.log(f"losses/{prefix}_expert", expert_loss, on_step=is_train, on_epoch=True, sync_dist=True)
    return loss + expert_loss
```

5. Extrair `_log_step_results(loss, output_for_metrics, hard_masks, prefix)`:
```python
def _log_step_results(self, loss: Tensor, output_for_metrics, hard_masks: Tensor, prefix: str) -> None:
    """Loga: total loss, individual losses, extra info, MoE diag, MEDOE diag, dual-head Kendall, métricas, per-class IoU."""
    is_train = prefix == "train"

    self.log(f"loss/{prefix}", loss, on_step=is_train, on_epoch=True, prog_bar=True, sync_dist=True)

    for name, val in self._individual_losses.items():
        self.log(f"losses/{prefix}_{name}", val, on_step=is_train, on_epoch=True, sync_dist=False)

    for name, extra_dict in self._extra_info.items():
        for key, value in extra_dict.items():
            self.log(f"extra/{prefix}_{name}_{key}", value, on_step=is_train, on_epoch=True, sync_dist=False)

    self._log_moe_diagnostics(hard_masks, prefix)
    self._log_medoe_diagnostics(hard_masks, prefix)
    self._log_dual_head_diagnostics(prefix)
    self._log_metrics(output_for_metrics, hard_masks, prefix)
    self._log_per_class_iou(output_for_metrics, hard_masks, prefix)

def _log_moe_diagnostics(self, hard_masks: Tensor, prefix: str) -> None: ...
def _log_medoe_diagnostics(self, hard_masks: Tensor, prefix: str) -> None: ...
def _log_dual_head_diagnostics(self, prefix: str) -> None: ...
def _log_metrics(self, output_for_metrics, hard_masks: Tensor, prefix: str) -> None: ...
def _log_per_class_iou(self, output_for_metrics, hard_masks: Tensor, prefix: str) -> None: ...
```

6. Extrair `_cleanup_step_state()`:
```python
def _cleanup_step_state(self) -> None:
    """Zera tensores armazenados em self.model após o step."""
    if self._is_dual_head:
        if hasattr(self.model, "last_logits_A"):
            self.model.last_logits_A = None
            self.model.last_logits_B = None

    if hasattr(self.model, "last_expert_outputs"):
        self.model.last_expert_outputs = None
        self.model.last_gate_weights = None
        if hasattr(self.model, "_gate_weights_with_grad"):
            self.model._gate_weights_with_grad = None
```

7. `_shared_step` vira 8 linhas:
```python
def _shared_step(self, batch, prefix: str) -> Tensor:
    self._individual_losses = {}
    self._extra_info = {}
    images, masks, hard_masks = self._unpack_and_prepare(batch)
    self._set_dual_head_context(hard_masks, batch)
    raw_output = self(images)
    output_for_loss, output_for_metrics = self._adapt_output(raw_output, prefix)
    loss = self._compute_final_loss(output_for_loss, masks, prefix)
    self._log_step_results(loss, output_for_metrics, hard_masks, prefix)
    self._cleanup_step_state()
    return loss
```

**Testes novos:**

`tests/test_model_step_decomposition.py`:
- `_unpack_and_prepare` com batch dict → extrai images e masks corretamente.
- `_unpack_and_prepare` com batch tuple → idem.
- `_unpack_and_prepare` com soft labels (float, B,C,H,W) → `hard_masks` via argmax, ignore=255.
- `_unpack_and_prepare` com hard labels (long, B,H,W) → `hard_masks == masks`.
- `_set_dual_head_context` sem dual-head → noop.
- `_set_dual_head_context` com dual-head + `hard_mask` em batch → usa batch["hard_mask"].
- `_adapt_output` com output dict com "alpha" → retorna probs para métricas, loga uncertainty.
- `_adapt_output` com tensor → retorna tensor para loss e métricas.
- `_compute_final_loss` com dual-head → usa `loss_function(output, masks, epoch=...)`.
- `_compute_final_loss` com OHEM ativo em train → usa `compute_ohem_loss`.
- `_compute_final_loss` com OHEM ativo em val → não usa OHEM.
- `_add_moe_aux_loss` sem `last_aux_loss` → loss inalterada.
- `_add_moe_aux_loss` com `last_aux_loss` → loss somada, log feito.
- `_cleanup_step_state` com dual-head → zera `last_logits_A/B`.
- `_cleanup_step_state` com MEDOE → zera `last_expert_outputs/last_gate_weights`.
- Log keys snapshot: todas as chaves de `self.log(...)` durante step são idênticas ao snapshot da Fase 0.

`tests/test_model_edl_step.py`:
- Step com modelo EDL → loga `edl/train_uncertainty` e `edl/val_uncertainty`.
- Step com modelo EDL → métricas calculadas com `probs`, não com `alpha`.

`tests/test_model_dual_head_step.py`:
- Step com dual-head → `last_hard_mask` populado antes do forward.
- Step com dual-head → loga `kendall/train_sigma_hard`, `kendall/train_sigma_soft`, `kendall/train_sigma_consist`.
- Step com dual-head → `last_logits_A/B` zerados após step.

`tests/test_model_moe_step.py`:
- Step com MoE → loga `losses/train_moe_aux`.
- Step com MoE → diagnostics logados a cada 50 steps.

`tests/test_model_medoe_step.py`:
- Step com MEDOE → `expert_loss` somado ao total.
- Step com MEDOE → `last_expert_outputs` zerado após step.

**Testes antigos que protegem:**
- `tests/test_edl_training_smoke.py`
- `tests/test_dual_head.py`
- `tests/test_moe.py`
- `tests/test_medoe.py`
- `tests/test_model_training_step.py`
- `tests/test_model_base_comprehensive.py`

**Critério de saída:** `_shared_step` tem 8 linhas. Cada método privado tem 20–40 linhas. Log keys snapshot inalterado. Testes antigos passando. Novos testes com 100% coverage dos métodos extraídos.

---

### Fase 6: Correção de `config_definitions`

**Objetivo:** Corrigir divergência entre `TrainConfig.optimizer` (List) e runtime (objeto único). Sem migrar YAMLs.

**Passos:**

1. Auditar todos os YAMLs em `conf/examples/` e `tests/test_configs/` para documentar formato real de `optimizer` usado.

2. Alterar `train_config.py`:
```python
from typing import Union

@dataclass
class TrainConfig:
    ...
    # ANTES: optimizer: List[OptimizerConfig]
    # DEPOIS: aceita ambos durante transição
    optimizer: Union[OptimizerConfig, Any] = field(default_factory=OptimizerConfig)
```

3. Atualizar `TrainConfigValidator._check_optimizer_format`:
   - Se `optimizer` é lista de 1 elemento → warning, aceita.
   - Se `optimizer` é lista de múltiplos → erro.
   - Se `optimizer` é objeto com `_target_` → aceita.
   - Documentar que o formato lista é legado.

4. Atualizar docstring de `TrainConfig.optimizer` explicando o histórico.

**Testes novos:**

`tests/test_train_config_dataclass.py`:
- `TrainConfig` com `optimizer` como objeto único → serializável via `OmegaConf.structured`.
- `TrainConfig` com `optimizer` como lista de 1 elemento → validator emite warning.
- `TrainConfig` com `optimizer` como lista de múltiplos → `ConfigValidationError`.

**Testes antigos:**
- `tests/test_configs/*.py` — todos devem continuar passando.

**Critério de saída:** Divergência documentada e tratada com warning. Nenhum YAML existente quebra.

---

### Fase 7: Dataset Split

**Objetivo:** Reduzir `dataset.py` de 3034 linhas em módulos com responsabilidade única. Imports antigos preservados.

**Pré-requisito obrigatório:** Antes de mover qualquer código, rodar:
```bash
python -c "
import ast, sys
from pathlib import Path
tree = ast.parse(Path('pytorch_segmentation_models_trainer/dataset_loader/dataset.py').read_text())
imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
for i in imports:
    print(ast.dump(i))
" > /tmp/dataset_imports.txt
```
Verificar que nenhum utilitário importado em `dataset.py` importa de `dataset_loader` (circular imports).

**Estrutura alvo:**
```text
dataset_loader/
  __init__.py
  augmentations.py      # _sanitize_aug_config, load_augmentation_object
  base.py               # AbstractDataset, _worker_init_fn
  readers.py            # leitura PIL/rasterio (funções puras)
  segmentation.py       # SegmentationDataset e subclasses
  detection.py          # DetectionDataset e subclasses
  windowed.py           # WindowedDataset, GridDataset, CsvWindowedDataset
  raster_patch_dataset.py  # RasterPatchDataset (já separado)
  class_balancing.py    # ClassBalancingDataset e utils
  soft_labels.py        # SoftLabelDataset e utils
  collate.py            # collate_fn customizados
  dataset.py            # APENAS reexports de compat — não deletar por 2 releases
```

**Passos:**

1. Criar `augmentations.py`:
   - Mover `_sanitize_aug_config` e `load_augmentation_object`.
   - Adicionar em `dataset.py`: `from .augmentations import _sanitize_aug_config, load_augmentation_object`.

2. Criar `base.py`:
   - Mover `AbstractDataset` e `_worker_init_fn`.
   - Adicionar em `dataset.py`: `from .base import AbstractDataset, _worker_init_fn`.

3. Criar `readers.py`:
   - Extrair funções de leitura PIL e rasterio (funções que não dependem de estado de dataset).
   - Adicionar reexport em `dataset.py`.

4. Mover classes de segmentação para `segmentation.py` (uma por vez, verificando testes após cada).

5. Mover classes de detecção para `detection.py`.

6. Mover windowed/grid para `windowed.py`.

7. Mover class balancing para `class_balancing.py`.

8. Mover soft labels para `soft_labels.py`.

9. Mover collate_fn para `collate.py`.

10. `dataset.py` final: apenas imports de compat.

11. Atualizar imports internos do framework (não públicos) para usar novos módulos.

**Regra de cada passo:** Mover uma classe. Rodar testes. Verificar cobertura. Só avançar se verde.

**Testes novos:**

`tests/test_dataset_import_compat.py`:
```python
def test_old_imports_work():
    from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
        _sanitize_aug_config,
        load_augmentation_object,
        AbstractDataset,
        _worker_init_fn,
        SegmentationDataset,  # ou qualquer classe pública
    )
    assert _sanitize_aug_config is not None

def test_new_imports_work():
    from pytorch_segmentation_models_trainer.dataset_loader.augmentations import (
        _sanitize_aug_config,
        load_augmentation_object,
    )
    from pytorch_segmentation_models_trainer.dataset_loader.base import AbstractDataset

def test_hydra_target_old_path_still_works(minimal_cfg):
    # cfg com _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
    # deve instanciar corretamente após o split
    ...
```

`tests/test_dataset_augmentations.py`:
- `_sanitize_aug_config` com `always_apply=True` → `p=1.0`, sem `always_apply`.
- `_sanitize_aug_config` com `always_apply=False` → remove chave, sem `p`.
- `load_augmentation_object` com lista de dicts → `A.Compose`.
- `load_augmentation_object` com `A.Compose` já pronto → retorna mesmo objeto.
- `load_augmentation_object` com seed → `set_random_seed` chamado.
- `load_augmentation_object` com bbox_params → `A.Compose` com bbox_params.

`tests/test_dataset_readers.py`:
- Leitura PIL com path válido → PIL.Image.
- Leitura PIL com mask → L mode.
- Fallback para rasterio mockado.
- `root_dir` join com filename relativo.

**Testes antigos que protegem:**
- `tests/test_dataset.py`
- `tests/test_csv_windowed_dataset.py`
- `tests/test_csv_windowed_image_dataset.py`
- `tests/test_raster_patch_dataset.py`
- `tests/test_class_balancing.py`
- `tests/test_soft_labels.py`
- `tests/test_grid_mode.py`

**Critério de saída:** `dataset.py` tem apenas reexports. Novos módulos têm 100% coverage. Imports antigos funcionam. `_target_` antigos em YAML instanciam.

---

### Fase 8: Evaluation Pipeline Split

**Objetivo:** Extrair componentes de `EvaluationPipeline` mantendo facade. 1093 linhas → facade + 5 classes especializadas.

**Estrutura alvo:**
```text
tools/evaluation/
  evaluation_pipeline.py     # facade — EvaluationPipeline.run() inalterado
  dataset_preparer.py        # DatasetPreparer
  prediction_runner.py       # PredictionRunner
  prediction_validator.py    # PredictionValidator
  experiment_evaluator.py    # ExperimentEvaluator
  report_writer.py           # ReportWriter
```

**Passos:**

1. Criar `DatasetPreparer`: extrair `_prepare_dataset`.
2. Criar `PredictionRunner`: extrair lógica de predição e subprocessos.
3. Criar `PredictionValidator`: extrair validação de pastas/arquivos.
4. Criar `ExperimentEvaluator`: extrair cálculo de métricas.
5. Criar `ReportWriter`: extrair escrita de relatório.
6. `EvaluationPipeline.run()` instancia e chama cada componente em sequência.
7. API pública de `EvaluationPipeline` permanece idêntica.

**Regra:** Mover um componente por vez. Rodar `tests/test_evaluation_pipeline.py` após cada movimento.

**Testes novos:**

`tests/test_evaluation_dataset_preparer.py`:
- Prepara dataset com CSV válido → dataset instanciado.
- CSV inválido → erro claro.

`tests/test_prediction_runner.py`:
- Runner com modelo mock → predicoes geradas.
- Subprocesso → resultado capturado.

`tests/test_prediction_validator.py`:
- Pasta existente com arquivos esperados → OK.
- Pasta vazia → erro específico.

`tests/test_report_writer.py`:
- Métricas → arquivo CSV gerado.
- Métricas → visualizações geradas (mock de matplotlib).

**Testes antigos que protegem:**
- `tests/test_evaluation_pipeline.py`
- `tests/test_csv_builder.py`
- `tests/test_metrics_calculator.py`
- `tests/test_results_aggregator.py`
- `tests/test_gpu_distributor.py`

**Critério de saída:** `EvaluationPipeline.run()` inalterado. Novos componentes com testes independentes.

---

### Fase 9: CLI Simplificado em `main.py`

**Objetivo:** Substituir if/elif por dict literal. Sem criar pacote `cli/`. 9 modos registrados explicitamente.

**Passos:**

1. Adicionar dict `_COMMANDS` em `main.py` antes da função `main`.
2. `main(cfg)` usa `_COMMANDS.get(cfg.mode)`.
3. Raise `NotImplementedError` com lista de modos disponíveis quando modo desconhecido.
4. Imports lazy preservados (para não importar torch antes de rasterio no Windows — lógica existente).

**Implementação:**
```python
def _dispatch(cfg, module_name, fn_name="main", package="pytorch_segmentation_models_trainer"):
    """Import lazy e executa função."""
    import importlib
    mod = importlib.import_module(f"{package}.{module_name}")
    return getattr(mod, fn_name)(cfg)


_COMMANDS = {
    "train": lambda cfg: _dispatch(cfg, "train"),
    "predict": lambda cfg: _dispatch(cfg, "predict", "predict"),
    "predict-from-batch": lambda cfg: _dispatch(cfg, "predict_from_batch", "predict_from_batch"),
    "predict-mod-polymapper-from-batch": lambda cfg: _dispatch(
        cfg, "predict_mod_polymapper_from_batch", "predict_mod_polymapper_from_batch"
    ),
    "validate-config": lambda cfg: _dispatch(cfg, "config_utils", "validate_config"),
    "build-mask": lambda cfg: _dispatch(cfg, "build_mask", "build_masks"),
    "convert-dataset": lambda cfg: _dispatch(cfg, "convert_ds", "convert_dataset"),
    "evaluate-experiments": lambda cfg: _dispatch(cfg, "evaluate_experiments", "evaluate"),
    "run-experiments": lambda cfg: _dispatch(
        cfg, "tools.experiments_runner.experiments_runner", "ExperimentsRunner"
    )(cfg).run(),
}
```

**Testes novos:**

`tests/test_main_dispatch.py`:
- Cada modo existente está registrado em `_COMMANDS`.
- Modo desconhecido levanta `NotImplementedError` com mensagem contendo lista de modos.
- Mock de função registrada: `_COMMANDS["train"]` pode ser monkeypatched e chamado.

**Testes antigos:**
- `tests/test_main.py` — deve passar sem alteração.

**Critério de saída:** if/elif removido. Dict explícito. Testes antigos passando. Novo teste de dispatch com 100% coverage do dict.

---

### Fase 10: Docs, Examples, Changelog

**Obrigatório pelas regras do projeto.**

**Passos:**

1. Criar ou atualizar `website/docs/advanced/architecture.md`:
   - Diagrama de fluxo YAML → factories → DataModule → LightningModule.
   - Explicação de mixins e como adicionar comportamento novo.
   - Tabela de fases da refatoração e o que mudou em cada.

2. Atualizar `website/docs/advanced/config-reference.md` (ou equivalente):
   - Documentar que `optimizer` aceita objeto único (formato atual) ou lista de 1 elemento (legado).
   - Documentar que `steps_per_epoch` em `OneCycleLR` pode ser explícito ou auto-calculado.
   - Documentar config validation: quais erros agora têm mensagens claras.

3. Revisar todos os YAMLs em `conf/examples/` e garantir que continuam funcionando. Não adicionar campos novos.

4. Atualizar `CHANGELOG.md` em `# Unreleased` com entradas agrupadas:
   - `## Arquitetura`
   - `## Factories`
   - `## DataModules`
   - `## Config Validation`
   - `## Dataset Split`
   - `## Bug Fixes` (IO em setup, bypass de __init__)

---

## Ordem de PRs Recomendada

| PR | Fase | Escopo | Risco |
|----|------|--------|-------|
| 1 | 0 | Baseline coverage + snapshots | Baixo |
| 2 | 0.5 | Config validators | Baixo |
| 3 | 1 | Unificar dataloader boilerplate | Baixo |
| 4 | 2 | factories.py (funções) | Médio |
| 5 | 3 | DataModules externos + remoção IO de Model | Alto |
| 6 | 4 | Mixins + DomainAdaptationModel sem bypass | Alto |
| 7 | 5a | _unpack_and_prepare, _adapt_output, _set_dual_head_context | Médio |
| 8 | 5b | _compute_final_loss + _add_moe_aux + _add_medoe_expert | Médio |
| 9 | 5c | _log_step_results + _cleanup_step_state | Médio |
| 10 | 6 | Correção config_definitions.optimizer | Baixo |
| 11 | 7a | dataset split: augmentations + base + readers | Baixo |
| 12 | 7b | dataset split: segmentation + detection | Médio |
| 13 | 7c | dataset split: windowed + class_balancing + soft_labels | Médio |
| 14 | 8 | Evaluation pipeline split | Baixo |
| 15 | 9 | CLI dict inline | Baixo |
| 16 | 10 | Docs + changelog final | Baixo |

---

## Riscos e Mitigações

### Risco: quebrar YAML existente
**Mitigação:** Fase 0 cria `test_yaml_examples_loadable.py`. Cada PR roda esse teste. Imports antigos preservados via reexport em `dataset.py` por 2 releases.

### Risco: alterar log keys
**Mitigação:** Fase 0 cria snapshot JSON de todos os `self.log(key, ...)`. Fase 5 verifica snapshot inalterado. CI quebra se log key sumir ou mudar.

### Risco: IO em setup() quebrar em inferência
**Mitigação:** Fase 3 remove `_compute_steps_from_config` de `Model`. Teste que cria `Model(cfg, inference_mode=True)` sem CSV disponível.

### Risco: DDP multi-GPU com DataModule externo
**Mitigação:** `SegmentationDataModule.setup()` respeita `stage` parameter do Lightning — correto para DDP. Adicionar teste com `trainer.num_devices=2` mockado.

### Risco: `save_hyperparameters` com Mixin não picklable
**Mitigação:** Mixins não têm estado próprio — são apenas métodos. `save_hyperparameters` salva `cfg`, não os mixins. Não há risco.

### Risco: circular imports no dataset split
**Mitigação:** Pré-requisito obrigatório na Fase 7: mapear imports antes de mover qualquer arquivo. Confirmado: `utils/polygonrnn_utils`, `utils/object_detection_utils`, `utils/dataframe_utils` não importam de `dataset_loader`.

### Risco: aumentar complexidade aparente
**Mitigação:** Factories são um arquivo (`factories.py`), não um pacote. Mixins são um arquivo (`mixins.py`). DataModules são um arquivo (`datamodules.py`). Nenhum pacote novo desnecessário.

### Risco: mudar ordem de cálculo de loss
**Mitigação:** Fase 5 decompõe `_shared_step` por extração mecânica — não muda ordem de chamadas. Testes de equivalência numérica: antes e depois produzem mesmo loss para mesma entrada.

---

## Política de Coverage

1. Antes da Fase 1, medir e salvar baseline em `coverage_baseline.txt`.
2. Cada PR roda:
```bash
uv run pytest tests/ -v --tb=short \
  --ignore=tests/test_detection_model.py \
  --ignore=tests/test_inference.py \
  --ignore=tests/test_predict.py \
  --ignore=tests/test_script.py \
  --cov=pytorch_segmentation_models_trainer \
  --cov-report=term-missing \
  --cov-fail-under=<baseline>
```
3. Módulos novos devem ter 100% de cobertura nos seus próprios testes.
4. Se código antigo movido reduzir coverage global, adicionar testes na mesma PR.
5. Não fazer squash de commits que reduzem coverage — investigar antes de mergear.

---

## Critério de Conclusão

Refatoração só é considerada concluída quando:

- [ ] YAMLs antigos continuam funcionando sem alteração.
- [ ] Nenhum YAML novo é necessário para ativar a arquitetura refatorada.
- [ ] `Model` não contém mais lógica específica de EDL/MoE/MEDOE/dual-head inline em `_shared_step`.
- [ ] `Model._shared_step` tem menos de 15 linhas.
- [ ] `DomainAdaptationModel` não herda de `Model`.
- [ ] `DomainAdaptationModel.__init__` não usa `pl.LightningModule.__init__` como workaround.
- [ ] Datasets principais não vivem em um único arquivo monolítico.
- [ ] `_compute_steps_from_config` não existe em `Model` (movido para DataModule).
- [ ] `Model` não chama `pd.read_csv` em nenhum método.
- [ ] Factories são funções em `factories.py`, não classes.
- [ ] DataModules são criados em `train.py`, não dentro de `Model.__init__`.
- [ ] `train.py` integra `TrainConfigValidator` antes de instanciar qualquer coisa.
- [ ] Coverage global não diminuiu em relação ao baseline.
- [ ] Testes antigos passam sem alteração relevante.
- [ ] Testes novos cobrem: factories, datamodules, validators, mixins, step decomposition, import compat.
- [ ] Log keys snapshot inalterado (CI verifica automaticamente).
- [ ] Docs em `website/docs/` explicam arquitetura nova.
- [ ] `CHANGELOG.md` tem entrada em `# Unreleased`.

---

## Comparação com Plano Anterior

| Decisão | REFACTOR.md (Codex) | REFACTOR_CLAUDE.md (este) | Motivo |
|---|---|---|---|
| Factories | Classes com método `build()` | Funções em módulo único | Classes sem estado e sem DI são overhead sem benefício |
| StepHandlers | Protocol com 3 hooks + StepContext dataclass | Métodos privados em `Model` | Protocol não cobre todos os pontos de extensão sem inflar StepContext |
| DataModule | Interno ao Model (`self._data_module`) | Externo (criado em `train.py`) | Interno não remove responsabilidade, apenas esconde |
| Herança DA | Nova `BaseSegmentationLightningModule` | Mixins independentes | Herança para resolver problema de herança é recursivo |
| Validators | Fase 5 (depois de factories) | Fase 0.5 (antes de factories) | Validators protegem a refatoração, não documentam depois |
| IO em Model | Não mencionado | Removido em Fase 3 | `pd.read_csv` em `setup()` é anti-pattern grave |
| CLI | Pacote `cli/` com registry | Dict inline em `main.py` | 9 entradas não justificam pacote separado |
| Dataset split | Move código | Mapeia deps antes de mover | Circular imports são risco real |
| Fase 1 | Factories direto | Unificar dataloader primeiro | Mover código ruim para factory perpetua o problema |
