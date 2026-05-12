# Plano de Refatoracao Arquitetural

## Contexto

O framework tem como requisito primordial a instanciacao dinamica via YAML/Hydra. Este plano preserva esse requisito: YAML continua sendo o contrato publico, `_target_` continua sendo o mecanismo principal de extensao, e `Model` continua existindo como `LightningModule` padrao.

Requisito absoluto de retrocompatibilidade: a refatoracao nao pode mudar a estrutura dos YAMLs existentes em hipotese alguma. Nenhum campo novo pode ser exigido. Nenhuma chave existente pode mudar de lugar, nome, tipo publico ou semantica. A arquitetura interna deve se adaptar ao YAML atual, nao o contrario.

A mudanca proposta nao remove dinamismo. Ela separa responsabilidades hoje concentradas em `Model`, datasets e pipelines grandes, para que a dinamica passe por factories, validadores e contratos explicitos.

Nota: `AGENTS.md` referencia `@RTK.md`, mas `RTK.md` nao foi encontrado no repositorio nem no diretorio pai durante esta analise. Nao sei se ha instrucoes adicionais fora deste checkout.

## Objetivos

1. Manter compatibilidade total com YAMLs existentes.
2. Reduzir acoplamento de `Model` sem quebrar sua API publica.
3. Preservar instanciacao dinamica via Hydra.
4. Transformar comportamentos implicitos baseados em `hasattr` em contratos explicitos.
5. Separar dados, factories, otimizacao, scheduler, loss, metricas e step logic.
6. Permitir evolucao de features como EDL, MoE, MEDOE, dual-head, OHEM e Domain Adaptation sem editar um metodo central gigante.
7. Garantir que coverage global nao diminua.
8. Nao alterar estrutura de YAML sob nenhuma hipotese durante a refatoracao.

## Diagnostico Atual

### `model_loader/model.py`

Responsabilidades atuais:

- Instancia modelo via Hydra.
- Instancia datasets.
- Constroi dataloaders.
- Constroi loss.
- Constroi optimizer e schedulers.
- Aplica fine tuning.
- Aplica GPU augmentations.
- Computa `steps_per_epoch`.
- Processa batch.
- Calcula loss.
- Trata outputs especiais de EDL.
- Trata dual-head.
- Trata OHEM.
- Trata MoE e MEDOE.
- Loga metricas, losses e diagnosticos.
- Aplica TTA em `test_step`.

Problema: `Model` virou orquestrador e implementacao de dominio ao mesmo tempo. A instanciacao dinamica via YAML e correta, mas o local onde a dinamica e resolvida esta concentrado demais.

### `model_loader/domain_adaptation_model.py`

Pontos positivos:

- Usa `BaseDomainAdaptationMethod` como estrategia.
- Mantem metodo de DA como `nn.Module`, separado do `LightningModule`.
- Tem contrato de retorno `DomainAdaptationLossOutput`.

Problema:

- Herda `Model`, mas precisa burlar `Model.__init__` chamando `pl.LightningModule.__init__` diretamente. Isso indica que a classe base assume estrutura de config/dataset que nem todo workflow usa.

### `dataset_loader/dataset.py`

Responsabilidades atuais:

- Base dataset.
- Leitura de imagem via PIL/rasterio.
- Leitura de metadata.
- Augmentation.
- Segmentacao.
- Deteccao.
- Window/crop/grid.
- Class balancing.
- Soft labels.
- Utilidades geoespaciais.

Problema: arquivo com muitas razoes para mudar. Dificulta teste isolado e reutilizacao.

### `tools/evaluation/evaluation_pipeline.py`

Responsabilidades atuais:

- Preparar dataset.
- Rodar predicoes.
- Validar predicoes.
- Calcular metricas.
- Agregar resultados.
- Gerar visualizacoes.
- Escrever relatorios.
- Controlar paralelismo/subprocessos.

Problema: `EvaluationPipeline` e facade util, mas acumula detalhes executaveis demais.

## Arquitetura Alvo

### Visao Geral

Fluxo alvo:

```text
YAML Hydra
  -> ConfigValidator
  -> Factories
     -> ModelFactory
     -> LossFactory
     -> MetricFactory
     -> OptimizerFactory
     -> SchedulerFactory
     -> DataModuleFactory
     -> StepHandlerFactory
  -> LightningModule fino
  -> DataModule
  -> Trainer
```

O YAML continua escolhendo implementacoes dinamicamente usando exatamente as chaves atuais:

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model

model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34

loss:
  _target_: torch.nn.CrossEntropyLoss
```

Handlers, factories e DataModules sao detalhes internos. Eles devem ser selecionados automaticamente a partir das chaves ja existentes (`model`, `loss`, `loss_params`, `train_dataset`, `val_dataset`, `test_dataset`, `domain_adaptation`, etc.). Nao adicionar `step_handlers`, `data_module`, `factory`, `capabilities` ou qualquer nova secao ao YAML como parte desta refatoracao.

## Modulos Novos

### `pytorch_segmentation_models_trainer/factories/`

#### Arquivos

- `factories/__init__.py`
- `factories/model_factory.py`
- `factories/loss_factory.py`
- `factories/metric_factory.py`
- `factories/optimizer_factory.py`
- `factories/scheduler_factory.py`
- `factories/callback_factory.py`

#### Motivo

Concentrar regras de instanciacao dinamica em componentes pequenos. `Model` continua consumindo YAML, mas nao precisa conhecer detalhes de construcao de tudo.

#### Contratos

```python
class ModelFactory:
    def build(self, cfg: DictConfig) -> nn.Module: ...

class LossFactory:
    def build(self, cfg: DictConfig) -> nn.Module: ...

class OptimizerFactory:
    def build(self, cfg: DictConfig, params) -> torch.optim.Optimizer: ...

class SchedulerFactory:
    def build(self, cfg: DictConfig, optimizer) -> list: ...
```

#### Regras

- Factories aceitam `DictConfig`.
- Factories usam `hydra.utils.instantiate`.
- Factories devem preservar comportamento atual.
- Casos especiais existentes entram primeiro como codigo movido, nao reescrito.
- Backward compatibility primeiro, limpeza depois.
- Factories nao introduzem novas chaves YAML. Toda decisao deve usar campos existentes.

### `pytorch_segmentation_models_trainer/data/`

#### Arquivos

- `data/__init__.py`
- `data/datamodules.py`
- `data/dataloader_factory.py`
- `data/batch.py`

#### Motivo

Separar dados de `LightningModule`. Hoje `Model` instancia datasets e dataloaders. Isso impede reutilizacao por workflows como Domain Adaptation.

#### Classes

```python
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig): ...
    def setup(self, stage=None): ...
    def train_dataloader(self): ...
    def val_dataloader(self): ...
    def test_dataloader(self): ...

class DomainAdaptationDataModule(pl.LightningDataModule):
    def train_dataloader(self) -> CombinedLoader: ...
    def val_dataloader(self): ...
```

#### Compatibilidade

`Model.train_dataloader()` continua existindo durante transicao, mas delega para um DataModule interno privado. O YAML nao ganha chave `data_module`; a criacao acontece automaticamente quando `train_dataset`, `val_dataset` ou `test_dataset` existem.

### `pytorch_segmentation_models_trainer/training/`

#### Arquivos

- `training/__init__.py`
- `training/contracts.py`
- `training/step_context.py`
- `training/step_result.py`
- `training/step_engine.py`
- `training/step_handlers.py`
- `training/output_adapters.py`
- `training/capabilities.py`

#### Motivo

Remover ifs especiais de `_shared_step` e trocar introspeccao implicita por contratos.

#### Contratos

```python
@dataclass
class StepContext:
    pl_module: pl.LightningModule
    batch: Any
    prefix: str
    images: torch.Tensor
    masks: torch.Tensor | dict | None
    hard_masks: torch.Tensor | None
    raw_output: Any
    output_for_loss: Any
    output_for_metrics: torch.Tensor | None
    loss: torch.Tensor | None = None
    individual_losses: dict = field(default_factory=dict)
    extra_info: dict = field(default_factory=dict)

class StepHandler(Protocol):
    def before_loss(self, context: StepContext) -> StepContext: ...
    def after_loss(self, context: StepContext) -> StepContext: ...
    def after_metrics(self, context: StepContext) -> StepContext: ...
```

#### Handlers iniciais

- `BaseLossHandler`
- `EDLOutputHandler`
- `DualHeadHandler`
- `OHEMHandler`
- `MoEAuxLossHandler`
- `MoEDiagnosticsHandler`
- `MEDOEHandler`
- `CompoundLossLoggingHandler`
- `MetricLoggingHandler`
- `PerClassIoUHandler`

#### Regra de transicao

Primeira fase move codigo atual para handlers preservando output/log keys. Refatoracao comportamental fica proibida nessa fase.

### `pytorch_segmentation_models_trainer/config_validation/`

#### Arquivos

- `config_validation/__init__.py`
- `config_validation/train_validator.py`
- `config_validation/domain_adaptation_validator.py`
- `config_validation/registry.py`

#### Motivo

Falhar cedo quando YAML esta inconsistente. Hoje varios erros aparecem tarde, dentro de treino.

#### Validacoes iniciais

- `cfg.model._target_` importavel.
- Exatamente uma forma valida de loss: `loss`, `loss_params.compound_loss` ou `loss_params.multi_loss`.
- `optimizer` aceita formato historico real e formato dataclass.
- `scheduler_list` com `OneCycleLR` exige `steps_per_epoch` calculavel ou explicito.
- Dataset configs precisam `_target_` quando presentes.
- Validadores nao devem exigir campos novos. Eles apenas validam estrutura atual e produzem mensagens melhores.

### `pytorch_segmentation_models_trainer/cli/`

#### Arquivos

- `cli/__init__.py`
- `cli/registry.py`
- `cli/commands.py`

#### Motivo

Substituir cadeia `if/elif` em `main.py` por registry. Facilita novos modos sem editar `main.py`.

#### Contrato

```python
COMMAND_REGISTRY = {
    "train": train,
    "predict": predict,
    ...
}

def dispatch(cfg):
    handler = COMMAND_REGISTRY.get(cfg.mode)
    if handler is None:
        raise NotImplementedError(...)
    return handler(cfg)
```

## Modulos Que Devem Mudar

### `model_loader/model.py`

#### Mudanca

Transformar `Model` em orquestrador fino.

#### Antes

`Model` implementa construcao, dados, loss, optimizacao, scheduler e step completo.

#### Depois

`Model`:

- recebe `cfg`;
- chama `ModelFactory`;
- chama `LossFactory`;
- cria ou recebe `DataModule`;
- delega optimizer/scheduler para factories;
- delega `_shared_step` para `StepEngine`;
- mantem metodos publicos existentes.

#### Passos

1. Criar factories copiando logica atual sem mudar comportamento.
2. Alterar `Model.get_model()` para usar `ModelFactory`.
3. Alterar `Model.get_loss_function()` para usar `LossFactory`.
4. Alterar `Model.get_optimizer()` para usar `OptimizerFactory`.
5. Alterar `Model.configure_optimizers()` para usar `SchedulerFactory`.
6. Criar `StepEngine` com codigo movido de `_shared_step`.
7. Fazer `_shared_step` chamar `self.step_engine.run(batch, prefix)`.
8. Manter wrappers antigos por uma versao:
   - `_compute_loss`
   - `_unpack_batch`
   - `_prepare_preds_for_metrics`
   - `_soft_to_hard_masks`
9. Marcar internamente pontos candidatos a deprecacao, sem warning publico ainda.

### `model_loader/domain_adaptation_model.py`

#### Mudanca

Remover dependencia rigida do `Model.__init__`.

#### Passos

1. Criar base compartilhada `BaseSegmentationLightningModule`.
2. Mover funcoes comuns de `Model` para base ou mixins:
   - model/loss factories;
   - metric setup;
   - loss computation;
   - optimizer/scheduler hooks.
3. Fazer `Model` herdar base e usar `SegmentationDataModule`.
4. Fazer `DomainAdaptationModel` herdar base diretamente.
5. Fazer `DomainAdaptationModel` usar `DomainAdaptationDataModule`.
6. Preservar `BaseDomainAdaptationMethod` e `DomainAdaptationLossOutput`.

### `dataset_loader/dataset.py`

#### Mudanca

Fatiar arquivo monolitico progressivamente.

#### Estrutura alvo

```text
dataset_loader/
  __init__.py
  base.py
  readers.py
  augmentations.py
  segmentation.py
  detection.py
  windowed.py
  raster_patch_dataset.py
  class_balancing.py
  soft_labels.py
  collate.py
  dataset.py              # compat imports temporarios
```

#### Passos

1. Criar `augmentations.py` e mover `_sanitize_aug_config`, `load_augmentation_object`.
2. Criar `base.py` e mover `AbstractDataset`.
3. Criar `readers.py` para PIL/rasterio path loading.
4. Mover classes uma a uma mantendo imports antigos em `dataset.py`.
5. Adicionar testes para imports antigos e novos.
6. Depois de duas releases, avaliar deprecacao dos imports antigos.

### `tools/evaluation/evaluation_pipeline.py`

#### Mudanca

Manter `EvaluationPipeline` como facade, extrair executores.

#### Estrutura alvo

```text
tools/evaluation/
  evaluation_pipeline.py     # facade
  dataset_preparer.py
  prediction_runner.py
  prediction_validator.py
  experiment_evaluator.py
  report_writer.py
```

#### Passos

1. Extrair `_prepare_dataset` para `DatasetPreparer`.
2. Extrair predicao/subprocessos para `PredictionRunner`.
3. Extrair validacao de pastas/arquivos para `PredictionValidator`.
4. Extrair escrita de relatorio para `ReportWriter`.
5. Manter API `EvaluationPipeline.run()` igual.

### `main.py`

#### Mudanca

Trocar dispatch manual por registry.

#### Passos

1. Criar `cli/registry.py`.
2. Mover mapping de modos para registry.
3. `main(cfg)` chama `dispatch(cfg)`.
4. Manter testes atuais de `test_main.py`.
5. Adicionar teste para registro de novo comando fake.

### `config_definitions/`

#### Mudanca

Alinhar dataclasses com YAML real.

#### Passos

1. Auditar campos usados por YAMLs em `conf/examples/` e `tests/test_configs/`.
2. Corrigir divergencia `TrainConfig.optimizer`: hoje dataclass sugere lista, runtime usa objeto.
3. A correcao deve refletir a estrutura YAML ja usada no projeto. Nao migrar YAMLs para novo formato.
4. Criar tipo flexivel durante transicao:
   - aceitar objeto unico;
   - aceitar lista historica se existir uso real.
5. Adicionar validadores para erro claro.
6. Atualizar docs sem alterar exemplos para nova estrutura.

## Plano Faseado

### Fase 0: Baseline e Guard Rails

Objetivo: medir antes de mexer.

Passos:

1. Rodar suite padrao:

```bash
uv run pytest tests/ -v --tb=short \
  --ignore=tests/test_detection_model.py \
  --ignore=tests/test_inference.py \
  --ignore=tests/test_predict.py \
  --ignore=tests/test_script.py
```

2. Rodar coverage baseline:

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

3. Salvar percentual global e arquivos com baixa cobertura.
4. Configurar CI para falhar com `--cov-fail-under=<baseline>`.
5. Adicionar snapshot de logs/keys para `Model._shared_step` antes da refatoracao.

### Fase 1: Factories Sem Mudanca Funcional

Objetivo: mover construcao para factories mantendo API.

Passos:

1. Criar pacote `factories`.
2. Mover logica de `get_model` para `ModelFactory`.
3. Mover logica de `get_loss_function` para `LossFactory`.
4. Mover logica de `get_optimizer` e `_get_param_groups_with_layer_decay` para `OptimizerFactory`.
5. Mover logica de `configure_optimizers` para `SchedulerFactory`.
6. `Model` chama factories.
7. Manter metodos antigos como delegadores.

Testes novos:

- `tests/test_model_factory.py`
  - instancia `torch.nn.Conv2d` via YAML;
  - aplica `replace_model_activation`;
  - aplica `fine_tuning` com mock;
  - preserva `inference_mode`.
- `tests/test_loss_factory.py`
  - `cfg.loss`;
  - `loss_params.compound_loss`;
  - `loss_params.multi_loss`;
  - erro sem loss.
- `tests/test_optimizer_factory.py`
  - optimizer simples;
  - layer-wise LR decay;
  - parametros congelados ignorados;
  - config object/list quando aplicavel.
- `tests/test_scheduler_factory.py`
  - sem scheduler;
  - `OneCycleLR` com `steps_per_epoch` explicito;
  - `OneCycleLR` auto;
  - warmup + scheduler normal.

Testes antigos que cobrem parte:

- `tests/test_model_base_comprehensive.py`
- `tests/test_model.py`
- `tests/test_losses_callbacks_tta.py`
- `tests/test_loss_builder.py`

Lacuna:

- Testes antigos cobrem comportamento via `Model`, mas nao contratos das factories. Precisam testes novos para manter coverage e facilitar falhas localizadas.

### Fase 2: DataModules e DataLoaderFactory

Objetivo: remover responsabilidade de dados do `Model`.

Passos:

1. Criar `DataloaderFactory`.
2. Mover `_prefetch_factor` e `_make_dataloader_generator`.
3. Criar `SegmentationDataModule`.
4. `Model.__init__` instancia `self._data_module` interno quando datasets existem.
5. `Model.train_dataloader/val_dataloader/test_dataloader` delegam.
6. Criar `DomainAdaptationDataModule`.
7. `DomainAdaptationModel` delega dataloaders ao novo module.

Testes novos:

- `tests/test_dataloader_factory.py`
  - `num_workers=0` remove `prefetch_factor`;
  - `num_workers>0` preserva `prefetch_factor`;
  - `persistent_workers`, `pin_memory`, `drop_last`;
  - `collate_fn` vindo do dataset;
  - generator deterministico com `seed`.
- `tests/test_segmentation_datamodule.py`
  - instancia train/val/test datasets via YAML;
  - retorna `None` quando val/test ausente;
  - respeita batch size e shuffle.
- `tests/test_domain_adaptation_datamodule.py`
  - train retorna `CombinedLoader`;
  - source/target keys preservadas;
  - val source-only, target-only, both.

Testes antigos que cobrem parte:

- `tests/test_model_training_step.py`
- `tests/test_model_test_dataset.py`
- `tests/test_seed_utils.py`
- `tests/test_domain_adaptation_model.py`
- `tests/test_dann_method.py`

Lacuna:

- Testes antigos validam dataloaders indiretamente via `Model` e `DomainAdaptationModel`. Nao cobrem DataModule como unidade isolada.

### Fase 3: StepEngine e StepHandlers

Objetivo: desmontar `_shared_step` sem mudar comportamento.

Passos:

1. Criar `StepContext` e `StepResult`.
2. Criar `StepEngine.run(batch, prefix)`.
3. Mover `_unpack_batch`, soft-label handling e forward para engine ou helpers.
4. Mover EDL para `EDLOutputHandler`.
5. Mover dual-head para `DualHeadHandler`.
6. Mover OHEM para `OHEMHandler`.
7. Mover MoE/MEDOE para handlers.
8. Mover logging de losses/extra/metrics para handlers.
9. `Model._shared_step` vira delegador.
10. Preservar keys de log exatamente.
11. Selecionar handlers internamente por comportamento ja existente:
   - EDL: output dict com `"alpha"`, como hoje;
   - dual-head: `loss_function.is_dual_head_loss`, como hoje;
   - OHEM: `model.compute_ohem_loss` e `model.ohem_ratio`, como hoje;
   - MoE/MEDOE: atributos/metodos existentes, encapsulados em adapters.
12. Nao adicionar `step_handlers` ao YAML.

Testes novos:

- `tests/test_step_engine.py`
  - retorna scalar loss;
  - suporta batch dict default;
  - suporta custom `image_key`/`mask_key`;
  - suporta tuple batch;
  - soft labels viram hard masks para metricas.
- `tests/test_step_handlers_edl.py`
  - dict com `alpha` usa `probs` para metricas;
  - loga `edl/train_uncertainty` e `edl/val_uncertainty`;
  - loss recebe output EDL original.
- `tests/test_step_handlers_dual_head.py`
  - seta `last_hard_mask`;
  - limpa `last_logits_A/B`;
  - loga Kendall diagnostics.
- `tests/test_step_handlers_moe_medoe.py`
  - soma `last_aux_loss`;
  - chama diagnostics;
  - limpa tensores MEDOE.
- `tests/test_step_handler_order.py`
  - ordem de handlers preserva loss final;
  - handlers default sao inferidos sem nenhuma chave YAML nova.

Testes antigos que cobrem parte:

- `tests/test_edl_training_smoke.py`
- `tests/test_dual_head.py`
- `tests/test_moe.py`
- `tests/test_medoe.py`
- `tests/test_model_training_step.py`
- `tests/test_model_base_comprehensive.py`

Lacuna:

- Testes antigos cobrem efeitos finais, nao ordem/contrato dos handlers. Precisam testes novos para evitar regressao interna.

### Fase 4: Base Lightning Compartilhada

Objetivo: remover necessidade de `DomainAdaptationModel` burlar `Model.__init__`.

Passos:

1. Criar `model_loader/base_lightning.py`.
2. Mover setup comum:
   - seed;
   - model factory;
   - loss factory;
   - metrics;
   - `save_hyperparameters`;
   - optimizer/scheduler;
   - `_compute_loss`;
   - `_prepare_preds_for_metrics`.
3. `Model(BaseSegmentationLightningModule)`.
4. `DomainAdaptationModel(BaseSegmentationLightningModule)`.
5. Garantir que `DomainAdaptationModel.__init__` nao chama mais `pl.LightningModule.__init__` diretamente.

Testes novos:

- `tests/test_base_lightning.py`
  - setup de metrics;
  - `_compute_loss` simples/compound;
  - optimizer/scheduler delegados;
  - checkpoint hyperparams sem objetos pesados.
- Atualizar `tests/test_domain_adaptation_model.py`
  - assert nao depende de bypass;
  - comportamento de training/validation igual.

Testes antigos que cobrem parte:

- `tests/test_domain_adaptation_model.py`
- `tests/test_da_model_comprehensive.py`
- `tests/test_dann_method.py`
- `tests/test_model_base_comprehensive.py`

Lacuna:

- Testes antigos nao protegem desenho de heranca. Precisam asserts diretos sobre base compartilhada.

### Fase 5: Config Validation

Objetivo: falhar cedo em YAML invalido sem limitar dinamismo.

Passos:

1. Criar validators.
2. Chamar validator em `train()` antes de instanciar model.
3. Chamar validator em `DomainAdaptationModel` ou no fluxo DA.
4. Adicionar modo existente `validate-config` usando mesmos validators.
5. Mensagens de erro devem apontar caminho YAML.

Testes novos:

- `tests/test_train_config_validator.py`
  - model target ausente;
  - loss ausente;
  - loss multipla ambigua;
  - optimizer formato legado aceito;
  - scheduler OneCycle sem steps calculavel falha cedo;
  - nenhuma chave nova e exigida para validação.
- `tests/test_domain_adaptation_config_validator.py`
  - source/target dataset obrigatorios;
  - method obrigatorio;
  - feature layer vazio com method que requer features gera warning ou erro configuravel.

Testes antigos que cobrem parte:

- `tests/test_config*.py`
- `tests/test_validate_evaluation_config.py`
- `tests/test_domain_adaptation_config.py`

Lacuna:

- Testes antigos cobrem dataclasses, nao validacao semantica cross-field.

### Fase 6: Dataset Split

Objetivo: reduzir `dataset.py` com compatibilidade.

Passos:

1. Mover augmentations.
2. Mover base dataset.
3. Mover readers.
4. Mover classes por grupo.
5. `dataset.py` reexporta nomes antigos.
6. Atualizar imports internos para novos modulos.
7. Manter imports publicos antigos funcionando.

Testes novos:

- `tests/test_dataset_import_compat.py`
  - imports antigos funcionam;
  - imports novos funcionam;
  - Hydra `_target_` antigo ainda instancia.
- `tests/test_dataset_readers.py`
  - PIL image;
  - PIL mask;
  - rasterio fallback mockado;
  - root_dir path join.
- `tests/test_dataset_augmentations.py`
  - `always_apply` sanitizado;
  - `A.Compose` com seed;
  - bbox params.

Testes antigos que cobrem parte:

- `tests/test_dataset.py`
- `tests/test_csv_windowed_dataset.py`
- `tests/test_csv_windowed_image_dataset.py`
- `tests/test_raster_patch_dataset.py`
- `tests/test_class_balancing.py`
- `tests/test_soft_labels.py`
- `tests/test_grid_mode.py`

Lacuna:

- Testes antigos cobrem comportamento de datasets, mas nao compatibilidade de import/reexport apos split.

### Fase 7: Evaluation Pipeline Split

Objetivo: extrair etapas preservando facade.

Passos:

1. Criar classes auxiliares.
2. Mover uma etapa por vez.
3. `EvaluationPipeline.run()` permanece contrato publico.
4. Manter config igual.

Testes novos:

- `tests/test_evaluation_dataset_preparer.py`
- `tests/test_prediction_runner.py`
- `tests/test_prediction_validator.py`
- `tests/test_report_writer.py`

Testes antigos que cobrem parte:

- `tests/test_evaluation_pipeline.py`
- `tests/test_csv_builder.py`
- `tests/test_metrics_calculator.py`
- `tests/test_results_aggregator.py`
- `tests/test_gpu_distributor.py`

Lacuna:

- Testes antigos cobrem facade, nao unidades extraidas.

### Fase 8: CLI Registry

Objetivo: extensibilidade de modos.

Passos:

1. Criar `cli/registry.py`.
2. Registrar modos existentes.
3. `main.py` chama `dispatch(cfg)`.
4. Preservar `entry()`.

Testes novos:

- `tests/test_cli_registry.py`
  - cada modo existente registrado;
  - modo desconhecido levanta `NotImplementedError`;
  - comando fake pode ser registrado em teste.

Testes antigos que cobrem parte:

- `tests/test_main.py`

Lacuna:

- `test_main.py` valida dispatch final, nao registry isolado.

### Fase 9: Docs, Examples, Changelog

Obrigatorio por regra do projeto.

Passos:

1. Atualizar `website/docs/advanced/architecture.md` ou criar se ausente.
2. Atualizar docs de configuracao com:
   - factories;
   - data modules;
   - step handlers;
   - config validation.
3. Revisar exemplos YAML existentes apenas para garantir que continuam validos. Nao adicionar exemplos que introduzam estrutura YAML nova para esta refatoracao.
4. Atualizar `CHANGELOG.md` em `# Unreleased`.

## Compatibilidade YAML

### Regras

1. Todo YAML existente deve continuar funcionando sem alteracao.
2. Nenhuma nova secao YAML deve ser adicionada como parte da refatoracao.
3. Nenhuma chave existente deve mudar de nome, lugar, tipo publico ou semantica.
4. Defaults internos devem reproduzir comportamento atual.
5. Imports antigos continuam via reexport.
6. Mudancas incompatíveis ficam fora do escopo desta refatoracao.
7. Novas abstrações internas devem ser inferidas a partir da estrutura YAML atual.

### Campos Novos

Nao propor campos novos nesta refatoracao. Especificamente, nao adicionar:

- `step_handlers`
- `data_module`
- `factories`
- `capabilities`
- `config_validation`

Se algum mecanismo precisar dessas informacoes, deve deriva-las internamente do YAML atual ou de contratos Python existentes.

## Analise: Testes Antigos Cobrem a Refatoracao?

Resposta curta: cobrem comportamento final em boa parte, mas nao cobrem arquitetura nova. Nao sao suficientes sozinhos.

### Cobertura forte existente

- `Model` init, optimizer, scheduler, `_shared_step`, dataloaders:
  - `tests/test_model.py`
  - `tests/test_model_base_comprehensive.py`
  - `tests/test_model_training_step.py`
  - `tests/test_model_methods.py`
  - `tests/test_model_test_dataset.py`
- EDL:
  - `tests/test_edl_training_smoke.py`
  - `tests/test_edl_wrapper.py`
  - `tests/test_edl_loss.py`
- Dual-head:
  - `tests/test_dual_head.py`
- MoE/MEDOE:
  - `tests/test_moe.py`
  - `tests/test_medoe.py`
- Domain Adaptation:
  - `tests/test_domain_adaptation_model.py`
  - `tests/test_da_model_comprehensive.py`
  - `tests/test_dann_method.py`
  - `tests/test_base_method.py`
- Datasets:
  - `tests/test_dataset.py`
  - `tests/test_grid_mode.py`
  - `tests/test_class_balancing.py`
  - `tests/test_soft_labels.py`
  - `tests/test_raster_patch_dataset.py`
- Evaluation:
  - `tests/test_evaluation_pipeline.py`
  - `tests/test_metrics_calculator.py`
  - `tests/test_results_aggregator.py`
  - `tests/test_gpu_distributor.py`

### Lacunas

- Factories isoladas nao existem, logo nao ha testes de contrato delas.
- DataModules isolados nao existem.
- Step handlers nao existem.
- Capabilities/protocols nao existem.
- Config validation semantica cross-field e limitada.
- Import compatibility apos split de datasets precisa teste novo.
- CLI registry nao existe.

### Conclusao de cobertura

Testes antigos devem ser mantidos como suite de regressao. Eles validam que refatoracao nao mudou comportamento publico. Mas cada modulo novo precisa testes unitarios proprios para que coverage nao caia e para localizar regressao em componente menor.

## Politica de Coverage

1. Antes da Fase 1, medir baseline global.
2. Cada PR/fase deve rodar coverage.
3. Coverage global nao pode ficar abaixo do baseline.
4. Todo modulo novo deve ter cobertura propria.
5. Se codigo antigo for movido, testes antigos devem continuar passando sem alteracao relevante.
6. Se linhas novas reduzirem coverage global, adicionar testes na mesma fase.

Comando recomendado:

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

## Riscos e Mitigacoes

### Risco: quebrar YAML existente

Mitigacao:

- Testar `conf/examples/*.yaml`.
- Testar `tests/test_configs/*.yaml`.
- Manter reexports e defaults.
- Validacao em modo permissivo inicialmente.

### Risco: alterar log keys

Mitigacao:

- Testes snapshot dos nomes usados em `self.log` e `self.log_dict`.
- Handlers devem preservar strings atuais.

### Risco: mudar ordem de calculo de loss

Mitigacao:

- Testes de equivalencia antes/depois para:
  - loss simples;
  - compound loss;
  - EDL;
  - dual-head;
  - MoE;
  - MEDOE;
  - OHEM.

### Risco: aumentar complexidade aparente

Mitigacao:

- Cada modulo novo precisa contrato curto.
- `Model` deve ficar menor.
- Factories nao devem virar novo monolito.

### Risco: overengineering

Mitigacao:

- Implementar por extracao mecanica primeiro.
- Nao mudar comportamento e arquitetura ao mesmo tempo.
- So criar handler para comportamento ja existente.

## Ordem Recomendada de PRs

1. Baseline coverage + tests snapshot de logs.
2. Factories.
3. DataLoaderFactory + SegmentationDataModule.
4. DomainAdaptationDataModule.
5. StepEngine com handlers basicos.
6. Handlers EDL/dual-head/OHEM/MoE/MEDOE.
7. Base Lightning compartilhada.
8. Config validation.
9. Dataset split.
10. Evaluation split.
11. CLI registry.
12. Docs/examples/changelog finais.

## Criterio de Conclusao

Refatoracao so deve ser considerada concluida quando:

- YAMLs antigos continuam funcionando.
- Nenhum YAML existente precisa ser editado para funcionar.
- Nenhum novo campo YAML e necessario para ativar a arquitetura refatorada.
- `Model` nao contem mais logica especifica de EDL/MoE/MEDOE/dual-head inline.
- `DomainAdaptationModel` nao precisa burlar `Model.__init__`.
- Datasets principais nao vivem todos em um arquivo monolitico.
- Coverage global nao diminuiu.
- Testes antigos passam.
- Testes novos cobrem factories, datamodules, handlers, validators, registry e import compatibility.
- Docs em `website/docs/` explicam arquitetura nova.
- `CHANGELOG.md` tem entrada em `# Unreleased`.
