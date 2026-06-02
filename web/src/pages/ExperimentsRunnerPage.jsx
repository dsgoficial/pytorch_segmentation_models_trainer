import { useState } from 'react'
import ModelSection from '../components/ModelSection'
import NormalizationSection from '../components/NormalizationSection'
import ClassDefinitionsSection from '../components/ClassDefinitionsSection'
import HyperparametersSection from '../components/HyperparametersSection'
import LossSection, { LOSS_DEFAULTS } from '../components/LossSection'
import OptimizerSection, { OPTIMIZER_DEFAULTS } from '../components/OptimizerSection'
import PLTrainerSection from '../components/PLTrainerSection'
import MetricsSection from '../components/MetricsSection'
import CallbacksSection, { CALLBACK_DEFAULTS } from '../components/CallbacksSection'
import DatasetSection from '../components/DatasetSection'
import ExperimentsRunnerSection from '../components/ExperimentsRunnerSection'
import ImportModal from '../components/ImportModal'
import YamlPreview from '../components/YamlPreview'

// Reuse the same initial values as TrainingPage for all training config fields.

const INITIAL_MODEL = {
  architecture: 'segmentation_models_pytorch.Unet',
  encoderName: 'resnet101',
  encoderWeights: 'None',
  inChannels: 12,
  classes: 7,
}

const INITIAL_NORM = {
  mean: [70.5, 77.6, 40.9, 70.5, 69.9, 41.2, 68.4, 67.4, 41.5, 75.2, 71.3, 42.6],
  std:  [27.0, 20.1, 18.0, 26.9, 20.8, 17.4, 26.9, 21.3, 17.2, 29.5, 22.4, 18.8],
}

const INITIAL_CLASSES = {
  names:  ['background', 'massa_dagua', 'area_edificada', 'terreno_exposto', 'vegetacao_baixa', 'floresta', 'veg_cultivada'],
  colors: ['#000000', '#4444FF', '#FF4444', '#8B4513', '#90EE90', '#005700', '#FFFF44'],
}

const INITIAL_HP = { batchSize: 200, epochs: 1000, maxLr: 0.0001 }

const INITIAL_LOSS = {
  target: 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss',
  params: LOSS_DEFAULTS['pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss'],
}

const INITIAL_OPTIMIZER = {
  target: 'torch.optim.AdamW',
  params: OPTIMIZER_DEFAULTS['torch.optim.AdamW'],
}

const INITIAL_TRAINER = {
  accelerator: 'gpu',
  devices: -1,
  precision: 16,
  syncBatchnorm: true,
}

const INITIAL_METRICS = [
  { target: 'torchmetrics.Accuracy',     params: { task: 'multiclass' } },
  { target: 'torchmetrics.JaccardIndex', params: { task: 'multiclass' } },
  { target: 'torchmetrics.F1Score',      params: { task: 'multiclass', average: 'macro' } },
]

const INITIAL_RUNNER = {
  outputBaseDir: '/tmp/experiments',
  seeds: [42, 101, 28],
  seedsText: '42, 101, 28',
  useNRuns: false,
  nRuns: 3,
  saveSummary: true,
  resume: false,
}

const DATALOADER_TRAIN = {
  shuffle: true, numWorkers: 4, pinMemory: true,
  dropLast: true, prefetchFactor: 2, persistentWorkers: true,
}

const DATALOADER_VAL = {
  shuffle: false, numWorkers: 2, pinMemory: false,
  dropLast: false, prefetchFactor: 1, persistentWorkers: false,
}

const INITIAL_TRAIN_DATASET = {
  csvPath: '', useRasterio: true, resetAugmentation: true,
  dataLoader: DATALOADER_TRAIN, augmentations: [],
}

const INITIAL_VAL_DATASET = {
  csvPath: '', useRasterio: true, resetAugmentation: true,
  dataLoader: DATALOADER_VAL, augmentations: [],
}

const INITIAL_CALLBACKS = [
  { target: 'pytorch_lightning.callbacks.EarlyStopping', params: CALLBACK_DEFAULTS['pytorch_lightning.callbacks.EarlyStopping'] },
  { target: 'pytorch_lightning.callbacks.ModelCheckpoint', params: CALLBACK_DEFAULTS['pytorch_lightning.callbacks.ModelCheckpoint'] },
  { target: 'pytorch_lightning.callbacks.LearningRateMonitor', params: CALLBACK_DEFAULTS['pytorch_lightning.callbacks.LearningRateMonitor'] },
]

// ── Helpers ────────────────────────────────────────────────────────────────────

function resizeArray(arr, len, fill = 0) {
  if (arr.length === len) return arr
  if (arr.length > len) return arr.slice(0, len)
  return [...arr, ...Array(len - arr.length).fill(fill)]
}

function serializeLoss(loss) {
  const p = loss.params
  const t = loss.target
  if (t.includes('WeightedDiceCrossEntropyLoss'))
    return { _target_: t, num_classes: '${model.classes}', dice_weight: p.diceWeight, ce_weight: p.ceWeight, smooth_factor: p.smoothFactor }
  if (t.endsWith('WeightedDiceSCELoss'))
    return { _target_: t, num_classes: '${model.classes}', dice_weight: p.diceWeight, sce_weight: p.sceWeight, smooth_factor: p.smoothFactor, sce_alpha: p.sceAlpha, sce_beta: p.sceBeta }
  if (t.endsWith('WeightedLovaszSCELoss'))
    return { _target_: t, num_classes: '${model.classes}', lovasz_weight: p.lovaszWeight, sce_weight: p.sceWeight, sce_alpha: p.sceAlpha, sce_beta: p.sceBeta, per_image: p.perImage, lovasz_classes: p.lovaszClasses }
  if (t.endsWith('WeightedJMLSCEGCBLLoss'))
    return { _target_: t, num_classes: '${model.classes}', jml_weight: p.jmlWeight, sce_weight: p.sceWeight, sce_alpha: p.sceAlpha, sce_beta: p.sceBeta, smooth: p.smooth, jml_classes: p.jmlClasses, label_smoothing: p.labelSmoothing, gcbl_weight: p.gcblWeight, gcbl_embed_dim: p.gcblEmbedDim, gcbl_temperature: p.gcblTemperature, gcbl_max_samples: p.gcblMaxSamples }
  if (t.endsWith('WeightedJMLSCELoss'))
    return { _target_: t, num_classes: '${model.classes}', jml_weight: p.jmlWeight, sce_weight: p.sceWeight, sce_alpha: p.sceAlpha, sce_beta: p.sceBeta, smooth: p.smooth, jml_classes: p.jmlClasses, label_smoothing: p.labelSmoothing }
  if (t.includes('SegLoss'))
    return { _target_: t, bce_coef: p.bceCoef, dice_coef: p.diceCoef, tversky_focal_coef: p.tverskFocalCoef }
  return { _target_: t, ...p }
}

function serializeOptimizer(opt) {
  const p = opt.params
  const base = { _target_: opt.target, lr: p.lr, weight_decay: p.weightDecay }
  if (opt.target === 'torch.optim.SGD') base.momentum = p.momentum
  return base
}

function serializeMetrics(metrics) {
  return metrics.map(m => ({ _target_: m.target, num_classes: '${model.classes}', ...m.params }))
}

function serializeCallbacks(callbacks) {
  return callbacks.map(cb => {
    const p = cb.params
    if (cb.target === 'pytorch_lightning.callbacks.EarlyStopping')
      return { _target_: cb.target, monitor: p.monitor, mode: p.mode, patience: p.patience, verbose: p.verbose, min_delta: p.minDelta }
    if (cb.target === 'pytorch_lightning.callbacks.ModelCheckpoint')
      return { _target_: cb.target, monitor: p.monitor, mode: p.mode, save_top_k: p.saveTopK, filename: p.filename, save_weights_only: p.saveWeightsOnly }
    if (cb.target === 'pytorch_lightning.callbacks.LearningRateMonitor')
      return { _target_: cb.target, logging_interval: p.loggingInterval }
    return { _target_: cb.target, ...p }
  })
}

function serializeAugmentations(augmentations) {
  const result = augmentations.map(aug => ({ _target_: aug.target, always_apply: false, ...aug.params }))
  result.push({ _target_: 'albumentations.Normalize', p: 1.0, mean: '${normalization_parameters.mean}', std: '${normalization_parameters.std}' })
  result.push({ _target_: 'albumentations.pytorch.transforms.ToTensorV2', always_apply: true })
  return result
}

function serializeDataset(dataset) {
  return {
    _target_: 'pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset',
    input_csv_path: dataset.csvPath,
    n_classes: '${model.classes}',
    use_rasterio: dataset.useRasterio,
    reset_augmentation_function: dataset.resetAugmentation,
    data_loader: {
      shuffle: dataset.dataLoader.shuffle,
      num_workers: dataset.dataLoader.numWorkers,
      pin_memory: dataset.dataLoader.pinMemory,
      drop_last: dataset.dataLoader.dropLast,
      prefetch_factor: dataset.dataLoader.prefetchFactor,
      persistent_workers: dataset.dataLoader.persistentWorkers,
    },
    augmentation_list: serializeAugmentations(dataset.augmentations),
  }
}

function serializeRunner(runner) {
  const block = {
    output_base_dir: runner.outputBaseDir,
    save_summary: runner.saveSummary,
    resume: runner.resume,
  }
  if (runner.useNRuns) {
    block.n_runs = runner.nRuns
  } else {
    block.seeds = runner.seeds.length > 0 ? runner.seeds : [42]
  }
  return block
}

function buildConfig(model, norm, classes, hp, loss, optimizer, trainer, metrics, callbacks, trainDataset, valDataset, runner) {
  return {
    mode: 'run-experiments',
    experiments_runner: serializeRunner(runner),
    pl_model: { _target_: 'pytorch_segmentation_models_trainer.model_loader.model.Model' },
    normalization_parameters: { mean: norm.mean, std: norm.std },
    class_definitions: { names: classes.names, colors: classes.colors },
    model: {
      _target_: model.architecture,
      encoder_name: model.encoderName,
      ...(model.encoderWeights !== 'None' && { encoder_weights: model.encoderWeights }),
      in_channels: model.inChannels,
      classes: model.classes,
    },
    loss: serializeLoss(loss),
    optimizer: serializeOptimizer(optimizer),
    hyperparameters: { batch_size: hp.batchSize, epochs: hp.epochs, max_lr: hp.maxLr },
    pl_trainer: {
      max_epochs: '${hyperparameters.epochs}',
      accelerator: trainer.accelerator,
      devices: trainer.devices,
      sync_batchnorm: trainer.syncBatchnorm,
      precision: trainer.precision,
    },
    callbacks: serializeCallbacks(callbacks),
    metrics: serializeMetrics(metrics),
    train_dataset: serializeDataset(trainDataset),
    val_dataset: serializeDataset(valDataset),
  }
}

// ── Componente ────────────────────────────────────────────────────────────────

export default function ExperimentsRunnerPage() {
  const [model, setModel]               = useState(INITIAL_MODEL)
  const [norm, setNorm]                 = useState(INITIAL_NORM)
  const [classes, setClasses]           = useState(INITIAL_CLASSES)
  const [hp, setHp]                     = useState(INITIAL_HP)
  const [loss, setLoss]                 = useState(INITIAL_LOSS)
  const [optimizer, setOptimizer]       = useState(INITIAL_OPTIMIZER)
  const [trainer, setTrainer]           = useState(INITIAL_TRAINER)
  const [metrics, setMetrics]           = useState(INITIAL_METRICS)
  const [callbacks, setCallbacks]       = useState(INITIAL_CALLBACKS)
  const [trainDataset, setTrainDataset] = useState(INITIAL_TRAIN_DATASET)
  const [valDataset, setValDataset]     = useState(INITIAL_VAL_DATASET)
  const [runner, setRunner]             = useState(INITIAL_RUNNER)
  const [showImport, setShowImport]     = useState(false)

  function handleModelChange(newModel) {
    setModel(newModel)
    if (newModel.inChannels !== model.inChannels) {
      setNorm(prev => ({
        mean: resizeArray(prev.mean, newModel.inChannels, 0),
        std:  resizeArray(prev.std,  newModel.inChannels, 1),
      }))
    }
  }

  const config = buildConfig(model, norm, classes, hp, loss, optimizer, trainer, metrics, callbacks, trainDataset, valDataset, runner)

  return (
    <>
      <div style={styles.toolbar}>
        <span style={styles.modeBadge}>mode: run-experiments</span>
      </div>

      {showImport && (
        <ImportModal onImport={() => {}} onClose={() => setShowImport(false)} />
      )}

      <main style={styles.main}>
        <div style={styles.formCol}>
          <ExperimentsRunnerSection runner={runner} onChange={setRunner} />
          <ModelSection            model={model}         onChange={handleModelChange} />
          <NormalizationSection    mean={norm.mean} std={norm.std} onChange={setNorm} />
          <ClassDefinitionsSection classes={classes}     onChange={setClasses} />
          <HyperparametersSection  hyperparameters={hp}  onChange={setHp} />
          <LossSection             loss={loss}           onChange={setLoss} />
          <OptimizerSection        optimizer={optimizer} onChange={setOptimizer} />
          <PLTrainerSection        trainer={trainer}     onChange={setTrainer} />
          <MetricsSection          metrics={metrics}     onChange={setMetrics} />
          <CallbacksSection        callbacks={callbacks} onChange={setCallbacks} />
          <DatasetSection label="Train Dataset" dataset={trainDataset} onChange={setTrainDataset} />
          <DatasetSection label="Val Dataset"   dataset={valDataset}   onChange={setValDataset} />
        </div>
        <div style={styles.previewCol}>
          <YamlPreview config={config} />
        </div>
      </main>
    </>
  )
}

const styles = {
  toolbar:    { padding: '8px 24px', borderBottom: '1px solid #e5e5e5', display: 'flex', alignItems: 'center', gap: 12, background: '#fff' },
  modeBadge:  { background: '#e8f5e9', color: '#2e7d32', padding: '3px 10px', borderRadius: 4, fontSize: '0.78rem', fontFamily: 'monospace', fontWeight: 600 },
  main:       { display: 'grid', gridTemplateColumns: '1fr 1fr', flex: 1, minHeight: 0 },
  formCol:    { padding: 24, overflowY: 'auto', borderRight: '1px solid #e5e5e5', background: '#f5f5f5' },
  previewCol: { padding: 0, background: '#1e1e1e', overflow: 'hidden' },
}
