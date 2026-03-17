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
import ImportModal from '../components/ImportModal'
import YamlPreview from '../components/YamlPreview'

// ── Estado inicial ─────────────────────────────────────────────────────────────

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
  { target: 'torchmetrics.Precision',    params: { task: 'multiclass', average: 'macro' } },
  { target: 'torchmetrics.Recall',       params: { task: 'multiclass', average: 'macro' } },
]

const TRAIN_AUG_DEFAULTS = [
  { target: 'albumentations.HorizontalFlip', params: { p: 0.5 } },
  { target: 'albumentations.RandomRotate90', params: { p: 0.5 } },
]

const DATALOADER_TRAIN = {
  shuffle: true, numWorkers: 4, pinMemory: true,
  dropLast: true, prefetchFactor: 2, persistentWorkers: true,
}

const DATALOADER_VAL = {
  shuffle: false, numWorkers: 2, pinMemory: false,
  dropLast: false, prefetchFactor: 1, persistentWorkers: false,
}

const INITIAL_TRAIN_DATASET = {
  csvPath: '',
  useRasterio: true,
  resetAugmentation: true,
  dataLoader: DATALOADER_TRAIN,
  augmentations: TRAIN_AUG_DEFAULTS,
}

const INITIAL_VAL_DATASET = {
  csvPath: '',
  useRasterio: true,
  resetAugmentation: true,
  dataLoader: DATALOADER_VAL,
  augmentations: [],
}

const INITIAL_CALLBACKS = [
  {
    target: 'pytorch_lightning.callbacks.EarlyStopping',
    params: CALLBACK_DEFAULTS['pytorch_lightning.callbacks.EarlyStopping'],
  },
  {
    target: 'pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.EnhancedImageSegmentationResultCallback',
    params: CALLBACK_DEFAULTS['pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.EnhancedImageSegmentationResultCallback'],
  },
  {
    target: 'pytorch_lightning.callbacks.ModelCheckpoint',
    params: CALLBACK_DEFAULTS['pytorch_lightning.callbacks.ModelCheckpoint'],
  },
  {
    target: 'pytorch_lightning.callbacks.LearningRateMonitor',
    params: CALLBACK_DEFAULTS['pytorch_lightning.callbacks.LearningRateMonitor'],
  },
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
  if (t.includes('WeightedDiceCrossEntropyLoss')) {
    return { _target_: t, num_classes: '${model.classes}', dice_weight: p.diceWeight, ce_weight: p.ceWeight, smooth_factor: p.smoothFactor }
  }
  if (t.includes('SegLoss')) {
    return { _target_: t, bce_coef: p.bceCoef, dice_coef: p.diceCoef, tversky_focal_coef: p.tverskFocalCoef }
  }
  return { _target_: t, ...p }
}

function serializeOptimizer(opt) {
  const p = opt.params
  const base = { _target_: opt.target, lr: p.lr, weight_decay: p.weightDecay }
  if (opt.target === 'torch.optim.SGD') base.momentum = p.momentum
  return base
}

function serializeMetrics(metrics) {
  return metrics.map(m => ({
    _target_: m.target,
    num_classes: '${model.classes}',
    ...m.params,
  }))
}

function serializeCallbacks(callbacks) {
  return callbacks.map(cb => {
    const p = cb.params
    if (cb.target === 'pytorch_lightning.callbacks.EarlyStopping') {
      return { _target_: cb.target, monitor: p.monitor, mode: p.mode, patience: p.patience, verbose: p.verbose, min_delta: p.minDelta }
    }
    if (cb.target === 'pytorch_lightning.callbacks.ModelCheckpoint') {
      return { _target_: cb.target, monitor: p.monitor, mode: p.mode, save_top_k: p.saveTopK, filename: p.filename, save_weights_only: p.saveWeightsOnly }
    }
    if (cb.target === 'pytorch_lightning.callbacks.LearningRateMonitor') {
      return { _target_: cb.target, logging_interval: p.loggingInterval }
    }
    if (cb.target.includes('EnhancedImageSegmentationResultCallback')) {
      return {
        _target_: cb.target,
        n_samples: p.nSamples,
        log_every_k_epochs: p.logEveryKEpochs,
        num_classes: '${model.classes}',
        normalized_input: p.normalizedInput,
        norm_params: { mean: '${normalization_parameters.mean}', std: '${normalization_parameters.std}' },
        band_indices: p.bandIndices.split(',').map(s => Number(s.trim())),
        alpha_mask: p.alphaMask,
        max_workers: p.maxWorkers,
        show_class_legend: p.showClassLegend,
        class_names: '${class_definitions.names}',
        class_colors: '${class_definitions.colors}',
      }
    }
    return { _target_: cb.target, ...p }
  })
}

function serializeAugmentations(augmentations) {
  const result = augmentations.map(aug => ({
    _target_: aug.target,
    always_apply: false,
    ...aug.params,
  }))
  result.push({
    _target_: 'albumentations.Normalize',
    p: 1.0,
    mean: '${normalization_parameters.mean}',
    std: '${normalization_parameters.std}',
  })
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

function deserializeDataset(d) {
  if (!d) return null
  const augmentations = (d.augmentation_list ?? [])
    .filter(a => !a._target_?.includes('Normalize') && !a._target_?.includes('ToTensorV2'))
    .map(({ _target_, always_apply, ...params }) => ({ target: _target_, params }))
  return {
    csvPath:           d.input_csv_path ?? '',
    useRasterio:       d.use_rasterio ?? false,
    resetAugmentation: d.reset_augmentation_function ?? false,
    dataLoader: {
      shuffle:            d.data_loader?.shuffle            ?? false,
      numWorkers:         d.data_loader?.num_workers        ?? 4,
      pinMemory:          d.data_loader?.pin_memory         ?? false,
      dropLast:           d.data_loader?.drop_last          ?? false,
      prefetchFactor:     d.data_loader?.prefetch_factor    ?? 1,
      persistentWorkers:  d.data_loader?.persistent_workers ?? false,
    },
    augmentations,
  }
}

function applyImport(data, setters) {
  const { setModel, setNorm, setClasses, setHp, setLoss, setOptimizer,
          setTrainer, setMetrics, setCallbacks, setTrainDataset, setValDataset } = setters

  if (data.model) {
    setModel({
      architecture:   data.model._target_,
      encoderName:    data.model.encoder_name,
      encoderWeights: data.model.encoder_weights ?? 'None',
      inChannels:     data.model.in_channels,
      classes:        data.model.classes,
    })
  }
  if (data.normalization_parameters) {
    setNorm({ mean: data.normalization_parameters.mean, std: data.normalization_parameters.std })
  }
  if (data.class_definitions) {
    setClasses({ names: data.class_definitions.names, colors: data.class_definitions.colors })
  }
  if (data.hyperparameters) {
    setHp({ batchSize: data.hyperparameters.batch_size, epochs: data.hyperparameters.epochs, maxLr: data.hyperparameters.max_lr })
  }
  if (data.loss) {
    const t = data.loss._target_
    let params = {}
    if (t.includes('WeightedDiceCrossEntropyLoss'))
      params = { diceWeight: data.loss.dice_weight, ceWeight: data.loss.ce_weight, smoothFactor: data.loss.smooth_factor }
    else if (t.includes('SegLoss'))
      params = { bceCoef: data.loss.bce_coef, diceCoef: data.loss.dice_coef, tverskFocalCoef: data.loss.tversky_focal_coef }
    else
      params = LOSS_DEFAULTS[t] ?? {}
    setLoss({ target: t, params })
  }
  if (data.optimizer) {
    const t = data.optimizer._target_
    const params = { lr: data.optimizer.lr, weightDecay: data.optimizer.weight_decay }
    if (t === 'torch.optim.SGD') params.momentum = data.optimizer.momentum
    setOptimizer({ target: t, params })
  }
  if (data.pl_trainer) {
    setTrainer({
      accelerator:   data.pl_trainer.accelerator,
      devices:       data.pl_trainer.devices,
      precision:     data.pl_trainer.precision,
      syncBatchnorm: data.pl_trainer.sync_batchnorm,
    })
  }
  if (data.metrics) {
    setMetrics(data.metrics.map(m => ({
      target: m._target_,
      params: { task: m.task ?? 'multiclass', ...(m.average ? { average: m.average } : {}) },
    })))
  }
  if (data.callbacks) {
    setCallbacks(data.callbacks.map(cb => {
      const t = cb._target_
      if (t === 'pytorch_lightning.callbacks.EarlyStopping')
        return { target: t, params: { monitor: cb.monitor, mode: cb.mode, patience: cb.patience, verbose: cb.verbose ?? true, minDelta: cb.min_delta ?? 0.001 } }
      if (t === 'pytorch_lightning.callbacks.ModelCheckpoint')
        return { target: t, params: { monitor: cb.monitor, mode: cb.mode, saveTopK: cb.save_top_k, filename: cb.filename ?? '', saveWeightsOnly: cb.save_weights_only ?? true } }
      if (t === 'pytorch_lightning.callbacks.LearningRateMonitor')
        return { target: t, params: { loggingInterval: cb.logging_interval ?? 'epoch' } }
      if (t.includes('EnhancedImageSegmentationResultCallback'))
        return { target: t, params: { nSamples: cb.n_samples, logEveryKEpochs: cb.log_every_k_epochs, normalizedInput: cb.normalized_input ?? true, bandIndices: (cb.band_indices ?? [0,1,2]).join(', '), alphaMask: cb.alpha_mask ?? 0.0, maxWorkers: cb.max_workers ?? 4, showClassLegend: cb.show_class_legend ?? true } }
      return { target: t, params: CALLBACK_DEFAULTS[t] ?? {} }
    }))
  }
  const td = deserializeDataset(data.train_dataset)
  if (td) setTrainDataset(td)
  const vd = deserializeDataset(data.val_dataset)
  if (vd) setValDataset(vd)
}

function buildConfig(model, norm, classes, hp, loss, optimizer, trainer, metrics, callbacks, trainDataset, valDataset) {
  return {
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

export default function TrainingPage() {
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

  function handleImport(data) {
    applyImport(data, {
      setModel, setNorm, setClasses, setHp, setLoss, setOptimizer,
      setTrainer, setMetrics, setCallbacks, setTrainDataset, setValDataset,
    })
  }

  const config = buildConfig(model, norm, classes, hp, loss, optimizer, trainer, metrics, callbacks, trainDataset, valDataset)

  return (
    <>
      <div style={styles.toolbar}>
        <button onClick={() => setShowImport(true)} style={styles.importBtn}>
          Import YAML
        </button>
      </div>

      {showImport && (
        <ImportModal onImport={handleImport} onClose={() => setShowImport(false)} />
      )}

      <main style={styles.main}>
        <div style={styles.formCol}>
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
  toolbar:    { padding: '8px 24px', borderBottom: '1px solid #e5e5e5', display: 'flex', justifyContent: 'flex-end', background: '#fff' },
  importBtn:  { background: 'none', border: '1px solid #3b82f6', color: '#3b82f6', borderRadius: 6, padding: '6px 14px', fontSize: '0.85rem', cursor: 'pointer', fontWeight: 500 },
  main:       { display: 'grid', gridTemplateColumns: '1fr 1fr', flex: 1, minHeight: 0 },
  formCol:    { padding: 24, overflowY: 'auto', borderRight: '1px solid #e5e5e5', background: '#f5f5f5' },
  previewCol: { padding: 0, background: '#1e1e1e', overflow: 'hidden' },
}
