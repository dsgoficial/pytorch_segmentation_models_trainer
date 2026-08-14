// Default params per callback _target_. Split from CallbacksSection.jsx so the
// component file only exports the component (react-refresh/only-export-components).
export const CALLBACK_DEFAULTS = {
  'pytorch_lightning.callbacks.EarlyStopping': {
    monitor: 'loss/val', mode: 'min', patience: 100, verbose: true, minDelta: 0.001,
  },
  'pytorch_lightning.callbacks.ModelCheckpoint': {
    monitor: 'val/MulticlassJaccardIndex', mode: 'max', saveTopK: 3,
    filename: 'best-{epoch:02d}-{val_JaccardIndex:.3f}', saveWeightsOnly: true,
  },
  'pytorch_lightning.callbacks.LearningRateMonitor': {
    loggingInterval: 'epoch',
  },
  'pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.EnhancedImageSegmentationResultCallback': {
    nSamples: 100, logEveryKEpochs: 1, normalizedInput: true,
    bandIndices: '0, 1, 2', alphaMask: 0.0, maxWorkers: 8,
    showClassLegend: true,
  },
}
