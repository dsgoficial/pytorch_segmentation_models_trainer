// Default params per loss _target_. Split from LossSection.jsx so the
// component file only exports the component (react-refresh/only-export-components).
export const LOSS_DEFAULTS = {
  'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss': {
    diceWeight: 0.25,
    ceWeight: 0.75,
    smoothFactor: 0.1,
  },
  'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceSCELoss': {
    diceWeight: 0.25,
    sceWeight: 0.75,
    smoothFactor: 0.0,
    sceAlpha: 1.0,
    sceBeta: 0.5,
  },
  'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedLovaszSCELoss': {
    lovaszWeight: 0.25,
    sceWeight: 0.75,
    sceAlpha: 1.0,
    sceBeta: 0.5,
    perImage: false,
    lovaszClasses: 'present',
  },
  'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedJMLSCELoss': {
    jmlWeight: 0.25,
    sceWeight: 0.75,
    sceAlpha: 1.0,
    sceBeta: 0.5,
    smooth: 0.001,
    jmlClasses: 'present',
    labelSmoothing: 0.0,
  },
  'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedJMLSCEGCBLLoss': {
    jmlWeight: 0.25,
    sceWeight: 0.75,
    sceAlpha: 1.0,
    sceBeta: 0.5,
    smooth: 0.001,
    jmlClasses: 'present',
    labelSmoothing: 0.0,
    gcblWeight: 0.1,
    gcblEmbedDim: 32,
    gcblTemperature: 0.07,
    gcblMaxSamples: 512,
  },
  'pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss': {
    bceCoef: 0.5,
    diceCoef: 0.5,
    tverskFocalCoef: 0.0,
  },
  'torch.nn.CrossEntropyLoss': {
    reduction: 'mean',
  },
}
