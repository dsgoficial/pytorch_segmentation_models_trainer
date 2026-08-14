// Default params per optimizer _target_. Split from OptimizerSection.jsx so the
// component file only exports the component (react-refresh/only-export-components).
export const OPTIMIZER_DEFAULTS = {
  'torch.optim.AdamW': { lr: 0.0001, weightDecay: 0.0001 },
  'torch.optim.Adam':  { lr: 0.0001, weightDecay: 0.0 },
  'torch.optim.SGD':   { lr: 0.01,   momentum: 0.9, weightDecay: 0.0001 },
}
