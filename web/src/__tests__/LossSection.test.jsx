import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import LossSection, { LOSS_DEFAULTS } from '../components/LossSection'

// Helpers
const CE_KEY  = 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss'
const SCE_KEY = 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceSCELoss'
const LSZ_KEY = 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedLovaszSCELoss'
const JML_KEY = 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedJMLSCELoss'
const GCBL_KEY = 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedJMLSCEGCBLLoss'
const SEG_KEY = 'pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss'
const TORCH_CE_KEY = 'torch.nn.CrossEntropyLoss'

function renderLoss(target, onChange = vi.fn()) {
  const params = LOSS_DEFAULTS[target]
  return render(<LossSection loss={{ target, params }} onChange={onChange} />)
}

describe('LOSS_DEFAULTS', () => {
  it('exports defaults for all 7 loss types', () => {
    expect(Object.keys(LOSS_DEFAULTS)).toHaveLength(7)
  })

  it('WeightedDiceCrossEntropyLoss defaults are correct', () => {
    expect(LOSS_DEFAULTS[CE_KEY]).toMatchObject({
      diceWeight: 0.25,
      ceWeight: 0.75,
      smoothFactor: 0.1,
    })
  })

  it('WeightedDiceSCELoss defaults are correct', () => {
    expect(LOSS_DEFAULTS[SCE_KEY]).toMatchObject({
      diceWeight: 0.25,
      sceWeight: 0.75,
      sceAlpha: 1.0,
      sceBeta: 0.5,
    })
  })

  it('WeightedLovaszSCELoss defaults are correct', () => {
    expect(LOSS_DEFAULTS[LSZ_KEY]).toMatchObject({
      lovaszWeight: 0.25,
      sceWeight: 0.75,
      perImage: false,
      lovaszClasses: 'present',
    })
  })

  it('WeightedJMLSCELoss defaults are correct', () => {
    expect(LOSS_DEFAULTS[JML_KEY]).toMatchObject({
      jmlWeight: 0.25,
      sceWeight: 0.75,
      jmlClasses: 'present',
      labelSmoothing: 0.0,
    })
  })

  it('WeightedJMLSCEGCBLLoss defaults include GCBL params', () => {
    expect(LOSS_DEFAULTS[GCBL_KEY]).toMatchObject({
      jmlWeight: 0.25,
      gcblWeight: 0.1,
      gcblEmbedDim: 32,
      gcblTemperature: 0.07,
      gcblMaxSamples: 512,
    })
  })

  it('SegLoss defaults are correct', () => {
    expect(LOSS_DEFAULTS[SEG_KEY]).toMatchObject({
      bceCoef: 0.5,
      diceCoef: 0.5,
      tverskFocalCoef: 0.0,
    })
  })

  it('CrossEntropyLoss defaults are correct', () => {
    expect(LOSS_DEFAULTS[TORCH_CE_KEY]).toMatchObject({ reduction: 'mean' })
  })
})

describe('LossSection rendering', () => {
  it('renders the "Loss Function" section heading', () => {
    renderLoss(CE_KEY)
    expect(screen.getByRole('heading', { name: 'Loss Function' })).toBeInTheDocument()
  })

  it('shows Hydra interpolation hint for num_classes', () => {
    renderLoss(CE_KEY)
    expect(screen.getByDisplayValue('${model.classes}')).toBeInTheDocument()
  })

  it('renders Dice Weight and CE Weight fields for WeightedDiceCrossEntropyLoss', () => {
    renderLoss(CE_KEY)
    expect(screen.getByLabelText('Dice Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('CE Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('Smooth Factor')).toBeInTheDocument()
  })

  it('renders Dice/SCE fields for WeightedDiceSCELoss', () => {
    renderLoss(SCE_KEY)
    expect(screen.getByLabelText('Dice Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('SCE Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('SCE Alpha (CE weight)')).toBeInTheDocument()
    expect(screen.getByLabelText('SCE Beta (RCE weight)')).toBeInTheDocument()
  })

  it('renders Lovász fields for WeightedLovaszSCELoss', () => {
    renderLoss(LSZ_KEY)
    expect(screen.getByLabelText('Lovász Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('SCE Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('Classes')).toBeInTheDocument()
    expect(screen.getByLabelText('Per Image')).toBeInTheDocument()
  })

  it('renders JML fields for WeightedJMLSCELoss', () => {
    renderLoss(JML_KEY)
    expect(screen.getByLabelText('JML Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('SCE Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('Label Smoothing')).toBeInTheDocument()
    expect(screen.getByLabelText('JML Classes')).toBeInTheDocument()
  })

  it('renders GCBL section and params for WeightedJMLSCEGCBLLoss', () => {
    renderLoss(GCBL_KEY)
    expect(screen.getByText('GCBL (Contrastive Boundary)')).toBeInTheDocument()
    expect(screen.getByLabelText('GCBL Weight')).toBeInTheDocument()
    expect(screen.getByLabelText('Embed Dim')).toBeInTheDocument()
    expect(screen.getByLabelText('Temperature')).toBeInTheDocument()
    expect(screen.getByLabelText('Max Samples')).toBeInTheDocument()
  })

  it('renders BCE/Dice/Tversky fields for SegLoss', () => {
    renderLoss(SEG_KEY)
    expect(screen.getByLabelText('BCE Coef')).toBeInTheDocument()
    expect(screen.getByLabelText('Dice Coef')).toBeInTheDocument()
    expect(screen.getByLabelText('Tversky/Focal Coef')).toBeInTheDocument()
  })

  it('renders Reduction select for CrossEntropyLoss', () => {
    renderLoss(TORCH_CE_KEY)
    expect(screen.getByLabelText('Reduction')).toBeInTheDocument()
  })
})

describe('LossSection interactions', () => {
  it('calls onChange when Dice Weight is updated', () => {
    const handleChange = vi.fn()
    renderLoss(CE_KEY, handleChange)

    const input = screen.getByLabelText('Dice Weight')
    fireEvent.change(input, { target: { value: '0.5' } })

    expect(handleChange).toHaveBeenCalledOnce()
    const [updated] = handleChange.mock.calls[0]
    expect(updated.params.diceWeight).toBe(0.5)
  })

  it('calls onChange when CE Weight is updated', () => {
    const handleChange = vi.fn()
    renderLoss(CE_KEY, handleChange)

    const input = screen.getByLabelText('CE Weight')
    fireEvent.change(input, { target: { value: '0.3' } })

    expect(handleChange).toHaveBeenCalledOnce()
    const [updated] = handleChange.mock.calls[0]
    expect(updated.params.ceWeight).toBe(0.3)
  })

  it('calls onChange when Lovász per_image checkbox is toggled', () => {
    const handleChange = vi.fn()
    renderLoss(LSZ_KEY, handleChange)

    const checkbox = screen.getByLabelText('Per Image')
    fireEvent.click(checkbox)

    expect(handleChange).toHaveBeenCalledOnce()
    const [updated] = handleChange.mock.calls[0]
    expect(updated.params.perImage).toBe(true)
  })

  it('calls onChange when Reduction is changed for CrossEntropyLoss', () => {
    const handleChange = vi.fn()
    renderLoss(TORCH_CE_KEY, handleChange)

    const select = screen.getByLabelText('Reduction')
    fireEvent.change(select, { target: { value: 'sum' } })

    expect(handleChange).toHaveBeenCalledOnce()
    const [updated] = handleChange.mock.calls[0]
    expect(updated.params.reduction).toBe('sum')
  })
})
