import SearchableSelect from './SearchableSelect'

const LOSS_TYPES = [
  {
    value: 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss',
    label: 'WeightedDiceCrossEntropyLoss',
  },
  {
    value: 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceSCELoss',
    label: 'WeightedDiceSCELoss (Dice + Symmetric CE)',
  },
  {
    value: 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedLovaszSCELoss',
    label: 'WeightedLovaszSCELoss (Lovász + Symmetric CE)',
  },
  {
    value: 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedJMLSCELoss',
    label: 'WeightedJMLSCELoss (Jaccard Metric + Symmetric CE)',
  },
  {
    value: 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedJMLSCEGCBLLoss',
    label: 'WeightedJMLSCEGCBLLoss (JML + SCE + Contrastive Boundary)',
  },
  {
    value: 'pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss',
    label: 'SegLoss (BCE + Dice + Tversky)',
  },
  {
    value: 'torch.nn.CrossEntropyLoss',
    label: 'CrossEntropyLoss (PyTorch)',
  },
]

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

export default function LossSection({ loss, onChange }) {
  function handleTypeChange(newType) {
    // Ao trocar o tipo, mantemos só o _target_ e aplicamos os defaults do novo tipo
    onChange({ target: newType, params: LOSS_DEFAULTS[newType] })
  }

  function handleParam(field, value) {
    onChange({ ...loss, params: { ...loss.params, [field]: value } })
  }

  return (
    <section style={styles.section}>
      <h2>Loss Function</h2>

      <div className="field">
        <label>Type (_target_)</label>
        <SearchableSelect
          value={loss.target}
          options={LOSS_TYPES}
          onChange={handleTypeChange}
        />
      </div>

      <div className="field">
        <label>num_classes</label>
        <input value="${model.classes}" readOnly style={styles.interpolated} />
        <span style={styles.hint}>Auto-reference to model.classes via Hydra interpolation.</span>
      </div>

      {loss.target === 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss' && (
        <div style={styles.paramsBox}>
          <div style={styles.row}>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-diceWeight">Dice Weight</label>
              <input
                id="loss-diceWeight"
                type="number" step="0.05" min={0} max={1}
                value={loss.params.diceWeight}
                onChange={e => handleParam('diceWeight', Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-ceWeight">CE Weight</label>
              <input
                id="loss-ceWeight"
                type="number" step="0.05" min={0} max={1}
                value={loss.params.ceWeight}
                onChange={e => handleParam('ceWeight', Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-smoothFactor">Smooth Factor</label>
              <input
                id="loss-smoothFactor"
                type="number" step="0.01" min={0}
                value={loss.params.smoothFactor}
                onChange={e => handleParam('smoothFactor', Number(e.target.value))}
              />
            </div>
          </div>
          <WeightBar dice={loss.params.diceWeight} ce={loss.params.ceWeight} />
        </div>
      )}

      {loss.target === 'pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss' && (
        <div style={styles.paramsBox}>
          <div style={styles.row}>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-bceCoef">BCE Coef</label>
              <input
                id="loss-bceCoef"
                type="number" step="0.05" min={0} max={1}
                value={loss.params.bceCoef}
                onChange={e => handleParam('bceCoef', Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-diceCoef">Dice Coef</label>
              <input
                id="loss-diceCoef"
                type="number" step="0.05" min={0} max={1}
                value={loss.params.diceCoef}
                onChange={e => handleParam('diceCoef', Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-tverskFocalCoef">Tversky/Focal Coef</label>
              <input
                id="loss-tverskFocalCoef"
                type="number" step="0.05" min={0} max={1}
                value={loss.params.tverskFocalCoef}
                onChange={e => handleParam('tverskFocalCoef', Number(e.target.value))}
              />
            </div>
          </div>
        </div>
      )}

      {loss.target === 'torch.nn.CrossEntropyLoss' && (
        <div style={styles.paramsBox}>
          <div className="field">
            <label htmlFor="loss-reduction">Reduction</label>
            <select
              id="loss-reduction"
              value={loss.params.reduction}
              onChange={e => handleParam('reduction', e.target.value)}
            >
              <option value="mean">mean</option>
              <option value="sum">sum</option>
              <option value="none">none</option>
            </select>
          </div>
        </div>
      )}

      {loss.target === 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceSCELoss' && (
        <div style={styles.paramsBox}>
          <div style={styles.row}>
            <NumField label="Dice Weight" field="diceWeight" value={loss.params.diceWeight} step={0.05} min={0} max={1} onChange={handleParam} />
            <NumField label="SCE Weight" field="sceWeight" value={loss.params.sceWeight} step={0.05} min={0} max={1} onChange={handleParam} />
            <NumField label="Smooth Factor" field="smoothFactor" value={loss.params.smoothFactor} step={0.01} min={0} onChange={handleParam} />
          </div>
          <div style={{ ...styles.row, marginTop: 8 }}>
            <NumField label="SCE Alpha (CE weight)" field="sceAlpha" value={loss.params.sceAlpha} step={0.1} min={0} onChange={handleParam} />
            <NumField label="SCE Beta (RCE weight)" field="sceBeta" value={loss.params.sceBeta} step={0.1} min={0} onChange={handleParam} />
          </div>
        </div>
      )}

      {loss.target === 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedLovaszSCELoss' && (
        <div style={styles.paramsBox}>
          <div style={styles.row}>
            <NumField label="Lovász Weight" field="lovaszWeight" value={loss.params.lovaszWeight} step={0.05} min={0} max={1} onChange={handleParam} />
            <NumField label="SCE Weight" field="sceWeight" value={loss.params.sceWeight} step={0.05} min={0} max={1} onChange={handleParam} />
          </div>
          <div style={{ ...styles.row, marginTop: 8 }}>
            <NumField label="SCE Alpha" field="sceAlpha" value={loss.params.sceAlpha} step={0.1} min={0} onChange={handleParam} />
            <NumField label="SCE Beta" field="sceBeta" value={loss.params.sceBeta} step={0.1} min={0} onChange={handleParam} />
          </div>
          <div style={{ ...styles.row, marginTop: 8 }}>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-lovaszClasses">Classes</label>
              <select id="loss-lovaszClasses" value={loss.params.lovaszClasses} onChange={e => handleParam('lovaszClasses', e.target.value)}>
                <option value="present">present</option>
                <option value="all">all</option>
              </select>
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-perImage">Per Image</label>
              <input id="loss-perImage" type="checkbox" checked={loss.params.perImage} onChange={e => handleParam('perImage', e.target.checked)} style={{ width: 'auto' }} />
            </div>
          </div>
        </div>
      )}

      {loss.target === 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedJMLSCELoss' && (
        <div style={styles.paramsBox}>
          <div style={styles.row}>
            <NumField label="JML Weight" field="jmlWeight" value={loss.params.jmlWeight} step={0.05} min={0} max={1} onChange={handleParam} />
            <NumField label="SCE Weight" field="sceWeight" value={loss.params.sceWeight} step={0.05} min={0} max={1} onChange={handleParam} />
            <NumField label="Smooth" field="smooth" value={loss.params.smooth} step={0.001} min={0} onChange={handleParam} />
          </div>
          <div style={{ ...styles.row, marginTop: 8 }}>
            <NumField label="SCE Alpha" field="sceAlpha" value={loss.params.sceAlpha} step={0.1} min={0} onChange={handleParam} />
            <NumField label="SCE Beta" field="sceBeta" value={loss.params.sceBeta} step={0.1} min={0} onChange={handleParam} />
            <NumField label="Label Smoothing" field="labelSmoothing" value={loss.params.labelSmoothing} step={0.01} min={0} max={1} onChange={handleParam} />
          </div>
          <div style={{ ...styles.row, marginTop: 8 }}>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="loss-jmlClasses">JML Classes</label>
              <select id="loss-jmlClasses" value={loss.params.jmlClasses} onChange={e => handleParam('jmlClasses', e.target.value)}>
                <option value="present">present</option>
                <option value="all">all</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {loss.target === 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedJMLSCEGCBLLoss' && (
        <div style={styles.paramsBox}>
          <span style={styles.subLabel}>JML + SCE</span>
          <div style={styles.row}>
            <NumField label="JML Weight" field="jmlWeight" value={loss.params.jmlWeight} step={0.05} min={0} max={1} onChange={handleParam} />
            <NumField label="SCE Weight" field="sceWeight" value={loss.params.sceWeight} step={0.05} min={0} max={1} onChange={handleParam} />
            <NumField label="Smooth" field="smooth" value={loss.params.smooth} step={0.001} min={0} onChange={handleParam} />
          </div>
          <div style={{ ...styles.row, marginTop: 8 }}>
            <NumField label="SCE Alpha" field="sceAlpha" value={loss.params.sceAlpha} step={0.1} min={0} onChange={handleParam} />
            <NumField label="SCE Beta" field="sceBeta" value={loss.params.sceBeta} step={0.1} min={0} onChange={handleParam} />
            <NumField label="Label Smoothing" field="labelSmoothing" value={loss.params.labelSmoothing} step={0.01} min={0} max={1} onChange={handleParam} />
          </div>
          <span style={{ ...styles.subLabel, marginTop: 12 }}>GCBL (Contrastive Boundary)</span>
          <div style={{ ...styles.row, marginTop: 4 }}>
            <NumField label="GCBL Weight" field="gcblWeight" value={loss.params.gcblWeight} step={0.05} min={0} onChange={handleParam} />
            <NumField label="Embed Dim" field="gcblEmbedDim" value={loss.params.gcblEmbedDim} step={8} min={8} onChange={handleParam} />
            <NumField label="Temperature" field="gcblTemperature" value={loss.params.gcblTemperature} step={0.01} min={0.01} onChange={handleParam} />
            <NumField label="Max Samples" field="gcblMaxSamples" value={loss.params.gcblMaxSamples} step={128} min={16} onChange={handleParam} />
          </div>
        </div>
      )}
    </section>
  )
}

function NumField({ label, field, value, step = 0.05, min, max, onChange }) {
  return (
    <div className="field" style={{ flex: 1 }}>
      <label htmlFor={field}>{label}</label>
      <input
        id={field}
        type="number"
        step={step}
        {...(min !== undefined ? { min } : {})}
        {...(max !== undefined ? { max } : {})}
        value={value}
        onChange={e => onChange(field, Number(e.target.value))}
      />
    </div>
  )
}

function WeightBar({ dice, ce }) {
  const total = dice + ce
  const dicePercent = total > 0 ? (dice / total) * 100 : 50
  const cePercent = 100 - dicePercent

  return (
    <div style={styles.weightBarWrapper}>
      <div style={{ ...styles.weightSegment, width: `${dicePercent}%`, background: '#3b82f6' }}>
        <span style={styles.weightLabel}>Dice {(dicePercent).toFixed(0)}%</span>
      </div>
      <div style={{ ...styles.weightSegment, width: `${cePercent}%`, background: '#f59e0b' }}>
        <span style={styles.weightLabel}>CE {(cePercent).toFixed(0)}%</span>
      </div>
    </div>
  )
}

const styles = {
  section: {
    background: '#fff',
    border: '1px solid #e5e5e5',
    borderRadius: 8,
    padding: 20,
    marginBottom: 16,
  },
  row: {
    display: 'flex',
    gap: 12,
  },
  paramsBox: {
    background: '#f9f9f9',
    border: '1px solid #efefef',
    borderRadius: 6,
    padding: 12,
    marginTop: 4,
  },
  interpolated: {
    background: '#f0f4ff',
    color: '#3b5bbf',
    fontFamily: 'monospace',
    fontSize: '0.82rem',
    cursor: 'default',
  },
  hint: {
    fontSize: '0.72rem',
    color: '#aaa',
    marginTop: 3,
    display: 'block',
  },
  subLabel: {
    display: 'block',
    fontSize: '0.72rem',
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    marginBottom: 6,
  },
  weightBarWrapper: {
    display: 'flex',
    borderRadius: 4,
    overflow: 'hidden',
    height: 22,
    marginTop: 4,
  },
  weightSegment: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'width 0.2s',
  },
  weightLabel: {
    fontSize: '0.7rem',
    color: '#fff',
    fontWeight: 600,
  },
}
