// Cada tipo de loss tem parâmetros diferentes.
// Renderização condicional: mostramos campos diferentes dependendo do tipo selecionado.
import SearchableSelect from './SearchableSelect'

const LOSS_TYPES = [
  {
    value: 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss',
    label: 'WeightedDiceCrossEntropyLoss',
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

// Parâmetros padrão para cada tipo — quando o usuário troca o tipo,
// resetamos os params para os defaults daquele tipo
export const LOSS_DEFAULTS = {
  'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss': {
    diceWeight: 0.25,
    ceWeight: 0.75,
    smoothFactor: 0.1,
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

      {/* num_classes usa interpolação Hydra — valor derivado de model.classes */}
      <div className="field">
        <label>num_classes</label>
        <input value="${model.classes}" readOnly style={styles.interpolated} />
        <span style={styles.hint}>Referência automática ao campo <em>classes</em> do model.</span>
      </div>

      {/* Renderização condicional: cada tipo exibe seus próprios campos */}

      {loss.target === 'pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss' && (
        <div style={styles.paramsBox}>
          <div style={styles.row}>
            <div className="field" style={{ flex: 1 }}>
              <label>Dice Weight</label>
              <input
                type="number" step="0.05" min={0} max={1}
                value={loss.params.diceWeight}
                onChange={e => handleParam('diceWeight', Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label>CE Weight</label>
              <input
                type="number" step="0.05" min={0} max={1}
                value={loss.params.ceWeight}
                onChange={e => handleParam('ceWeight', Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label>Smooth Factor</label>
              <input
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
              <label>BCE Coef</label>
              <input
                type="number" step="0.05" min={0} max={1}
                value={loss.params.bceCoef}
                onChange={e => handleParam('bceCoef', Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label>Dice Coef</label>
              <input
                type="number" step="0.05" min={0} max={1}
                value={loss.params.diceCoef}
                onChange={e => handleParam('diceCoef', Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label>Tversky/Focal Coef</label>
              <input
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
            <label>Reduction</label>
            <select
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
    </section>
  )
}

// Barra visual mostrando a proporção dice vs ce
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
