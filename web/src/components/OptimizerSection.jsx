// Mesmo padrão do LossSection: tipo determina quais campos aparecem
import schema from '../assets/schema.json'
import SearchableSelect from './SearchableSelect'

const schemaOptimizers = schema.torch?.optimizers ?? ['Adam', 'AdamW', 'SGD', 'RMSprop']
const OPTIMIZER_TYPES = schemaOptimizers.map(o => ({ value: `torch.optim.${o}`, label: o }))

export const OPTIMIZER_DEFAULTS = {
  'torch.optim.AdamW': { lr: 0.0001, weightDecay: 0.0001 },
  'torch.optim.Adam':  { lr: 0.0001, weightDecay: 0.0 },
  'torch.optim.SGD':   { lr: 0.01,   momentum: 0.9, weightDecay: 0.0001 },
}

export default function OptimizerSection({ optimizer, onChange }) {
  function handleTypeChange(newType) {
    const defaults = OPTIMIZER_DEFAULTS[newType] ?? { lr: 0.001, weightDecay: 0.0001 }
    onChange({ target: newType, params: defaults })
  }

  function handleParam(field, value) {
    onChange({ ...optimizer, params: { ...optimizer.params, [field]: value } })
  }

  return (
    <section style={styles.section}>
      <h2>Optimizer</h2>

      <div className="field">
        <label>Type (_target_)</label>
        <SearchableSelect
          value={optimizer.target}
          options={OPTIMIZER_TYPES}
          onChange={handleTypeChange}
        />
      </div>

      <div style={styles.row}>
        <div className="field" style={{ flex: 1 }}>
          <label>Learning Rate (lr)</label>
          <input
            type="number"
            step="any"
            value={optimizer.params.lr}
            onChange={e => handleParam('lr', Number(e.target.value))}
          />
        </div>

        <div className="field" style={{ flex: 1 }}>
          <label>Weight Decay</label>
          <input
            type="number"
            step="any"
            value={optimizer.params.weightDecay}
            onChange={e => handleParam('weightDecay', Number(e.target.value))}
          />
        </div>
      </div>

      {optimizer.target === 'torch.optim.SGD' && (
        <div className="field">
          <label>Momentum</label>
          <input
            type="number"
            step="0.01"
            min={0}
            max={1}
            value={optimizer.params.momentum}
            onChange={e => handleParam('momentum', Number(e.target.value))}
          />
        </div>
      )}
    </section>
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
  row: { display: 'flex', gap: 12 },
}
