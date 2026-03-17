export default function HyperparametersSection({ hyperparameters, onChange }) {
  function handleChange(field, value) {
    onChange({ ...hyperparameters, [field]: value })
  }

  return (
    <section style={styles.section}>
      <h2>Hyperparameters</h2>

      <div style={styles.row}>
        <div className="field" style={{ flex: 1 }}>
          <label>Batch Size</label>
          <input
            type="number"
            min={1}
            value={hyperparameters.batchSize}
            onChange={e => handleChange('batchSize', Number(e.target.value))}
          />
        </div>

        <div className="field" style={{ flex: 1 }}>
          <label>Epochs</label>
          <input
            type="number"
            min={1}
            value={hyperparameters.epochs}
            onChange={e => handleChange('epochs', Number(e.target.value))}
          />
        </div>
      </div>

      <div className="field">
        <label>Max LR</label>
        <input
          type="number"
          step="any"
          value={hyperparameters.maxLr}
          onChange={e => handleChange('maxLr', Number(e.target.value))}
        />
        <span style={styles.hint}>ex: 0.0001 (= 1e-4)</span>
      </div>
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
  row: {
    display: 'flex',
    gap: 12,
  },
  hint: {
    fontSize: '0.72rem',
    color: '#aaa',
    marginTop: 3,
    display: 'block',
  },
}
