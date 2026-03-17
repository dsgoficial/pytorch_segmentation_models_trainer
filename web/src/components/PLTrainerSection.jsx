export default function PLTrainerSection({ trainer, onChange }) {
  function handleChange(field, value) {
    onChange({ ...trainer, [field]: value })
  }

  return (
    <section style={styles.section}>
      <h2>PL Trainer</h2>

      <div className="field">
        <label>max_epochs</label>
        <input value="${hyperparameters.epochs}" readOnly style={styles.interpolated} />
        <span style={styles.hint}>Derivado de <em>hyperparameters.epochs</em>.</span>
      </div>

      <div style={styles.row}>
        <div className="field" style={{ flex: 1 }}>
          <label>Accelerator</label>
          <select value={trainer.accelerator} onChange={e => handleChange('accelerator', e.target.value)}>
            <option value="gpu">gpu</option>
            <option value="cpu">cpu</option>
            <option value="auto">auto</option>
            <option value="tpu">tpu</option>
          </select>
        </div>

        <div className="field" style={{ flex: 1 }}>
          <label>Devices</label>
          <input
            type="number"
            min={-1}
            value={trainer.devices}
            onChange={e => handleChange('devices', Number(e.target.value))}
          />
          <span style={styles.hint}>-1 = todos os GPUs</span>
        </div>

        <div className="field" style={{ flex: 1 }}>
          <label>Precision</label>
          <select value={trainer.precision} onChange={e => handleChange('precision', e.target.value)}>
            <option value={32}>32 (float32)</option>
            <option value={16}>16 (float16 / AMP)</option>
            <option value="bf16">bf16</option>
            <option value="16-mixed">16-mixed</option>
            <option value="32-true">32-true</option>
          </select>
        </div>
      </div>

      <div className="field" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <input
          type="checkbox"
          id="sync_batchnorm"
          checked={trainer.syncBatchnorm}
          onChange={e => handleChange('syncBatchnorm', e.target.checked)}
          style={{ width: 'auto' }}
        />
        <label htmlFor="sync_batchnorm" style={{ margin: 0 }}>
          sync_batchnorm <span style={styles.hint}>(recomendado para multi-GPU)</span>
        </label>
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
  row: { display: 'flex', gap: 12 },
  interpolated: {
    background: '#f0f4ff',
    color: '#3b5bbf',
    fontFamily: 'monospace',
    fontSize: '0.82rem',
    cursor: 'default',
  },
  hint: { fontSize: '0.72rem', color: '#aaa', marginTop: 3, display: 'inline' },
}
