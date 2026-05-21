export default function ExperimentsRunnerSection({ runner, onChange }) {
  function handleField(field, value) {
    onChange({ ...runner, [field]: value })
  }

  function handleSeedsText(text) {
    const parsed = text
      .split(',')
      .map(s => s.trim())
      .filter(s => s !== '')
      .map(Number)
      .filter(n => !isNaN(n))
    onChange({ ...runner, seeds: parsed, seedsText: text, useNRuns: false })
  }

  function handleNRuns(value) {
    onChange({ ...runner, nRuns: value, seeds: [], seedsText: '', useNRuns: true })
  }

  return (
    <section style={styles.section}>
      <h2>Experiments Runner</h2>

      <div className="field">
        <label>output_base_dir</label>
        <input
          type="text"
          value={runner.outputBaseDir}
          onChange={e => handleField('outputBaseDir', e.target.value)}
          placeholder="/path/to/output/experiments"
        />
      </div>

      <div style={styles.modeRow}>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="seedMode"
            checked={!runner.useNRuns}
            onChange={() => onChange({ ...runner, useNRuns: false })}
          />
          {' '}Explicit seeds
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="seedMode"
            checked={runner.useNRuns}
            onChange={() => onChange({ ...runner, useNRuns: true })}
          />
          {' '}Random n_runs
        </label>
      </div>

      {!runner.useNRuns && (
        <div className="field">
          <label>seeds (comma-separated)</label>
          <input
            type="text"
            value={runner.seedsText}
            onChange={e => handleSeedsText(e.target.value)}
            placeholder="42, 101, 28"
          />
          <span style={styles.hint}>
            {runner.seeds.length > 0
              ? `${runner.seeds.length} run(s): [${runner.seeds.join(', ')}]`
              : 'Enter at least one seed.'}
          </span>
        </div>
      )}

      {runner.useNRuns && (
        <div className="field">
          <label htmlFor="runner-nRuns">n_runs</label>
          <input
            id="runner-nRuns"
            type="number"
            min={1}
            value={runner.nRuns}
            onChange={e => handleNRuns(Number(e.target.value))}
          />
          <span style={styles.hint}>Seeds will be generated randomly at runtime.</span>
        </div>
      )}

      <div style={styles.checkboxRow}>
        <CheckboxField
          id="runner-save-summary"
          label="save_summary"
          checked={runner.saveSummary}
          onChange={v => handleField('saveSummary', v)}
        />
        <CheckboxField
          id="runner-resume"
          label="resume"
          checked={runner.resume}
          onChange={v => handleField('resume', v)}
        />
      </div>
    </section>
  )
}

function CheckboxField({ id, label, checked, onChange }) {
  return (
    <div style={styles.checkItem}>
      <input
        type="checkbox"
        id={id}
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        style={{ width: 'auto', margin: 0 }}
      />
      <label htmlFor={id} style={styles.checkLabel}>{label}</label>
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
  modeRow: {
    display: 'flex',
    gap: 20,
    marginBottom: 14,
  },
  radioLabel: {
    fontSize: '0.85rem',
    color: '#444',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  checkboxRow: {
    display: 'flex',
    gap: 16,
    marginTop: 8,
    flexWrap: 'wrap',
  },
  checkItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  checkLabel: {
    fontSize: '0.8rem',
    color: '#444',
    margin: 0,
    cursor: 'pointer',
  },
  hint: {
    fontSize: '0.72rem',
    color: '#aaa',
    marginTop: 3,
    display: 'block',
  },
}
