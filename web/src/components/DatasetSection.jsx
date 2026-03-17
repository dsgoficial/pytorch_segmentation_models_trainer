// DatasetSection é usado duas vezes no App: uma para train, uma para val.
// A prop "label" muda o título. Todo o resto funciona igual.
// Isso é reutilização de componente: mesmo código, comportamento diferente via props.

import AugmentationList from './AugmentationList'

export default function DatasetSection({ label, dataset, onChange }) {
  // Atualiza um campo de primeiro nível (input_csv_path, use_rasterio etc.)
  function handleField(field, value) {
    onChange({ ...dataset, [field]: value })
  }

  // Atualiza um campo dentro de data_loader (estado aninhado)
  // Precisamos copiar tanto o objeto raiz quanto o sub-objeto
  function handleLoader(field, value) {
    onChange({
      ...dataset,
      dataLoader: { ...dataset.dataLoader, [field]: value },
    })
  }

  return (
    <section style={styles.section}>
      {/* O título muda via prop — "Train Dataset" ou "Val Dataset" */}
      <h2>{label}</h2>

      <div className="field">
        <label>input_csv_path</label>
        <input
          type="text"
          value={dataset.csvPath}
          onChange={e => handleField('csvPath', e.target.value)}
          placeholder="/path/to/dataset.csv"
        />
      </div>

      <div className="field">
        <label>n_classes</label>
        <input value="${model.classes}" readOnly style={styles.interpolated} />
      </div>

      <div style={styles.checkboxRow}>
        <CheckboxField
          id={`${label}-rasterio`}
          label="use_rasterio"
          checked={dataset.useRasterio}
          onChange={v => handleField('useRasterio', v)}
        />
        <CheckboxField
          id={`${label}-reset-aug`}
          label="reset_augmentation_function"
          checked={dataset.resetAugmentation}
          onChange={v => handleField('resetAugmentation', v)}
        />
      </div>

      {/* Estado aninhado: data_loader é um sub-objeto dentro de dataset */}
      <div style={styles.subSection}>
        <span style={styles.subLabel}>data_loader</span>

        <div style={styles.loaderGrid}>
          <NumField label="num_workers"     value={dataset.dataLoader.numWorkers}   min={0}  onChange={v => handleLoader('numWorkers', v)} />
          <NumField label="prefetch_factor" value={dataset.dataLoader.prefetchFactor} min={1} onChange={v => handleLoader('prefetchFactor', v)} />
          <CheckboxField id={`${label}-shuffle`}   label="shuffle"            checked={dataset.dataLoader.shuffle}           onChange={v => handleLoader('shuffle', v)} />
          <CheckboxField id={`${label}-pin`}       label="pin_memory"         checked={dataset.dataLoader.pinMemory}         onChange={v => handleLoader('pinMemory', v)} />
          <CheckboxField id={`${label}-drop`}      label="drop_last"          checked={dataset.dataLoader.dropLast}          onChange={v => handleLoader('dropLast', v)} />
          <CheckboxField id={`${label}-persist`}   label="persistent_workers" checked={dataset.dataLoader.persistentWorkers} onChange={v => handleLoader('persistentWorkers', v)} />
        </div>
      </div>

      <div style={styles.subSection}>
        <AugmentationList
          augmentations={dataset.augmentations}
          onChange={augs => handleField('augmentations', augs)}
        />
      </div>
    </section>
  )
}

// ── Sub-componentes utilitários ──────────────────────────────────────────────

function NumField({ label, value, min = 0, onChange }) {
  return (
    <div style={styles.fieldCompact}>
      <span style={styles.compactLabel}>{label}</span>
      <input
        type="number"
        min={min}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={styles.compactInput}
      />
    </div>
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

// ── Estilos ──────────────────────────────────────────────────────────────────

const styles = {
  section: {
    background: '#fff',
    border: '1px solid #e5e5e5',
    borderRadius: 8,
    padding: 20,
    marginBottom: 16,
  },
  interpolated: {
    background: '#f0f4ff',
    color: '#3b5bbf',
    fontFamily: 'monospace',
    fontSize: '0.82rem',
    cursor: 'default',
  },
  checkboxRow: {
    display: 'flex',
    gap: 16,
    marginBottom: 14,
    flexWrap: 'wrap',
  },
  subSection: {
    background: '#f9f9f9',
    border: '1px solid #efefef',
    borderRadius: 6,
    padding: 12,
    marginBottom: 12,
  },
  subLabel: {
    display: 'block',
    fontSize: '0.72rem',
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    marginBottom: 10,
  },
  loaderGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '8px 16px',
    alignItems: 'center',
  },
  fieldCompact: {
    display: 'flex',
    flexDirection: 'column',
    gap: 3,
  },
  compactLabel: {
    fontSize: '0.72rem',
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase',
  },
  compactInput: {
    padding: '5px 8px',
    border: '1px solid #d0d0d0',
    borderRadius: 5,
    fontSize: '0.8rem',
    width: '100%',
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
}
