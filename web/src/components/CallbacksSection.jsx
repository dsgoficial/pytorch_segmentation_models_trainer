// CallbacksSection: o mesmo padrão de lista de objetos do MetricsSection,
// mas cada callback tem campos completamente diferentes.
// Isso reforça: renderização condicional + lista de objetos com formas distintas.
import SearchableSelect from './SearchableSelect'
import { CALLBACK_DEFAULTS } from './callbacksDefaults'

const CALLBACK_TYPES = [
  { value: 'pytorch_lightning.callbacks.EarlyStopping',      label: 'EarlyStopping' },
  { value: 'pytorch_lightning.callbacks.ModelCheckpoint',    label: 'ModelCheckpoint' },
  { value: 'pytorch_lightning.callbacks.LearningRateMonitor',label: 'LearningRateMonitor' },
  {
    value: 'pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.EnhancedImageSegmentationResultCallback',
    label: 'EnhancedImageSegmentationResultCallback',
  },
]

function newCallback(target) {
  return { target, params: { ...CALLBACK_DEFAULTS[target] } }
}

export default function CallbacksSection({ callbacks, onChange }) {
  function updateCallback(index, updated) {
    onChange(callbacks.map((cb, i) => (i === index ? updated : cb)))
  }

  function handleTypeChange(index, newTarget) {
    updateCallback(index, newCallback(newTarget))
  }

  function handleParam(index, field, value) {
    const cb = callbacks[index]
    updateCallback(index, { ...cb, params: { ...cb.params, [field]: value } })
  }

  function addCallback() {
    onChange([...callbacks, newCallback('pytorch_lightning.callbacks.EarlyStopping')])
  }

  function removeCallback(index) {
    onChange(callbacks.filter((_, i) => i !== index))
  }

  const shortLabel = target => CALLBACK_TYPES.find(t => t.value === target)?.label ?? target

  return (
    <section style={styles.section}>
      <div style={styles.titleRow}>
        <h2>Callbacks</h2>
        <span style={styles.count}>{callbacks.length} callbacks</span>
      </div>

      {callbacks.map((cb, index) => (
        <div key={index} style={styles.card}>
          <div style={styles.cardHeader}>
            <span style={styles.badge}>{shortLabel(cb.target)}</span>
            <SearchableSelect
              value={cb.target}
              options={CALLBACK_TYPES}
              onChange={newTarget => handleTypeChange(index, newTarget)}
            />
            <button onClick={() => removeCallback(index)} style={styles.removeBtn} title="Remover">×</button>
          </div>

          <div style={styles.paramsBox}>
            {cb.target === 'pytorch_lightning.callbacks.EarlyStopping' && (
              <EarlyStoppingParams params={cb.params} onChange={(f, v) => handleParam(index, f, v)} />
            )}
            {cb.target === 'pytorch_lightning.callbacks.ModelCheckpoint' && (
              <ModelCheckpointParams params={cb.params} onChange={(f, v) => handleParam(index, f, v)} />
            )}
            {cb.target === 'pytorch_lightning.callbacks.LearningRateMonitor' && (
              <LRMonitorParams params={cb.params} onChange={(f, v) => handleParam(index, f, v)} />
            )}
            {cb.target.includes('EnhancedImageSegmentationResultCallback') && (
              <ImageCallbackParams params={cb.params} onChange={(f, v) => handleParam(index, f, v)} />
            )}
          </div>
        </div>
      ))}

      <button onClick={addCallback} style={styles.addBtn}>+ Add Callback</button>
    </section>
  )
}

// ── Sub-componentes de params ────────────────────────────────────────────────
// Cada callback delega a renderização dos params para um componente dedicado.
// Isso mantém o componente pai limpo e cada callback com sua própria lógica.

function EarlyStoppingParams({ params, onChange }) {
  return (
    <div style={styles.grid2}>
      <Field label="monitor">
        <input value={params.monitor} onChange={e => onChange('monitor', e.target.value)} style={styles.input} />
      </Field>
      <Field label="mode">
        <ModeSelect value={params.mode} onChange={v => onChange('mode', v)} />
      </Field>
      <Field label="patience">
        <input type="number" min={1} value={params.patience} onChange={e => onChange('patience', Number(e.target.value))} style={styles.input} />
      </Field>
      <Field label="min_delta">
        <input type="number" step="any" value={params.minDelta} onChange={e => onChange('minDelta', Number(e.target.value))} style={styles.input} />
      </Field>
    </div>
  )
}

function ModelCheckpointParams({ params, onChange }) {
  return (
    <>
      <div style={styles.grid2}>
        <Field label="monitor">
          <input value={params.monitor} onChange={e => onChange('monitor', e.target.value)} style={styles.input} />
        </Field>
        <Field label="mode">
          <ModeSelect value={params.mode} onChange={v => onChange('mode', v)} />
        </Field>
        <Field label="save_top_k">
          <input type="number" min={1} value={params.saveTopK} onChange={e => onChange('saveTopK', Number(e.target.value))} style={styles.input} />
        </Field>
        <Field label="save_weights_only">
          <CheckboxField checked={params.saveWeightsOnly} onChange={v => onChange('saveWeightsOnly', v)} />
        </Field>
      </div>
      <Field label="filename" style={{ marginTop: 6 }}>
        <input value={params.filename} onChange={e => onChange('filename', e.target.value)} style={styles.input} />
      </Field>
    </>
  )
}

function LRMonitorParams({ params, onChange }) {
  return (
    <Field label="logging_interval">
      <select value={params.loggingInterval} onChange={e => onChange('loggingInterval', e.target.value)} style={styles.input}>
        <option value="epoch">epoch</option>
        <option value="step">step</option>
      </select>
    </Field>
  )
}

function ImageCallbackParams({ params, onChange }) {
  return (
    <div style={styles.grid2}>
      <Field label="n_samples">
        <input type="number" min={1} value={params.nSamples} onChange={e => onChange('nSamples', Number(e.target.value))} style={styles.input} />
      </Field>
      <Field label="log_every_k_epochs">
        <input type="number" min={1} value={params.logEveryKEpochs} onChange={e => onChange('logEveryKEpochs', Number(e.target.value))} style={styles.input} />
      </Field>
      <Field label="band_indices (CSV)">
        <input value={params.bandIndices} onChange={e => onChange('bandIndices', e.target.value)} style={styles.input} placeholder="0, 1, 2" />
      </Field>
      <Field label="alpha_mask">
        <input type="number" step="0.1" min={0} max={1} value={params.alphaMask} onChange={e => onChange('alphaMask', Number(e.target.value))} style={styles.input} />
      </Field>
      <Field label="max_workers">
        <input type="number" min={1} value={params.maxWorkers} onChange={e => onChange('maxWorkers', Number(e.target.value))} style={styles.input} />
      </Field>
      <Field label="show_class_legend">
        <CheckboxField checked={params.showClassLegend} onChange={v => onChange('showClassLegend', v)} />
      </Field>
    </div>
  )
}

// ── Componentes utilitários ──────────────────────────────────────────────────

function Field({ label, children, style }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3, ...style }}>
      <span style={styles.paramLabel}>{label}</span>
      {children}
    </div>
  )
}

function ModeSelect({ value, onChange }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} style={styles.input}>
      <option value="min">min</option>
      <option value="max">max</option>
    </select>
  )
}

function CheckboxField({ checked, onChange }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', height: 30 }}>
      <input
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        style={{ width: 'auto', margin: 0 }}
      />
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
  titleRow: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 },
  count: { fontSize: '0.75rem', color: '#888', background: '#f0f0f0', padding: '2px 8px', borderRadius: 10 },
  card: { border: '1px solid #efefef', borderRadius: 6, marginBottom: 8, overflow: 'hidden' },
  cardHeader: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: '#f9f9f9', borderBottom: '1px solid #efefef', padding: '6px 10px',
  },
  badge: {
    fontSize: '0.7rem', fontWeight: 700, color: '#3b5bbf',
    background: '#e8eeff', padding: '2px 7px', borderRadius: 4, flexShrink: 0,
  },
  typeSelect: {
    flex: 1, padding: '4px 8px', border: '1px solid #d0d0d0',
    borderRadius: 5, fontSize: '0.78rem', background: '#fff', color: '#555',
  },
  removeBtn: {
    background: 'none', border: '1px solid #e0e0e0', borderRadius: 4,
    color: '#999', fontSize: '1rem', cursor: 'pointer',
    width: 26, height: 26, display: 'flex', alignItems: 'center',
    justifyContent: 'center', padding: 0, flexShrink: 0,
  },
  paramsBox: { padding: '10px 12px' },
  grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 12px' },
  paramLabel: { fontSize: '0.72rem', fontWeight: 600, color: '#888', textTransform: 'uppercase', letterSpacing: '0.04em' },
  input: {
    width: '100%', padding: '5px 8px', border: '1px solid #d0d0d0',
    borderRadius: 5, fontSize: '0.8rem', outline: 'none', background: '#fff',
  },
  addBtn: {
    marginTop: 4, background: 'none', border: '1px dashed #3b82f6',
    borderRadius: 6, color: '#3b82f6', padding: '6px 14px',
    fontSize: '0.8rem', cursor: 'pointer', width: '100%', fontWeight: 500,
  },
}
