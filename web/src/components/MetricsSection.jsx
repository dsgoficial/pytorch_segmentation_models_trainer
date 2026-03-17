// MetricsSection: lista de objetos onde cada item tem _target_ + params próprios.
// Conceito central: atualizar, adicionar e remover itens em uma lista de objetos.
import schema from '../assets/schema.json'
import SearchableSelect from './SearchableSelect'

const METRIC_TYPES = (schema.torchmetrics?.metrics ?? [
  'Accuracy', 'JaccardIndex', 'F1Score', 'Precision', 'Recall',
]).map(m => ({ value: `torchmetrics.${m}`, label: m }))

// Params comuns a todas as métricas multiclass
const COMMON_DEFAULTS = { task: 'multiclass', average: 'macro' }

// Algumas métricas têm params extras
const METRIC_EXTRA = {
  'torchmetrics.Accuracy':     {},
  'torchmetrics.JaccardIndex': {},
  'torchmetrics.F1Score':      { average: 'macro' },
  'torchmetrics.Precision':    { average: 'macro' },
  'torchmetrics.Recall':       { average: 'macro' },
}

function newMetric(target) {
  return { target, params: { ...COMMON_DEFAULTS, ...METRIC_EXTRA[target] } }
}

export default function MetricsSection({ metrics, onChange }) {
  // Atualiza um item na posição index — padrão que vamos reusar em Callbacks também
  function updateMetric(index, updated) {
    onChange(metrics.map((m, i) => (i === index ? updated : m)))
  }

  function handleTypeChange(index, newTarget) {
    updateMetric(index, newMetric(newTarget))
  }

  function handleParam(index, field, value) {
    const m = metrics[index]
    updateMetric(index, { ...m, params: { ...m.params, [field]: value } })
  }

  function addMetric() {
    onChange([...metrics, newMetric('torchmetrics.JaccardIndex')])
  }

  function removeMetric(index) {
    onChange(metrics.filter((_, i) => i !== index))
  }

  return (
    <section style={styles.section}>
      <div style={styles.titleRow}>
        <h2>Metrics</h2>
        <span style={styles.count}>{metrics.length} métricas</span>
      </div>

      {metrics.map((metric, index) => (
        <div key={index} style={styles.card}>
          <div style={styles.cardHeader}>
            <SearchableSelect
              value={metric.target}
              options={METRIC_TYPES}
              onChange={newTarget => handleTypeChange(index, newTarget)}
            />
            <button onClick={() => removeMetric(index)} style={styles.removeBtn} title="Remover">×</button>
          </div>

          <div style={styles.paramsRow}>
            {/* num_classes é sempre interpolado */}
            <div style={styles.paramField}>
              <label style={styles.paramLabel}>num_classes</label>
              <input value="${model.classes}" readOnly style={styles.interpolated} />
            </div>

            <div style={styles.paramField}>
              <label style={styles.paramLabel}>task</label>
              <select
                value={metric.params.task}
                onChange={e => handleParam(index, 'task', e.target.value)}
                style={styles.paramInput}
              >
                <option value="multiclass">multiclass</option>
                <option value="binary">binary</option>
                <option value="multilabel">multilabel</option>
              </select>
            </div>

            {/* average só existe em F1, Precision e Recall */}
            {'average' in metric.params && (
              <div style={styles.paramField}>
                <label style={styles.paramLabel}>average</label>
                <select
                  value={metric.params.average}
                  onChange={e => handleParam(index, 'average', e.target.value)}
                  style={styles.paramInput}
                >
                  <option value="macro">macro</option>
                  <option value="micro">micro</option>
                  <option value="weighted">weighted</option>
                  <option value="none">none</option>
                </select>
              </div>
            )}
          </div>
        </div>
      ))}

      <button onClick={addMetric} style={styles.addBtn}>+ Add Metric</button>
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
  titleRow: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 },
  count: { fontSize: '0.75rem', color: '#888', background: '#f0f0f0', padding: '2px 8px', borderRadius: 10 },
  card: {
    border: '1px solid #efefef',
    borderRadius: 6,
    marginBottom: 8,
    overflow: 'hidden',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    background: '#f9f9f9',
    borderBottom: '1px solid #efefef',
    padding: '6px 10px',
  },
  typeSelect: {
    flex: 1,
    padding: '4px 8px',
    border: '1px solid #d0d0d0',
    borderRadius: 5,
    fontSize: '0.82rem',
    background: '#fff',
  },
  removeBtn: {
    background: 'none',
    border: '1px solid #e0e0e0',
    borderRadius: 4,
    color: '#999',
    fontSize: '1rem',
    cursor: 'pointer',
    width: 26,
    height: 26,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    flexShrink: 0,
  },
  paramsRow: {
    display: 'flex',
    gap: 10,
    padding: '8px 10px',
    flexWrap: 'wrap',
  },
  paramField: { display: 'flex', flexDirection: 'column', gap: 3, minWidth: 100 },
  paramLabel: { fontSize: '0.72rem', fontWeight: 600, color: '#888', textTransform: 'uppercase', letterSpacing: '0.04em' },
  paramInput: {
    padding: '4px 7px',
    border: '1px solid #d0d0d0',
    borderRadius: 5,
    fontSize: '0.8rem',
    background: '#fff',
  },
  interpolated: {
    background: '#f0f4ff',
    color: '#3b5bbf',
    fontFamily: 'monospace',
    fontSize: '0.78rem',
    padding: '4px 7px',
    border: '1px solid #d0d7ff',
    borderRadius: 5,
    cursor: 'default',
    width: '100%',
  },
  addBtn: {
    marginTop: 4,
    background: 'none',
    border: '1px dashed #3b82f6',
    borderRadius: 6,
    color: '#3b82f6',
    padding: '6px 14px',
    fontSize: '0.8rem',
    cursor: 'pointer',
    width: '100%',
    fontWeight: 500,
  },
}
