// As opções vêm do schema.json gerado pelo generate_schema.py.
// Em desenvolvimento local, schema.json contém valores de fallback.
// No build de CI, contém os valores reais das bibliotecas instaladas.
import schema from '../assets/schema.json'
import SearchableSelect from './SearchableSelect'

const SMP_PREFIX = 'segmentation_models_pytorch.'

// Deriva as listas do schema em vez de hardcodar
const ARCHITECTURES = (schema.smp?.architectures ?? []).map(a => SMP_PREFIX + a)
const ENCODERS      = Object.keys(schema.smp?.encoders ?? {})

function weightsForEncoder(encoderName) {
  const settings = schema.smp?.encoders?.[encoderName]?.pretrained_settings ?? []
  return ['None', ...settings]
}

export default function ModelSection({ model, onChange }) {
  function handleChange(field, value) {
    onChange({ ...model, [field]: value })
  }

  // Quando o encoder muda, verifica se o peso atual ainda existe para o novo encoder.
  // Se não existir, reseta para 'None'.
  function handleEncoderChange(newEncoder) {
    const validWeights = weightsForEncoder(newEncoder)
    const weight = validWeights.includes(model.encoderWeights) ? model.encoderWeights : 'None'
    onChange({ ...model, encoderName: newEncoder, encoderWeights: weight })
  }

  const availableWeights = weightsForEncoder(model.encoderName)

  return (
    <section style={styles.section}>
      <div style={styles.titleRow}>
        <h2>Model</h2>
        {schema.generated_at && (
          <span style={styles.schemaTag} title={`Schema gerado em ${schema.generated_at}`}>
            smp {schema.library_versions?.segmentation_models_pytorch ?? '?'}
          </span>
        )}
      </div>

      <div className="field">
        <label>Architecture (_target_)</label>
        <SearchableSelect
          value={model.architecture}
          options={ARCHITECTURES.map(a => ({ value: a, label: a.replace(SMP_PREFIX, '') }))}
          onChange={v => handleChange('architecture', v)}
        />
      </div>

      <div className="field">
        <label>Encoder ({ENCODERS.length} disponíveis)</label>
        <SearchableSelect
          value={model.encoderName}
          options={ENCODERS.map(e => ({ value: e, label: e }))}
          onChange={handleEncoderChange}
        />
      </div>

      <div className="field">
        <label>Encoder Weights</label>
        <SearchableSelect
          value={model.encoderWeights}
          options={availableWeights.map(w => ({ value: w, label: w }))}
          onChange={v => handleChange('encoderWeights', v)}
        />
        <span style={styles.hint}>
          Pesos disponíveis para <em>{model.encoderName}</em>
        </span>
      </div>

      <div style={styles.row}>
        <div className="field" style={{ flex: 1 }}>
          <label>Input Channels (in_channels)</label>
          <input
            type="number"
            min={1}
            value={model.inChannels}
            onChange={e => handleChange('inChannels', Number(e.target.value))}
          />
        </div>
        <div className="field" style={{ flex: 1 }}>
          <label>Classes</label>
          <input
            type="number"
            min={1}
            value={model.classes}
            onChange={e => handleChange('classes', Number(e.target.value))}
          />
        </div>
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
  titleRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    marginBottom: 12,
  },
  schemaTag: {
    fontSize: '0.7rem',
    background: '#e8f5e9',
    color: '#2e7d32',
    padding: '2px 7px',
    borderRadius: 4,
    fontFamily: 'monospace',
    cursor: 'default',
  },
  row: { display: 'flex', gap: 12 },
  hint: { fontSize: '0.72rem', color: '#aaa', marginTop: 3, display: 'block' },
}
