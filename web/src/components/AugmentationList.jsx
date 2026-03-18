import schema from '../assets/schema.json'
import SearchableSelect from './SearchableSelect'

const schemaTransforms = schema.albumentations?.transforms ?? {}

const FIXED_EXCLUDE = new Set(['Normalize', 'ToTensorV2'])
const AUGMENTATION_OPTIONS = Object.entries(schemaTransforms)
  .filter(([name]) => !FIXED_EXCLUDE.has(name))
  .map(([name, info]) => ({ value: info.module, label: name }))

// Extrai os defaults do schema (chaves já são snake_case)
function defaultParamsFromSchema(transformName) {
  const info = schemaTransforms[transformName]
  const params = { p: 0.5 }
  if (!info) return params
  for (const [key, meta] of Object.entries(info.params ?? {})) {
    if (meta.default !== null && meta.default !== undefined) {
      params[key] = meta.default
    }
  }
  return params
}

function newAug(target) {
  const name = target.split('.').pop() // 'albumentations.HorizontalFlip' → 'HorizontalFlip'
  return { target, params: defaultParamsFromSchema(name) }
}

export default function AugmentationList({ augmentations, onChange }) {
  function updateAug(index, updated) {
    onChange(augmentations.map((a, i) => (i === index ? updated : a)))
  }

  function handleTypeChange(index, newTarget) {
    updateAug(index, newAug(newTarget))
  }

  function handleParam(index, key, value) {
    const a = augmentations[index]
    updateAug(index, { ...a, params: { ...a.params, [key]: value } })
  }

  function addAug() {
    onChange([...augmentations, newAug('albumentations.HorizontalFlip')])
  }

  function removeAug(index) {
    onChange(augmentations.filter((_, i) => i !== index))
  }

  return (
    <div>
      <span style={styles.subLabel}>augmentation_list</span>

      {augmentations.map((aug, index) => {
        const name = aug.target.split('.').pop()
        const schemaParams = schemaTransforms[name]?.params ?? {}

        return (
          <div key={index} style={styles.augCard}>
            <div style={styles.augHeader}>
              <SearchableSelect
                value={aug.target}
                options={AUGMENTATION_OPTIONS}
                onChange={newTarget => handleTypeChange(index, newTarget)}
              />
              <button onClick={() => removeAug(index)} style={styles.removeBtn}>×</button>
            </div>

            <div style={styles.augParams}>
              {/* p: sempre presente, renderizado como slider */}
              <ProbField
                value={aug.params.p ?? 0.5}
                onChange={v => handleParam(index, 'p', v)}
              />

              {/* Restante dos params: renderizados dinamicamente a partir do schema */}
              {Object.entries(schemaParams).map(([key]) => (
                <DynamicParamField
                  key={key}
                  label={key}
                  value={aug.params[key]}
                  onChange={v => handleParam(index, key, v)}
                />
              ))}
            </div>
          </div>
        )
      })}

      <button onClick={addAug} style={styles.addBtn}>+ Add Transform</button>

      {/* Transforms fixos — sempre presentes, não editáveis aqui */}
      <FixedTransform
        name="albumentations.Normalize"
        detail="mean/std → ${normalization_parameters.mean/std}"
      />
      <FixedTransform
        name="albumentations.pytorch.transforms.ToTensorV2"
        detail="always_apply: true"
      />
    </div>
  )
}

// ── Sub-componentes ──────────────────────────────────────────────────────────

function ProbField({ value, onChange }) {
  return (
    <div style={styles.paramField}>
      <span style={styles.paramLabel}>p (probabilidade)</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <input
          type="range"
          min={0} max={1} step={0.05}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          style={{ flex: 1, accentColor: '#3b82f6' }}
        />
        <span style={styles.probValue}>{Number(value).toFixed(2)}</span>
      </div>
    </div>
  )
}

// Renderiza um campo de acordo com o tipo do valor atual
function DynamicParamField({ label, value, onChange }) {
  const type = typeof value

  return (
    <div style={styles.paramField}>
      <span style={styles.paramLabel}>{label}</span>
      {type === 'boolean' ? (
        <input
          type="checkbox"
          checked={!!value}
          onChange={e => onChange(e.target.checked)}
          style={{ width: 'auto', margin: 0 }}
        />
      ) : type === 'number' ? (
        <input
          type="number"
          step="any"
          value={value ?? 0}
          onChange={e => onChange(Number(e.target.value))}
          style={styles.paramInput}
        />
      ) : (
        <input
          type="text"
          value={value ?? ''}
          onChange={e => onChange(e.target.value)}
          style={styles.paramInput}
        />
      )}
    </div>
  )
}

function FixedTransform({ name, detail }) {
  return (
    <div style={styles.fixedCard}>
      <span style={styles.fixedTag}>fixo</span>
      <span style={styles.fixedName}>{name}</span>
      <span style={styles.fixedDetail}>{detail}</span>
    </div>
  )
}

// ── Estilos ──────────────────────────────────────────────────────────────────

const styles = {
  subLabel: {
    display: 'block',
    fontSize: '0.72rem',
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    marginBottom: 8,
  },
  augCard: {
    border: '1px solid #e8e8e8',
    borderRadius: 6,
    marginBottom: 6,
    overflow: 'visible',
  },
  augHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    background: '#f9f9f9',
    borderBottom: '1px solid #efefef',
    padding: '5px 8px',
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
  augParams: {
    padding: '8px 10px',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  paramField: { display: 'flex', flexDirection: 'column', gap: 2 },
  paramLabel: { fontSize: '0.7rem', fontWeight: 600, color: '#aaa', textTransform: 'uppercase' },
  paramInput: {
    padding: '4px 7px',
    border: '1px solid #d0d0d0',
    borderRadius: 4,
    fontSize: '0.8rem',
    width: '100%',
  },
  probValue: {
    fontSize: '0.78rem',
    fontFamily: 'monospace',
    color: '#555',
    minWidth: 30,
    textAlign: 'right',
  },
  addBtn: {
    marginBottom: 6,
    background: 'none',
    border: '1px dashed #3b82f6',
    borderRadius: 6,
    color: '#3b82f6',
    padding: '5px 14px',
    fontSize: '0.8rem',
    cursor: 'pointer',
    width: '100%',
    fontWeight: 500,
  },
  fixedCard: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    border: '1px solid #e8e8e8',
    borderRadius: 6,
    padding: '6px 10px',
    marginBottom: 6,
    background: '#fafafa',
    opacity: 0.75,
  },
  fixedTag: {
    fontSize: '0.65rem',
    fontWeight: 700,
    color: '#888',
    background: '#ebebeb',
    padding: '1px 5px',
    borderRadius: 3,
    flexShrink: 0,
  },
  fixedName: {
    fontSize: '0.78rem',
    color: '#555',
    fontFamily: 'monospace',
    flex: 1,
  },
  fixedDetail: {
    fontSize: '0.7rem',
    color: '#bbb',
    fontStyle: 'italic',
  },
}
