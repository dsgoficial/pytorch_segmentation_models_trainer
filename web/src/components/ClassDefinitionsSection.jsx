// ClassDefinitionsSection: lista dinâmica de classes com nome e cor
// Ensina: adicionar/remover itens de arrays no estado React

const DEFAULT_COLORS = [
  '#000000', '#4444FF', '#FF4444', '#8B4513',
  '#90EE90', '#005700', '#FFFF44', '#FF8C00',
  '#800080', '#00CED1',
]

export default function ClassDefinitionsSection({ classes, onChange }) {
  function handleNameChange(index, value) {
    const newNames = classes.names.map((n, i) => (i === index ? value : n))
    onChange({ ...classes, names: newNames })
  }

  function handleColorChange(index, value) {
    const newColors = classes.colors.map((c, i) => (i === index ? value : c))
    onChange({ ...classes, colors: newColors })
  }

  function addClass() {
    const nextColor = DEFAULT_COLORS[classes.names.length % DEFAULT_COLORS.length]
    onChange({
      names: [...classes.names, `class_${classes.names.length}`],
      colors: [...classes.colors, nextColor],
    })
  }

  function removeClass(index) {
    // filter retorna um novo array sem o item na posição removida
    onChange({
      names: classes.names.filter((_, i) => i !== index),
      colors: classes.colors.filter((_, i) => i !== index),
    })
  }

  return (
    <section style={styles.section}>
      <div style={styles.titleRow}>
        <h2>Class Definitions</h2>
        <span style={styles.count}>{classes.names.length} classes</span>
      </div>

      <div style={styles.grid}>
        <span style={styles.colHeader}>ID</span>
        <span style={styles.colHeader}>Name</span>
        <span style={styles.colHeader}>Color</span>
        <span />
      </div>

      {classes.names.map((name, index) => (
        <div key={index} style={styles.grid}>
          <span style={styles.idLabel}>{index}</span>

          <input
            type="text"
            value={name}
            onChange={e => handleNameChange(index, e.target.value)}
            style={styles.input}
          />

          <div style={styles.colorWrapper}>
            <input
              type="color"
              value={classes.colors[index] ?? '#000000'}
              onChange={e => handleColorChange(index, e.target.value)}
              style={styles.colorInput}
              title={classes.colors[index]}
            />
            <span style={styles.colorHex}>{classes.colors[index]}</span>
          </div>

          {/* Não permite remover a classe 0 (background) */}
          {index === 0 ? (
            <span style={styles.bgLabel}>bg</span>
          ) : (
            <button onClick={() => removeClass(index)} style={styles.removeBtn} title="Remover classe">
              ×
            </button>
          )}
        </div>
      ))}

      <button onClick={addClass} style={styles.addBtn}>
        + Add Class
      </button>
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
  count: {
    fontSize: '0.75rem',
    color: '#888',
    background: '#f0f0f0',
    padding: '2px 8px',
    borderRadius: 10,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '30px 1fr 110px 28px',
    gap: 8,
    marginBottom: 6,
    alignItems: 'center',
  },
  colHeader: {
    fontSize: '0.75rem',
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
  },
  idLabel: {
    fontSize: '0.8rem',
    color: '#aaa',
    textAlign: 'center',
    fontVariantNumeric: 'tabular-nums',
  },
  input: {
    width: '100%',
    padding: '5px 8px',
    border: '1px solid #d0d0d0',
    borderRadius: 5,
    fontSize: '0.8rem',
    outline: 'none',
  },
  colorWrapper: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  colorInput: {
    width: 28,
    height: 28,
    border: '1px solid #d0d0d0',
    borderRadius: 4,
    padding: 1,
    cursor: 'pointer',
    background: 'none',
  },
  colorHex: {
    fontSize: '0.72rem',
    color: '#666',
    fontFamily: 'monospace',
  },
  removeBtn: {
    background: 'none',
    border: '1px solid #e0e0e0',
    borderRadius: 4,
    color: '#999',
    fontSize: '1rem',
    cursor: 'pointer',
    width: 28,
    height: 28,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
  },
  bgLabel: {
    fontSize: '0.7rem',
    color: '#bbb',
    textAlign: 'center',
  },
  addBtn: {
    marginTop: 8,
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
