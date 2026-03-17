// NormalizationSection recebe mean[] e std[] do pai
// O número de bandas vem de model.inChannels — quando o usuário muda inChannels,
// o App ajusta o tamanho dos arrays automaticamente (ver App.jsx)

export default function NormalizationSection({ mean, std, onChange }) {
  function handleMean(index, value) {
    // Cria um novo array com o valor atualizado na posição correta
    // (nunca mutamos o array original — React precisa de um novo objeto para detectar a mudança)
    const newMean = mean.map((v, i) => (i === index ? Number(value) : v))
    onChange({ mean: newMean, std })
  }

  function handleStd(index, value) {
    const newStd = std.map((v, i) => (i === index ? Number(value) : v))
    onChange({ mean, std: newStd })
  }

  return (
    <section style={styles.section}>
      <h2>Normalization Parameters</h2>
      <p style={styles.hint}>Uma linha por banda. O número de bandas vem de <em>in_channels</em>.</p>

      <div style={styles.grid}>
        <span style={styles.colHeader}>Banda</span>
        <span style={styles.colHeader}>Mean</span>
        <span style={styles.colHeader}>Std</span>
      </div>

      {/* map transforma cada item do array em um elemento JSX */}
      {mean.map((meanVal, index) => (
        <div key={index} style={styles.grid}>
          <span style={styles.bandLabel}>Band {index + 1}</span>

          <input
            type="number"
            step="any"
            value={meanVal}
            onChange={e => handleMean(index, e.target.value)}
            style={styles.input}
          />

          <input
            type="number"
            step="any"
            value={std[index] ?? 1}
            onChange={e => handleStd(index, e.target.value)}
            style={styles.input}
          />
        </div>
      ))}
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
  hint: {
    fontSize: '0.78rem',
    color: '#888',
    marginBottom: 12,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '60px 1fr 1fr',
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
  bandLabel: {
    fontSize: '0.8rem',
    color: '#555',
    fontWeight: 500,
  },
  input: {
    width: '100%',
    padding: '5px 8px',
    border: '1px solid #d0d0d0',
    borderRadius: 5,
    fontSize: '0.8rem',
    outline: 'none',
  },
}
