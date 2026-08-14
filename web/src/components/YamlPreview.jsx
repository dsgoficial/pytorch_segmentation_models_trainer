import { dump } from 'js-yaml'

// Recebe o objeto de configuração e exibe como YAML
export default function YamlPreview({ config }) {
  const yamlText = dump(config, { lineWidth: -1, quotingType: '"' })

  function handleDownload() {
    const blob = new Blob([yamlText], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'train_config.yaml'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>YAML Preview</span>
        <button onClick={handleDownload} style={styles.button}>
          Download YAML
        </button>
      </div>
      <pre style={styles.pre}>{yamlText}</pre>
    </div>
  )
}

const styles = {
  container: {
    background: '#1e1e1e',
    borderRadius: 8,
    overflow: 'hidden',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px 16px',
    background: '#2d2d2d',
    borderBottom: '1px solid #3d3d3d',
  },
  title: {
    color: '#ccc',
    fontSize: '0.8rem',
    fontWeight: 600,
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
  },
  button: {
    background: '#3b82f6',
    color: '#fff',
    border: 'none',
    borderRadius: 5,
    padding: '5px 12px',
    fontSize: '0.8rem',
    cursor: 'pointer',
    fontWeight: 500,
  },
  pre: {
    flex: 1,
    padding: 16,
    color: '#d4d4d4',
    fontSize: '0.8rem',
    lineHeight: 1.6,
    overflowY: 'auto',
    fontFamily: "'SF Mono', 'Fira Code', monospace",
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
}
