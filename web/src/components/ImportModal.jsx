import { useState } from 'react'
import { load } from 'js-yaml'

export default function ImportModal({ onImport, onClose }) {
  const [text, setText]   = useState('')
  const [error, setError] = useState(null)

  function handleImport() {
    try {
      const data = load(text)
      if (!data || typeof data !== 'object') {
        setError('YAML inválido: o conteúdo não é um objeto.')
        return
      }
      onImport(data)
      onClose()
    } catch (e) {
      setError(`Erro ao parsear YAML: ${e.message}`)
    }
  }

  // Fecha ao clicar no overlay (fora do modal)
  function handleOverlayClick(e) {
    if (e.target === e.currentTarget) onClose()
  }

  return (
    <div style={styles.overlay} onClick={handleOverlayClick}>
      <div style={styles.modal}>
        <div style={styles.header}>
          <h2 style={styles.title}>Importar YAML</h2>
          <button onClick={onClose} style={styles.closeBtn}>×</button>
        </div>

        <p style={styles.hint}>
          Cole um YAML existente. Os campos do formulário serão preenchidos automaticamente.
          Campos não reconhecidos serão ignorados.
        </p>

        <textarea
          value={text}
          onChange={e => { setText(e.target.value); setError(null) }}
          placeholder="model:&#10;  _target_: segmentation_models_pytorch.Unet&#10;  encoder_name: resnet101&#10;  ..."
          style={styles.textarea}
          spellCheck={false}
        />

        {error && <div style={styles.error}>{error}</div>}

        <div style={styles.footer}>
          <button onClick={onClose}  style={styles.btnCancel}>Cancelar</button>
          <button onClick={handleImport} style={styles.btnImport} disabled={!text.trim()}>
            Importar
          </button>
        </div>
      </div>
    </div>
  )
}

const styles = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.45)',
    zIndex: 200,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modal: {
    background: '#fff',
    borderRadius: 10,
    width: 640,
    maxWidth: '95vw',
    maxHeight: '85vh',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 8px 40px rgba(0,0,0,0.2)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 20px',
    borderBottom: '1px solid #e5e5e5',
  },
  title: { fontSize: '1rem', fontWeight: 600, margin: 0 },
  closeBtn: {
    background: 'none',
    border: 'none',
    fontSize: '1.4rem',
    color: '#888',
    cursor: 'pointer',
    lineHeight: 1,
    padding: 0,
  },
  hint: {
    fontSize: '0.8rem',
    color: '#888',
    padding: '10px 20px 0',
    margin: 0,
  },
  textarea: {
    flex: 1,
    margin: '12px 20px',
    padding: 12,
    border: '1px solid #d0d0d0',
    borderRadius: 6,
    fontFamily: "'SF Mono', 'Fira Code', monospace",
    fontSize: '0.8rem',
    lineHeight: 1.6,
    resize: 'none',
    outline: 'none',
    minHeight: 300,
    color: '#1a1a1a',
  },
  error: {
    margin: '0 20px',
    padding: '8px 12px',
    background: '#fff0f0',
    border: '1px solid #fca5a5',
    borderRadius: 6,
    fontSize: '0.8rem',
    color: '#b91c1c',
  },
  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: 10,
    padding: '14px 20px',
    borderTop: '1px solid #e5e5e5',
  },
  btnCancel: {
    background: 'none',
    border: '1px solid #d0d0d0',
    borderRadius: 6,
    padding: '7px 16px',
    fontSize: '0.875rem',
    cursor: 'pointer',
  },
  btnImport: {
    background: '#3b82f6',
    border: 'none',
    borderRadius: 6,
    padding: '7px 20px',
    fontSize: '0.875rem',
    color: '#fff',
    cursor: 'pointer',
    fontWeight: 500,
    opacity: 1,
  },
}
