// SearchableSelect: combo box com campo de busca embutido.
//
// Novos conceitos React:
//   useRef    → acessa o elemento DOM diretamente (para detectar clique fora)
//   useEffect → executa código como efeito colateral (adiciona/remove event listener)
//   cleanup   → a função retornada pelo useEffect remove o listener quando o componente desmonta

import { useState, useRef, useEffect } from 'react'

// options: [{ value: string, label: string }]
export default function SearchableSelect({ value, options, onChange, placeholder }) {
  const [open, setOpen]   = useState(false)
  const [query, setQuery] = useState('')
  const containerRef      = useRef(null)   // referência ao elemento DOM raiz
  const inputRef          = useRef(null)   // referência ao input de busca

  const currentLabel = options.find(o => o.value === value)?.label ?? value

  const filtered = query
    ? options.filter(o => o.label.toLowerCase().includes(query.toLowerCase()))
    : options

  // useEffect: roda depois de cada render.
  // Aqui usamos para adicionar um listener que fecha o dropdown ao clicar fora.
  useEffect(() => {
    function handleClickOutside(event) {
      // containerRef.current é o div raiz deste componente.
      // Se o clique foi FORA dele, fechamos.
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setOpen(false)
        setQuery('')
      }
    }

    document.addEventListener('mousedown', handleClickOutside)

    // Cleanup: o React chama essa função antes de remover o listener
    // (evita memory leak se o componente for desmontado com o dropdown aberto)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, []) // [] = roda só na montagem, não em cada render

  function handleOpen() {
    setOpen(true)
    // Foca o input automaticamente ao abrir (setTimeout porque o input
    // ainda não está no DOM no momento em que setOpen(true) é chamado)
    setTimeout(() => inputRef.current?.focus(), 0)
  }

  function handleSelect(optValue) {
    onChange(optValue)
    setOpen(false)
    setQuery('')
  }

  return (
    <div ref={containerRef} style={styles.container}>
      {/* Trigger: mostra o valor atual ou o campo de busca quando aberto */}
      <div
        onClick={open ? undefined : handleOpen}
        style={{ ...styles.trigger, ...(open ? styles.triggerOpen : {}) }}
      >
        {open ? (
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder={placeholder ?? `Buscar em ${options.length} opções...`}
            style={styles.searchInput}
          />
        ) : (
          <span style={styles.valueLabel}>{currentLabel}</span>
        )}
        <span style={styles.arrow} onClick={open ? () => { setOpen(false); setQuery('') } : handleOpen}>
          {open ? '▲' : '▼'}
        </span>
      </div>

      {/* Dropdown com as opções filtradas */}
      {open && (
        <div style={styles.dropdown}>
          {filtered.length === 0 ? (
            <div style={styles.noResults}>Nenhum resultado para "{query}"</div>
          ) : (
            filtered.map(opt => (
              <div
                key={opt.value}
                onClick={() => handleSelect(opt.value)}
                style={{
                  ...styles.option,
                  ...(opt.value === value ? styles.optionActive : {}),
                }}
              >
                {opt.label}
              </div>
            ))
          )}
          {filtered.length > 0 && query && (
            <div style={styles.counter}>{filtered.length} / {options.length}</div>
          )}
        </div>
      )}
    </div>
  )
}

const styles = {
  container: {
    position: 'relative',
    width: '100%',
  },
  trigger: {
    display: 'flex',
    alignItems: 'center',
    border: '1px solid #d0d0d0',
    borderRadius: 6,
    background: '#fff',
    cursor: 'pointer',
    minHeight: 34,
    padding: '0 8px',
    transition: 'border-color 0.15s',
  },
  triggerOpen: {
    borderColor: '#3b82f6',
    boxShadow: '0 0 0 2px rgba(59,130,246,0.15)',
  },
  valueLabel: {
    flex: 1,
    fontSize: '0.875rem',
    color: '#1a1a1a',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  searchInput: {
    flex: 1,
    border: 'none',
    outline: 'none',
    fontSize: '0.875rem',
    background: 'transparent',
    padding: '6px 0',
    width: '100%',
  },
  arrow: {
    fontSize: '0.6rem',
    color: '#aaa',
    marginLeft: 6,
    flexShrink: 0,
    userSelect: 'none',
  },
  dropdown: {
    position: 'absolute',
    top: '100%',
    left: 0,
    right: 0,
    zIndex: 100,
    background: '#fff',
    border: '1px solid #d0d0d0',
    borderRadius: 6,
    boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
    maxHeight: 260,
    overflowY: 'auto',
    marginTop: 3,
  },
  option: {
    padding: '8px 12px',
    fontSize: '0.85rem',
    cursor: 'pointer',
    color: '#333',
    borderBottom: '1px solid #f5f5f5',
  },
  optionActive: {
    background: '#eff6ff',
    color: '#1d4ed8',
    fontWeight: 600,
  },
  noResults: {
    padding: '10px 12px',
    fontSize: '0.82rem',
    color: '#aaa',
    textAlign: 'center',
  },
  counter: {
    padding: '4px 12px',
    fontSize: '0.7rem',
    color: '#bbb',
    textAlign: 'right',
    borderTop: '1px solid #f0f0f0',
  },
}
