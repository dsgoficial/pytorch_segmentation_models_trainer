import { useState } from 'react'
import TrainingPage from './pages/TrainingPage'
import PredictPage from './pages/PredictPage'

const TABS = [
  { id: 'training', label: 'Training' },
  { id: 'predict',  label: 'Predict' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('training')

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>PyTorch Segmentation Models Trainer — Config Builder</h1>
        <nav style={styles.tabs}>
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                ...styles.tab,
                ...(activeTab === tab.id ? styles.tabActive : {}),
              }}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <div style={styles.links}>
          <a
            href="/pytorch_segmentation_models_trainer/"
            style={styles.link}
            title="Documentation"
          >
            Docs
          </a>
          <a
            href="https://github.com/dsgoficial/pytorch_segmentation_models_trainer"
            style={styles.link}
            target="_blank"
            rel="noopener noreferrer"
            title="GitHub"
          >
            GitHub
          </a>
        </div>
      </header>

      <div style={styles.content}>
        {activeTab === 'training' && <TrainingPage />}
        {activeTab === 'predict'  && <PredictPage />}
      </div>
    </div>
  )
}

const styles = {
  app: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    background: '#fff',
    borderBottom: '1px solid #e5e5e5',
    padding: '0 24px',
    position: 'sticky',
    top: 0,
    zIndex: 10,
    display: 'flex',
    alignItems: 'center',
    gap: 24,
    height: 53,
  },
  title: {
    fontSize: '0.95rem',
    fontWeight: 600,
    margin: 0,
    whiteSpace: 'nowrap',
  },
  tabs: {
    display: 'flex',
    gap: 4,
  },
  tab: {
    background: 'none',
    border: 'none',
    borderBottom: '2px solid transparent',
    padding: '0 12px',
    height: 53,
    fontSize: '0.875rem',
    cursor: 'pointer',
    color: '#555',
    fontWeight: 500,
    transition: 'color 0.15s',
  },
  tabActive: {
    borderBottomColor: '#3b82f6',
    color: '#3b82f6',
  },
  content: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    height: 'calc(100vh - 53px)',
  },
  links: {
    marginLeft: 'auto',
    display: 'flex',
    gap: 16,
    alignItems: 'center',
  },
  link: {
    fontSize: '0.875rem',
    color: '#555',
    textDecoration: 'none',
    fontWeight: 500,
  },
}
