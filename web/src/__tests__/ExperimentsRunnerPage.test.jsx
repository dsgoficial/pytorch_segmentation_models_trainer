import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ExperimentsRunnerPage from '../pages/ExperimentsRunnerPage'

describe('ExperimentsRunnerPage', () => {
  it('renders the mode badge with "mode: run-experiments"', () => {
    render(<ExperimentsRunnerPage />)
    expect(screen.getByText('mode: run-experiments')).toBeInTheDocument()
  })

  it('renders all section headings', () => {
    render(<ExperimentsRunnerPage />)
    expect(screen.getByRole('heading', { name: 'Experiments Runner' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Model' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Loss Function' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Optimizer' })).toBeInTheDocument()
  })

  it('renders train and val dataset sections', () => {
    render(<ExperimentsRunnerPage />)
    expect(screen.getByText('Train Dataset')).toBeInTheDocument()
    expect(screen.getByText('Val Dataset')).toBeInTheDocument()
  })

  it('renders YAML preview containing "mode: run-experiments"', () => {
    render(<ExperimentsRunnerPage />)
    const pre = document.querySelector('pre')
    expect(pre).not.toBeNull()
    expect(pre.textContent).toContain('mode: run-experiments')
  })

  it('YAML preview contains experiments_runner block', () => {
    render(<ExperimentsRunnerPage />)
    const pre = document.querySelector('pre')
    expect(pre.textContent).toContain('experiments_runner:')
  })

  it('YAML preview contains output_base_dir field', () => {
    render(<ExperimentsRunnerPage />)
    const pre = document.querySelector('pre')
    expect(pre.textContent).toContain('output_base_dir:')
  })

  it('YAML preview contains seeds array by default', () => {
    render(<ExperimentsRunnerPage />)
    const pre = document.querySelector('pre')
    expect(pre.textContent).toContain('seeds:')
  })

  it('YAML preview contains pl_trainer block', () => {
    render(<ExperimentsRunnerPage />)
    const pre = document.querySelector('pre')
    expect(pre.textContent).toContain('pl_trainer:')
  })

  it('YAML preview contains normalization_parameters block', () => {
    render(<ExperimentsRunnerPage />)
    const pre = document.querySelector('pre')
    expect(pre.textContent).toContain('normalization_parameters:')
  })

  it('YAML preview contains train_dataset and val_dataset blocks', () => {
    render(<ExperimentsRunnerPage />)
    const pre = document.querySelector('pre')
    expect(pre.textContent).toContain('train_dataset:')
    expect(pre.textContent).toContain('val_dataset:')
  })
})
