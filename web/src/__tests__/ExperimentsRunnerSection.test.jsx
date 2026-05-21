import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ExperimentsRunnerSection from '../components/ExperimentsRunnerSection'

const BASE_RUNNER = {
  outputBaseDir: '/tmp/experiments',
  seeds: [42, 101, 28],
  seedsText: '42, 101, 28',
  useNRuns: false,
  nRuns: 3,
  saveSummary: true,
  resume: false,
}

describe('ExperimentsRunnerSection rendering', () => {
  it('renders the section heading', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.getByRole('heading', { name: 'Experiments Runner' })).toBeInTheDocument()
  })

  it('renders output_base_dir input with current value', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.getByDisplayValue('/tmp/experiments')).toBeInTheDocument()
  })

  it('renders "Explicit seeds" and "Random n_runs" radio options', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.getByText('Explicit seeds')).toBeInTheDocument()
    expect(screen.getByText('Random n_runs')).toBeInTheDocument()
  })

  it('shows seeds text input when useNRuns is false', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.getByDisplayValue('42, 101, 28')).toBeInTheDocument()
  })

  it('hides seeds text input when useNRuns is true', () => {
    const runner = { ...BASE_RUNNER, useNRuns: true, seeds: [], seedsText: '' }
    render(<ExperimentsRunnerSection runner={runner} onChange={vi.fn()} />)
    expect(screen.queryByPlaceholderText('42, 101, 28')).not.toBeInTheDocument()
  })

  it('shows n_runs number input when useNRuns is true', () => {
    const runner = { ...BASE_RUNNER, useNRuns: true, seeds: [], seedsText: '' }
    render(<ExperimentsRunnerSection runner={runner} onChange={vi.fn()} />)
    expect(screen.getByLabelText('n_runs')).toBeInTheDocument()
  })

  it('hides n_runs input when useNRuns is false', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.queryByLabelText('n_runs')).not.toBeInTheDocument()
  })

  it('renders save_summary and resume checkboxes', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.getByLabelText('save_summary')).toBeInTheDocument()
    expect(screen.getByLabelText('resume')).toBeInTheDocument()
  })

  it('save_summary checkbox reflects current state', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.getByLabelText('save_summary')).toBeChecked()
  })

  it('resume checkbox reflects current state', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.getByLabelText('resume')).not.toBeChecked()
  })

  it('shows parsed seeds hint text', () => {
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={vi.fn()} />)
    expect(screen.getByText(/3 run\(s\): \[42, 101, 28\]/)).toBeInTheDocument()
  })
})

describe('ExperimentsRunnerSection interactions', () => {
  it('calls onChange with updated outputBaseDir', () => {
    const handleChange = vi.fn()
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={handleChange} />)

    const input = screen.getByDisplayValue('/tmp/experiments')
    fireEvent.change(input, { target: { value: '/data/runs' } })

    expect(handleChange).toHaveBeenCalledOnce()
    expect(handleChange.mock.calls[0][0].outputBaseDir).toBe('/data/runs')
  })

  it('switching to "Random n_runs" sets useNRuns true', () => {
    const handleChange = vi.fn()
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={handleChange} />)

    const radios = screen.getAllByRole('radio')
    const nRunsRadio = radios[1]
    fireEvent.click(nRunsRadio)

    expect(handleChange).toHaveBeenCalledOnce()
    expect(handleChange.mock.calls[0][0].useNRuns).toBe(true)
  })

  it('switching to "Explicit seeds" sets useNRuns false', () => {
    const runner = { ...BASE_RUNNER, useNRuns: true, seeds: [], seedsText: '' }
    const handleChange = vi.fn()
    render(<ExperimentsRunnerSection runner={runner} onChange={handleChange} />)

    const radios = screen.getAllByRole('radio')
    const seedsRadio = radios[0]
    fireEvent.click(seedsRadio)

    expect(handleChange).toHaveBeenCalledOnce()
    expect(handleChange.mock.calls[0][0].useNRuns).toBe(false)
  })

  it('typing seeds text parses comma-separated integers into seeds array', () => {
    const handleChange = vi.fn()
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={handleChange} />)

    const input = screen.getByDisplayValue('42, 101, 28')
    fireEvent.change(input, { target: { value: '1, 2, 3, 4' } })

    expect(handleChange).toHaveBeenCalledOnce()
    const updated = handleChange.mock.calls[0][0]
    expect(updated.seeds).toEqual([1, 2, 3, 4])
    expect(updated.seedsText).toBe('1, 2, 3, 4')
    expect(updated.useNRuns).toBe(false)
  })

  it('seeds parser ignores non-numeric tokens', () => {
    const handleChange = vi.fn()
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={handleChange} />)

    const input = screen.getByDisplayValue('42, 101, 28')
    fireEvent.change(input, { target: { value: '10, abc, 20' } })

    const updated = handleChange.mock.calls[0][0]
    expect(updated.seeds).toEqual([10, 20])
  })

  it('toggling save_summary calls onChange with updated value', () => {
    const handleChange = vi.fn()
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={handleChange} />)

    const cb = screen.getByLabelText('save_summary')
    fireEvent.click(cb)

    expect(handleChange).toHaveBeenCalledOnce()
    expect(handleChange.mock.calls[0][0].saveSummary).toBe(false)
  })

  it('toggling resume calls onChange with updated value', () => {
    const handleChange = vi.fn()
    render(<ExperimentsRunnerSection runner={BASE_RUNNER} onChange={handleChange} />)

    const cb = screen.getByLabelText('resume')
    fireEvent.click(cb)

    expect(handleChange).toHaveBeenCalledOnce()
    expect(handleChange.mock.calls[0][0].resume).toBe(true)
  })

  it('changing n_runs value calls onChange', () => {
    const runner = { ...BASE_RUNNER, useNRuns: true, seeds: [], seedsText: '' }
    const handleChange = vi.fn()
    render(<ExperimentsRunnerSection runner={runner} onChange={handleChange} />)

    const input = screen.getByLabelText('n_runs')
    fireEvent.change(input, { target: { value: '5' } })

    expect(handleChange).toHaveBeenCalledOnce()
    expect(handleChange.mock.calls[0][0].nRuns).toBe(5)
    expect(handleChange.mock.calls[0][0].useNRuns).toBe(true)
  })
})
