import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App'

describe('App', () => {
  it('renders the app title', () => {
    render(<App />)
    expect(screen.getByText(/Config Builder/)).toBeInTheDocument()
  })

  it('renders all three navigation tabs', () => {
    render(<App />)
    expect(screen.getByRole('button', { name: 'Training' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Predict' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Experiments Runner' })).toBeInTheDocument()
  })

  it('shows Training page content by default', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Model' })).toBeInTheDocument()
  })

  it('switches to Experiments Runner page when tab is clicked', async () => {
    render(<App />)
    await userEvent.click(screen.getByRole('button', { name: 'Experiments Runner' }))
    expect(screen.getByText('mode: run-experiments')).toBeInTheDocument()
  })

  it('switches back to Training page from Experiments Runner', async () => {
    render(<App />)
    await userEvent.click(screen.getByRole('button', { name: 'Experiments Runner' }))
    await userEvent.click(screen.getByRole('button', { name: 'Training' }))
    expect(screen.queryByText('mode: run-experiments')).not.toBeInTheDocument()
  })

  it('renders GitHub and Docs links', () => {
    render(<App />)
    expect(screen.getByTitle('GitHub')).toBeInTheDocument()
    expect(screen.getByTitle('Documentation')).toBeInTheDocument()
  })
})
