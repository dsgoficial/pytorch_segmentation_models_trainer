import '@testing-library/jest-dom'
import { vi } from 'vitest'

// jsdom doesn't implement URL.createObjectURL — stub it so download tests don't throw
globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock')
globalThis.URL.revokeObjectURL = vi.fn()
