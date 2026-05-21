import '@testing-library/jest-dom'

// jsdom doesn't implement URL.createObjectURL — stub it so download tests don't throw
global.URL.createObjectURL = vi.fn(() => 'blob:mock')
global.URL.revokeObjectURL = vi.fn()
