import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  resolve: {
    extensions: ['.js', '.ts', '.tsx', '.jsx'],
    alias: {
      '@assets': fileURLToPath(new URL('./src/assets', import.meta.url)),
      '@entities': fileURLToPath(new URL('./src/components/entities', import.meta.url)),
      '@styles': fileURLToPath(new URL('./src/styles', import.meta.url)),
      '@components': fileURLToPath(new URL('./src/components', import.meta.url)),
      '@models': fileURLToPath(new URL('./src/models', import.meta.url)),
      '@pages': fileURLToPath(new URL('./src/pages', import.meta.url)),
      '@utilities': fileURLToPath(new URL('./src/utilities', import.meta.url)),
      '@thorpi': fileURLToPath(new URL('./src/thorpi', import.meta.url)),
    },
  },
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'node',
    setupFiles: ['./vitest.setup.ts'],
    testTimeout: 30000,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      include: ['src/utilities/**', 'src/models/**'],
    },
  },
});
