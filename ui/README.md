# Thorium Frontend

Thorium's web application is a React 19 + TypeScript SPA built with Vite and styled-components.

## Installation

Install the required node packages:
```bash
npm install
```

## Development

### Running locally

Point the frontend at a Thorium API instance and start the dev server:
```bash
export THORIUM_API_URL=https://[thorium.DOMAIN]
npm run dev
```

The dev server runs on port 8000 by default.

### Local dev environment (minithor)

For a self-contained development cluster, use minithor (see root `CLAUDE.md` for full instructions):

**First-time setup:**
```bash
minithor/minithor minikube install  # install minikube and start the k8s cluster
minithor/minithor deploy            # deploy Thorium services (creates test user test/INSECURE_DEV_PASSWORD)
minithor/minithor expose            # port-forward API to localhost:8080
export THORIUM_API_URL=http://localhost:8080
npm run dev
```

**After a reboot or `minithor stop`:**
```bash
minithor/minithor start             # restart the cluster
minithor/minithor expose            # re-establish port-forwards
export THORIUM_API_URL=http://localhost:8080
npm run dev
```

## Project Structure

- [index.html](./index.html): Entrypoint HTML file
- [public](./public): Static files served from root path (e.g. `/ferris-scientist.png`)
- [src/](./src): Project source code
  - [main.tsx](./src/main.tsx): React entrypoint — loads global styles and mounts `<Thorium/>`
  - [Thorium.tsx](./src/Thorium.tsx): Root component with routes, error handling, and auth
  - [assets](./src/assets): Static assets imported by components (icons, banner text)
  - [components](./src/components): All non-page UI components
  - [models](./src/models): TypeScript interfaces, types, and enums for API data shapes
  - [pages](./src/pages): Route-level page components
  - [styles](./src/styles): Global SCSS styles — theme colors, spacing, and library overrides
  - [thorpi](./src/thorpi): Thorium API client (axios-based, one module per resource)
  - [utilities](./src/utilities): Shared TypeScript utility functions
- [e2e/](./e2e): Playwright E2E tests (see [TESTING.md](e2e/TESTING.md))
- [vite.config.ts](vite.config.ts): Vite configuration including path aliases (`@pages`, `@components`, etc.)
- [mitre_tags](mitre_tags): Static MBC and ATT&CK tag data for tag select dropdowns

## Formatting and Linting

Run the formatter and linter before committing:

```bash
npm run format             # prettier write
npm run validate-style     # prettier check (CI-safe)
npm run lint               # eslint check
npm run fix                # eslint --fix
```

## Building

```bash
npm run build              # production build (tsc + vite)
npm run build-check        # full type-check including .jsx checkJs
```

To verify the production bundle locally:
```bash
npm run build-preview
npm run preview
```

## Testing

### Unit Tests (Vitest)

```bash
npm run test               # single run
npm run test:watch         # watch mode
```

### E2E Tests (Playwright)

E2E tests run against a live Thorium instance. See [e2e/TESTING.md](e2e/TESTING.md) for full details.

```bash
THORIUM_API_URL=http://localhost:8080 npm run test:e2e           # headless
THORIUM_API_URL=http://localhost:8080 npm run test:e2e:headed    # visible browser
```
