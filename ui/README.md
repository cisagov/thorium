# Thorium Frontend

Built using React and served with Vite.


## Installation

Install the required node packages by running,
```bash
npm install
```

## Running Webapp locally on CLI
For development, run the webapp locally using,
```bash
export REACT_APP_API_URL=https://[thorium.DOMAIN]/api
npm run dev
```

## Project Structure

- [index.html](./index.html): Entrypoint HTML file for the Thorium web application
- [public](./public): Files that are hosted from the root path of the site (e.g. `/ferris-scientist.png`)
- [src/](./src): Project source code and imported assets
  - [assets](./src/assets): Static assets imported by site components
  - [components](./src/components): All non-page UI components
  - [main.tsx](./src/main.tsx): Entrypoint for the React app. This loads global styles and an instance of <Thorium/>.
  - [models](./src/models): Thorium data structures as Typescript interfaces, types, and enums
  - [pages](./src/pages): Thorium Site pages
  - [styles](./src/styles): Global styles including theme colors, spacing, and bootstrap component overrides for the theme.
  - [thorium.tsx](./src/thorium.tsx): Root site component that includes routes, global error handling and auth
  - [thorpi](./src/thorpi): Thorium API client
  - [utilities](./src/utilities): Typescript utility functions
- [vite.config.ts](vite.config.ts): Vite project configuration including import aliases (`@pages`)
- [mitre_tags](mitre_tags): Static MBC and ATT&CK tags list used for Tag select options dropdowns and associated crawl scripts

## Formatting and Testing

You should always run the formatter before committing code for final review:

```bash
npm run format
```

You can also lint your code to help identify potential issues:

```bash
npm run lint
```

Once you are confident your code works as intended, run the preview version of the UI. This should be much closer to a production release and help identify issues that crop up during the bundling process. Sometimes the dev server will have access to static files and other assets that the final bundle won't. This should catch these types of issues.

```bash
npm run build-preview
npm run preview
```


