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

- Files in `public/static` e.g. `public/static/file.ext` are accessible in source files using `/static/file.ext`
- `src/` is the root directory of all React source files
  - `src/components/` contains reusable React components with accompanying stylesheets written in SCSS
  - `src/pages/` contains React source files for each page
  - `src/styles` contains stylesheets for required pages. Contains stylesheets which define color and typography variables
  - `main.js` main entrypoint for React app. Associated stylesheet (for whole app) which handles all includes: `main.scss`
  - `app.js` contains source code of `App` which is loaded by `main.js`. Contains React Router and maintains routes, sidebar and OUO banner. Loads different pages based on specified route
  - `index.html` is the HTML entrypoint of the webapp. `main.js` mounts a container using `#mount` in `index.html`


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


