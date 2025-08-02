import { defineConfig, loadEnv} from 'vite'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import eslintPlugin from 'vite-plugin-eslint'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // eslint-disable-next-line no-undef
  const env = loadEnv(mode, process.cwd(), '');
  return {
    //build: {
    //  sourcemap: true, // remove after done debugging prod builds
    //},
    resolve: {
      extensions: ['.js', '.ts', '.tsx', '.jsx', 'scss'],
      alias: {
        "@assets": fileURLToPath(new URL("./src/assets", import.meta.url)),
        "@client": fileURLToPath(new URL("./src/client", import.meta.url)),
        "@components": fileURLToPath(new URL("./src/components", import.meta.url)),
        "@models": fileURLToPath(new URL("./src/models", import.meta.url)),
        "@pages": fileURLToPath(new URL("./src/pages", import.meta.url)),
        "@styles": fileURLToPath(new URL("./src/styles", import.meta.url)),
        "@utilities": fileURLToPath(new URL("./src/utilities", import.meta.url)),
        "@thorpi": fileURLToPath(new URL("./src/thorpi", import.meta.url)),
      },
    },
    css: {
      preprocessorOptions: {
        scss: {
          api: 'modern'
      }
      }
    },
    assetsInclude: ['src/assets/*.txt', 'mitre_tags/*.tags'],
    plugins: [
      react(),
      //eslintPlugin({
      //  cache: false,
      //  include: ['./src/**/*.js', './src/**/*.jsx', './src/**/*.ts', './src/**/*/tsx']
      //})
    ],
    server: {
      port: 8000,
      strictPort: true,
      cors: false,
      headers: {
        "Access-Control-Allow-Origin": "*",
      },
    },
    preview: {
      cors: false,
      strictPort: false,
      headers: {
        "Access-Control-Allow-Origin": "*",
      },
    },
    define: {
      'process.env.REACT_APP_API_URL': env && env.REACT_APP_API_URL ? JSON.stringify(env.REACT_APP_API_URL) : JSON.stringify("http://127.0.0.1:80"),
      'process.env.VERSION': JSON.stringify(env.npm_package_version),
    }
  };
});
