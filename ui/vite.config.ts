import { defineConfig, loadEnv} from 'vite'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import fs from "node:fs";
import path from "node:path";
//import { visualizer } from 'rollup-plugin-visualizer'
//import eslintPlugin from 'vite-plugin-eslint'

type ChunkMap = Record<string, string[]>;

function loadChunkMap(relPath: string): ChunkMap {
  const p = path.resolve(__dirname, relPath);
  const raw = fs.readFileSync(p, "utf8");
  return JSON.parse(raw) as ChunkMap;
}

// Build an index: packageName -> chunkName
function invertChunkMap(chunkMap: ChunkMap): Map<string, string> {
  const packageToChunk = new Map<string, string>();
  for (const [chunkName, packages] of Object.entries(chunkMap)) {
    for (const pkg of packages) {
      if (packageToChunk.has(pkg) && packageToChunk.get(pkg) !== chunkName) {
        throw new Error(
          `Package "${pkg}" is assigned to multiple chunks: ` +
          `"${packageToChunk.get(pkg)}" and "${chunkName}".`
        );
      }
      packageToChunk.set(pkg, chunkName);
    }
  }
  return packageToChunk;
}

// Extract package name from a node_modules path.
// Examples:
//   .../node_modules/react/index.js           -> react
//   .../node_modules/@babel/runtime/helpers   -> @babel/runtime
function packageNameFromId(id: string): string | null {
  const module = id.lastIndexOf("/node_modules/");
  if (module === -1) return null;
  const rest = id.slice(module + "/node_modules/".length);
  const parts = rest.split("/");
  if (parts[0]?.startsWith("@")) {
    if (parts.length < 2) return null;
    return `${parts[0]}/${parts[1]}`;
  }
  return parts[0] ?? null;
}

// load in package/chunk name map
const chunkMap = loadChunkMap("./bundle/chunks.json");
// invert chunk list to map of package -> chunk 
const chunkPackageMap = invertChunkMap(chunkMap);

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    build: {
      sourcemap: false,
      minify: true,
      modulePreload: true,
      rollupOptions: {
        output: {
          manualChunks(id) {
            // preload-helper is not in the node_modules
            if (id.includes("vite/preload-helper")) return "vendors";
            if (!id.includes('node_modules')) return;
            //if (id.includes("domhandler")) return "vendors"; // chunk with sanitize-html still loads from index-*.js
            if (id.includes("/node_modules/@babel/runtime/")) return "vendors";
            const pkg = packageNameFromId(id);
            if (!pkg) return;
            const chunk = chunkPackageMap.get(pkg);
            if (chunk == 'vendors') return;

            return chunk;
          },
        },
      },
    },
    resolve: {
      extensions: ['.js', '.ts', '.tsx', '.jsx', '.scss', '.ttf'],
      alias: {
        "@assets": fileURLToPath(new URL("./src/assets", import.meta.url)),
        "@entities": fileURLToPath(new URL("./src/components/entities", import.meta.url)),
        "@styles": fileURLToPath(new URL("./src/styles", import.meta.url)),
        "@components": fileURLToPath(new URL("./src/components", import.meta.url)),
        "@models": fileURLToPath(new URL("./src/models", import.meta.url)),
        "@pages": fileURLToPath(new URL("./src/pages", import.meta.url)),
        "@utilities": fileURLToPath(new URL("./src/utilities", import.meta.url)),
        "@thorpi": fileURLToPath(new URL("./src/thorpi", import.meta.url))
      },
    },
    css: {
      preprocessorOptions: {
        scss: {
          // hides warnings from deps on build especially deprecated bootsrap.scss
          quietDeps: true,
        },
      },
    },
    assetsInclude: ['src/assets/*.txt', 'mitre_tags/*.tags'],
    plugins: [
      react({
        babel: {
          plugins: [
            ['babel-plugin-react-compiler', {}],
          ],
        },
      }),
      // use for debugging bundle size and chunking, remove for production builds
      //visualizer({ filename: 'dist/stats.html', gzipSize: true, brotliSize: true }),
      //eslintPlugin({
      //  cache: false,
      //  include: ['./src/**/*.js', './src/**/*.jsx', './src/**/*.ts', './src/**/*.tsx']
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
      'process.env.REACT_APP_API_URL': env && env.REACT_APP_API_URL ? JSON.stringify(env.REACT_APP_API_URL) : undefined,
    }
  };
});
