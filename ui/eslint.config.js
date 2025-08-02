import { globalIgnores } from "eslint/config";
import globals from 'globals'
import js from "@eslint/js";
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from 'typescript-eslint';
import prettier from 'eslint-plugin-prettier';
import eslint from '@eslint/js';
import tsParser from '@typescript-eslint/parser';

export default tseslint.config(
  { ignores: ['**/node_modules/', 'dist/*', '**/*.css', '**/*.scss'] },
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  //...tseslint.configs.strictTypeChecked,

  globalIgnores(['dist/', 'node_modules/']),
  {
    languageOptions: {
      globals: {
        ...globals.node,
        ...globals.browser,
      },
      ecmaVersion: 'latest',
      sourceType: "module",
      parser: tsParser,
      parserOptions: {
        project: ['**/tsconfig.json', '**/tsconfig.lint.json'],
      }
    }
  },
  {
    files: ['**/*.{ts,tsx}'],
    plugins: {
      js, react, reactHooks, reactRefresh, tseslint, prettier,
    },
    rules: {
      'max-len': ['error', 140, 2, {ignoreUrls: true}],
      "@typescript-eslint/no-explicit-any": "off",
    }
  },
);
