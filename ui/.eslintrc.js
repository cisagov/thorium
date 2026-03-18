module.exports = {
  'env': {
    'browser': true,
    'es2021': true,
  },
  'extends': [
    'eslint:recommended',
    'plugin:jsdoc/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:import/errors',
    'plugin:import/warnings',
    'plugin:@typescript-eslint/recommended',
    'prettier',
  ],
  'parser': '@typescript-eslint/parser',
  'parserOptions': {
    'ecmaFeatures': {
      'jsx': true,
    },
    'ecmaVersion': 12,
    'sourceType': 'module',
  },
  'plugins': ['react', 'jsdoc', 'react-hooks', 'import', '@typescript-eslint'],
  'rules': {
    'max-len': ['error', 100, 2, {ignoreUrls: true}],
    'react/prop-types': 'off',
    'react/react-in-jsx-scope': 'off',
    'valid-jsdoc': [2, {
      'prefer': {
        'return': 'returns',
      },
    }],
    '@typescript-eslint/explicit-module-boundary-types': 'off',
  },
  'settings': {
    'react': {
      'version': 'detect',
    },
    'import/resolver': {
      'typescript': true,
      'node': true,
    },
  },
  'overrides': [
    {
      'files': ['*.ts', '*.tsx'],
      'parser': '@typescript-eslint/parser',
      'parserOptions': {
        'project': ['./tsconfig.json'],
      },
    },
  ],
};
