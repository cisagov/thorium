import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'core-js/stable';
import 'regenerator-runtime/runtime';

// project imports
import Thorium from './thorium';
import '@styles/main.scss';

createRoot(document.getElementById('thorium')!).render(
  <StrictMode>
    <Thorium />
  </StrictMode>,
);
