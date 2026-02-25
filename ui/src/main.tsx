import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'core-js/stable';

// project imports
import Thorium from './Thorium';
import '@styles/main.scss';

createRoot(document.getElementById('thorium')!).render(
  <StrictMode>
    <Thorium />
  </StrictMode>,
);
