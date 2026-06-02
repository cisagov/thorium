import { useContext } from 'react';

// project imports
import { WindowManagerContext } from './WindowManagerContext';

export const useWindowManager = () => {
  const context = useContext(WindowManagerContext);
  if (context === undefined) {
    throw new Error('useWindowManager must be used within a WindowManager');
  }
  return context;
};
