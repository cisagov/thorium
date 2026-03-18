import { useContext } from 'react';

// project imports
import { WindowManagerContext } from './WindowManagerContext';

/**
 * Use hook for getting window manager context
 */
export const useWindowManager = () => {
  const context = useContext(WindowManagerContext);
  if (context === undefined) {
    throw new Error('useWindowManager must be used within a WindowManager');
  }
  return context;
};
